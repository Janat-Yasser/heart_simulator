"""
Beat classifier (1-D CNN) trained on the MIT-BIH Arrhythmia Database.

Uses the standard INTER-PATIENT split (de Chazal et al., 2004): training and
test beats come from different people, so the reported accuracy reflects how
the model does on a patient it has never seen. (A random beat-level split puts
beats from the same patient in both sets and inflates accuracy.)

Beats are grouped into the 5 AAMI EC57 classes: N, S, V, F, Q.

The trained weights are saved to ai/ecg_cnn.pt. NOTE: the live monitor doesn't
use this model yet; it runs the rule-based analyser in ai/classifier.py.
Integrating the CNN (resampling 500 -> 360 Hz, beat segmentation) is on the roadmap.

Requirements (in addition to main requirements):
    pip install torch wfdb scikit-learn imbalanced-learn

Usage:
    python train_cnn.py
"""

import numpy as np
import os

# ── Check dependencies ───────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("PyTorch not installed. Run: pip install torch")

try:
    import wfdb
    WFDB_OK = True
except ImportError:
    WFDB_OK = False
    print("wfdb not installed. Run: pip install wfdb")

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    SK_OK = True
except ImportError:
    SK_OK = False


# ── Beat labels: MIT-BIH symbols -> AAMI EC57 classes ────────────────────────
BEAT_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',          # normal & bundle branch
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',                    # supraventricular ectopic
    'V': 'V', 'E': 'V',                                        # ventricular ectopic
    'F': 'F',                                                  # fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q',                              # paced / unclassifiable
}

# de Chazal et al. (2004) inter-patient split. Paced records (102, 104, 107, 217)
# are excluded, as is standard.
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
       '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
       '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']


# ── Data loading ─────────────────────────────────────────────────────────────

def load_mitbih(records, seg_len=180, data_dir='data/mitbih'):
    """
    Download (from PhysioNet) and segment MIT-BIH records.
    Each beat is seg_len samples (0.5 s at 360 Hz) centred on its annotated R peak.
    Returns X (n_beats, seg_len), y (n_beats,) AAMI class letters.
    """
    if not (WFDB_OK and SK_OK):
        raise RuntimeError("wfdb and scikit-learn required for training")

    os.makedirs(data_dir, exist_ok=True)

    X, y = [], []
    for rec in records:
        print(f"  Loading record {rec}…", end='', flush=True)
        try:
            record = wfdb.rdrecord(rec, pn_dir='mitdb', sampto=None)
            ann    = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
            # Use the MLII lead by name (record 114 stores it in channel 1)
            ch     = record.sig_name.index('MLII') if 'MLII' in record.sig_name else 0
            sig    = record.p_signal[:, ch]
            half   = seg_len // 2
            for idx, sym in zip(ann.sample, ann.symbol):
                if sym not in BEAT_MAP:
                    continue
                s = idx - half
                e = idx + half
                if s < 0 or e > len(sig):
                    continue
                seg = sig[s:e]
                seg = (seg - seg.mean()) / (seg.std() + 1e-8)  # normalise
                X.append(seg)
                y.append(BEAT_MAP[sym])
            print(f" {len(X)} beats total")
        except Exception as ex:
            print(f" SKIP ({ex})")

    return np.array(X, dtype=np.float32), np.array(y)


# ── 1-D CNN model ─────────────────────────────────────────────────────────────

class ECGNet(nn.Module):
    def __init__(self, n_classes, seg_len=180):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),                                       # → 90
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                                       # → 45
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),                               # → 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)    # (B, 1, L)
        return self.classifier(self.features(x))


# ── Training loop ─────────────────────────────────────────────────────────────

def train(epochs=25, batch_size=256):
    if not TORCH_OK:
        return

    print("Loading MIT-BIH training patients (DS1)…")
    X_tr, ytr_str = load_mitbih(DS1)
    print("Loading MIT-BIH test patients (DS2)…")
    X_te, yte_str = load_mitbih(DS2)
    print(f"Train beats: {len(X_tr)}  |  Test beats: {len(X_te)}  (different patients)")

    le = LabelEncoder()
    le.fit(np.concatenate([ytr_str, yte_str]))
    y_tr, y_te = le.transform(ytr_str), le.transform(yte_str)

    ds_tr = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long))
    ds_te = TensorDataset(torch.tensor(X_te), torch.tensor(y_te, dtype=torch.long))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size)

    model   = ECGNet(n_classes=len(le.classes_))
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Training on {device}  |  {len(le.classes_)} classes")

    opt    = optim.Adam(model.parameters(), lr=1e-3)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)
    # Normal beats outnumber everything else ~10:1, so weight the loss by inverse frequency
    counts  = np.bincount(y_tr, minlength=len(le.classes_)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0, 0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
        sched.step()
        print(f"  Epoch {epoch:2d}/{epochs}  loss={total_loss/n:.4f}  acc={correct/n:.3f}")

    # Evaluate
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            all_pred.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())
    print("\nInter-patient test results (DS2, patients never seen in training):")
    print(classification_report(all_true, all_pred, target_names=le.classes_, zero_division=0))

    # Save
    save_path = 'ai/ecg_cnn.pt'
    torch.save({'model': model.state_dict(), 'classes': list(le.classes_), 'fs': 360, 'seg_len': 180}, save_path)
    print(f"\nWeights saved → {save_path}  (load with ECGNet + load_state_dict)")


if __name__ == '__main__':
    train(epochs=25)
