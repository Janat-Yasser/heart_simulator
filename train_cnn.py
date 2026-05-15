"""
Phase 4: CNN Trainer on MIT-BIH Arrhythmia Database
Run this script once to download data and train the classifier.
The trained model is saved to ai/ecg_cnn.pt and auto-loaded by the app.

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
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    SK_OK = True
except ImportError:
    SK_OK = False


# ── Beat label mapping (AAMI standard) ──────────────────────────────────────
BEAT_MAP = {
    'N': 'Normal',   'L': 'LBBB',     'R': 'RBBB',
    'A': 'APB',      'a': 'AberAPB',  'J': 'NodJunct',
    'S': 'SV',       'V': 'PVC',      'F': 'FusionNV',
    'e': 'AtrEsc',   'j': 'NodEsc',   '/': 'Paced',
    'Q': 'Unclass',
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_mitbih(n_records=10, seg_len=180, data_dir='data/mitbih'):
    """
    Download and segment MIT-BIH records.
    Each beat is extracted as seg_len samples centred on the R-peak.
    Returns X (n_beats, seg_len), y (n_beats,) label strings.
    """
    if not (WFDB_OK and SK_OK):
        raise RuntimeError("wfdb and scikit-learn required for training")

    os.makedirs(data_dir, exist_ok=True)
    records = [
        '100','101','102','103','104','105','106','107','108','109',
        '111','112','113','114','115','116','117','118','119','121',
    ][:n_records]

    X, y = [], []
    for rec in records:
        print(f"  Loading record {rec}…", end='', flush=True)
        try:
            record = wfdb.rdrecord(rec, pn_dir='mitdb', sampto=None)
            ann    = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
            sig    = record.p_signal[:, 0]  # Lead I
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

def train(n_records=20, epochs=25, batch_size=256):
    if not TORCH_OK:
        return

    print("Loading MIT-BIH data…")
    X, y_str = load_mitbih(n_records=n_records)
    print(f"Total beats: {len(X)}")

    le = LabelEncoder()
    y  = le.fit_transform(y_str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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
    loss_fn = nn.CrossEntropyLoss()

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
    print("\nTest set results:")
    print(classification_report(all_true, all_pred, target_names=le.classes_))

    # Save
    save_path = 'ai/ecg_cnn.pt'
    torch.save({'model': model.state_dict(), 'classes': list(le.classes_)}, save_path)
    print(f"\nModel saved → {save_path}")
    print("Restart the app — it will auto-load the trained model.")


if __name__ == '__main__':
    train(n_records=20, epochs=25)
