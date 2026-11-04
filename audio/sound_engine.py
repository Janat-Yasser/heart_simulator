"""
Audio Engine v3
Supports both synthesised sounds and downloaded audio files.

DOWNLOADED SOUND FILES:
  Place these files in the  assets/sounds/  folder:
    heartbeat.wav   — real heartbeat (lub-dub)   e.g. freesound.org #47935
    beep.wav        — clean monitor beep          e.g. freesound.org #542074
    alarm.wav       — cardiac alarm               e.g. freesound.org #366102

  If the files are not found, synthesised versions are used automatically.
"""

import numpy as np
import threading
import queue
import time
import os

try:
    import pygame
    import pygame.sndarray
    import pygame.mixer
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

SAMPLE_RATE = 44100
_q: queue.Queue = queue.Queue(maxsize=12)
_running = False
_cache: dict = {}

SOUND_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'sounds')

# Public state
_enabled   = True
_mode      = "beep"      # beep | heart | both | mute
_volume    = 0.7
_alarm_on  = False


def init_audio() -> bool:
    global _running
    if not PYGAME_OK:
        return False
    try:
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, 512)
        pygame.mixer.init()
        _running = True
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return True
    except Exception:
        return False


def shutdown():
    global _running
    _running = False
    if PYGAME_OK:
        try: pygame.mixer.quit()
        except: pass


# ── Synthesis ────────────────────────────────────────────────────────────────

def _sine(freq, dur, amp=0.6):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    return amp * np.sin(2 * np.pi * freq * t)

def _env(sig, atk=0.005, rel=0.04):
    n = len(sig)
    a = min(int(SAMPLE_RATE * atk), n // 4)
    r = min(int(SAMPLE_RATE * rel), n // 4)
    e = np.ones(n)
    e[:a]  = np.linspace(0, 1, a)
    e[-r:] = np.linspace(1, 0, r)
    return sig * e

def _synth_beep(freq=880, dur=0.065, amp=0.55):
    return _env(_sine(freq, dur, amp))

def _synth_heart(amp=0.7):
    from scipy.signal import butter, sosfilt
    dur = 0.14
    n   = int(SAMPLE_RATE * dur)
    t   = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n)
    sos   = butter(4, [25, 90], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    base  = sosfilt(sos, noise) * amp
    s1 = np.exp(-((t - 0.022) / 0.016) ** 2)
    s2 = np.exp(-((t - 0.072) / 0.013) ** 2) * 0.6
    return base * (s1 + s2)

def _synth_alarm(dur=1.0):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    f = 880 + 220 * np.sign(np.sin(2 * np.pi * 2.5 * t))
    s = 0.45 * np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    return _env(s, 0.01, 0.05)

def _synth_alarm_pattern(priority='high'):
    """IEC 60601-1-8 style bursts: high = 3+2 pulses at ~990 Hz, medium = 3 pulses at ~660 Hz."""
    f0, gaps = (988, [0, .12, .24, .52, .64]) if priority == 'high' else (660, [0, .20, .40])
    dur = gaps[-1] + 0.16
    out = np.zeros(int(SAMPLE_RATE * dur))
    for g in gaps:
        tone = _env(0.5 * (np.sin(2*np.pi*f0*np.arange(int(SAMPLE_RATE*.1))/SAMPLE_RATE)
                           + 0.3*np.sin(2*np.pi*3*f0*np.arange(int(SAMPLE_RATE*.1))/SAMPLE_RATE)), .008, .03)
        i = int(g * SAMPLE_RATE); out[i:i+len(tone)] += tone
    return out

def _spo2_freq(spo2):
    """Pulse-tone pitch drops about a semitone per SpO2 point below 100 (as on bedside monitors)."""
    return 880 * 2 ** ((spo2 - 99) / 12)

def _to_sound(arr):
    pcm = (np.clip(arr, -1, 1) * 32767 * _volume).astype(np.int16)
    # pygame mixer may be stereo — duplicate mono to both channels
    if pygame.mixer.get_init() and pygame.mixer.get_init()[2] == 2:
        pcm = np.column_stack((pcm, pcm))
    return pygame.sndarray.make_sound(pcm)


def _try_load_file(filename):
    path = os.path.join(SOUND_DIR, filename)
    if os.path.exists(path):
        try:
            return pygame.mixer.Sound(path)
        except:
            pass
    return None


def _build_cache():
    global _cache
    if not PYGAME_OK:
        return

    # Try real files first, fall back to synthesis
    hb = _try_load_file('heartbeat.wav')
    bp = _try_load_file('beep.wav')
    al = _try_load_file('alarm.wav')

    _cache = {
        'beep_normal': bp  or _to_sound(_synth_beep(880,  0.065, 0.55)),
        'beep_brady':       _to_sound(_synth_beep(660,  0.090, 0.50)),
        'beep_tachy':       _to_sound(_synth_beep(1100, 0.045, 0.55)),
        'beep_vtach':       _to_sound(_synth_beep(1320, 0.035, 0.65)),
        'heart':       hb  or _to_sound(_synth_heart()),
        'alarm':       al  or _to_sound(_synth_alarm()),
        'alarm_high':  al  or _to_sound(_synth_alarm_pattern('high')),
        'alarm_medium':     _to_sound(_synth_alarm_pattern('medium')),
    }
    for v in range(70, 101):
        _cache[f'tone_{v}'] = _to_sound(_synth_beep(_spo2_freq(v), 0.06, 0.5))


def _worker():
    _build_cache()
    while _running:
        try:
            cmd = _q.get(timeout=0.1)
            _play_cmd(cmd)
        except queue.Empty:
            pass


def _play_cmd(cmd):
    if not PYGAME_OK or not _cache:
        return
    snd = _cache.get(cmd)
    if snd:
        try: snd.play()
        except: pass


# ── Public API ───────────────────────────────────────────────────────────────

def set_mode(mode: str):    global _mode;    _mode = mode
def set_enabled(v: bool):   global _enabled; _enabled = v
def set_volume(v: float):   global _volume;  _volume = max(0.0, min(1.0, v))


def beat(rhythm_type, spo2=None):
    """One QRS tone. With spo2 given, the pitch tracks saturation like a real monitor."""
    if not _enabled or not PYGAME_OK or _mode == 'mute':
        return
    from core.ecg_generator import RhythmType
    if spo2 is not None:                         bkey = f'tone_{max(70, min(100, int(spo2)))}'
    elif rhythm_type == RhythmType.VTACH:       bkey = 'beep_vtach'
    elif rhythm_type == RhythmType.BRADYCARDIA: bkey = 'beep_brady'
    elif rhythm_type == RhythmType.TACHYCARDIA: bkey = 'beep_tachy'
    else:                                        bkey = 'beep_normal'

    if _mode in ('beep',  'both'): _enq(bkey)
    if _mode in ('heart', 'both'): _enq('heart')


_alarm_thread = None
_alarm_priority = 'high'

def start_alarm(priority='high'):
    """Loop an alarm pattern: 'high' repeats every ~2.5 s, 'medium' every ~6 s."""
    global _alarm_on, _alarm_thread, _alarm_priority
    _alarm_priority = priority
    if _alarm_on: return
    _alarm_on = True
    def _loop():
        while _alarm_on and PYGAME_OK and _enabled:
            pr = _alarm_priority
            _enq('alarm_' + pr)
            for _ in range(25 if pr == 'high' else 60):
                if not _alarm_on or _alarm_priority != pr: break
                time.sleep(0.1)
    _alarm_thread = threading.Thread(target=_loop, daemon=True)
    _alarm_thread.start()

def stop_alarm():
    global _alarm_on
    _alarm_on = False


def _enq(cmd):
    try: _q.put_nowait(cmd)
    except queue.Full: pass
