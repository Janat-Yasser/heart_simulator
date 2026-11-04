"""
Monitor alarms, ranked the way bedside monitors rank them (IEC 60601-1-8):
red = high priority (act now), yellow = medium (attend soon), cyan = advisory.

Alarms come from what the analysis *measured* on screen, not from the rhythm
the simulator was told to generate, so they behave like a real monitor's:
they appear a few seconds after a rhythm starts and they can be wrong.
"""

from dataclasses import dataclass

RED, YELLOW, ADVISORY = 2, 1, 0

HR_LOW, HR_HIGH = 50, 120          # adjustable limits, shown in the HR box
SPO2_LOW = 90


@dataclass(frozen=True)
class Alarm:
    priority: int
    text: str

    @property
    def sounds(self) -> bool:
        return self.priority > ADVISORY


def evaluate(result, spo2=None, hr_low=HR_LOW, hr_high=HR_HIGH):
    """result: ai.classifier.ClassificationResult. Returns alarms, most urgent first."""
    out = []
    if result is None or result.label == 'No Signal':
        return [Alarm(YELLOW, "ECG leads off")]
    hr = result.hr_bpm
    lab = result.label

    if lab == "Ventricular Tachycardia":
        out.append(Alarm(RED, "V-Tach"))
    elif hr < 40:
        out.append(Alarm(RED, f"Extreme brady {hr:.0f}"))
    elif hr > 150:
        out.append(Alarm(RED, f"Extreme tachy {hr:.0f}"))
    elif hr > hr_high:
        out.append(Alarm(YELLOW, f"HR {hr:.0f} > {hr_high}"))
    elif hr < hr_low:
        out.append(Alarm(YELLOW, f"HR {hr:.0f} < {hr_low}"))

    if spo2 is not None and spo2 < 85:
        out.append(Alarm(RED, f"Desat {spo2}"))
    elif spo2 is not None and spo2 < SPO2_LOW:
        out.append(Alarm(YELLOW, f"SpO₂ {spo2} < {SPO2_LOW}"))

    if lab == "Atrial Fibrillation":
        out.append(Alarm(YELLOW, "AFib"))
    if lab == "Premature Ventricular Contractions":
        out.append(Alarm(YELLOW, "PVCs"))
    if lab == "Wide-Complex Tachycardia":
        out.append(Alarm(YELLOW, "Wide QRS tachy"))

    # ST and QT can't be judged on wide (ventricular) complexes
    for a in ([] if result.qrs_wide else result.alerts):
        if a.startswith("ST elevation"):
            out.append(Alarm(YELLOW, f"ST II {result.st_level:+.1f} mV"))
        elif a.startswith("ST depression"):
            out.append(Alarm(YELLOW, f"ST II {result.st_level:+.1f} mV"))
        elif a.startswith("Prolonged QTc"):
            out.append(Alarm(YELLOW, f"QTc {result.qt_ms:.0f} high"))

    if lab == "1st Degree AV Block":
        out.append(Alarm(ADVISORY, f"PR {result.pr_ms:.0f} ms"))

    out.sort(key=lambda a: -a.priority)
    return out
