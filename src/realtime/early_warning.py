from __future__ import annotations

from typing import Sequence


class EarlyWarning:
    """Memicu warning setelah N prediksi kelelahan berturut-turut (thesis 2.3.5)."""

    def __init__(self, threshold: int = 5, fatigue_classes: Sequence[str] = ("light", "heavy")):
        self.threshold = threshold
        self.fatigue_classes = set(fatigue_classes)
        self.counter = 0
        self.active = False

    def update(self, label: str) -> bool:
        if label in self.fatigue_classes:
            self.counter += 1
        else:
            self.counter = 0
        self.active = self.counter >= self.threshold
        return self.active

    def reset(self) -> None:
        self.counter = 0
        self.active = False
