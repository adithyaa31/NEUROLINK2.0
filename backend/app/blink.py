"""Debounced temporal OPEN -> CLOSED -> OPEN blink-event detector."""
from dataclasses import dataclass


@dataclass
class BlinkEventDetector:
    min_closed_ms: int = 50
    max_closed_ms: int = 500
    cooldown_ms: int = 800
    state: str = "unknown"
    closed_at: float | None = None
    last_event_at: float = float("-inf")

    def update(self, eye_state: str, now_ms: float):
        if eye_state not in {"open", "closed"}: raise ValueError("eye_state must be open or closed")
        if eye_state == "closed":
            if self.state != "closed": self.closed_at = now_ms
            self.state = "closed"
            return False
        event = False
        if self.state == "closed" and self.closed_at is not None:
            duration = now_ms - self.closed_at
            event = self.min_closed_ms <= duration <= self.max_closed_ms and now_ms - self.last_event_at >= self.cooldown_ms
            if event: self.last_event_at = now_ms
        self.state, self.closed_at = "open", None
        return event
