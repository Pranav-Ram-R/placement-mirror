"""Session state machine: choose question, calibrate, answer, processing, report.

choose_question --select--> calibrate --calibrated--> ready --start--> answering
answering --stop (user, auto stop or end of replay)--> processing --processed--> report

From ready or report the user can pick another question (back to calibrate) or
recalibrate. Any other transition raises InvalidTransition.
"""

from __future__ import annotations

import threading
import time
from enum import Enum


class State(str, Enum):
    CHOOSE_QUESTION = "choose_question"
    CALIBRATE = "calibrate"
    READY = "ready"
    ANSWERING = "answering"
    PROCESSING = "processing"
    REPORT = "report"


ALLOWED = {
    "select": {State.CHOOSE_QUESTION, State.CALIBRATE, State.READY, State.REPORT},
    "recalibrate": {State.READY, State.REPORT},
    "calibrated": {State.CALIBRATE},
    "start": {State.READY},
    "stop": {State.ANSWERING},
    "processed": {State.PROCESSING},
}
TARGET = {
    "select": State.CALIBRATE,
    "recalibrate": State.CALIBRATE,
    "calibrated": State.READY,
    "start": State.ANSWERING,
    "stop": State.PROCESSING,
    "processed": State.REPORT,
}


class InvalidTransition(ValueError):
    pass


class SessionMachine:
    def __init__(self):
        self.state = State.CHOOSE_QUESTION
        self.question: dict | None = None
        self.history: list[tuple[str, str, float]] = [("init", State.CHOOSE_QUESTION.value, time.monotonic())]
        self._lock = threading.Lock()

    def fire(self, event: str, question: dict | None = None) -> State:
        with self._lock:
            if self.state not in ALLOWED[event]:
                raise InvalidTransition(f"cannot {event} in state {self.state.value}")
            if event == "select":
                if question is None:
                    raise InvalidTransition("select needs a question")
                self.question = question
            self.state = TARGET[event]
            self.history.append((event, self.state.value, time.monotonic()))
            return self.state
