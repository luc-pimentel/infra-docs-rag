"""What people said about an answer: one line per signal in `data/feedback.jsonl`, keyed by the trace
the answer wrote. `POST /feedback` on the service records it; the online sampler (stage 19) reads it
as the dissatisfaction signal."""

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

FEEDBACK = Path("data/feedback.jsonl")


class Feedback(BaseModel):
    trace_id: str
    helpful: bool
    note: str = Field(default="", max_length=2000)
    at: str = ""


class FeedbackLog:
    """Append-only JSON lines."""

    def __init__(self, path: Path = FEEDBACK) -> None:
        self.path = path

    def append(self, fb: Feedback) -> Feedback:
        if not fb.at:
            fb = fb.model_copy(update={"at": datetime.now(UTC).isoformat(timespec="seconds")})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            out.write(fb.model_dump_json() + "\n")
        return fb

    def read(self) -> list[Feedback]:
        if not self.path.exists():
            return []
        return [Feedback.model_validate_json(line) for line in self.path.read_text().splitlines() if line]

    def by_trace(self) -> dict[str, Feedback]:
        """The last signal per trace."""
        out: dict[str, Feedback] = {}
        for fb in self.read():
            out[fb.trace_id] = fb
        return out
