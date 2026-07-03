"""Corpus progress reporting for EM learning (Rich with tqdm fallback)."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


class LearnProgressReporter:
    """Track corpus example processing with elapsed time and ETA."""

    def __init__(self, total: int, *, enabled: bool = True) -> None:
        """Prepare a reporter for *total* corpus examples."""
        self._total = total
        self._enabled = enabled and total > 0
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._tqdm = None
        self._completed = 0

    def __enter__(self) -> LearnProgressReporter:
        """Start the progress display."""
        if not self._enabled:
            return self
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("|"),
                TimeElapsedColumn(),
                TextColumn("|"),
                TimeRemainingColumn(),
                expand=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("Learning", total=self._total)
        except ImportError:
            from tqdm import tqdm

            self._tqdm = tqdm(total=self._total, unit="ex", dynamic_ncols=True)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop and tear down the progress display."""
        _ = (exc_type, exc, tb)
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

    def begin_example(self, label: str) -> None:
        """Update the display to show the example currently being processed."""
        if not self._enabled:
            return
        text = _truncate_label(label)
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                description=f"[{self._completed + 1}/{self._total}] {text}",
            )
        elif self._tqdm is not None:
            self._tqdm.set_description(f"[{self._completed + 1}/{self._total}] {text}")

    def finish_example(self) -> None:
        """Advance after one corpus example has been processed."""
        if not self._enabled:
            return
        self._completed += 1
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)
        elif self._tqdm is not None:
            self._tqdm.update(1)


def _truncate_label(label: str, max_len: int = 60) -> str:
    """Shorten *label* for narrow terminal columns."""
    stripped = " ".join(label.split())
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1] + "…"
