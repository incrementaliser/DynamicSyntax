"""Optional speech-act inference rules file (Java `SpeechActInferenceGrammar`)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeechActInferenceGrammar:
    """Loads `speech-act-inference-grammar.txt` when present; otherwise stays empty."""

    FILE_NAME = "speech-act-inference-grammar.txt"

    def __init__(self, resource_dir: str | Path) -> None:
        path = Path(resource_dir) / self.FILE_NAME
        if path.is_file():
            logger.info("Speech-act inference grammar present at %s (rules not loaded in v0)", path)
        else:
            logger.info("No speech act inference file %s; using empty SA grammar.", path)
