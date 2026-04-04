"""Dialogue context with DAG state (partial Java `Context`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

from dylan.action.speech_act_inference_grammar import SpeechActInferenceGrammar
from dylan.dag.dag_edge import DAGEdge
from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.word_level_context_dag import WordLevelContextDAG

T = TypeVar("T", bound=DAGTuple)
E = TypeVar("E", bound=DAGEdge)


class Context(Generic[T, E]):
    """Holds participants, floor state, and the word-level DAG (Eshghi et al. 2015)."""

    def __init__(
        self,
        dag: WordLevelContextDAG,
        sa_grammar: SpeechActInferenceGrammar | None = None,
        *participants: str,
    ) -> None:
        """Wire a fresh DAG, participants, and optional SA grammar."""
        self._dag: WordLevelContextDAG = dag
        dag.set_context(self)
        self.my_name = participants[0] if participants else "Dylan"
        self.who_has_floor: str | None = None
        self._participants: set[str] = set(participants) if participants else {self.my_name}
        self.sa_inf_grammar = sa_grammar or SpeechActInferenceGrammar(Path("."))
        self._dialogue_words: list[Any] = []

    def get_name(self) -> str:
        """Return the default agent name for this context."""
        return self.my_name

    def get_dag(self) -> WordLevelContextDAG:
        """Return the word-level context DAG."""
        return self._dag

    def set_dag(self, dag: WordLevelContextDAG) -> None:
        """Replace the active DAG and attach this context to it."""
        self._dag = dag
        dag.set_context(self)

    def get_participants(self) -> set[str]:
        """Return dialogue participant ids."""
        return set(self._participants)

    def get_current_tuple(self) -> DAGTuple:
        """Return the tuple under the DAG cursor."""
        return self._dag.get_current_tuple()

    def floor_is_open(self) -> bool:
        """Return True when no participant holds the floor."""
        return self.who_has_floor is None

    def open_floor(self) -> None:
        """Mark the floor as unassigned."""
        self.who_has_floor = None

    def set_who_has_floor(self, speaker: str) -> None:
        """Assign the conversational floor to `speaker`."""
        self.who_has_floor = speaker

    def set_repair_processing(self, repairing: bool) -> None:
        """Enable or disable repair handling on the DAG."""
        self._dag.set_repair_processing(repairing)

    def repair_initiated(self) -> bool:
        """Return True when the top stack token is the repair-init marker."""
        return self._dag.repair_initiated()

    def append_word(self, w: Any) -> None:
        """Record a word in dialogue history (Java `Context.appendWord`)."""
        self._dialogue_words.append(w)

    def init(self) -> None:
        """Reset DAG and variable pools (simplified vs Java `Context.init`)."""
        self._dag = WordLevelContextDAG()
        self._dag.set_context(self)
        self._dialogue_words.clear()

    def init_participants(self, participants: list[str]) -> None:
        """Reset participant set and rebuild an empty DAG."""
        self._participants = set(participants)
        self.my_name = participants[0] if participants else "Dylan"
        self.init()
