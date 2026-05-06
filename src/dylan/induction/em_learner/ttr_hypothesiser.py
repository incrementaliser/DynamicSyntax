"""TTR-specific hypothesiser for EM induction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import Word, sentence_from_text
from dylan.induction.em_learner.hypothesiser import Hypothesiser
from dylan.induction.em_learner.tree_hypothesis import TreeHypothesis
from dylan.induction.em_learner.type_lattice import TypeLattice
from dylan.induction.em_learner.type_lattice_increment import TypeLatticeIncrement
from dylan.tree.node_address import NodeAddress
from dylan.type.dstype import DSType


class TTRHypothesiser(Hypothesiser):
    """Hypothesiser that derives target abstraction trees from TTR semantics."""

    def __init__(
        self,
        resource_dir_or_url: str | Path | None = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
    ) -> None:
        """Create a TTR hypothesiser."""
        super().__init__(resource_dir_or_url, top_n, load_learnt_lexicon)
        self.target_type: TTRRecordType | None = None
        self.lattice: TypeLattice | None = None
        self.all_words: list[Word] = []

    def load_training_example(self, sentence: str | Iterable[str | Word], target: TTRRecordType) -> None:
        """Load a sentence paired with target TTR semantics."""
        super().load_training_example(sentence, target)
        self.all_words = sentence_from_text(sentence) if isinstance(sentence, str) else [Word(str(w)) for w in sentence]
        self.target_type = target
        self.lattice = TypeLattice(target)
        self.initialise()

    def initialise(self) -> None:
        """Initialise tree-hypothesis children from target lattice increments."""
        if self.target_type is None or self.lattice is None:
            return
        for increment_set in self.lattice.get_increments(self.target_type.get_head_field().label):
            whole_increment = self.flatten(increment_set)
            head_field = whole_increment.get_head_field()
            filtered = head_field is not None and head_field.ds_type == DSType.es
            trees = whole_increment.get_maximal_filtered_abstractions(NodeAddress(), DSType.t, filtered)
            for tree in trees:
                child = self.state.get_new_tuple(self.state.get_current_tuple().get_tree().clone())
                self.state.add_child(child, TreeHypothesis(list(increment_set), tree))

    @staticmethod
    def flatten(increments: Iterable[TypeLatticeIncrement]) -> TTRRecordType:
        """Flatten lattice increments into one record type."""
        return TypeLattice.flatten(increments)


TTRHypothesiser.loadTrainingExample = TTRHypothesiser.load_training_example  # type: ignore[attr-defined]
