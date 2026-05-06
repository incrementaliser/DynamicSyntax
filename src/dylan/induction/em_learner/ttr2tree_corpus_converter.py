"""Convert TTR corpus targets to abstraction-tree targets."""

from __future__ import annotations

from dylan.induction.em_learner.corpus import Corpus
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.tree.node_address import NodeAddress
from dylan.type.dstype import DSType


class TTR2TreeCorpusConverter:
    """Convert :class:`RecordTypeCorpus` examples into tree-target examples."""

    def convert_corpus(self, corpus: RecordTypeCorpus) -> Corpus[object]:
        """Return a corpus whose targets are first maximal abstraction trees."""
        out: Corpus[object] = Corpus()
        for words, rt in corpus:
            trees = rt.get_maximal_filtered_abstractions(NodeAddress(), DSType.t, False)
            if trees:
                out.append((words, trees[0]))
        return out


TTR2TreeCorpusConverter.convertCorpus = TTR2TreeCorpusConverter.convert_corpus  # type: ignore[attr-defined]
