"""``infer-sa`` — run speech-act inference rules (Java ``InferSpeechAct``)."""

from __future__ import annotations

import logging
from typing import Any

from dylan.action.atomic.effect import Effect
from dylan.context.context import Context
from dylan.dag.dag_tuple import DAGTuple
from dylan.tree.label.labels import Label, SpeechActLabel
from dylan.tree.tree import Tree

logger = logging.getLogger(__name__)


class InferSpeechAct(Effect):
    """Try SA grammar rules on a clone of the tree (Java ``InferSpeechAct``)."""

    FUNCTOR = "infer-sa"

    def exec(self, tree: Tree, context: Context[DAGTuple, Any] | None) -> Tree | None:
        if context is None:
            raise ValueError("infer-sa requires Context")
        sag = context.get_speech_act_grammar()
        clone = tree.clone()
        _strip_speech_act_labels(clone.pointed_node)
        for action in sag:
            logger.debug("trying speech act rule: %s", action.get_name())
            result = action.exec(clone, context)
            if result is not None:
                return result
        return tree.clone()

    def exec_tuple_context(self, tree: Tree, context: Any) -> Tree | None:
        raise ValueError("infer-sa does not support tuple-only context")

    def instantiate(self) -> Effect:
        return InferSpeechAct()

    def __str__(self) -> str:
        return self.FUNCTOR


def _strip_speech_act_labels(node: Any) -> None:
    """Remove :class:`SpeechActLabel` instances from *node* (Java ``removeSAAnnotations``)."""
    keep: list[Label] = [lab for lab in node.labels if not isinstance(lab, SpeechActLabel)]
    node.labels = keep
