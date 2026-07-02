"""Parse lattices and supervised examples for NeSy DS-VSS learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from dylan.action.lexical_action import LexicalAction
from dylan.dag.groundable_edge import GroundableEdge
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.tree import Tree
from dylan.vss.svo_roles import SVORoles

from dynamicsyntax.parse_result import ParseResult


@dataclass(frozen=True, slots=True)
class ParseStateKey:
    """Hashable DS parse state for lattice nodes."""

    tree_key: tuple
    pointer: str

    @classmethod
    def from_tree(cls, tree: Tree) -> ParseStateKey:
        """Build a key from a :class:`~dylan.tree.tree.Tree` snapshot."""
        items: list[tuple[str, tuple[str, ...]]] = []
        for addr in sorted(tree.keys(), key=lambda a: a.address):
            node = tree[addr]
            labels_key = tuple(str(lab) for lab in node.labels)
            items.append((addr.address, labels_key))
        return cls(tree_key=tuple(items), pointer=tree.pointer.address)


@dataclass(frozen=True, slots=True)
class LatticeEdge:
    """One lexical transition in a parse lattice."""

    word: str
    action_index: int
    action_name: str
    parent: ParseStateKey
    child: ParseStateKey


@dataclass
class LatticeSpec:
    """Finite parse lattice for one sentence (DAG over parse states)."""

    sentence: str
    words: tuple[str, ...]
    edges_by_parent: dict[ParseStateKey, list[LatticeEdge]] = field(default_factory=dict)
    root: ParseStateKey | None = None
    gold_edge_indices: tuple[int, ...] = ()
    gold_action_names: tuple[str, ...] = ()

    def edges_at(self, parent: ParseStateKey) -> list[LatticeEdge]:
        """Outgoing edges from *parent*."""
        return list(self.edges_by_parent.get(parent, []))

    @property
    def num_words(self) -> int:
        """Number of surface words."""
        return len(self.words)


@dataclass
class SupervisedParseExample:
    """Gold parse trace for supervised NeSy training."""

    sentence: str
    grammar_path: Path
    words: tuple[str, ...]
    gold_action_names: tuple[str, ...]
    gold_semantics: TTRRecordType | None = None
    gold_roles: SVORoles | None = None
    lattice: LatticeSpec | None = None


def _active_path_edges(parser: InteractiveContextParser) -> list[GroundableEdge]:
    """Return DAG edges from root to the parser's current tuple."""
    dag = parser.get_state()
    cur = dag.get_current_tuple()
    edges: list[GroundableEdge] = []
    while True:
        edge = dag.get_parent_edge(cur)
        parent = dag.get_parent(cur)
        if edge is None or parent is None:
            break
        edges.append(edge)
        cur = parent
    edges.reverse()
    return edges


def examples_from_parse_result(
    result: ParseResult,
    *,
    grammar_path: Path,
) -> SupervisedParseExample | None:
    """Build a :class:`SupervisedParseExample` from a traced :class:`~dynamicsyntax.parse_result.ParseResult`."""
    if not result.ok or not result.sentence.strip():
        return None
    gold_names = tuple(
        step.action_name for step in result.action_steps if step.word is not None
    )
    roles: SVORoles | None = None
    if result.semantics is not None and result.tree is not None:
        from dylan.vss.svo_roles import roles_from_parse

        try:
            words = tuple(result.trace_step_labels)
            roles = roles_from_parse(
                result.semantics,
                result.tree,
                words,
                allow_dataset_fallback=False,
            )
        except ValueError:
            roles = None
    return SupervisedParseExample(
        sentence=result.sentence,
        grammar_path=grammar_path,
        words=tuple(result.trace_step_labels),
        gold_action_names=gold_names,
        gold_semantics=result.semantics,
        gold_roles=roles,
    )


def toy_three_word_lattice() -> LatticeSpec:
    """Minimal 3-step linear lattice for tests and smoke scripts."""
    root = ParseStateKey(tree_key=(("0", ("?Ty(t)",)),), pointer="0")
    s1 = ParseStateKey(tree_key=(("0", ("Ty(e)",)),), pointer="01")
    s2 = ParseStateKey(tree_key=(("0", ("Ty(e)", "Ty(e)",)),), pointer="011")
    s3 = ParseStateKey(tree_key=(("0", ("Ty(t)",)),), pointer="0")
    spec = LatticeSpec(
        sentence="w1 w2 w3",
        words=("w1", "w2", "w3"),
        root=root,
        gold_edge_indices=(0, 1, 0),
        gold_action_names=("a0", "b1", "c0"),
    )
    spec.edges_by_parent[root] = [
        LatticeEdge("w1", 0, "a0", root, s1),
        LatticeEdge("w1", 1, "a1", root, s1),
    ]
    spec.edges_by_parent[s1] = [
        LatticeEdge("w2", 0, "b0", s1, s2),
        LatticeEdge("w2", 1, "b1", s1, s2),
    ]
    spec.edges_by_parent[s2] = [
        LatticeEdge("w3", 0, "c0", s2, s3),
        LatticeEdge("w3", 1, "c1", s2, s3),
    ]
    return spec


class ParseLatticeBuilder:
    """Build parse lattices from a loaded :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser`."""

    def __init__(
        self,
        parser: InteractiveContextParser,
        *,
        max_states: int = 500,
    ) -> None:
        """Attach to *parser* with a cap on explored states."""
        self._parser = parser
        self._max_states = max_states

    def gold_lattice_from_parse(self, result: ParseResult) -> LatticeSpec:
        """Replay the successful parse path as a linear lattice."""
        if result.parser is None:
            raise ValueError("ParseResult must retain parser for gold lattice replay")
        parser = result.parser
        spec = LatticeSpec(
            sentence=result.sentence,
            words=tuple(result.trace_step_labels),
            gold_action_names=tuple(
                s.action_name for s in result.action_steps if s.word is not None
            ),
        )
        if not result.trace_trees:
            return spec
        root_key = ParseStateKey.from_tree(result.trace_trees[0])
        spec.root = root_key
        parent_key = root_key
        gold_indices: list[int] = []
        lexical_steps = [s for s in result.action_steps if s.word is not None]
        if lexical_steps:
            for step in lexical_steps:
                actions = self._legal_actions(step.before_tree, step.word)
                if not actions:
                    continue
                try:
                    ai = next(
                        i for i, a in enumerate(actions) if a.get_name() == step.action_name
                    )
                except StopIteration:
                    ai = 0
                gold_indices.append(ai)
                child_key = ParseStateKey.from_tree(step.after_tree)
                edge = LatticeEdge(
                    word=step.word,
                    action_index=ai,
                    action_name=actions[ai].get_name(),
                    parent=parent_key,
                    child=child_key,
                )
                spec.edges_by_parent.setdefault(parent_key, []).append(edge)
                parent_key = child_key
        else:
            labels = list(result.trace_step_labels)
            trees = list(result.trace_trees)
            for i, word in enumerate(labels):
                if i + 1 >= len(trees):
                    break
                before, after = trees[i], trees[i + 1]
                actions = self._legal_actions(before, word)
                if not actions:
                    actions = list(self._parser.lexicon.lookup(word))
                    actions = [a for a in actions if isinstance(a, LexicalAction)]
                if not actions:
                    continue
                ai = self._match_action_index(before, after, word, actions)
                gold_indices.append(ai)
                child_key = ParseStateKey.from_tree(after)
                edge = LatticeEdge(
                    word=word,
                    action_index=ai,
                    action_name=actions[ai].get_name(),
                    parent=parent_key,
                    child=child_key,
                )
                spec.edges_by_parent.setdefault(parent_key, []).append(edge)
                parent_key = child_key
        spec.gold_edge_indices = tuple(gold_indices)
        return spec

    def _match_action_index(
        self,
        before: Tree,
        after: Tree,
        word: str,
        actions: list[LexicalAction],
    ) -> int:
        """Pick the lexical action whose application best matches *after*."""
        target = ParseStateKey.from_tree(after)
        for i, act in enumerate(actions):
            clone = before.clone()
            nxt = self._parser.apply_actions(clone, [act])
            if nxt is not None and ParseStateKey.from_tree(nxt) == target:
                return i
        return 0

    def beam_lattice(
        self,
        sentence: str,
        *,
        speaker: str = "Dylan",
    ) -> LatticeSpec:
        """Bounded beam expansion over lexical actions per word."""
        from dylan.nlp.types import utterance_from_text

        parser = self._parser
        parser.init()
        parser.new_sentence()
        utt = utterance_from_text(speaker, sentence.strip())
        words = tuple(w.word for w in utt.words)
        spec = LatticeSpec(sentence=sentence.strip(), words=words)
        frontier: list[tuple[ParseStateKey, Tree]] = [
            (ParseStateKey.from_tree(parser.get_best_tuple().get_tree()), parser.get_best_tuple().get_tree())
        ]
        spec.root = frontier[0][0]
        seen: set[ParseStateKey] = {frontier[0][0]}
        states_explored = 1

        for word in words:
            next_frontier: list[tuple[ParseStateKey, Tree]] = []
            for parent_key, tree in frontier:
                for ai, action in enumerate(self._legal_actions(tree, word)):
                    clone = tree.clone()
                    nxt = parser.apply_actions(clone, [action])
                    if nxt is None:
                        continue
                    child_key_probe = ParseStateKey.from_tree(nxt)
                    if child_key_probe in seen and states_explored >= self._max_states:
                        continue
                    seen.add(child_key_probe)
                    states_explored += 1
                    child_key = ParseStateKey.from_tree(nxt)
                    edge = LatticeEdge(
                        word=word,
                        action_index=ai,
                        action_name=action.get_name(),
                        parent=parent_key,
                        child=child_key,
                    )
                    spec.edges_by_parent.setdefault(parent_key, []).append(edge)
                    next_frontier.append((child_key, nxt))
                    if states_explored >= self._max_states:
                        break
                if states_explored >= self._max_states:
                    break
            frontier = next_frontier
            if not frontier:
                break
        return spec

    def _legal_actions(self, tree: Tree, word: str) -> list[LexicalAction]:
        """Return lexical actions for *word* that succeed on *tree*."""
        out: list[LexicalAction] = []
        for act in self._parser.lexicon.lookup(word):
            if not isinstance(act, LexicalAction):
                continue
            clone = tree.clone()
            if self._parser.apply_actions(clone, [act]) is not None:
                out.append(act)
        return out
