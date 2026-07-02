"""Orchestrate DS incremental parsing with vector-space semantic composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.tree import Tree
from dylan.vss.compose_svo import compose_svo
from dylan.vss.embedding_store import EmbeddingStore, embedding_store_from_config
from dylan.vss.lexicon_vss import LexicalVSSBinding, LexiconVSSIndex
from dylan.vss.svo_roles import roles_from_parse, surface_svo_sentence
from dylan.vss.tree_vss import TreeVSSState, decorate_tree
from dylan.vss.types import (
    GS2013Sentence,
    IncrementalComposition,
    SVORoles,
    UnderspecMethod,
    VSSConfig,
)

_VSS_DIR = Path(__file__).resolve().parent
_DEFAULT_VSS_GRAMMAR = _VSS_DIR / "resources" / "vss-transitive"


@dataclass(frozen=True, slots=True)
class IncrementalVSSStep:
    """One word-aligned DS parse step with VSS state."""

    word: str | None
    tree: Tree
    vss_tree: TreeVSSState
    composition: IncrementalComposition | None = None
    roles: SVORoles | None = None
    lexical_action_name: str | None = None


@dataclass
class DSVSSParseResult:
    """Full incremental DS-VSS parse outcome."""

    ok: bool
    sentence: str
    steps: list[IncrementalVSSStep] = field(default_factory=list)
    semantics: TTRRecordType | None = None
    final_roles: SVORoles | None = None
    action_steps: tuple[object, ...] = field(default_factory=tuple)


class DSVSSSession:
    """Combine incremental DS parsing with VSS composition via :class:`LexiconVSSIndex`."""

    def __init__(
        self,
        grammar: str | Path | None = None,
        embedding_store: EmbeddingStore | None = None,
        config: VSSConfig | None = None,
        *,
        parser: InteractiveContextParser | None = None,
    ) -> None:
        """Configure grammar path, embeddings, and VSS options."""
        self._config = config or VSSConfig()
        self._store = embedding_store or embedding_store_from_config(self._config)
        g = grammar or self._config.grammar_path or _DEFAULT_VSS_GRAMMAR
        self._grammar_path = Path(g) if not isinstance(g, Path) else g
        self._parser: InteractiveContextParser | None = parser
        self._lexicon_index: LexiconVSSIndex | None = None

    def _ensure_parser(self) -> InteractiveContextParser:
        """Lazily construct an :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser`."""
        if self._parser is None:
            self._parser = InteractiveContextParser.from_resource_dir(
                self._grammar_path,
                top_n=self._config.top_n,
            )
        return self._parser

    @property
    def lexicon_index(self) -> LexiconVSSIndex:
        """Lexicon-backed VSS bindings for the loaded grammar."""
        if self._lexicon_index is None:
            self._lexicon_index = LexiconVSSIndex(self._ensure_parser().lexicon)
        return self._lexicon_index

    @property
    def grammar_path(self) -> Path:
        """Filesystem path to the active grammar resource directory."""
        return self._grammar_path

    def parse_incremental(
        self,
        sentence: str,
        *,
        dataset_hint: GS2013Sentence | None = None,
        speaker: str = "Dylan",
    ) -> DSVSSParseResult:
        """Parse *sentence* word-by-word and compute incremental VSS after each word."""
        from dynamicsyntax._parse import _run_parse_core

        parser = self._ensure_parser()
        result = _run_parse_core(parser, sentence.strip(), speaker=speaker, trace=True)
        steps: list[IncrementalVSSStep] = []
        labels = list(result.trace_step_labels)
        trees = list(result.trace_trees)
        words_so_far: list[str] = []
        incr_comp: IncrementalComposition | None = None
        final_roles: SVORoles | None = None
        lex_idx = self.lexicon_index if self._config.use_lexicon_vss_hints else None

        action_by_word_index: dict[int, str] = {}
        ai = 0
        for step in result.action_steps:
            if step.word:
                action_by_word_index[ai] = step.action_name
                ai += 1

        for i, tree in enumerate(trees):
            word = labels[i - 1] if i > 0 else None
            if word:
                words_so_far.append(word)
            semantics = None
            if result.ok and i == len(trees) - 1:
                try:
                    semantics = parser.get_final_semantics()
                except Exception:
                    semantics = result.semantics
            else:
                semantics = result.semantics if i == len(trees) - 1 else None

            roles: SVORoles | None = None
            if words_so_far:
                try:
                    roles = roles_from_parse(
                        semantics,
                        tree,
                        tuple(words_so_far),
                        fallback=dataset_hint,
                        allow_dataset_fallback=self._config.allow_dataset_role_fallback,
                    )
                except ValueError:
                    roles = None

            step_bindings: list[LexicalVSSBinding] = []
            action_name: str | None = None
            if i > 0 and lex_idx is not None:
                action_name = action_by_word_index.get(i - 1)
                if action_name:
                    b = lex_idx.lookup_by_action_name(action_name)
                    if b is not None:
                        step_bindings.append(b)

            vss_tree = decorate_tree(
                tree,
                self._store,
                tuple(words_so_far),
                subj=roles.subj if roles else None,
                verb=roles.landmark if roles else None,
                obj=roles.obj if roles else None,
                underspec=self._config.underspec,
                lexicon_index=lex_idx,
                step_bindings=step_bindings,
            )
            if roles and len(words_so_far) >= 3:
                try:
                    incr_comp = compose_svo(
                        self._store,
                        roles.subj,
                        roles.landmark,
                        roles.obj,
                        underspec=self._config.underspec,
                    )
                except KeyError:
                    incr_comp = None
                final_roles = roles

            steps.append(
                IncrementalVSSStep(
                    word=word,
                    tree=tree.clone(),
                    vss_tree=vss_tree,
                    composition=incr_comp,
                    roles=roles,
                    lexical_action_name=action_name,
                )
            )

        return DSVSSParseResult(
            ok=result.ok,
            sentence=sentence,
            steps=steps,
            semantics=result.semantics,
            final_roles=final_roles,
            action_steps=result.action_steps,
        )

    def parse_gs2013_sentence(
        self,
        sent: GS2013Sentence,
        *,
        use_landmark: bool = True,
    ) -> DSVSSParseResult:
        """Parse a GS2013 item using subj + landmark + obj surface order."""
        verb = sent.landmark if use_landmark else sent.verb
        surface = surface_svo_sentence(sent.subj, verb, sent.obj)
        return self.parse_incremental(surface, dataset_hint=sent)
