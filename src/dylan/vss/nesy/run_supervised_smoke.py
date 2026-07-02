"""Smoke script: supervised NeSy fit on vss-transitive traced sentences."""

from __future__ import annotations

from pathlib import Path

from dylan.vss.embedding_store import DemoEmbeddingStore
from dylan.vss.nesy.learner import NesyDSVSSLearner
from dylan.vss.nesy.parse_lattice import SupervisedParseExample, toy_three_word_lattice
from dylan.vss.nesy.trainer import SupervisedTrainConfig

_VSS_DIR = Path(__file__).resolve().parents[1]
_GRAMMAR = _VSS_DIR / "resources" / "vss-transitive"
_SENTENCES = (
    "table draw eye",
    "boy push ball",
    "girl find book",
    "dog chase cat",
    "man lift box",
)


def main() -> None:
    """Fit a small supervised model and print path MAP accuracy."""
    learner = NesyDSVSSLearner(_GRAMMAR, store=DemoEmbeddingStore())
    examples: list[SupervisedParseExample] = []
    for sent in _SENTENCES:
        ex = learner.example_from_sentence(sent)
        if ex is not None and ex.lattice is not None and ex.lattice.gold_edge_indices:
            examples.append(ex)
    if not examples:
        toy = toy_three_word_lattice()
        examples.append(
            SupervisedParseExample(
                sentence=toy.sentence,
                grammar_path=_GRAMMAR,
                words=toy.words,
                gold_action_names=toy.gold_action_names,
                lattice=toy,
            )
        )
    result = learner.fit_supervised(
        examples,
        config=SupervisedTrainConfig(epochs=10, lr=0.1, prefer_cirkit=True),
    )
    correct = 0
    for ex in examples:
        if ex.lattice is None or not ex.lattice.gold_edge_indices:
            continue
        try:
            pred = learner.predict_parse(ex.sentence)
        except ValueError:
            continue
        if pred.map_indices == ex.lattice.gold_edge_indices:
            correct += 1
    print(f"trained epochs={len(result.epoch_losses)} final_nll={result.final_parse_nll:.4f}")
    print(f"path_accuracy={correct}/{len(examples)}")
    for ex in examples:
        try:
            sem = learner.predict_semantics(ex.sentence)
            print(f"sample_semantics_shape={tuple(sem.tensor.shape)}")
            break
        except ValueError:
            continue


if __name__ == "__main__":
    main()
