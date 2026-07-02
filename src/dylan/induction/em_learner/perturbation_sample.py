"""Perturbation sample helper (Java ``qmul.ds.learn.PerturbationSample``).

Holds a single perturbed (sentence, semantics) example used to evaluate the
generator's robustness to mid-utterance goal switches.  Provides static
helpers to load/filter/write batches of samples from disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Generic, Iterable, TypeVar

from dylan.formula.ttr_record_type import TTRRecordType

logger = logging.getLogger(__name__)

ANSI_RESET = "\u001b[0m"
ANSI_RED = "\u001b[31m"
ANSI_GREEN = "\u001b[32m"

T = TypeVar("T")


@dataclass
class PerturbationSample:
    """Single perturbation sample (Java ``PerturbationSample`` data class).

    ``rG`` is the original goal semantics, ``rP`` the perturbed goal.  ``pI``
    indexes the last successfully generated word before the perturbation.
    """

    original_sent: str | None = None
    r_g: TTRRecordType | None = None
    r_p: TTRRecordType | None = None
    perturbed_sent: str | None = None
    p_i: int = -1
    is_forward: bool = False
    distance: int = -1
    pos: str | None = None

    @staticmethod
    def load_perturbation_data(file_name: "str | Path") -> "list[PerturbationSample]":
        """Java ``loadPerturbationData``: parse 8-line records from *file_name*."""
        path = Path(file_name)
        samples: list[PerturbationSample] = []
        if not path.exists():
            logger.warning("perturbation file not found: %s", path)
            return samples
        with path.open("r", encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh.readlines()]
        i = 0
        while i + 7 < len(lines):
            try:
                samples.append(
                    PerturbationSample(
                        original_sent=lines[i],
                        r_g=TTRRecordType.parse(lines[i + 1]),
                        r_p=TTRRecordType.parse(lines[i + 2]),
                        perturbed_sent=lines[i + 3],
                        p_i=int(lines[i + 4]),
                        is_forward=lines[i + 5].strip().lower() == "true",
                        distance=int(lines[i + 6]),
                        pos=lines[i + 7],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("could not parse perturbation block at line %d: %s", i, exc)
            i += 9

        return samples

    @staticmethod
    def find_generatable_samples(
        all_samples: "Iterable[PerturbationSample]",
        generator: object | None = None,
    ) -> "list[PerturbationSample]":
        """Java ``findGeneratableSamples``: filter to samples the generator can produce.

        Without an ``InteractiveProbabilisticGenerator`` available, the Python
        port logs a warning and returns the input unchanged for parity.
        """
        items = list(all_samples)
        if generator is None:
            logger.warning(
                "InteractiveProbabilisticGenerator unavailable; returning all %d samples",
                len(items),
            )
            return items
        out: list[PerturbationSample] = []
        for sample in items:
            try:
                generator.init()  # type: ignore[attr-defined]
                generator.set_goal(sample.r_p)  # type: ignore[attr-defined]
                if generator.generate():  # type: ignore[attr-defined]
                    out.append(sample)
            except Exception as exc:  # noqa: BLE001
                logger.debug("generator failed on sample: %s", exc)
        logger.info("Number of generatable samples: %d", len(out))
        return out

    @staticmethod
    def write_perturbation_data_to_file(
        samples: "Iterable[PerturbationSample]",
        file_name: "str | Path",
    ) -> None:
        """Java ``writePerturbationDataTofFile``: dump samples one per block."""
        path = Path(file_name)
        try:
            with path.open("w", encoding="utf-8") as out:
                for sample in samples:
                    out.write(sample.to_string_as_rows())
                    out.write("\n")
        except OSError as exc:
            logger.error("Couldn't write to %s: %s", path, exc)

    @staticmethod
    def write_eval_output_to_file(
        samples: "list[PerturbationSample]",
        file_name: "str | Path",
        generated: "list[str]",
    ) -> None:
        """Java ``writeEvalOutputTofFile``: write samples + generator outputs."""
        path = Path(file_name)
        try:
            with path.open("w", encoding="utf-8") as out:
                for sample, generated_sentence in zip(samples, generated):
                    out.write(sample.to_string_as_rows())
                    out.write(generated_sentence + "\n\n")
        except OSError as exc:
            logger.error("Couldn't write to %s: %s", path, exc)

    def to_string_as_rows(self) -> str:
        """Java ``toStringAsRows``: 8-line block representation suitable for round-trip."""
        return (
            f"{self.original_sent}\n"
            f"{self.r_g}\n"
            f"{self.r_p}\n"
            f"{self.perturbed_sent}\n"
            f"{self.p_i}\n"
            f"{str(self.is_forward).lower()}\n"
            f"{self.distance}\n"
            f"{self.pos}\n"
        )

    def __str__(self) -> str:
        """Multi-line debug rendering matching Java ``toString``."""
        return (
            f"\noriginalSent= {self.original_sent}"
            f",\nrG= {self.r_g}"
            f",\nrP= {self.r_p}"
            f",\nperturbedSent= {self.perturbed_sent}"
            f",\npI= {self.p_i}"
            f",\nisForward= {self.is_forward}"
            f",\ndistance= {self.distance}"
            f",\npos= {self.pos}"
        )


@dataclass
class PerturbationSampleBag(Generic[T]):
    """Lightweight random-sample helper retained for backward compatibility."""

    items: list[T] = field(default_factory=list)
    seed: int | None = None

    def sample(self, n: int) -> list[T]:
        """Return up to *n* sampled items."""
        rng = Random(self.seed)
        if n >= len(self.items):
            return list(self.items)
        return rng.sample(self.items, n)

    @classmethod
    def from_iterable(cls, items: Iterable[T], seed: int | None = None) -> "PerturbationSampleBag[T]":
        """Build a sample helper from *items*."""
        return cls(list(items), seed)


PerturbationSample.loadPerturbationData = PerturbationSample.load_perturbation_data  # type: ignore[attr-defined]
PerturbationSample.findGeneratableSamples = PerturbationSample.find_generatable_samples  # type: ignore[attr-defined]
PerturbationSample.writePerturbationDataTofFile = PerturbationSample.write_perturbation_data_to_file  # type: ignore[attr-defined]
PerturbationSample.writeEvalOutputTofFile = PerturbationSample.write_eval_output_to_file  # type: ignore[attr-defined]
PerturbationSample.toStringAsRows = PerturbationSample.to_string_as_rows  # type: ignore[attr-defined]
