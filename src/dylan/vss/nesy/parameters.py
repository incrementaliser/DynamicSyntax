"""Parameter factories for NeSy parse circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cirkit.symbolic.parameters import ParameterFactory


def softmax_logits_factory() -> ParameterFactory:
    """Return a cirkit factory that builds normalized categorical logits."""
    from cirkit.symbolic.initializers import NormalInitializer
    from cirkit.symbolic.parameters import Parameter, SoftmaxParameter, TensorParameter

    def factory(shape: tuple[int, ...]) -> Parameter:
        """Build a :class:`~cirkit.symbolic.parameters.SoftmaxParameter` graph."""
        raw = Parameter.from_input(
            TensorParameter(*shape, initializer=NormalInitializer())
        )
        return SoftmaxParameter(raw)

    return factory
