"""Einsum Networks: fast and scalable learning of tractable probabilistic circuits.

A self-contained PyTorch implementation of the model of

    Peharz, R., Lang, S., Vergari, A., Stelzner, K., Molina, A., Trapp, M.,
    Van den Broeck, G., Kersting, K. & Ghahramani, Z. (2020).
    "Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic
    Circuits." ICML 2020 (PMLR 119:7563–7574),

following the probabilistic-circuits framework of Choi, Vergari & Van den
Broeck (2020).  The circuit alternates monolithic *einsum* layers — products
computed as sums in log-space, sums computed with the log-einsum-exp trick —
over a random binary region graph (:mod:`dylan.pc.regions`), with categorical
input distributions.  Because the circuit is smooth and decomposable,
**marginals and conditionals are exact** and cost a single feed-forward pass.

Training supports both SGD (Adam on negative log-likelihood) and the paper's
EM algorithm implemented via automatic differentiation, including stochastic
online EM (Sato 1999).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dylan.pc.regions import PCStructure, random_region_graph

try:  # optional dependency: the PC subsystem requires torch
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "dylan.pc requires PyTorch; install the optional extra with "
        "`pip install dynamicsyntax[pc]` (or `pip install torch`)."
    ) from exc

#: Evidence value marking a marginalised (unobserved) variable.
UNKNOWN = -1


@dataclass
class EiNetConfig:
    """Structural hyper-parameters of an :class:`EinsumNetwork`."""

    num_vars: int
    num_categories: int | tuple[int, ...]
    num_sums: int = 8
    num_input_dists: int = 4
    num_repetitions: int = 2
    #: Std of the Gaussian parameter initialisation.  Values below ~0.5
    #: make deep circuits start almost uniform (channels average out inside
    #: the log-einsum-exp layers), stalling training — the well-known
    #: vanishing-signal phenomenon of deep SPNs.
    init_scale: float = 1.0
    seed: int | None = None


class CategoricalLeaf(nn.Module):
    """Categorical input distributions, one set of channels per variable.

    Parameters are ``(num_vars, num_channels, max_categories)`` logits;
    variables with fewer categories than the maximum are masked with
    ``-inf`` on the invalid categories.  Marginalised inputs (value
    :data:`UNKNOWN`) yield log-density 0 (density 1), which is what makes
    exact marginal inference a single forward pass.
    """

    def __init__(
        self,
        num_vars: int,
        num_channels: int,
        num_categories: Sequence[int],
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.num_channels = num_channels
        self.num_categories = tuple(int(c) for c in num_categories)
        max_cat = max(self.num_categories)
        self.logits = nn.Parameter(torch.randn(num_vars, num_channels, max_cat) * init_scale)
        mask = torch.zeros(num_vars, 1, max_cat)
        for v, c in enumerate(self.num_categories):
            if c < max_cat:
                mask[v, 0, c:] = float("-inf")
        self.register_buffer("category_mask", mask)

    def log_probs(self) -> torch.Tensor:
        """Normalised log-probabilities, ``(num_vars, channels, max_cat)``."""
        return torch.log_softmax(self.logits + self.category_mask, dim=-1)

    def forward(self, x: torch.Tensor, log_p: torch.Tensor | None = None) -> torch.Tensor:
        """Log-density of evidence *x*, ``(batch, num_vars, channels)``.

        ``x[v] < 0`` marks a marginalised variable (output 0 = log 1).
        """
        if log_p is None:
            log_p = self.log_probs()
        idx = x.clamp(min=0).long()  # (batch, num_vars)
        log_p_e = log_p.unsqueeze(0).expand(x.shape[0], -1, -1, -1)  # (B, V, K, C)
        gather_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_channels, 1)
        gathered = torch.gather(log_p_e, 3, gather_idx).squeeze(-1)  # (B, V, K)
        return torch.where((x < 0).unsqueeze(-1), torch.zeros_like(gathered), gathered)


class EinsumLayer(nn.Module):
    """One monolithic product+sum layer with the log-einsum-exp trick.

    For parent region ``p`` with child outputs ``l`` (left) and ``r``
    (right), each of ``I`` channels, the layer computes ::

        pair_log[b, p, i, j] = l[b, p, i] + r[b, p, j]
        out[b, p, o] = logsumexp_ij( log_w[p, o, i*I + j] + pair_log[b, p, i, j] )

    Carried-over single children are products with a dummy unit region whose
    log-density is ``[0, -inf, -inf, …]``, so spurious pairs vanish inside
    the logsumexp.  Weights are normalised with a log-softmax over the input
    dimension, keeping every sum node normalised.
    """

    def __init__(
        self,
        left: Sequence[int],
        right: Sequence[int | None],
        num_regions_prev: int,
        in_channels: int,
        out_channels: int,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_parents = len(left)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Index of the appended dummy unit region is num_regions_prev.
        self.register_buffer("left_idx", torch.tensor(list(left), dtype=torch.long))
        right_resolved = [num_regions_prev if r is None else r for r in right]
        self.register_buffer("right_idx", torch.tensor(right_resolved, dtype=torch.long))
        self.logits = nn.Parameter(
            torch.randn(self.num_parents, out_channels, in_channels * in_channels) * init_scale
        )
        # Carried-over parents pair their single child with a dummy unit
        # region; only dummy channel 0 is valid.  Mask the other pairs to
        # -inf *before* the log-softmax so that sum weights normalise over
        # the valid inputs only (otherwise the partition function leaks).
        mask = torch.zeros(self.num_parents, 1, in_channels * in_channels)
        for p, r in enumerate(right):
            if r is None:
                keep = torch.arange(in_channels) * in_channels  # (i, j=0) entries
                mask[p, 0, :] = float("-inf")
                mask[p, 0, keep] = 0.0
        self.register_buffer("weight_mask", mask)

    def log_weights(self) -> torch.Tensor:
        return torch.log_softmax(self.logits + self.weight_mask, dim=-1)

    def forward(self, x: torch.Tensor, log_w: torch.Tensor | None = None) -> torch.Tensor:
        """``x``: ``(batch, num_regions_prev, in_channels)`` → ``(batch, P, out_channels)``."""
        batch = x.shape[0]
        unit = torch.zeros(batch, 1, self.in_channels, dtype=x.dtype, device=x.device)
        unit[:, :, 1:] = float("-inf")
        x_ext = torch.cat([x, unit], dim=1)
        left = x_ext[:, self.left_idx, :]
        right = x_ext[:, self.right_idx, :]
        pair_log = left.unsqueeze(3) + right.unsqueeze(2)
        pair_log = pair_log.reshape(batch, self.num_parents, -1)
        if log_w is None:
            log_w = self.log_weights()
        return torch.logsumexp(pair_log.unsqueeze(2) + log_w.unsqueeze(0), dim=-1)


class EinsumNetwork(nn.Module):
    """Einsum Network over categorical random variables.

    :param config: an :class:`EiNetConfig`; ``num_categories`` may be a
        single int (shared by all variables) or a per-variable tuple.

    .. note::
        Training trade-off (cf. Peharz et al. 2020, sec. 5–6): full-batch
        EM can plateau on very wide circuits trained on small data (the
        deterministic M-step preserves the near-uniform routing of
        initialisation); SGD escapes it reliably.  Moderate widths (the
        default 4–8 sums) learn well with either method.
    """

    def __init__(self, config: EiNetConfig) -> None:
        super().__init__()
        self.config = config
        if isinstance(config.num_categories, int):
            categories = (config.num_categories,) * config.num_vars
        else:
            categories = tuple(config.num_categories)
            if len(categories) != config.num_vars:
                raise ValueError(
                    f"num_categories tuple has {len(categories)} entries but "
                    f"num_vars={config.num_vars}"
                )
        self.categories = categories
        self.structure: PCStructure = random_region_graph(
            config.num_vars, config.num_repetitions, seed=config.seed
        )
        k_leaf = config.num_input_dists * config.num_repetitions
        k_inner = config.num_sums * config.num_repetitions
        self.leaf = CategoricalLeaf(config.num_vars, k_leaf, categories, config.init_scale)
        self.einsum_layers = nn.ModuleList()
        prev_regions = config.num_vars
        prev_channels = k_leaf
        for spec in self.structure.layers:
            self.einsum_layers.append(
                EinsumLayer(
                    spec.left,
                    spec.right,
                    prev_regions,
                    prev_channels,
                    k_inner,
                    config.init_scale,
                )
            )
            prev_regions = spec.num_parents
            prev_channels = k_inner
        # Root sum: mixes all channels (incl. repetitions) into one output.
        self.root_logits = nn.Parameter(torch.randn(1, prev_channels) * config.init_scale)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _forward_with_weights(
        self,
        x: torch.Tensor,
        leaf_log_p: torch.Tensor | None = None,
        layer_log_ws: list[torch.Tensor] | None = None,
        root_log_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.leaf(x, leaf_log_p)
        for i, layer in enumerate(self.einsum_layers):
            lw = None if layer_log_ws is None else layer_log_ws[i]
            out = layer(out, lw)
        top = out[:, 0, :]  # (batch, root_channels)
        if root_log_w is None:
            root_log_w = torch.log_softmax(self.root_logits, dim=-1)  # (1, root_channels)
        ll = torch.logsumexp(top.unsqueeze(1) + root_log_w.unsqueeze(0), dim=-1)
        return ll[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood ``log p(evidence)`` per row of *x* (shape ``(batch,)``)."""
        return self._forward_with_weights(x)

    def conditional_log_probs(
        self, evidence: torch.Tensor, var: int, values: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Exact ``log p(X_var = value | evidence)`` for each value.

        Computed by exact marginalisation: one forward pass per candidate
        value (vectorised as a batch), normalised by the evidence
        likelihood — the tractable-inference selling point of PCs.
        """
        if values is None:
            values = torch.arange(self.categories[var], device=evidence.device)
        base = evidence.unsqueeze(0).expand(values.shape[0], -1).clone()
        base[:, var] = values.to(base.device)
        joint = self.forward(base)
        evid = self.forward(evidence.unsqueeze(0))[0]
        return joint - evid

    def predict(self, evidence: torch.Tensor, var: int) -> tuple[int, torch.Tensor]:
        """MAP value of *var* given *evidence*, with the full log-posterior."""
        log_p = self.conditional_log_probs(evidence, var)
        best = int(torch.argmax(log_p).item())
        return best, log_p

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def fit_sgd(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-2,
        verbose: bool = False,
    ) -> list[float]:
        """Maximum-likelihood training with Adam on the negative log-likelihood."""
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = data.shape[0]
        history: list[float] = []
        for epoch in range(epochs):
            perm = torch.randperm(n, device=data.device)
            total = 0.0
            for start in range(0, n, batch_size):
                batch = data[perm[start : start + batch_size]]
                loss = -self.forward(batch).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * batch.shape[0]
            history.append(total / n)
            if verbose:
                print(f"[einet/sgd] epoch {epoch + 1}: nll={history[-1]:.4f}")
        return history

    def fit_em(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        step_size: float = 0.3,
        pseudocount: float = 1e-2,
        verbose: bool = False,
    ) -> list[float]:
        """EM training via automatic differentiation (Peharz et al. 2020, sec. 5).

        The gradient of the log-likelihood with respect to the *normalised*
        log-parameters of a sum node yields the expected sufficient
        statistics (edge flows); the M-step re-normalises them.  Mini-batch
        updates with ``step_size`` implement stochastic online EM
        (Sato 1999).
        """
        n = data.shape[0]
        history: list[float] = []
        for epoch in range(epochs):
            perm = torch.randperm(n, device=data.device)
            total = 0.0
            for start in range(0, n, batch_size):
                batch = data[perm[start : start + batch_size]]
                with torch.no_grad():
                    total += float(-self.forward(batch).sum())
                flows = self._em_flows(batch)
                with torch.no_grad():
                    for param, w, flow in zip(
                        self._em_parameters(), self._current_weights(), flows
                    ):
                        # M-step: normalised expected counts.  The pseudo-
                        # count is distributed uniformly over the edges so
                        # that its total mass does not scale with layer width.
                        n_edges = flow.shape[-1]
                        new_w = (flow + pseudocount / n_edges) / (
                            flow.sum(-1, keepdim=True) + pseudocount
                        )
                        w.mul_(1.0 - step_size).add_(new_w, alpha=step_size)
                        param.copy_(torch.log(w.clamp_min(1e-30)))
            history.append(total / n)
            if verbose:
                print(f"[einet/em] epoch {epoch + 1}: nll={history[-1]:.4f}")
        return history

    def _em_parameters(self) -> list[nn.Parameter]:
        params = [self.leaf.logits]
        params.extend(layer.logits for layer in self.einsum_layers)
        params.append(self.root_logits)
        return params

    def _current_weights(self) -> list[torch.Tensor]:
        """Effective normalised weights (probability space), masks included."""
        with torch.no_grad():
            weights = [self.leaf.log_probs().exp()]
            weights.extend(layer.log_weights().exp() for layer in self.einsum_layers)
            weights.append(torch.softmax(self.root_logits, dim=-1))
        return weights

    def _em_flows(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Expected sufficient statistics of *batch* via one backward pass."""
        leaf_log_p = self.leaf.log_probs().detach().requires_grad_()
        layer_ws = [layer.log_weights().detach().requires_grad_() for layer in self.einsum_layers]
        root_w = torch.log_softmax(self.root_logits, dim=-1).detach().requires_grad_()
        ll = self._forward_with_weights(batch, leaf_log_p, layer_ws, root_w)
        # Gradient of the (positive) log-likelihood w.r.t. the normalised
        # log-parameters = expected sufficient statistics (edge flows).
        ll.sum().backward()
        flows = [leaf_log_p.grad]
        flows.extend(w.grad for w in layer_ws)
        flows.append(root_w.grad)
        # Numerical noise may produce tiny negatives; flows are counts.
        return [f.clamp_min(0.0) for f in flows]

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save config + weights to *path* (torch checkpoint)."""
        torch.save(
            {
                "config": self.config.__dict__,
                "categories": self.categories,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "EinsumNetwork":
        """Load a model saved with :meth:`save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(EiNetConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model


__all__ = [
    "UNKNOWN",
    "CategoricalLeaf",
    "EinsumLayer",
    "EinsumNetwork",
    "EiNetConfig",
]
