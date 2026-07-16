"""Correctness tests for the Einsum Network (Peharz, Vergari et al. 2020)."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from dylan.pc.einet import UNKNOWN, EinsumNetwork, EiNetConfig  # noqa: E402


def make_net(num_vars=3, categories=3, seed=7, **kw):
    defaults = dict(num_sums=4, num_input_dists=2, num_repetitions=2, seed=seed)
    defaults.update(kw)
    return EinsumNetwork(
        EiNetConfig(num_vars=num_vars, num_categories=categories, **defaults)
    )


def all_configs(num_vars, categories):
    if isinstance(categories, int):
        categories = (categories,) * num_vars
    return torch.tensor(list(itertools.product(*(range(c) for c in categories))))


@pytest.mark.parametrize("num_vars,cats", [(2, 3), (3, 3), (4, 2), (5, 2), (3, (2, 3, 4))])
def test_partition_function_is_one(num_vars, cats):
    """Smooth + decomposable circuit: all configurations sum to 1."""
    torch.manual_seed(0)
    net = make_net(num_vars, cats)
    log_z = torch.logsumexp(net(all_configs(num_vars, cats)), dim=0)
    assert float(log_z) == pytest.approx(0.0, abs=1e-4)


def test_partition_function_survives_training():
    torch.manual_seed(0)
    net = make_net()
    data = torch.tensor([[0, 0, 0], [1, 1, 1]] * 30)
    net.fit_em(data, epochs=3)
    log_z = torch.logsumexp(net(all_configs(3, 3)), dim=0)
    assert float(log_z) == pytest.approx(0.0, abs=1e-4)


def test_conditionals_match_brute_force():
    torch.manual_seed(0)
    net = make_net()
    evidence = torch.tensor([1, UNKNOWN, UNKNOWN])
    log_p = net.conditional_log_probs(evidence, var=2)
    for c in range(3):
        brute = net(torch.tensor([[1, UNKNOWN, c]])) - net(evidence.unsqueeze(0))
        assert float(log_p[c]) == pytest.approx(float(brute), abs=1e-5)
    assert float(log_p.exp().sum()) == pytest.approx(1.0, abs=1e-5)


def test_em_decreases_nll_and_learns():
    torch.manual_seed(0)
    net = make_net()
    data = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 1], [1, 1, 0]] * 20)
    history = net.fit_em(data, epochs=6, step_size=0.5)
    assert history[-1] < history[0]
    # the model concentrates probability on the observed third-variable values
    probs = net.conditional_log_probs(torch.tensor([0, 0, UNKNOWN]), var=2).exp()
    assert float(probs[2]) < 0.05


def test_full_m_step_decreases_nll():
    torch.manual_seed(0)
    net = make_net(seed=11)
    data = torch.tensor([[0, 0, 0], [1, 1, 1]] * 20)
    history = net.fit_em(data, epochs=4, step_size=1.0)
    assert history[-1] <= history[0]


def test_sgd_decreases_nll():
    torch.manual_seed(0)
    net = make_net(seed=5)
    data = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]] * 15)
    history = net.fit_sgd(data, epochs=25, lr=0.05)
    assert history[-1] < history[0]


def test_masked_categories_get_zero_probability():
    torch.manual_seed(0)
    net = make_net(3, (2, 3, 4))
    x = torch.tensor([[0, 0, 0], [1, 2, 3]] * 10)
    net.fit_em(x, epochs=2)
    # variable 0 has only 2 categories; category 2 must be impossible
    evidence = torch.tensor([UNKNOWN, 0, 0])
    log_p = net.conditional_log_probs(evidence, 0, values=torch.tensor([0, 1, 2]))
    assert float(log_p.exp()[2]) == pytest.approx(0.0, abs=1e-6)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = make_net()
    x = torch.tensor([[0, 1, 2], [2, 0, 1]])
    before = net(x)
    net.save(str(tmp_path / "net.pt"))
    loaded = EinsumNetwork.load(str(tmp_path / "net.pt"))
    assert torch.allclose(before, loaded(x))


def test_predict_map():
    torch.manual_seed(0)
    net = make_net()
    data = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]] * 20)
    net.fit_em(data, epochs=6)
    best, log_p = net.predict(torch.tensor([0, 0, UNKNOWN]), var=2)
    assert best == 0
    assert log_p.shape[0] == 3
