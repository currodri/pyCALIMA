"""pahchem.hydrogen_observables: per-state weights of what the hydrogenation changes."""
import numpy as np
import pytest

from pycalima.pahchem import hydrogen_observables as ho


def test_ch_counts():
    nh0 = 12
    nh = np.arange(0, nh0 + 25)
    aro, ali = ho.ch_counts(nh, nh0)
    assert np.all(aro + ali == nh) and np.all(aro >= 0) and np.all(ali >= 0)
    assert np.all(aro[: nh0 + 1] == nh[: nh0 + 1]) and np.all(ali[: nh0 + 1] == 0)   # dehydrogenated: aromatic only
    j = np.arange(1, nh0 + 1)
    assert np.all(ali[nh0 + j] == 2 * j)                                             # each extra H makes a CH2
    assert ali[-1] == nh[-1]                                                         # fully aliphatic past +N_H0


# the joint chain as pahchem builds it by default, and its deep super-hydrogenated chain (to +8, with the C2H2 loss of
# the super-hydrogenated states, whose carbon the weights must carry too)
CHAINS = {"andrews16": {}, "deep": dict(superh_chemistry="deep", superh_levels=8, superh_carbon_loss="murga22")}


@pytest.fixture(scope="module", params=list(CHAINS))
def joint_pop(request):
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_suite, population

    kw = CHAINS[request.param]
    if any(not hasattr(population.Params, k) for k in kw):
        pytest.skip(f"this pahchem has no {request.param} super-hydrogenated chain")
    try:
        lib = population.library_m20(ames_suite.ascii_path())
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed")
    mols = sorted((m for m in lib if m.ea_ev is not None), key=lambda m: m.nc)
    return population.Population([mols[0], mols[len(mols) // 2]], {"all": (1.0, 1e4)},
                                 params=population.Params(hydrogen_chain="joint", **kw))


ENVS = (("lit", 1e4, 1e4, 10.0, 500.0), ("stripping", 1e4, 1e2, 1.0, 1000.0), ("dense", 0.1, 1e5, 1e-2, 30.0),
        ("cnm", 1.7, 30.0, 0.03, 100.0))


@pytest.mark.parametrize("cond", ENVS, ids=[c[0] for c in ENVS])
def test_weights_reproduce_the_population_steady_state(joint_pop, cond):
    """At F = joint_fractions the weights give Population's own functions."""
    from pycalima.pahchem import population
    from pycalima.pahchem.baseline import Environment

    pop = joint_pop
    env = Environment(cond[0], *cond[1:], 40000)
    pop.prepare(population.field_shape(env), workers=1)
    o = ho.steady_observables(pop, env)
    summary = pop.hydrogenation_summary(env)
    nh0 = np.array([m.nh for m in pop.library])
    exact = {"heating": pop.photoelectric_heating(env)[0], "cooling": pop.recombination_cooling(env)[0],
             "h2": summary["h2"], "bare": summary["bare"], "superh": summary["superh"], "at_cap": summary["at_cap"],
             "n_h": summary["x"] * nh0}
    for k, v in exact.items():
        assert o[k] == pytest.approx(v, rel=1e-9, abs=1e-300), k
    # Population's "dehydrogenated" counts the bare skeleton too
    assert o["dehydrogenated"] + o["bare"] == pytest.approx(summary["dehydrogenated"], rel=1e-9, abs=1e-300)
    # pahchem's joint_state_observables carries the carbon exactly; the fallback (older pahchem) to 1e-3, as that
    # pahchem took the deposits' C2H2 as a product of marginals
    rel = 1e-9 if hasattr(pop, "joint_state_observables") else 1e-3
    assert o["carbon"] == pytest.approx(pop.destruction_rates(env)["carbon"], rel=rel, abs=1e-300)
    shares = o["bare"] + o["dehydrogenated"] + o["intact"] + o["superh"]
    assert shares == pytest.approx(1.0, abs=1e-12)
    absorbed = pop.photoelectric_heating(env)
    eps = absorbed[2]
    assert o["absorbed"] == pytest.approx(np.divide(absorbed[0], eps, out=np.zeros_like(eps), where=eps > 0), rel=1e-9)


def test_weights_are_linear_in_any_state(joint_pop):
    """O = sum W F holds for a state off the steady state; h2 is the network's own H2 rate."""
    from pycalima.pahchem import population
    from pycalima.pahchem.baseline import Environment

    pop = joint_pop
    env = Environment("cnm", 1.7, 30.0, 0.03, 100.0, 40000)
    pop.prepare(population.field_shape(env), workers=1)
    nets = pop.joint_networks(env)
    W = ho.state_weights(pop, env, nets)
    rng = np.random.default_rng(1)
    for w, net in zip(W, nets):
        F = rng.random((3,) + net.shape)
        F /= F.sum(axis=(1, 2), keepdims=True)
        o = ho.observe(w, F)
        assert o["h2"] == pytest.approx([net.h2_formation_rate(f) for f in F], rel=1e-12)
        assert np.all(o["cation"] + o["anion"] <= 1.0 + 1e-12)
        assert np.all(o["uv_aro"] + o["uv_ali"] <= o["absorbed"] * max(net.nh_values) * (1 + 1e-12))
