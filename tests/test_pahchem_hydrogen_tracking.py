"""pahchem.hydrogen_tracking: invariants on small synthetic (Z, N_H) networks (no data needed)."""
import numpy as np
import pytest

from pycalima.pahchem import hydrogen_tracking as ht
from pycalima.pahchem.network import PAHNetwork


def toy(charge_scale=1.0, nh_dependent_charge=True, nh_max=4, nh0=3):
    """Z = 0, +1; N_H = 0 ... nh_max: H loss faster for cations, H addition, ionisation that rises with N_H."""
    nh = np.arange(nh_max + 1)
    net = PAHNetwork((0, 1), nh)
    ion = 1e-3 * (1.0 + (nh if nh_dependent_charge else 0 * nh))
    net.add("ionisation", charge_scale * np.array([ion, 0 * ion]), dz=+1)
    net.add("recombination", charge_scale * np.array([0 * ion, np.full(nh.size, 5e-3)]), dz=-1)
    loss = np.where(nh > 0, 1e-6 * (1 + (nh > nh0) * 30.0), 0.0)
    net.add("H loss", np.array([loss, 3.0 * loss]), dnh=-1)
    add = np.where(nh < nh_max, 2e-6, 0.0)
    net.add("H addition", np.array([add, 0.5 * add]), dnh=+1)
    net.add("H2 loss", np.array([np.where(nh >= 2, 1e-7, 0.0)] * 2), dnh=-2, h2_released=1)
    return net, nh0


def test_relaxation_rate_of_a_two_state_chain():
    net = PAHNetwork((0,), (0, 1))
    net.add("loss", [[0.0, 3e-3]], dnh=-1)
    net.add("add", [[1e-3, 0.0]], dnh=+1)
    assert ht.relaxation_rates(net, 1)[0] == pytest.approx(4e-3, rel=1e-10)


def test_charge_rates_are_the_charge_steps_alone():
    net, _ = toy()
    r = ht.charge_rates(net)
    ion = 1e-3 * (1.0 + np.arange(5))
    assert np.allclose(r, ion + 5e-3, rtol=1e-10)              # a two-state charge chain at each N_H


def test_fast_charge_elimination_is_exact_in_the_limit():
    net, nh0 = toy(charge_scale=1e6)
    full = net.steady_state()
    chain, fz = ht.eliminate_charge(net, True)
    lifted = fz * chain.steady_state()[0][None, :]
    assert np.allclose(lifted, full, atol=1e-6)
    # time evolution from the intact neutral
    f0 = net.zeros()
    f0[0, nh0] = 1.0
    t = np.geomspace(1e3, 1e8, 6)
    p = chain.evolve(f0.sum(axis=0)[None, :], t)
    assert np.allclose(fz[None] * p, net.evolve(f0, t), atol=1e-5)


def test_intact_charge_elimination_differs_only_where_charge_depends_on_nh():
    net, nh0 = toy(nh_dependent_charge=False)
    a, fa = ht.eliminate_charge(net, True)
    b, fb = ht.eliminate_charge(net, False, nh0)
    assert np.allclose(fa, fb) and np.allclose(a.steady_state(), b.steady_state())
    net, nh0 = toy(nh_dependent_charge=True)
    _, fa = ht.eliminate_charge(net, True)
    _, fb = ht.eliminate_charge(net, False, nh0)
    assert not np.allclose(fa, fb)
    assert np.allclose(fb, fa[:, [nh0]])


def test_closure_is_exact_in_steady_state_and_flows_towards_it():
    net, nh0 = toy()
    cl = ht.Closure(net, nh0)
    o = ht.observables(net, net.steady_state(), nh0)
    assert cl.x_eq == pytest.approx(o["x"], rel=1e-12)
    assert np.allclose(cl.distribution(cl.x_eq), net.steady_state(), atol=1e-3)
    assert cl.dxdt(0.5 * cl.x_eq) > 0 > cl.dxdt(0.5 * (cl.x_eq + cl.x[-1]))
    x = cl.step(cl.x[0], 1e9)
    assert x == pytest.approx(cl.x_eq, abs=1e-6)
    assert cl.linear_rate() > 0


def test_closure_of_a_two_state_chain_is_exact():
    net = PAHNetwork((0,), (0, 1))
    net.add("loss", [[0.0, 3e-3]], dnh=-1)
    net.add("add", [[1e-3, 0.0]], dnh=+1)
    cl = ht.Closure(net, 1)
    for x0, dt in ((0.0, 100.0), (1.0, 300.0)):
        x_eq = 0.25
        exact = x_eq + (x0 - x_eq) * np.exp(-4e-3 * dt)
        assert cl.step(x0, dt) == pytest.approx(exact, abs=2e-4)
    assert cl.linear_rate() == pytest.approx(4e-3, rel=1e-3)
    assert cl.exp_step(1.0, 300.0) == pytest.approx(0.25 + 0.75 * np.exp(-1.2), abs=2e-4)


def test_exp_step_converges_to_the_closure_ode():
    net, nh0 = toy()
    cl = ht.Closure(net, nh0)
    x0, T = cl.x[1], 3e5
    exact = cl.step(x0, T)
    errs = []
    for n in (1, 10, 100):
        x = x0
        for _ in range(n):
            x = cl.exp_step(x, T / n)
        errs.append(abs(x - exact))
    assert errs[2] < errs[0] and errs[2] < 1e-2 * abs(x0 - exact) + 1e-6


def test_evolve_piecewise_conserves_and_matches_evolve():
    net, nh0 = toy()
    f0 = net.zeros()
    f0[1, 0] = 1.0
    f = ht.evolve_piecewise([net, net], [1e5, 2e5], f0)
    assert np.allclose(f.sum(axis=(1, 2)), 1.0)
    assert np.allclose(f[-1], net.evolve(f0, [3e5])[0], atol=1e-10)


def test_settling_time():
    t = np.arange(10.0)
    y = np.exp(-t)
    assert ht.settling_time(t, y, 0.9, 0.0) == pytest.approx(3.0)          # e^-3 < 0.1 < e^-2
    assert ht.settling_time(t, np.ones(10)) == 0.0
    assert ht.settling_time(t, np.linspace(0, 1, 10), 0.9, 2.0) == np.inf


def test_observables_of_single_states():
    net, nh0 = toy()
    f = net.zeros()
    f[1, nh0 + 1] = 1.0                                 # one extra H on a cation
    o = ht.observables(net, f, nh0)
    assert o["x"] == pytest.approx((nh0 + 1) / nh0)
    assert (o["f_superh"], o["f_cation"], o["f_intact"], o["f_bare"]) == (1.0, 1.0, 0.0, 0.0)
    assert o["r_ali"] == pytest.approx(2.0 / (nh0 - 1))  # a CH2 out of one of the aromatic CH
    f = net.zeros()
    f[0, 0] = 1.0
    assert ht.observables(net, f, nh0)["f_bare"] == 1.0


def test_population_chains_reproduce_the_population_steady_state():
    from pycalima._datasets import DatasetUnavailable
    from pycalima.pahchem import ames_suite, population
    from pycalima.pahchem.baseline import Environment

    try:
        lib = population.library_m20(ames_suite.ascii_path())
    except DatasetUnavailable:
        pytest.skip("PAHdb v3.20 not installed")
    mol = min((m for m in lib if m.ea_ev is not None), key=lambda m: m.nc)
    pop = population.Population([mol], {"all": (1.0, 1e4)})
    for env in (Environment("lit", 1e4, 1e4, 10.0, 500.0, 40000), Environment("dim", 3.0, 1e2, 0.1, 100.0, 40000)):
        pop.prepare(population.field_shape(env), workers=1)
        (chain,) = ht.population_chains(pop, env)
        assert np.allclose(chain.steady_state()[0], pop._chain_distributions(env)[0], rtol=1e-6, atol=1e-12)
