"""What the hydrogenation of PAHs changes, per (Z, N_H) state: the observables of the equilibrium tables and their scoring.

The RAMSES tables of the PAH hydrogenation state are equilibrium tables, built offline from pahchem's joint (Z, N_H)
chain (``Params.hydrogen_chain = "joint"``). They are judged on what the hydrogenation changes, not on the state shares
(docs/physics/pah-hydrogen.md; the literature review in results/carbon_plan/pah_hydrogenation_review/, 2026-11-15).
Per molecule [per second, or erg s^-1]:

- **carbon**: C atoms lost (``Population.destruction_rates``' "carbon"): the bare skeleton's destruction channels, the
  C2H2 loss of the emitting states N_H0 ... N_H0-3 (the super-hydrogenated states lose C2H2 as the parent), the
  explosions past the top charge and the passages beyond the chain (a whole molecule each), the thermal ions'
  knock-outs;
- **heating**: photoelectric heating (``Population.photoelectric_heating``); **cooling**: recombination cooling
  (``Population.recombination_cooling``);
- **h2**: H2 released (H2 loss and Eley-Rideal abstraction: the network's processes with ``h2_released``);
  **h_uptake**: H atoms taken from the gas (the processes with dN_H > 0, H addition and associative detachment);
- **absorbed**: the UV power absorbed (the denominator of ``photoelectric_heating``'s efficiency);
- **uv_aro**, **uv_ali**: absorbed power times the aromatic and the aliphatic CH count of the state. Each extra H
  makes a CH2 out of an aromatic CH (N_H0 - j aromatic CH, 2j aliphatic) until no aromatic CH is left; a
  dehydrogenated state has N_H aromatic CH. The 3.3 and 3.4 um bands follow these first moments (linear in each,
  Murga+22 App. A); their ratio times the band-strength ratio A3.4/A3.3 per bond (an input) is a proxy for I3.4/I3.3;
- the shares **bare** (N_H = 0), **dehydrogenated** (0 < N_H < N_H0), **intact**, **superh** (> N_H0), **at_cap**
  (the top of the ladder, N_H0 + ``superh_levels``: where it holds more than a few per cent the cap, not the chemistry,
  sets the super-hydrogenation), **cation**, **anion**, and **n_h** (<N_H>).

Each is a weight W[Z, N_H] per molecule (``state_weights``) on its ``Population.joint_networks`` network's charges and
nh_values, so that O = sum W F (``observe``) for any state F: a steady state, or one evolved along a path
(``hydrogen_tracking.evolve_piecewise``). At F = ``joint_fractions`` the weights give Population's own steady-state
functions (tests/test_pahchem_hydrogen_observables.py), except the C2H2 of the deposits: Population takes it as the
N_H marginal times a deposit rate over the charge marginal (``loss_rates``), these from each source state's own share,
linear in F.

**Carbon, heating, cooling and absorbed** come from ``Population.joint_state_observables`` where it exists (pahchem's
answer to the request of 2026-11-15, results/carbon_plan/request_pahchem_deep_superh.md; branch
feature/pahchem-joint-observables). It takes heating, cooling and explosions from the network's own rates: each N_H's
shifted threshold, the per-N_H attachment. Without it, a provisional version is built from Population internals
(``_pe_stack``, ``_capture_rates``, ``_capture_energy``, ``_bare_stack``, ``_ladders``, ``_deposit_flows``,
``_deposit_losses``, ``_past_top``, ``_interp_g0``). That version uses the intact molecule's threshold for every N_H:
stripped C90 at Z = +3, which the network holds there by its raised threshold, is given the intact +3 state's
photoelectric heating, about 36x too much. The test compares either with the public functions.
"""
from __future__ import annotations

import numpy as np

OBSERVABLES = ("carbon", "heating", "cooling", "h2", "h_uptake", "absorbed", "uv_aro", "uv_ali",
               "bare", "dehydrogenated", "intact", "superh", "at_cap", "cation", "anion", "n_h")


def ch_counts(nh, nh0):
    """(aromatic, aliphatic) CH counts of N_H = ``nh`` (array) for a parent with N_H0 = ``nh0`` (module docstring)."""
    nh = np.asarray(nh)
    extra = np.maximum(nh - nh0, 0)
    aro = np.maximum(np.minimum(nh, nh0) - extra, 0)
    return aro, nh - aro


def _unit_slots(pop, i):
    """f[s, slot]: 1 in slot i for every molecule, the charge fractions that pick one source charge out of a linear map."""
    f = np.zeros((len(pop.library), len(pop.zs)))
    f[:, i] = 1.0
    return f


def _carbon_slots(pop, env):
    """(bare[s, slot], every[s, slot], c2h2[s, slot, k], c2h2_dep[s, slot, k]): C atoms lost per second by a molecule in
    charge slot at N_H = 0 (``destruction_rates``' bare channels, photon-driven and after the deposits) and at every N_H
    (explosions, passages beyond, knock-outs); C2H2 losses per second in slot at N_H0 - k (the loss tables), and of the
    deposits out of slot (``_deposit_losses``, its source charge)."""
    from pycalima.pahchem import population as P

    shape = P.field_shape(env)
    n, m = len(pop.library), len(pop.zs)
    nc_lost = {name: k for mol in pop.library for name, k, *_ in P.bare_channels(mol, pop.p)}
    bare = np.zeros((n, m))
    for c, tab in pop._bare_stack(shape).items():
        bare += nc_lost.get(c, 0) * pop._interp_g0(tab, env.g0)
    use = lambda fl: fl[2] != "rec" or pop.p.recombination_dissociation      # noqa: E731
    flows = {i: [fl for fl in pop._deposit_flows(env, _unit_slots(pop, i)) if use(fl)] for i in range(m)}
    top = {}                     # one loss grid per molecule, as destruction_rates (a yield depends on the bins below it)
    for fls in flows.values():
        for s, z, which, *_ in fls:
            energies, _ = pop._flow_energies(pop.library[s], z, which)
            top[s] = max(top.get(s, P.LOSS_E_MAX_EV), float(energies.max()) + 5.0)
    for i, fls in flows.items():
        for s, z, which, idx, rate, _ in fls:
            mol = pop.library[s]
            sp = P.bare_species(mol, z)
            energies, W = pop._flow_energies(mol, z, which)
            e_cl, Y = P.bare_event_yields(sp.freqs_cm, sp.einstein_a, P.bare_channels(mol, pop.p), top[s])
            for c, y in Y.items():
                p = np.interp(energies * P.dos.CM_PER_EV, e_cl, y)
                pe = float(W[idx] @ p) if W is not None else float(p[idx])
                bare[s, i] += nc_lost.get(c, 0) * rate * pe
    nc = np.asarray(pop.nc, dtype=float)[:, None]
    every = np.zeros((n, m))
    past = pop._past_top(env, pop._single_up(env))[0] * nc
    routed = pop._routed()[:, None]
    bare += np.where(routed, past, 0.0)                     # past a barrier-limited top only the bare skeleton explodes
    every += np.where(routed, 0.0, past)
    cr, hot = pop._cr_rates(env), pop._hot_rates(env)
    if cr is not None:
        every += nc * cr["beyond"]
    if hot is not None:
        every += nc * hot["beyond"] + hot["knockout_c"]
    c2h2 = pop._interp_g0(pop._ladders(shape)[2], env.g0)
    c2h2_dep = np.zeros_like(c2h2)
    for i in range(m):
        c2h2_dep[:, i] = pop._deposit_losses(env, _unit_slots(pop, i), by_state=True)[2].sum(axis=1)
    return bare, every, c2h2, c2h2_dep


def state_weights(pop, env, nets=None):
    """[{name: W[Z, N_H]}] of every molecule of ``pop`` (a joint-chain ``population.Population``) at ``env``: the weights
    of ``OBSERVABLES`` on the charges and nh_values of its ``joint_networks`` (``nets``, built if not given)."""
    from pycalima.pahchem import chemistry, photo
    from pycalima.pahchem import population as P

    if not pop._joint_on():
        raise ValueError('state_weights needs Params.hydrogen_chain = "joint"')
    nets = pop.joint_networks(env) if nets is None else nets
    native = getattr(pop, "joint_state_observables", None)
    if native is not None:              # pahchem's own weights (feature/pahchem-joint-observables, 2026-11-15)
        rates = native(env)
    else:
        shape = P.field_shape(env)
        T = env.temperature_k
        G, A = pop._pe_stack(shape, photo.PHOTOELECTRON_PARTITION)
        heat, absorbed = (env.g0 * x * chemistry.EV_ERG for x in (G, A))
        cool = env.n_e * pop._capture_rates(T) * pop._capture_energy(T) * chemistry.EV_ERG
        bare_c, every_c, c2h2, c2h2_dep = _carbon_slots(pop, env)
    out = []
    for s, (mol, net) in enumerate(zip(pop.library, nets)):
        lo = pop.zs.index(net.charges[0])
        sl = slice(lo, lo + len(net.charges))
        nh, z = np.asarray(net.nh_values), np.asarray(net.charges)
        nh0 = mol.nh
        if native is not None:
            w = {k: np.asarray(rates[s][k], dtype=float) for k in ("carbon", "heating", "cooling", "absorbed")}
            if any(v.shape != net.shape for v in w.values()):
                raise ValueError(f"{mol.name}: joint_state_observables' weights are not on its network's states")
        else:
            row = lambda v: np.repeat(np.asarray(v)[sl, None], nh.size, axis=1)      # noqa: E731  (per charge)
            # C2H2 of N_H0 - k (the loss tables and the deposits), k < min(N_EMITTING, N_H0); super-H states as N_H0
            k = np.where(nh > nh0, 0, nh0 - nh)
            emit = (k < min(P.N_EMITTING, nh0)) & (nh >= 1)
            kk = np.clip(k, 0, P.N_EMITTING - 1)
            cc = np.where(emit[None, :], c2h2[s, sl][:, kk] + c2h2_dep[s, sl][:, kk], 0.0)
            carbon = row(every_c[s]) + 2.0 * cc
            carbon[:, nh == 0] += bare_c[s, sl][:, None]
            w = {"carbon": carbon, "heating": row(heat[s]), "cooling": row(cool[s]), "absorbed": row(absorbed[s])}
        aro, ali = ch_counts(nh, nh0)
        w["uv_aro"], w["uv_ali"] = w["absorbed"] * aro[None, :], w["absorbed"] * ali[None, :]
        w["h2"] = sum((q.h2_released * q.rate for q in net.processes if q.h2_released), np.zeros(net.shape))
        w["h_uptake"] = sum((q.dnh * q.rate for q in net.processes if q.dnh > 0), np.zeros(net.shape))
        ones = np.ones(net.shape)
        for name, sel in (("bare", nh == 0), ("dehydrogenated", (nh > 0) & (nh < nh0)), ("intact", nh == nh0),
                          ("superh", nh > nh0), ("at_cap", (nh == nh.max()) & (nh > nh0))):
            w[name] = ones * sel[None, :]
        w["cation"], w["anion"] = ones * (z > 0)[:, None], ones * (z < 0)[:, None]
        w["n_h"] = ones * nh[None, :]
        out.append(w)
    return out


def observe(weights, f):
    """{name: sum W F} of one molecule's weights and its state f[Z, N_H] (or a stack [..., Z, N_H])."""
    f = np.asarray(f, dtype=float)
    return {k: (w * f).sum(axis=(-2, -1)) for k, w in weights.items()}


def steady_observables(pop, env):
    """{name: [s]}: ``OBSERVABLES`` of every molecule in the joint chain's steady state at ``env``: ``steady_state_closed``
    of its ``joint_networks``, built once (``joint_fractions`` solves the same networks)."""
    nets = pop.joint_networks(env)
    W = state_weights(pop, env, nets)
    obs = [observe(w, net.steady_state_closed()) for w, net in zip(W, nets)]
    return {k: np.array([o[k] for o in obs]) for k in OBSERVABLES}
