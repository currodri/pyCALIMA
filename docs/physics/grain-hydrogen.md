(physics-grain-hydrogen)=
# Hydrogen of a-C(:H) grains: tabulate or track?

With rung M1 the covalent carbon bins (1–53 nm) carry the hydrogen of their grains
($M_{\rm H}$ beside $M_{\rm C}$), and the H/C sets each grain's band gap and so its
optics and charging ({doc}`carbon-optics`, {doc}`charging-cm20`). This page asks
whether that hydrogen needs to be tracked, or whether a table in local equilibrium
would do. It covers the grain half of the plan's P2 tests (T1, T4–T6); the PAH half
waits for the molecular network. Montillaud et al. (2013, Sect. 5.3) find the same
question open for large PAHs, whose hydrogenation takes $10^2$–$10^4$ yr in NGC 7023.

## The laboratory laws

`gsd.hydrogen` evolves a bin's carbon-weighted H/C $x$ as

$$
\frac{dx}{dt} = -k_{\rm UV}(x)\,(x - x_{\min}) + k_{\rm H}\,(x_{\max} - x).
$$

Every number is a laboratory one, and each is labelled in the module:

- **UV loss** is first order in the C–H bonds (Mennella et al. 2001). The rate is
  $k_{\rm UV} = Y_{\rm CH}\times$ (photons absorbed per C atom, 6.8–13.6 eV), with
  $Y_{\rm CH} = 0.02$ bonds per absorbed photon (Alata et al. 2014; Maté 2014; range
  0.014–0.025). This gives $\sigma_d \approx 1.4\times10^{-19}$ cm², inside the
  measured $1$–$3\times10^{-19}$. The absorption is the bin's own at its current band
  gap, so $k_{\rm UV}$ depends on $x$. The lamps stop at 10.5 eV; 10.5–13.6 eV is
  assumed to act alike.
- **The floor** $x_{\min}$ is the lab limit of UV processing, $X_{\rm H}\approx 0.05$
  ($E_g$ = 0.215 eV, Jones 2012 III).
- **Re-hydrogenation** by H atoms is $k_{\rm H} = \sigma_f(T)\,n({\rm H\,I})\,v_{\rm H}/4$
  (the H flux per unit surface). It uses $\sigma_f = 1.9\times10^{-18}$ cm² (Mennella et
  al. 2002) with Mennella (2006)'s 70 K activation energy, normalised at 300 K, and
  saturates at Mennella's plateau, bulk H/C = 0.65 ($E_g$ = 1.69 eV).

Mennella (2002) estimated a diffuse-ISM balance of $\Gamma = k_{\rm UV}/k_{\rm H}
\approx 0.5$ with an H flux of $8\times10^6$ cm⁻² s⁻¹, which is $n v$, not $n v/4$. With
the flux per unit surface and the temperature dependence, re-hydrogenation in the CNM
is ~10 times slower than that, and the covalent grains sit near the dehydrogenated end
(bulk $E_g$ 0.3–0.65 eV) in diffuse gas.

## T1: how fast it relaxes

Relaxation time $1/(k_{\rm UV}+k_{\rm H})$ at equilibrium, bin 4 (1.06–2.82 nm), 100 K
(`diagnostics/gsd/check_hydrogen_tracking.py`):

| FUV [Mathis] \ $n$(H I) [cm⁻³] | 0.1 | 3.2 | 100 | 3200 | $10^5$ |
|---|---|---|---|---|---|
| $10^{-3}$ | $2\times10^6$ yr | $2\times10^5$ | $7\times10^3$ | 230 | 7 |
| $10^{-1}$ | $3\times10^4$ | $2.5\times10^4$ | $6\times10^3$ | 230 | 7 |
| 1 | 2900 | 2900 | 2000 | 210 | 7 |
| $10^2$ | 29 | 29 | 29 | 25 | 6 |
| $10^4$ | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |

The equilibrium is the UV floor wherever FUV/$n$(H I) $\gtrsim 10^{-2}$, and the H
plateau where it is $\lesssim 10^{-4}$. Bins 5–7 behave alike; bin 7 (20–53 nm) is about
1.7 times slower, because it absorbs less per carbon atom. Charge relaxes in hours to
weeks, so it can stay tabulated, conditional on the hydrogen.

## T4: parcels crossing a PDR

Parcels are carried at 0.3, 1 and 3 km/s through the Orion Bar (isobaric) and Horsehead
profiles of Murga et al. (2020; their Fig. 2 data on Figshare), into the cloud and out
of it. Outward parcels start at the H plateau: hydrogenated material leaving the cloud.
A third case cycles between CNM and WNM every half-period. The true H/C (implicit ODE)
is compared with the equilibrium a table would give
(`results/carbon_plan/check_hydrogen_tracking_paths.png`):

- **Orion Bar:** relaxation takes under a year to $A_V \approx 4$, and the grains follow
  equilibrium. The large errors in the script's summary come from the profile's last
  point ($A_V \approx 5$), where $n$(H I) rises again and equilibrium jumps to the plateau.
- **Horsehead, inward:** equilibrium holds in the lit layer ($A_V \lesssim 1.5$). Deeper,
  the relaxation time passes the time left to the edge, and the grains keep their
  dehydrogenated state: 0.24–0.44 eV against the 0.215 eV floor (|Δ$E_g$| 0.03–0.48 eV).
- **Horsehead, outward:** cloud material stays hydrogenated until $A_V \approx 2.5$–3 and
  reaches equilibrium only by $A_V \approx 1.5$–2. Equilibrium is wrong by 0.5–1.4 eV in
  between.
- **CNM ↔ WNM:** with periods of $10^4$–$10^5$ yr the equilibrium is off by 0.14–0.28 eV;
  at $10^6$ yr by 0.04–0.09 eV.

These errors matter because the far-IR opacity of a-C(:H) is extremely sensitive to the
band gap near the a-C end. 0.1 eV is a factor ~2–3 in $\kappa$(100 µm); the largest
errors above reach $\Delta\ln\kappa_{100} \approx 8$–9. At 3.4 µm and in the visible the
factors are ~3–10 times smaller in the logarithm.

## T6: the step RAMSES would take

RAMSES would apply the exact solution over a hydro step with the rates frozen at the
current state (`BinHydrogen.step`). Against the exact solution under the same held
conditions, at the step ends, it is within 0.05 eV for every path and for steps of
$10^2$–$10^4$ yr. It is unconditionally stable and needs no sub-cycling. What remains
is the sampling of the conditions themselves (held over a step while the parcel moves),
up to ~1 eV for $10^3$–$10^4$ yr steps across the Orion Bar. That is the spatial
resolution any cell-by-cell process has.

## T5: mixing

When two parcels mix, a single tracked H/C is the mean. Grains of different histories,
though, keep their own H, and the far-IR is far from linear in it. Mixing 1:1 at equal
carbon gives $\kappa$(mean H/C) / mean $\kappa$ − 1:

| bin | pair | 3.4 µm | 0.55 µm | 100 µm |
|---|---|---|---|---|
| 4 | CNM equilibrium + WNM equilibrium | −3% | 0% | −24% |
| 4 | CNM equilibrium + H plateau (cloud material) | −67% | −30% | −99.6% |
| 7 | CNM equilibrium + WNM equilibrium | −14% | −3% | −55% |
| 7 | CNM equilibrium + H plateau | −67% | −27% | −99% |

In lit gas the mixture relaxes in $\lesssim 3\times10^3$ yr, so the bias lasts only where
grains of different histories meet in shielded gas.

## Treatment map (grains)

| where | treatment |
|---|---|
| UV-lit gas (FUV $\gtrsim 1$ Mathis, or dense atomic gas), changing on $\gtrsim 10^5$ yr | **E**: an equilibrium table is enough (relaxation $\lesssim 3\times10^3$ yr) |
| PDR transition layers ($A_V \approx 1.5$–3), outflows from clouds, phase cycling faster than $10^5$ yr, shielded gas | **M**: one tracked H/C with the exact step |
| shielded gas where grains of different histories mix, for far-IR observables | **F**: one scalar is not enough (mixing bias up to −99% at 100 µm) |

For grains, then, M1 (tracking $M_{\rm H}$) is needed beyond the lit layers, and the
step is cheap and exact. Whether the F cases matter depends on the far-IR weight of the
1–53 nm grains in shielded gas, which the T4 benchmarks will show.

## Caveats

- The laws are laboratory ones, extrapolated to 13.6 eV and to grains.
- The nano-grains' thermal (multi-photon) channels are missing; they need the P1
  network.
- Accretion of C + H in dense gas (a-C:H mantles) is not here: growth adds carbon only,
  and hydrogen comes only from H atoms. The "H plateau" start for cloud material is
  therefore an assumption.
- The FUV is the Mathis spectrum scaled by the profile's $\chi$, and the profiles are
  Murga et al.'s as given.
