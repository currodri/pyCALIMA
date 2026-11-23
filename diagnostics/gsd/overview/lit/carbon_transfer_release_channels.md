# Literature: release channels for PAHs adsorbed on a-C nano-grains (2026-11-23)

Sonnet subagent, read-only. Full text (grep'd parts only): Jones & Ysard 2025, Hirashita & Murga 2020, Narayanan+26,
lit_release.md. Everything else abstract or snippet; A&A pages (Dartois+22, Pino+19, Dartois+23) returned 403; Maté+23
text unreadable. M = measured, C = computed, A = assumed.

Bottom line: no measured or modelled release channel works for a C24-C100 PAH on a 3-20 nm carbon grain. Only
non-thermal UV photodesorption (unmeasured) and cosmic-ray track sputtering (measured only for thick PAH films) are
plausible; neither reaches Myr in dense clouds on literature numbers.

| channel | CNM (n_H 30, G0 1) | dense (1e4, G0 1e-3, zeta 1e-16) | robustness |
|---|---|---|---|
| non-thermal UV photodesorption | 0.5 yr / Y_abs | 450 yr / Y_abs | Y unmeasured (no PAH yield from carbon or ice found) |
| CR track sputtering | ~30-100 Myr (~1/zeta) | ~30-100 Myr | agent's estimate (Fe flux ~1e-2 cm^-2 s^-1, 3 nm track); yields not read |
| CR spot / whole-grain heating | inf | inf | Ivlev+15: explosive desorption ~ a^5, few tens of K; 3 nm grain 100-200 K peak |
| reactive desorption | >> Gyr | >> Gyr | agent's argument (Minissale+16 applies to small products on bare surfaces) |
| grain-grain collisions | inf (coagulation buries) | inf | Brownian ~2 m/s, ~1e-3 eV |
| shocks | >~ 1e8 yr | >~ 1e8 yr | carbon grain lifetimes 0.6-2.6e8 yr (Bocchio+14), PAH few 1e8 yr (Micelotta+10) |
| host photo-fragmentation | inf (>= 3 nm) | inf | Jones & Ysard 2025: larger grains stable; fragments C4H6-C5H7, not PAHs |
| thermal single-photon | ~inf (3 nm) | ~inf | lit_release.md |
| turbulent mixing into UV-lit gas | ~10-30 Myr (A) | ~10-30 Myr | assumed, no PAH-specific numbers |

Photodesorption scaling (A/C): a C50 absorbs ~5e-16 cm^2; CNM (1.3e8 FUV photons cm^-2 s^-1): one absorption per ~0.5
yr, tau_rel ~ 0.5 yr / Y_abs: Y ~ 1e-6 gives ~0.5 Myr, ~1e-7 balances freeze-out; dense (~1.3e5 FUV + ~1e4 CR-UV):
tau_rel ~ 450 yr / Y_abs, release in < 1 Myr needs Y >~ 5e-4 (unlikely given fast quenching on a conducting substrate,
A). Adsorbed benzene/toluene are more UV-stable than in the gas (review, snippet). Ice VUV yields of larger molecules are
tiny (formic acid < ~5e-6 per photon, Bertin+23 as quoted).

CR sputtering: Dartois+22 (abstract): perylene and coronene films under swift ions; Breuer+16: coronene films sputter
"largely ... nearly fragment-free" as intact neutrals; Pino+19: swift heavy ions on soot release neutral PAHs of up to
several tens of C and fullerenes (no absolute yields). Maté+23: the CR route needs 1e-3 to 1e-4 naphthalene fractions in
ice mantles to explain TMC-1; a 2026 TMC-1 review calls CR sputtering "the main desorption mechanism in UV-shielded cold
clouds" but the implied ice reservoir "unrealistically large".

Models: Hirashita & Murga 2020 and Narayanan+26: coagulation removes PAHs in dense gas, no release term; Lange+23 (discs)
the only PAH release term found; no paper argues freeze-out balances release in molecular clouds.

Ranking for balancing freeze-out at GMC scale: 1. non-thermal photodesorption in UV-lit gas (Y ~1e-7 suffices in the CNM;
dense clouds would need ~1e-3, unlikely) - needs lab data; 2. turbulent mixing (~10-30 Myr, unquantified); 3. CR track
sputtering (30-100 Myr, too slow at n_H 1e4); others negligible. Also check whether the model's sticking or burial is
overestimated.
