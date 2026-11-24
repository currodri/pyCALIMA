# Literature: electron sticking on PAH-sized cations and small carbon grains versus size (2026-11-23)

Sonnet subagent, read-only. Read from local text: WD01 (Sect. 3.1), Andrews+16 App. A, Biennier+06 Sect. 3.4, Bierbaum+11
excerpt, DustEM doc, Thi+19/20 App. B, Bakes & Tielens 1998, the DESIREE C60+/C60- paper (arXiv 2409.11851, abstract and
Sect. 3). Web abstracts/snippets only: Viggiano+10, Smith/Spanel, Tobita. Not fetched (unverified): Draine & Sutin 1987,
Bakes & Tielens 1994 (scanned), Ibáñez-Mejía+19, Hoang, Carelli, Gianturco, Lindén+16. Kimura+16's s_e not found.

Bottom line: no measurement of electron recombination on cations beyond N_C = 16; lab_cold's saturation (s ~ 1 by N_C
~35) and WD01's 0.2-0.3 are both extrapolations. The only data above N_C 16 are C60, for attachment to the neutral,
revised upward ~1000x in 2010. WD01's cation law (eq. 30) has no small-molecule factor; the logistic is only in the
neutral law (eq. 28).

WD01 (ApJS 134, 263, Sect. 3.1): "We will assume that p_es ~ 0.5, so that the maximum possible value of s_e would be
(1 - p_es) ~ 0.5" (assumption); inelastic scattering "1 - e^{-a/le}, where le ~ 10 A" (assumption); neutral eq. 27
s_e = (1 - p_es)(1 - e^{-a/le}) p_rad with p_rad "poorly known", so eq. 28 s_e(0) = 0.5 (1 - e^{-a/le}) / (1 + e^{20 -
N_C}), fitted to neutral attachment data (Tobita+92 N_C 13-20, anthracene, C60/C70; "tetracene and perylene would be
larger by an order of magnitude at T ~ 2000 K"; C60/C70 "remarkably low ... unusual rigidity and symmetry", 0.26 eV
barrier); cation eq. 30 s_e = 0.5 (1 - e^{-a/le}), compared with 300 K recombination of small hydrocarbon cations and
alkanes: "agree ... within about a factor of 2 except ... naphthalene, smaller than predicted by about a factor of 4";
Z < 0: s_e(0) above Z_min, 0 at Z_min. DustEM keeps alpha = 0.5, beta = 1, l_e from Kimura+16.

Data (implied sticking from Draine & Sutin neutral-limit geometry, agent's computation, not verified):
| species | N_C | z | T | k [cm3/s] | implied s | source | robustness |
|---|---|---|---|---|---|---|---|
| C6H6+ ... C16H10+ (9) | 6-16 | +1 | 300 K | 3e-7 - 4.1e-6 | lab fit | pah_electron_recombination.csv | measured; naphthalene 3x systematic |
| C60 | 60 | 0 | ~300 K | ~1e-6, small negative T coefficient | ~1 | Viggiano+10 (abstract) | measured; supersedes the 1990s data |
| C60 (old) | 60 | 0 | 500-4500 K | ~3e-7 at 4500 K | low | Smith/Spanel, used by WD01 | superseded |
| anthracene | 14 | 0 | 48-300 K | ~1e-9 - 3e-9 | ~3e-3 | Canosa+94, Moustefaoui+98 | measured; low EA |
| C60+ + e- | 60 | +1 | - | no measured rate | - | 2409.11851 | DESIREE measured C60+ + C60- only |
| circumcoronene+ | 54 | +1 | 300 K | 1.4e-5 | model | Montillaud+13 (snippet) | computed |

Other models: Bakes & Tielens 1998: neutral PAHs > ~25 C have s close to unity (EA > 1 eV); recombination rates "are
expected to approach the theoretical grain rates with increasing size". Andrews+16 App. A: cations sticking unity
"given the attractive interaction". Thi+20: "S close to unity" for N_C > 20. Biennier+06: s -> 1 for nm particles
"reasonable", "no way of estimating this coefficient for smaller molecules". Thi+19: S_e = 0.5 (Umebayashi & Nakano
1980, "most cases larger than 0.3"). Viggiano+10: capture theory overshoots C60 and needs empirical corrections.

Recommended (judgement): z > 0: s_+ = s_lab(N_C) for N_C <= 16; beyond, between WD01 (~0.2-0.3) and the lab fit (1):
a range 0.3-1 at N_C 35-400, uncertainty ~x3 for N_C 20-400; neutrals/anions governed by EA (keep WD01's logistic and
Z_min). Next cheap step: read Viggiano+10 in full.
