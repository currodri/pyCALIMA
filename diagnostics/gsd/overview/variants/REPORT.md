# PL2M variants at equal memory (N and M per bin)

Report of an Opus subagent (extra-high effort), 2026-11-17/18, saved by the main session (subagents cannot write
report files). Code: `diagnostics/gsd/pl2m_variants.py`, `bench_pl2m_variants.py`, `bench_pl2m_figures.py` (new files;
no existing file edited). Numbers in `results/gsd_overview/variants/*.txt`; figures `variants_*.pdf`. Everything is
measured unless marked [inferred]. The main session spot-checked `growth.txt`, `a0.txt`, `cost.txt`, `drift.txt` against
the tables below and viewed every figure.

Reproduce from the worktree root: activate the venv, `export PYTHONPATH=$PWD/src:$PWD/diagnostics/gsd`, then
`python diagnostics/gsd/bench_pl2m_variants.py {growth|stress|kernels|drift|cost|figures}`, `... a0 --workers 3`,
`... rates` (with `CALIMA_DUSTEM_DIR` set).

## 0. Ranked recommendation

1. **Adopt EA (edge-aware closure), ε = 1e-6.** A bin whose neighbour is (nearly) empty is treated as a front: its
   profile continues the neighbour's power-law exponent, cut off inside the bin. A bin with both neighbours empty is a
   spike at its mean mass (moving pivot).
   - Growth (McKinnon Fig. 10, 8 bins, 100 steps): 14.2 % → 1.6 %; exact to round-off from 64 bins.
   - Sinusoid: no spread past the edge (PL2M: 3 bins at 128). Narrow population grown ×10–×100: 17–20 % → 1e-5.
   - RAMSES-like steps on the real 8-bin silicate and 11-bin carbon grids: mass 0.8 % → 0.12–0.16 %, surface 4 % → 1 %,
     independent of 1e3–1e5 steps.
   - A0 at 8 bins, worst observable: 12.1 % → 4.4 % (accretion 8.1 → 2.8, coagulation 5.9 → 0.3, shattering 7.8 → 4.4,
     sputtering 12.1 → 0.6, SN 3.4 unchanged).
   - Caveats: about half of the A0 gain needs the optics/rate tables to see the EA shape (EA states through PL2M
     tables: worst 7.7 %); ε must be small (ε = 1e-2 at 4–6 bins, η = 53–390, takes MRN slopes for fronts: A0 up to
     92 %); do not use the isolated-bin spike for drift velocities (it freezes in-bin segregation; keep the local
     closure there).
2. **Adopt with EA for rates and tables: 2-point Gauss nodes in radius** (match ∫a^k dN for k = 0–3: N, M and surface
   exact, same pair cost). With EA the worst rate error at 8 bins drops from 6.2 % / 7.8 % (PL2M exact in-bin integral /
   two nodes) to 1.0 % / 2.1 %; drift two-node error 15/23 % → 9/12 % (= per-moment velocities). In collisions it is
   mixed (Golovin 8 bins 37 → 4.3 %; McKinnon coagulation 32 bins 3.3 → 1.1 %; but constant kernel 8 bins 2.3 → 12 %,
   McKinnon coagulation 8 bins 0.4 → 2.6 %, coagulation threshold sweep worse): do not switch the collision nodes yet.
3. **Test further: kernel averaging KA (4×4 sub-slices per node pair)**, products and donor split weighted by rate over
   the sub-pairs. Fixes the sharp shattering threshold (McKinnon Fig. 15, 8 bins: worst bin 15 % → 0.6 %, misplaced
   2.1 % → 0.26 %; threshold swept through a bin: max 2.1 % → 0.6 %); Golovin 37 → 12 % (8 bins), 4.5 → 2.0 % (32);
   step-control sub-steps 10× fewer in threshold cases (33k → 2.8k). Cost n² kernel/product evaluations per pair:
   5.5× one collision-term evaluation here. Does nothing for the coagulation threshold.
4. **Table-friendly option: EAF** (EA with a fixed front exponent −3.5): A0 worst at 8 bins 4.4 %, growth 2.2 %; all
   front shapes depend on s only, so tables stay 1-D (≈3–4× the optics-table memory [inferred]).
5. **Drop:** TC (truncate beyond a slope window; small gains, none in A0); affine-with-switch and max-entropy
   exponential closures (worse on MRN-like profiles and in A0 at 4–6 bins, up to 97 %, at 6–11× the closure cost);
   overflow carried as a truncated power law (not robust: boxcar +1.8 → −0.3 %, MRN +1.4 → −2.0 %).
6. **Not fixed by any variant:** the coagulation-threshold bin (worst bin 16 % at 32 bins; misplaced mass 3.3 % with a
   properly aligned reference, not the stored 5.9 %); the jump of the reconstruction at bin edges in smooth regions.
   Both are limits of two degrees of freedom per bin.

## 1. Machinery checks
Closures are written as power-law pieces (support [u1, u2] inside the bin) or spikes, so one exact remap kernel (a
per-piece copy of the production one) serves all. `patched()` routes the production exchange, sputtering and collision
operators through a variant at run time. With the PL2M closure the piece remap reproduces `remap.remap` to 3e-15 and the
nodes reproduce `bin_nodes` to 3e-14; a patched PL2M A0 rerun reproduces the stored run bit for bit. All closures keep N
and M to ≤ 4e-14, stay positive and inside the bin on random 4–32-bin states (η up to 390), including subnormal bins.

## 2. Literature
Used for leads only. The affine switch, EA/EAF and KA are the subagent's own designs; the affine closure is "in the
spirit of" the two-size-moment method of Laurent et al. (2016), whose realisability treatment was not read. Bott's flux
method, the cell average technique and the Lombart & Laibe DG limiter were not used.

## 3. Candidates
Implemented: EA, EAF, TC, affine with switch, max-entropy exponential, Gauss nodes (radius and mass), KA, overflow as a
truncated power law. Considered, not implemented:

| candidate | reason |
|---|---|
| log-quadratic / PPM-like profile with neighbour curvature | errors are dominated by fronts; a non-local smooth closure would break the s-only tables |
| anti-diffusive remap (THINC, MPDATA, Bott) | the PL2M remap is exact for a given profile; its only diffusion is the re-imposed closure, which EA fixes |
| remap only what crossed an edge | needs a stored in-bin shape (more memory) |
| 3–4 explicit nodes per bin | pair cost ∝ (nodes per bin)²; KA gives the sub-node rate integral with one deposit per pair |
| cell average technique | PL2M already keeps N and M with spread deposits |
| order-1 DG | coagulation-only, explicit CFL-limited stepping; for growth its information equals McKinnon's PL (benchmarked) |

## 4. Equations (u = a/a_L ∈ [1, u_R], r = M/(N m_L))
- **TC(L):** λ limited to [−L, L]; beyond, λ = ±L and the support is cut at u_c with ⟨u³⟩ = r (continuous in r).
- **EA(ε):** neighbour emptiness q = (N_nb/Δa_nb)/(N_i/Δa_i) (grid ends count as empty); front weights
  e = max(0, 1 − q/ε); profile (1−e_L)(1−e_R) f_base + e_R(1−e_L) f_Rcut + e_L(1−e_R) f_Lcut + e_L e_R δ(u − r^{1/3}),
  each component with the bin's own N and M. f_Rcut = u^λref on [1, u_c], λref = (1 − e_L,i−1) λ_{i−1} + e_L,i−1 λ0,
  λ0 = 0; f_Lcut the mirror image. Infeasible cuts fall back to f_base (TC or PL2M). Scanned: ε 1e-8–0.3, λ0 −3.5–2,
  L 4–∞.
- **EAF:** EA with λref fixed.
- **Affine:** dn/da linear while positive, else a triangle with partial support (stored as 8 fitted sub-pieces).
- **Exponential:** dN/dm ∝ e^{−κm} (8 sub-pieces).
- **Gauss nodes:** z₁,₂ = (γ ∓ √(γ² + 4))/2 from the mean, variance and skewness of the in-bin measure in a (or m).
- **KA(n):** K, the products and the donor split averaged over n×n equal-number sub-slices, weighted by rate.
- N and M exact and positive in every variant; realisability holds.

## 5. Benchmarks

### 5.1 Growth (McKinnon Fig. 10), 100 steps
| variant | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| PC (McKinnon) | 22 | 3.0 | 0.88 | 0.34 |
| PL (McKinnon) | 0.51 | 0.18 | 0.048 | 0.012 |
| PL2M | 0.142 | 0.046 | 0.012 | 2.5e-3 |
| TC4 | 0.124 | 0.033 | 5.5e-3 | 6e-4 |
| **EA** | **0.016** | **5.2e-3** | **1.5e-3** | **1e-13** |
| EAF | 0.022 | 5.1e-3 | 1.1e-3 | – |
| affine | 0.092 | 0.029 | 7.1e-3 | 1.3e-3 |
| exponential | 0.084 | 0.035 | 0.011 | 2.3e-3 |

Versus steps at 8 bins: PL2M 0 → 3.9 % (3) → 14.2 % (100) → 14.8 % (1e4); EA 0 → 0.28 % → 1.62 % → 1.65 %; errors
saturate after ~100 steps. EA's residual comes from phases where the whole boxcar sits inside one bin (a spike). ε
1e-8–1e-2 makes no difference here; λ0 gives 1.6–2.3 %. Sinusoid at 128 bins: spread 3 → 0; W1 of the mass distribution
1.4e-3 → 2e-6.

### 5.2 Overflow (McKinnon Fig. 11)
+1.8 / +0.4 / −0.1 % at a_max = 1 / 2 / 4 µm for every closure: the error is in the carried spike, not the closure.

### 5.3 Stress tests (PL2M → EA)
- S1 narrow ±1 % top-hat grown ×10, 8 bins: 17.5 % → 9e-6; ×100: 20 % → 1e-6 (same at 1000 and 5000 steps).
- S2 10 growth–shrink periods, 16 bins: boxcar L1(M) 0.074 → 0.31 (PL2M, 1 → 10 periods) vs 0 (EA); MRN segment
  0.064 → 0.23 vs 0.000 → 0.004.
- S3 equilibrium radius (da/dt ∝ a_eq/a − 1), 8 bins, 10τ: L1(M) 0.019 → 0.000; W1 0.035 → 0.019.
- S4 cut-off at f = 0.05–0.95 of a bin, max W1: 8 bins growth 0.159 → 0.040, shrinking 0.223 → 0.059; 16 bins
  0.070 → 0.021, 0.027 → 0.002.
- S5 mixing of two cells: static surface errors ~1.3 % for both; 200 grow-and-mix cycles: L1(M) 0.119 → 0.016, surface
  −3.3 % → −0.2 %.
- S8 RAMSES-like steps (1e2–1e4 yr): silicate 8 bins (η = 18): mass 0.81 → 0.12 %, L1(M) 0.042 → 0.009, surface
  −4.1 → −1.0 %; carbon layout (η = 2.1, 18.8): mass 0.81 → 0.16 %, surface −4.3 → −0.9 %; independent of the step.

### 5.4 Collisions
Reference correction: references rebuilt with 640 fine bins so that coarse edges align (the stored
`check_mckinnon18_tests` reference used 400 fine bins, which do not align with 32 coarse bins, and matched by
searchsorted on round-off-equal edges): coagulation 32 bins worst bin 16.3 % (stored 15.7 %), misplaced 3.3 % (stored
5.9 %); shattering 8 bins 15.3 % / 2.1 % (stored 14.8 % / 2.1 %).

| variant | shatter 8 | shatter 16 | coag 8 | coag 32 | coag 32 sub-steps |
|---|---|---|---|---|---|
| PL2M | 0.153 / 0.021 | 0.067 / 0.009 | 0.052 / 0.004 | 0.163 / 0.033 | 33,086 |
| Gauss (a) | 0.199 / 0.026 | 0.060 / 0.008 | 0.302 / 0.026 | 0.071 / 0.011 | 3,058 |
| Gauss (m) | 0.206 / 0.025 | 0.030 / 0.004 | 0.744 / 0.061 | 0.208 / 0.023 | – |
| **KA4** | **0.006 / 0.003** | **0.002 / 0.0004** | 0.247 / 0.021 | 0.223 / 0.029 | 2,843 |
| KA8 | 0.013 / 0.002 | 0.000 / 0.0001 | 0.246 / 0.021 | 0.236 / 0.030 | – |
| TC8, EA, affine, exponential | ≈ PL2M | ≈ PL2M | 0.05–0.21 / 0.004–0.017 | ≈ PL2M | – |

(worst bin / mass misplaced). Threshold swept through a bin (7 positions), max / mean misplaced: shattering 8 bins PL2M
0.021 / 0.012, KA4 0.006 / 0.004, KA8 0.003 / 0.002, Gauss 0.025 / 0.014; coagulation 16 bins PL2M 0.040 / 0.027,
KA4 0.037 / 0.027, Gauss 0.054 / 0.035. [Partly inferred] Sticking empties the light side of the threshold bin, an in-bin
edge no two-parameter closure represents; KA with products left at the nodes made it worse (0.4 → 4 % at 8 bins).
Constant kernel, misplaced at τ = 100, 8 / 16 / 32 bins: PL2M 0.023 / 0.009 / 0.002; Gauss 0.122 / 0.053 / 0.020; KA4 =
PL2M; EA 0.004 / 0.009 / 0.002; affine 0.001 / 0.013 / 0.003; exponential 0.021 / 0.011 / 0.002. Golovin, τ = 3: PL2M
0.367 / 0.164 / 0.045; Gauss 0.043 / 0.051 / 0.051; KA4 0.120 / 0.065 / 0.020; EA, TC = PL2M; affine, exponential worse
(0.47 / 0.49 at 8 bins). Total number exact for every variant.

### 5.5 A0 cases, worst observable at 8 bins [EA states seen through PL2M tables]
| case | PL2M | EA [PL2M tables] | EAF | EA + Gauss | Gauss | TC8 | affine | exp |
|---|---|---|---|---|---|---|---|---|
| sn_diffuse | 0.034 | 0.034 [0.034] | 0.034 | 0.029 | 0.029 | 0.034 | 0.068 | 0.057 |
| mrn_accretion | 0.081 | **0.028** [0.054] | 0.041 | 0.028 | 0.081 | 0.081 | 0.111 | 0.088 |
| mrn_coagulation | 0.059 | **0.003** [0.058] | 0.003 | 0.006 | 0.058 | 0.059 | 0.039 | 0.032 |
| mrn_shattering | 0.078 | **0.044** [0.077] | 0.044 | 0.051 | 0.091 | 0.078 | 0.108 | 0.107 |
| mrn_sputtering | 0.121 | **0.006** [0.063] | 0.006 | 0.006 | 0.121 | 0.121 | 0.109 | 0.139 |
| **worst** | 0.121 | **0.044** [0.077] | 0.044 | 0.051 | 0.121 | 0.121 | 0.111 | 0.139 |

Worst over cases, PL2M / EA / EAF / EA + Gauss: 4 bins 0.246 / 0.240 / 0.241 / 0.238; 6 bins 0.164 / 0.166 / 0.172 /
0.117; 12 bins 0.055 / 0.046 / 0.063 / 0.051. EA improves accretion, coagulation and sputtering at every bin count (4
bins: accretion 0.232 → 0.089, sputtering 0.212 → 0.062). With ε = 1e-2 at 4 bins: accretion 0.92, coagulation 0.64,
sputtering 0.63.

### 5.6 Grain rates on each variant's own A0 end states (exact in-bin integral / two nodes)
| variant | 4 | 6 | 8 | 12 |
|---|---|---|---|---|
| PL2M | 0.169 / 0.264 | 0.141 / 0.218 | 0.062 / 0.078 | 0.030 / 0.030 |
| EA | 0.102 / 0.274 | 0.142 / 0.219 | 0.050 / 0.085 | 0.019 / 0.032 |
| Gauss | 0.169 / 0.167 | 0.104 / 0.104 | 0.062 / 0.063 | 0.030 / 0.030 |
| **EA + Gauss** | **0.045 / 0.043** | **0.027 / 0.023** | **0.010 / 0.021** | **0.010 / 0.012** |

### 5.7 Drift (one bin, w ∝ a), L1 of N / M
Per-moment velocities: PL2M 0.091 / 0.120 (= stored); TC16 0.091 / 0.120; TC8 0.092 / 0.123; **TC4 0.062 / 0.085**;
affine 0.154 / 0.203; exponential 0.145 / 0.197. Two nodes: median split 0.154 / 0.230 (= stored); **Gauss in radius
0.090 / 0.118**. EA not applicable (drift caveat above).

### 5.8 Cost and memory
Memory unchanged (two scalars per bin). Compiled time per cell and step at 8 bins (numba, one thread; laptop shared,
relative only):

| variant | closure [µs] | remap kernel [µs] | total vs PL2M |
|---|---|---|---|
| PL2M | 6.1 | 3.33 | 1.00 |
| TC | 4.3–5.2 | 3.1–3.4 | 0.79–0.91 |
| **EA** | 8.3 | 3.20 | **1.22** |
| EA on TC8 | 6.9 | 3.18 | 1.07 |
| EAF | 8.3 | 3.17 | 1.22 |
| affine | 37 | 17 | 5.8 |
| exponential | 70 | 18 | 9.3 |

With clean fronts EA has as many non-empty pieces as PL2M, so the remap costs the same. [Inferred, Fortran] EA adds two
divisions per bin and one root solve (like PL2M's) per front bin. Collision terms (8 bins, coagulation, Python level):
production 0.12 ms; KA 2/4/8 = 0.47 / 0.66 / 1.05 ms. [Inferred] EA observables need front-aware tables: 2-D (s, λref)
per bin, or with EAF two extra 1-D tables per bin plus spike point values.

## 6. Fixes and breakages
EA fixes fronts and narrow populations and is neutral where bins are filled by collisions; weaknesses: non-local in size
(reads neighbouring bins of the same cell), the isolated-bin spike loses the in-bin spread, ε must be small, one rare
discontinuity when a neighbour becomes a spike. Gauss nodes: good for rates and drift, mixed in collisions. KA: fixes
separable thresholds, n² cost. TC: marginal. Affine and exponential: worse overall.

## 7. Measured vs inferred; not run
Inferred: Fortran operation counts, table-memory factors, the closure-limit reading of the coagulation threshold. Not
run: DG, log-quadratic/PPM closures, 3–4 explicit nodes per bin, KA inside the production GrainCollisions model in A0.
The McKinnon collision tests ran EA with ε = 1e-2; the kernel tests ε = 1e-6. The exponential sub-pieces are unreliable
only for η ≳ 1e3 (no grid here). The McKinnon and sweep tables were rebuilt from the complete run's log
(`collisions_mk_sweep.log`) after the kernel section crashed on subnormal-N bins; the kernel section was rerun after
fixing the node routine.

## Figures
- `variants_growth.pdf` — growth error vs bins and vs steps
- `variants_edges.pdf` — edge shapes per closure; sinusoid
- `variants_collisions.pdf` — McKinnon collision tests, threshold sweeps, constant and Golovin kernels
- `variants_stress.pdf` — stress-test summary
- `variants_a0_rates_drift.pdf` — A0 vs bins, rates, drift
- `variants_cost.pdf` — error vs compiled cost
