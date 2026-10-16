# Numerical schemes for a fully discretised, multi-composition grain-size-distribution (GSD) solver
## Literature review and design recommendation for pyCALIMA

Date: 2026-10-15. Scope: growth/sublimation advection in size space, nucleation, coagulation, shattering, sputtering, coupling to gas-phase depletion, and time integration for AGN dust-rim conditions (n_H ~ 1e7–1e12 cm^-3, days–Myr).

**How the references were checked.** Every reference in Section 8 was checked against at least one primary metadata source: Crossref DOI records (publisher metadata), arXiv abstract pages (authors, title, DOI), publisher/HAL pages, or GitHub API/licence files. For the papers marked [read], I downloaded the arXiv or author-hosted PDF and read the cited passages. ADS pages could not be opened (CAPTCHA), so the old A&A papers that have no DOI were confirmed through publisher pages or search snippets. Anything I could not confirm is marked **[unverified]**. Statements labelled *(our analysis)* or *(recommendation)* are my own derivations or design suggestions. They do not come from a cited paper.

---

## 0. Executive summary

* **Two-moment sections should be the state variable.** Store number N_i and mass M_i per bin per composition, so that the mean grain mass m̄_i = M_i/N_i is known exactly. This one change gives three things:
  * exact number and mass conservation under nucleation, sublimation losses, coagulation and fragmentation;
  * bin-collision products that do not need splitting between two collectors (the splitting is what causes Ohtsuki-type artificial acceleration);
  * narrow nucleation bursts that stay sharp, because a monodisperse cohort is represented exactly by (N, M) inside a bin.

  The literature supports this from several directions:
  * the fixed-pivot vs moving-pivot comparison of Kumar & Ramkrishna (1996a,b);
  * the two-moment collection scheme of Tzivion et al. (1987);
  * the piecewise-linear (N, M, slope) GSD of McKinnon et al. (2018);
  * the "two-size-moment" multi-fluid sprays of Laurent et al. (2016);
  * the finding that the moving-center structure (Jacobson 1997a) is the best fixed-grid sectional method for condensation (Zhang et al. 1999).
* **Growth, sublimation and sputtering should use a conservative semi-Lagrangian (characteristic) remap on a fixed geometric mass grid.** This generalises your existing shift-and-remap to signed, size-dependent ȧ(a) and to two moments. It is unconditionally stable, exactly conservative, and has very low diffusion. Eulerian upwind is demonstrably diffusive: Hirashita (2012) reports that his upwind scheme with 512 bins smooths a sharp accretion peak through numerical diffusion.
* **Nucleation should be injected along characteristics within each sub-step** (a "top-hat" source segment), in the spirit of the formation-time (Lagrangian) treatment of Nozawa & Kozasa (2013). It must be integrated in the same implicit gas-coupling sub-step as growth and sublimation, because all three compete for the same vapour.
* **Coagulation and shattering should be two-moment pair transfers.** Use precomputed fragment mass- and number-distribution tensors, and a positivity-preserving, mass-conservative implicit integrator. Two options:
  * modified Patankar–Runge–Kutta MPRK22: second order, unconditionally positive, conservative (Burchard et al. 2003; Kopecz & Meister 2018a,b);
  * the cheaper first-order, non-iterative Jacobson et al. (1994) semi-implicit sequential scheme.

  Your current explicit Euler sub-stepping becomes prohibitive in the stiff case of continuous nucleation plus scavenging by large grains.
* **Splitting.** Use Lie/Strang splitting between a "collisions" operator and an "exchange" operator (nucleation + growth/sublimation + sputtering + gas). Put the stiff exchange operator last in each sequence, which is Sportisse's (2000) result for stiff splitting. Choose macro-steps adaptively.

---

## 1. Governing equations (notation)

For composition s, let f_s(m,t) be the number density per unit grain mass. Then

∂f/∂t + ∂(ṁ f)/∂m = J(t) δ(m − m*) + C_coag[f] + C_shat[f] (+ cross-composition terms),

where ṁ = 4πa²ρ_gr ȧ. In log-mass μ = ln m, with σ ≡ m f (the mass per unit ln m) and a size-independent ȧ, this becomes (Hirashita 2012, eq. 10) [read]

∂σ/∂t + μ̇ ∂σ/∂μ = (1/3) μ̇ σ,  μ̇ = 3ȧ/a.

Two features of this form drive the numerics:

* The advection speed in log space, μ̇ ∝ 1/a, is largest at the nucleation end. An explicit Eulerian scheme has CFL Δt < Δμ·a_min/(3|ȧ|), which is set by ~0.3 nm grains. *(our analysis)*
* For sublimation, ȧ(a) is signed and strongly size-dependent. Two reasons:
  * the Kelvin factor enhances evaporation of small grains;
  * in an AGN radiation field T_d depends on a.

  Schematically (standard Hertz–Knudsen form), ȧ = V₀[α n_X v̄_X/4 − α n_sat(T_d(a)) v̄_X(T_d)/4 · exp(a_K/a)]. *(standard form, not tied to one cited paper)*

  1D characteristics never cross, but they can **converge** (pile-up at a stable size) or **diverge** from a critical size a_c where ȧ(a_c) = 0 (the LSW-type instability). A scheme must handle compression and rarefaction of the distribution without losing number. *(our analysis)*

---

## 2. Size-space advection (growth, sublimation, sputtering)

### 2.1 Eulerian upwind and its numerical diffusion

* **Method.** First-order donor-cell in μ, as in Hirashita (2012) Appendix A1, "the difference is evaluated based on upwind differencing" (N = 512 bins, a = 2×10⁻⁴–0.3 µm) [read]. The modified equation gives a numerical diffusivity D_num = (μ̇Δμ/2)(1 − C), with Courant number C = μ̇Δt/Δμ.
* **Smearing estimate** *(our analysis)*. Over a total log-mass displacement Λ = ∫μ̇dt, a δ-burst spreads to a Gaussian of variance ≈ Δμ·Λ(1 − C̄). Example: growth from 0.5 nm to 0.1 µm (Λ = ln(200³) ≈ 15.9) at 20 bins per mass decade (Δμ ≈ 0.115) gives σ_μ ≈ 1.35, which is a factor ≈ 3.9 in mass (≈ 1.6 in radius) at 1σ. That is unacceptable for nucleation bursts.
* **Evidence.** Hirashita (2012) states explicitly that "the sharp peak is smoothed because of the numerical diffusion" when compared with the Hirashita & Kuo (2011) analytic accretion solution [read].
* **Higher-order flux-limited / positive-definite schemes** (e.g. Bott 1989, used for condensation by air-quality models) reduce diffusion. However, Zhang et al. (1999) found that the Bott-scheme model "significantly overpredicts the particle volume below 0.1 µm" in their haze test [read]. Higher-order Eulerian schemes also keep the CFL restriction at small a.

### 2.2 Full-moving / Lagrangian bins

* **Method.** Bin edges move with the growth law (method of characteristics). The only error is the sectional discretisation itself. Zhang et al. (1999) use this "full-moving" approach as their "exact" reference [read].
* **Drawbacks:**
  * it is incompatible with a fixed grid shared between spatial cells or time series with mixing;
  * coagulation on a moving grid needs re-gridding;
  * sublimation collapses bins to zero width;
  * nucleation needs new bins.
* **How Kumar & Ramkrishna (1997, "III") handle nucleation on a moving grid** [read]. They add a new zero-population bin at the nucleus size whenever the first bin's lower edge has moved past it. They note that this "correctly accounts for both the mass and the number of the nuclei for constant and linear growth rates". However, for constant growth rates the added bins form a linear grid, so "the number of equations that need to be solved increases very rapidly and the technique becomes computation-intensive".
* **Lagrangian tracking in formation time.**
  * Nozawa et al. (2006) follow the trajectories of grains of each initial radius through the post-shock flow when computing sputtering [read; trajectories by initial radius are discussed in their Section 6].
  * Nozawa & Kozasa (2013, "NK13") combine a discrete n-mer master equation (J_n = α_{n−1}c_{n−1}c_1 − β_n c_n for 2 ≤ n ≤ n*) with moment-like integrals over formation time t′. These are K_i(t) = ∫ I_*(t′) (a(t,t′)/a₀)^i dt′ with dK_i/dt = I_* n*^{i/3} + (i/a₀)(da/dt)K_{i−1}, and n* = 100 [read, eqs. 1, 6, 48–49].
  * This is exact for growth when ȧ is size-independent, and it is the natural model for injecting nucleation along characteristics (Section 2.6).

### 2.3 Shift-and-remap (semi-Lagrangian, conservative)

* **Method.** Advect the reconstructed sub-bin distribution exactly along characteristics over Δt, then integrate it over the fixed bins. McKinnon et al. (2018, AREPO) formalise this in their Section 3.1–3.2 [read]:
  * N_j(t+Δt) = Σ_i ∫_{I_i ∩ (I_j − ȧΔt)} (∂n/∂a) da;
  * M_j(t+Δt) = Σ_i ∫ m(a + ȧΔt)(∂n/∂a) da.

  Two extra bins collect grains that cross a_min or a_max. The scheme "conserves total grain number" before boundary rebinning, and for ȧ < 0 with Δt → ∞, m(a + ȧΔt) → 0, so mass is fully destroyed.
* **Reconstruction.** Piecewise linear in a with slopes s_i, giving N_i, M_i ↔ (N_i, s_i) per bin. A slope limiter keeps n ≥ 0 and preserves bin mass. With piecewise-linear vs piecewise-constant reconstruction the mass error in their growth test was "roughly 40 per cent for the piecewise constant discretisation and just 1 per cent for the piecewise linear one", with first- vs second-order convergence [read].
* **Limitation.** It assumes size-independent ȧ (accretion, thermal sputtering). For size-dependent ȧ(a), replace a + ȧΔt by the numerical flow map X_Δt(a), obtained by integrating da/dt = ȧ(a,t) for each bin edge or quadrature node. *(our generalisation; it is exact up to ODE error and remains monotone because characteristics do not cross)*
* **Your current scheme** is the one-moment version of this. Its weakness is that N_i is not an independent variable: the number follows from the neighbour-slope reconstruction, so number is conserved only approximately, and a narrow burst's location inside a bin is lost after one remap.

### 2.4 Fixed-grid sectional methods from aerosol science

* **Gelbard & Seinfeld (1980) / Gelbard, Tambour & Seinfeld (1980) sectional method.** This is a one-moment (mass per section) fixed grid with inter-sectional coefficients computed once.
  * Zhang et al. (1999) report that in its original form "it is subject to significant numerical diffusion". With 12 sections between 0.001 and 10 µm the average absolute error could reach 35%, falling to 12% with three times more sections [read].
* **Moving-center structure** (Jacobson 1997a; described in Jacobson 2002 and in Zhang et al. 1999) [read].
  * Bin edges are fixed, but "the mass of each particle size section is assigned to a single particle size … allowed to grow as condensation occurs. When this particle size grows out of the section, all particles of that section are transferred into the next section" and averaged there.
  * In Jacobson's (2002) words, the single-particle volume "can vary between the low- and high-edge volume" of the bin.
  * This is a two-moment (number + volume) scheme with a δ-reconstruction. It conserves number and mass and is minimally diffusive.
  * Among fixed-grid sectional schemes, GATOR (moving-center) "comes the closest to reproducing" the nuclei-mode peak. But reproducing that narrow mode still needed "about 100 size sections over the range of 0.001 to 10 µm" (Zhang et al. 1999) [read].
  * Weaknesses: jaggedness, because whole populations jump bins, and averaging errors when populations from different cells mix.
* **Hybrid structures.**
  * Jacobson & Turco (1995) introduced a "combined moving and stationary size grid" [title verified; the PDF could not be text-extracted, so details are **[unverified]**].
  * In UAM-AERO/SAQM-AERO, "hybrid" means full-moving growth over short steps followed by redistribution onto the fixed grid at transport steps. Zhang et al. (1999) note it "may lead to significant numerical diffusion for time steps commensurate with 3-D transport" [read].
* **Kumar & Ramkrishna fixed pivot (1996a) and moving pivot (1996b)** [1996b read].
  * The fixed pivot preserves two chosen moments but "consistently over-predict[s] number densities for the largest particles".
  * The moving pivot "preserves any two arbitrarily chosen properties of the population" through a varying pivot per interval. In one product-kernel test it gave a 33% under-prediction of the front position against a 1800% over-prediction for the fixed pivot [read].
  * The moving pivot is the coagulation analogue of the moving-center structure.
* **Cell-average technique (Kumar et al. 2006).** It computes the average size of newly formed particles entering each cell before redistributing them to pivots, preserving number and mass [title/journal verified; mechanism summary from secondary knowledge, **details unverified**].
* **Two-size-moment (TSM) multi-fluid (Laurent, Sibra & Doisneau 2016)** [abstract read, HAL].
  * It uses two size moments per section with "a piecewise affine reconstruction in size … thoroughly analyzed in terms of stability and accuracy". Realizability of the moments is ensured, and it describes evaporating sprays accurately "resorting to a lower number of sections compared to one size moment methods".
  * This is the closest published analogue of what we need for evaporation (sublimation).
  * Kah et al. (2012) is a related high-order moment method for evaporation [title/journal verified].

### 2.5 Moment methods

* **Methods.** Gail & Sedlmayr-type moment equations (Gail, Keller & Sedlmayr 1984; Gail & Sedlmayr 2013 monograph) and Woitke & Helling (2003)-type moments evolve K_j = ∫ a^j f da. Variants:
  * MOMIC interpolative closure (Frenklach & Harris 1987), which Mattsson (2016) applies to ISM dust including condensation, coagulation, shattering and sputtering [read];
  * QMOM / DQMOM (McGraw 1997; Marchisio & Fox 2005).
* **Strengths.** They are very cheap and handle nucleation + growth naturally.
* **Weaknesses.** They lose the GSD needed for optics and for size-dependent T_d. Closure is required for non-integer moments and for collision terms (Mattsson 2016 §"closure") [read]. Evaporation is a known difficulty: the moment equations need the boundary flux at the smallest size, which moments do not contain. This motivated EMSM/TSM work (Kah et al. 2012; Laurent et al. 2016) *(our reading of the motivation; we did not extract a quote)*.
* **Use for us.** Not as the main solver. They are useful as a cross-check (exact moment ODEs for constant ȧ + J).
* **Note on Mattsson (2020).** It concerns stochastic (Ornstein–Uhlenbeck) gas-density fluctuations producing a lognormal GSD. It is a physics paper on turbulent growth with a binned Euler–Maruyama scheme, not a numerical moment method [read].

### 2.6 Which advection scheme handles nucleation + condensation + evaporation together?

*Assessment from the evidence above plus our analysis:*

| Scheme | N cons. | M cons. | Diffusion | CFL-free | Signed, size-dependent ȧ | Nucleation bursts | Fixed grid (needed by coag/mixing) |
|---|---|---|---|---|---|---|---|
| 1-moment upwind (Gelbard–Seinfeld; Hirashita 2012) | approx. | yes | high | no | yes | smeared | yes |
| Bott/flux-limited Eulerian | approx. | yes | moderate | no | yes | moderate | yes |
| Full-moving (Lagrangian bins) | yes | yes | none | yes | collapse/expansion problems | needs bin creation (KR-III) | no |
| Moving-center (Jacobson) | yes | yes | low (but jagged) | yes | yes | good | yes |
| 1-moment shift-and-remap (current pyCALIMA) | approx. | yes | low–moderate | yes | constant ȧ only | position lost in bin | yes |
| **2-moment semi-Lagrangian remap** (McKinnon 2018; TSM; generalised) | **yes** | **yes** | **low** | **yes** | **yes, via the flow map** | **good** | **yes** |
| Moments (GS, MOMIC, QMOM) | yes | yes | n/a | yes | closure issues | natural | n/a |

The two-moment semi-Lagrangian remap is the best fit. Nucleation within a sub-step [t, t+Δt] is injected as the image of the source under the flow map. With constant J and ȧ this is a uniform segment n(a) = J/ȧ on [a*, a* + ȧΔt]. In general, use (N, M) = (∫J dt′, ∫J·m(X_{t+Δt−t′}(a*)) dt′), evaluated by Gauss quadrature over t′. This reproduces the exact nucleation + growth "moving top-hat" solution and is the binned analogue of NK13's formation-time integral.

---

## 3. Coagulation discretisations

### 3.1 Classical sectional and pair-splitting schemes

* **Kovetz & Olund (1969).** A sectional scheme that splits the product mass between bins (title as verified: "The effect of coalescence and condensation on rain formation in a cloud of finite vertical extent"). Lombart & Laibe (2021) use it to illustrate the "growth over-diffusion problem": order-0 schemes "over-estimate the formation of large grains at low resolution" [read, their Fig. 1].
* **Brauer et al. (2008) / Birnstiel et al. (2010) / DustPy.** The product m_i + m_j is split linearly between the two neighbouring grid masses so that mass is conserved (Stammler & Birnstiel 2022, eq. 15 ff.) [read].
  * DustPy rewrites the terms analytically for m_j ≪ m_k, where "a computer would falsely calculate m_k + m_j = m_k". Using K_kjk − 1 = −m_j/(m_{k+1} − m_k) removes the precision loss [read, eqs. 19–22]. This limit is accretion of small grains onto large grains, which is effectively advection.
  * The resolution requirement is "at least seven mass bins per decade for simple collision models (Ohtsuki et al. 1990)" [read]. Ohtsuki et al. (1990) is titled "Artificial acceleration in accumulation due to coarse mass-coordinate divisions".
  * cuDisc keeps > 7 bins per mass decade for the same reason [read].
* **Marchand et al. (2021), Appendix B** (your current coagulation scheme) [read].
  * Power-law slope per bin from the harmonic mean of neighbour slopes. The product of bins i, j is distributed over bins C and C+1 with C = ⌊log((m_{i,−} + m_{j,−})/m_min)/log η⌋ + 1.
  * The transferred fraction υ comes from analytic double integrals of (m + m′)K(m,m′)m^{β_i}m′^{β_j}, with numerical care for steep β (their code is "Ishinisan").
  * The main paper is a path-integral method that is "mathematically exact" for pure coagulation but "cannot account for processes other than coagulation" [read]. It is therefore not usable once growth, sublimation or shattering are present.
* **Jacobson et al. (1994) semi-implicit, volume-conserving scheme** (as generalised in Jacobson 2002) [read].
  * Update: n_{k,t} = (n_{k,t−h} + h(T₁ + T₂))/(1 + hT₃). The production terms use already-updated concentrations of smaller bins, so the bins are solved sequentially from small to large.
  * Products are partitioned with volume fractions f_{i,j,k} between bracketing bins.
  * It is "volume-conserving, noniterative, unconditionally stable", first order, and positive.
  * It also works with the moving-center structure, where single-particle volumes change each step (f is recomputed).
* **Two-moment sectional collection** (Tzivion, Feingold & Levin 1987; multicomponent framework of Chen & Lamb 1994) [titles/journals verified; method content from general knowledge, **details unverified**]. Number and mass per bin are evolved, so the product of a bin pair is deposited as (ΔN, ΔM) into the bin that contains it. This is the coagulation counterpart of 2.4.
* **Finite-volume conservative form** (Filbet & Laurençot 2004; flux form of Tanaka et al. 1996 as cited by Lombart & Laibe) [abstract read]. The mass flux across bin interfaces is computed so that mass conservation is built in.

### 3.2 High-order discontinuous Galerkin (Lombart & Laibe 2021; Lombart, Hutchison & Lee 2022)

[read]

* **Method.**
  * DG on the conservative mass-flux form, with Legendre polynomials of order k per bin.
  * Analytic integrals for polynomial kernels.
  * A scaling limiter for positivity.
  * SSPRK3 time stepping under a positivity CFL condition (Δt_coag = ½Δt_CFL in practice).
* **Results.**
  * "Mass is conserved to machine precision."
  * Errors of 0.1–1% with ~15–20 bins over 12 decades in mass.
  * "Creating aggregates of masses larger than the mass of the reservoir is mathematically excluded."
  * Extended to non-linear fragmentation in Lombart et al. (2022) and to general non-linear fragmentation in Lombart, Bréhier, Hutchison & Lee (arXiv:2404.11851) [arXiv only; journal **unverified**].
* **Drawbacks for us** *(our analysis)*:
  * the time stepping is explicit and CFL-limited, and the scheme is designed for coagulation/fragmentation only, not for signed advection plus nucleation;
  * number is not an independent conserved variable;
  * the analytic kernel integrals become complicated for Coulomb-modified, piecewise kernels.
* **Use for us.** An excellent accuracy benchmark, and its code (Coala, CeCILL 2.1) can produce reference solutions.

### 3.3 Implicit integration of coagulation/fragmentation

* **Birnstiel et al. (2010), Appendix A** [read].
  * Linearised implicit Euler, (1/Δt − J)ΔN = S, with Jacobian J_ki = Σ_j(M̃_ijk + M̃_jik)N_j.
  * This is extended to a fully implicit radial transport + coagulation system solved with a sparse iterative ILU solver.
  * The motivation was that splitting caused the solution to "flip-flop" between transport and coagulation.
* **DustPy** [read].
  * Implicit Euler Σ^{n+1} = (1 − ΔtJ)^{-1}(Σ^n + ΔtS_ext) with a SuperLU sparse solve (or GMRES, or explicit Cash–Karp).
  * But it limits Δt = 0.1·min(Σ/Σ̇⁻), "such that neither the gas nor the dust densities could become negative in a first-order Euler scheme". Positivity is not structural, so this limit is restrictive.
* **cuDisc** [read]. Explicit embedded Bogacki–Shampine 3(2) with adaptive sub-cycling inside the global step, with 1% relative tolerance by default.
* **Cost and stiffness in our regime** *(our analysis; order-of-magnitude only)*.
  * Example: all Si (~3×10⁻⁵ n_H) condensed into 100-monomer clusters at n_H = 1e10 cm^-3 gives n_d ~ 3×10³ cm^-3. A Brownian kernel ~1e-9 cm³ s⁻¹ then gives t_coag ~ days, and hours at 1e12.
  * For pure self-coagulation t_coag grows ∝ t, so adaptive explicit stepping costs only ~ln(t_end/t₀)/ε steps.
  * The genuinely stiff case is continuous nucleation feeding nm grains while large grains scavenge them. The small-bin loss rate is then fast and steady, and explicit sub-stepping needs ~t_end/t_loss steps (up to ~10⁸ for Myr runs).
  * Hence an implicit, positivity-preserving integrator is needed.

### 3.4 Positivity-preserving conservative integrators (production–destruction systems)

* **Modified Patankar–Euler and MPRK22** (Burchard, Deleersnijder & Meister 2003). Write the system as dy_k/dt = Σ_j p_kj(y) − Σ_j d_kj(y) with p_kj = d_jk. Production and destruction terms are weighted by y^{n+1}/y^{(*)}, which yields a linear system per step.
  * The schemes are unconditionally positive and exactly conservative.
  * Third-order MPRK43 and the order conditions are in Kopecz & Meister (2018a,b).
  * Huang & Shu (2019) give second- and third-order positivity-preserving alternatives.
  * Sandu (2001) analyses positive integration for chemical kinetics.
  * [all titles/journals verified]
* **Application to coagulation and shattering** *(recommendation)*. In mass variables both are exactly production–destruction systems: mass moves from bins i, j to bin(s) k. MPRK22 in M with a consistent Patankar update of N (number is a non-conserved PDS with net sinks, for which Patankar weighting still guarantees positivity) gives positivity, exact mass conservation and second order. It needs one dense N_b × N_b linear solve per species per stage, which is ~1e6 flops for 100 bins and negligible.
* **SSP Runge–Kutta** (Gottlieb, Shu & Tadmor 2001) keeps positivity only under a CFL-type restriction (the approach Lombart & Laibe use).

---

## 4. Shattering and fragmentation

### 4.1 Physical fragment models

* **Jones, Tielens & Hollenbach (1996, JTH96) and Tielens et al. (1994)**, as implemented by Hirashita & Yan (2009, HY09) [read].
  * Shocked mass for a target–projectile collision: M/m_j = [(1 + 2R)/(1 + R)^{9/16}] (1/σ_{1i}^{1/9}) (M_r²/(σ₁M₁²))^{8/9}, with M_r = v/c₀, M₁ = 2φ₁/(1 + (1 + 4sφ₁)^{1/2}), φ₁ = P₁/(ρ_gr c₀²), and σ(M) = 0.30(s + M⁻¹ − 0.11)^{0.13}/(s + M⁻¹ − 1).
  * If M > m_k/2 the whole target fragments. Otherwise the ejected mass is M_ej = 0.40M (for z = 3.4). The whole projectile fragments.
  * Fragments follow n_frag(a) ∝ a^{−3.3} between a_f,min = 0.03 a_f,max and a_f,max = 0.22 a_j (v_cat/v), with v_cat = c₀(m_k/((1 + 2R)m_j))^{9/16} σ₁^{1/2} σ_{1i}^{1/16}(1 + R)M₁ for the projectile.
  * Material parameters (HY09 Table 2; silicate): ρ = 3.3, c₀ = 5 km s⁻¹, s = 1.2, v_shat = 2.7 km s⁻¹, P₁ = 3×10¹¹ dyn cm⁻². Graphite has v_shat = 1.2 km s⁻¹ and P₁ = 4×10¹⁰ (Hirashita & Kobayashi 2013).
  * JTH96 argue α_f = 3.0–3.4 (Hirashita & Kobayashi 2013) [read].
* **Kobayashi & Tanaka (2010, KT10) simplification**, as used by Hirashita & Kobayashi (2013) [read].
  * M_ej = φ/(1 + φ)·m₁, with φ = E_imp/(m₁Q*_D), E_imp = ½ m₁m₂v²/(m₁ + m₂), and Q*_D ≈ P₁/(2ρ_gr).
  * Largest fragment a_f,max = (0.0204 f)^{1/3} a₁ with f = M_ej/m₁, following Guillet et al. (2009). Fragments below a_min are removed.
  * KT10 versus JTH96 differ by ≤ 1.8 (silicate) / 1.4 (carbon) in small-grain production. The fragment slope has "minor effects as long as α_f ≲ 3.5", and P₁/t is the controlling quantity.
  * KT10's own abstract: the model has parameters for total ejecta mass, largest fragment and fragment-size exponent. Cratering is more effective than catastrophic disruption, and depletion time is governed mainly by total ejecta mass [abstract via search; journal verified].
* **Guillet, Pineau des Forêts & Jones (2011, Shocks in dense clouds III)** [abstract via search; journal verified]. Shattering/vaporisation/sputtering coupled to the GSD in C-shocks. Grain–grain processing becomes important above a density threshold, which is relevant to n ≥ 1e7.
* **Protoplanetary-disc recipes** (Brauer et al. 2008; Birnstiel et al. 2010, 2011; DustPy) [read].
  * Fragment distribution n(m)dm ∝ m^{−ξ}dm with ξ = 1.83 (equivalent to n(a) ∝ a^{2−3ξ} = a^{−3.5}).
  * Cratering with m_out = χ m_s (χ = 1).
  * Fragmentation probability p_f ramps linearly from 0 to 1 over a width δu below u_f.
  * Kernel decomposition K = Δu σ_geo p_c and L = Δu σ_geo p_f.

### 4.2 Numerical implementation in binned schemes

* **HY09 shattering equation** (in mass density per bin) [read]: dρ̃_i/dt = −m_iρ̃_iΣ_kα_kiρ̃_k + ΣΣα_kjρ̃_kρ̃_j m^{kj}_shat(i), with α_ki = σ_kiv_ki/(m_im_k) for v > v_shat.
  * m^{kj}_shat(i) = ∫_bin m n_frag da.
  * Each step is split into four sub-steps with v = v_i + v_k, |v_i − v_k|, v_i, v_k to mimic directions.
  * Fragments with a_f,max < a_min "are put in the bin with the smallest size". HK13 instead removes them. pyCALIMA must decide and record explicitly whether such mass returns to the gas.
  * HY09 use N = 32 bins over 0.001–0.25 µm. Kirchschlager et al. (2019, Paperboats) use 40 bins plus a "collector bin" i = 0 for fragments below a₁ and vaporised mass ("dusty gas") [read].
* **Exact power-law transfer integrals for two-moment bins** *(recommendation)*. For n_frag = C a^{−α} in bin [a_L, a_R]:
  * ΔN = C(a_R^{1−α} − a_L^{1−α})/(1−α);
  * ΔM = (4π/3)ρC(a_R^{4−α} − a_L^{4−α})/(4−α).

  Precompute a tensor F[i,j,k,(v-class)] → (ΔN_k, ΔM_k) per unit collision rate, normalised so that Σ_k ΔM_k equals the fragment mass exactly. The remnant m_k − M_ej is deposited as a single grain. Mass is then conserved to round-off, and number is exact.
* **Cost reduction.** Rafikov, Silsbee & Booth (2020, "A fast O(N²) fragmentation algorithm"): explicit redistribution is O(N³) in general and O(N²) when the fragment spectrum is self-similar with a single characteristic mass scale. It is used by DustPy and cuDisc [abstract read]. Our power-law fragment spectra with a_f,max as the only scale qualify.
* **Analytic tests.** Exact coagulation–fragmentation solutions (Dubovskii, Galkin & Stewart 1992); steady-state collisional cascades (Dohnanyi 1969; Tanaka, Inaba & Nakazawa 1996); coagulation–fragmentation equilibrium fits (Birnstiel, Ormel & Dullemond 2011).

### 4.3 Relative-velocity sources

* **Brownian motion.** Δu_BM = (8kT(m₁ + m₂)/(πm₁m₂))^{1/2} (Birnstiel et al. 2010, eq. 44) [read].
* **Turbulence.**
  * Ormel & Cuzzi (2007): closed-form relative velocities in terms of Stokes numbers, with an intermediate regime used by Marchand et al. (2021, eq. B.20).
  * Yan, Lazarian & Draine (2004): hydrodrag + gyroresonance in compressible MHD turbulence, as used by HY09/HK13.
* **Coulomb modification.** Okuzumi (2009), and Draine & Sutin (1987) for charging.
* **AGN-rim estimate** *(our analysis; rough)*. With Epstein stopping time t_s = ρ_gr a/(ρ_g v_th), a 0.1 µm grain at n ~ 1e10 cm^-3 and T ~ 1500 K has t_s ~ 10³ s. Grains are therefore extremely well coupled (St ≪ 1), and turbulent relative velocities are tiny. Shattering-level velocities (≳ 1–3 km s⁻¹) would most plausibly come from size-dependent radiation-pressure drift. A drift term Δv_rad(a₁, a₂) should be a first-class kernel ingredient.

---

## 5. Coupling to the gas phase, splitting and stiffness

* **Analytical Predictor of Condensation (APC)** (Jacobson 1997b; restated in Jacobson 2002) [read].
  * Each bin is integrated implicitly as c_i^t = c_i^{t−h} + h k_i(C^t − S′_iC_{s,i}). The unknown final gas concentration is closed with the mass balance C^t + Σc_i^t = C_tot.
  * This gives a "noniterative, unconditionally stable" update: C^t = (C^{t−h} + hΣk_iS′_iC_{s,i})/(1 + hΣk_i) for condensation.
  * This is exactly the structure needed for growth and sublimation with gas depletion. *(our restatement of the closed form)*
* **Nucleation** is extremely nonlinear in supersaturation. Zhang et al. (1999) conclude that "simulation of relative rates of condensation on existing particles and nucleation is a preferable approach" over parameterised nucleation [read]. Nucleation and growth must therefore be solved together in the same implicit sub-step. *(recommendation)* Solve one scalar monotone equation for n_X^{n+1}:

  n_X^{n+1} + Σ_{s,i}ΔM_{s,i}(n_X^{n+1})/m_X + n*·J(n_X^{n+1})Δt = n_X,tot.

  Use bracketed Brent/bisection, then perform the remap with the converged ȧ. Positivity and exact element conservation follow by construction.
* **Operator splitting in the stiff case.** Sportisse (2000) shows that classical splitting-error analysis fails when Δt ≫ the fastest time scales, and that "the stiff part should be applied last" [abstract via search; journal verified].
  * Birnstiel et al. (2010) saw "flip-flop" errors when splitting transport and coagulation and moved to an unsplit implicit solve [read].
  * For a single cell (no transport) splitting collisions from exchange is acceptable if the exchange operator (fastest: vapour depletion over seconds to hours at high n) is placed last and the macro-step is controlled adaptively.
* **Fully coupled alternative.** Method of lines on (N_i, M_i, n_X) with BDF/Radau/Rosenbrock (dense Jacobian of size 2N_bS + N_gas).
  * Advantages: robust and simple; fine for a single long time series.
  * Disadvantages: growth must then be an Eulerian flux (losing Lagrangian sharpness), positivity is not guaranteed, and cost is ~(2N_bS)³ per Jacobian factorisation. *(our analysis)*

---

## 6. Open-source codes (licence, relevance)

Licences come from the GitHub API and LICENSE files, checked 2026-10-15.

| Code | Licence | What it offers | Relevance |
|---|---|---|---|
| DustPy (github.com/stammler/dustpy; Stammler & Birnstiel 2022) | GPL-3.0 | Smoluchowski coagulation + fragmentation, Brauer-type 2-bin splitting with machine-precision fixes, Rafikov O(N²) fragmentation, implicit Euler with sparse Jacobian | Algorithms and tests (constant/additive kernels); copying code would impose GPL |
| two-pop-py (github.com/birnstiel/two-pop-py; Birnstiel, Klahr & Ercolano 2012) | GPL-3.0 | Two-population parametric model | Not a full GSD; benchmark only |
| cuDisc (github.com/cuDisc/cuDisc; Robinson, Booth & Owen 2024) | GPL-3.0 | CUDA coagulation/fragmentation (Brauer 2008 + Rafikov 2020), explicit adaptive RK3(2) sub-cycling | GPU ideas; explicit time stepping |
| Coala (github.com/mlombart/Coala; Lombart & Laibe 2021) | CeCILL 2.1 (GPL-compatible copyleft) | High-order DG coagulation/fragmentation | Reference solutions |
| PartMC (github.com/compdyn/partmc; Riemer et al. 2009); PyPartMC | GPL-2.0; GPL-3.0 | Particle-resolved Monte Carlo aerosol (coagulation, condensation, nucleation; MOSAIC chemistry) | Independent stochastic reference |
| PySDM (github.com/open-atmos/PySDM) | GPL-3.0 | Super-droplet Monte Carlo microphysics | Stochastic reference |
| CARMA (github.com/ESCOMP/CARMA; Turco et al. 1979; Toon et al. 1988; Bardeen et al. 2008) | Apache-2.0 (repo described as "Stand-alone tests for the CARMA aerosol model") | Sectional nucleation/growth/evaporation/coagulation microphysics | Permissive; mature sectional design |
| MOSAIC (Zaveri et al. 2008) | **[licence unverified]** (distributed within WRF-Chem) | Sectional aerosol thermodynamics/dynamics | Concepts only |
| Paperboats (Kirchschlager et al. 2019) | **No public repository found [unverified]** | Post-processing dust destruction: 40 bins + collector bin, sputtering + grain–grain collisions | Design reference |
| AREPO dust GSD (McKinnon et al. 2018) | **Public availability unverified** | Two-moment piecewise-linear remap | Algorithm (paper is detailed) |
| Ishinisan (Marchand et al. 2021) | **[unverified]** | Two-collector power-law coagulation | Already reimplemented by you |
| Shark (github.com/ICRAR/shark) | GPL-3.0 | Semi-analytic galaxy model | Not a GSD solver as far as we can tell; not relevant |
| "GRAINSIZE" | Not found | — | — |

Licence note: GPL/CeCILL code should be studied rather than copied unless pyCALIMA is released under a compatible licence. CARMA (Apache-2.0) is the only mature permissive sectional microphysics code we identified.

---

## 7. Recommended architecture for the pyCALIMA GSD solver

### 7.1 State and grid

* **State.** Per composition s ∈ {C, silicate(s), Fe, FeS, Al₂O₃, …} and bin i: N_{s,i} [cm⁻³] and M_{s,i} [g cm⁻³]. Gas key-species/elemental densities n_X are carried alongside. A per-composition **gas collector** records mass returned by sublimation, sub-a_min fragments and sputtering, mapped stoichiometrically back to elements (e.g. FeS → Fe + S).
* **Realizability invariant.** m_{L,i} ≤ m̄_i ≤ m_{R,i}. If an operation violates it, move the whole population to the neighbour bin (moving-center rule) or split it with a two-moment-consistent rule.
* **Grid.** Fixed and geometric in mass, from a_min ≈ 0.25–0.3 nm (≈ 10 monomers; one C atom ≈ 0.13 nm equivalent radius) to a_max ≈ 10–30 µm. That is ~13–14 decades in mass. Use **8–10 bins per mass decade (≈ 110–140 bins)**, above the ≥ 7/decade minimum of Ohtsuki et al. (1990)/DustPy, and verify convergence at 5, 7, 10, 14 per decade. Below ~10–100 monomers consider a discrete n-mer block (NK13 used n* = 100), or start the continuum at the nucleation size.

### 7.2 Sub-bin reconstruction (recommendation)

* **Two-moment power law (PL2M).** Within the bin, f ∝ m^β, with β the unique root of m̄(β) = M/N. Here m̄(β) = m_L·((β+1)/(β+2))·(η^{β+2} − 1)/(η^{β+1} − 1) with η = m_R/m_L (special cases β = −1, −2), which is monotone in β. Tabulate or solve by Newton.
* Properties *(our analysis)*:
  * always positive;
  * spans the full realizable range: β → ±∞ is a δ at a bin edge;
  * local, so no neighbour slopes;
  * reuses your power-law integral machinery (moment formulas as in Marchand et al. 2021, eq. B.15).
* **Alternative.** Affine-in-m reconstruction with a positivity limiter (McKinnon 2018; Laurent 2016).

### 7.3 Exchange operator E (nucleation + growth + sublimation + sputtering + gas)

The operator is sub-stepped, with the sub-step limited by |Δln S| and the burst time scale. Each sub-step:

1. Solve the scalar implicit gas equation (Section 5) for n_X^{n+1}, which gives ȧ_s(a) for each composition (Kelvin, T_d(a), sputtering yield).
2. Compute the flow map X_Δt at bin edges plus 2–4 Gauss nodes per bin:
   * analytic for size-independent ȧ in a;
   * otherwise RK4/analytic per node.
3. Remap PL2M segments onto the fixed bins. Overlap integrals give (N, M) exactly for constant ȧ and via Gauss quadrature otherwise. Grains mapped below a_min (or to a ≤ 0) go to the gas collector with their full mass.
4. Inject nucleation as the mapped source segment (Section 2.6).
5. Enforce n_X = n_X,tot − Σ dust/collector exactly. This takes up round-off and leaves conservation exact.

Characteristics are monotone, so the remap needs no CFL condition. Stability is unconditional, and accuracy is controlled by the gas-coupling step.

### 7.4 Collision operator C (coagulation + shattering/cratering, optional cross-composition)

* **Pair rates.** R_ij = K_ij N_iN_j/(1 + δ_ij). The kernel is K = σ_geo·Δv·p·(Coulomb factor), with Δv² summing Brownian, turbulent (Ormel & Cuzzi 2007), radiation-pressure drift and settling contributions. Evaluate it at 2 Gauss nodes per bin, which captures in-bin structure, or at m̄. Cache when T and ρ are unchanged.
* **Coagulation.** Deposit (ΔN, ΔM) = (R, R(m₁ + m₂)) into the bin containing m₁ + m₂, and remove (R, Rm₁) and (R, Rm₂) from the sources. When the target is the large-grain bin itself (m_j ≪ m_k), add only ΔM_k = Rm_j and skip the cancelling ±R. This is the two-moment equivalent of DustPy's precision fix.
* **Shattering/cratering.** Use JTH96/HY09 or KT10 ejecta with precomputed power-law fragment tensors (Section 4.2) and the HY09 four-direction velocity averaging. Fragments below a_min go to the collector (record the choice).
* **Integrator.** MPRK22 in mass with consistent Patankar weighting of N, with adaptive Δt from the embedded MPE/MPRK22 difference. A fallback/fast mode is the Jacobson et al. (1994) sequential semi-implicit scheme. Both are positive, and both are exactly conservative in mass.
* **Cross-composition collisions.** This is a design choice:
  * neglect them, as HY09 do;
  * create a "mixed" population (Jacobson 2002's P/Q/L interaction flags);
  * assign the product to the dominant-mass composition while tracking element masses.
* **Cost.** O(S²N_b²) pair evaluations, ~2.5×10⁵ for S = 5 and N_b = 100, which is vectorisable to around a millisecond, plus O(N_b³) per species for the MPRK solve.

### 7.5 Splitting and step control

* **Macro-step.** C(Δt) followed by E(Δt), stiff exchange last (Sportisse 2000). Optionally use Strang ½C – E – ½C for second order if stiffness permits; compare both.
* **Step control.** Choose Δt from limits on the relative changes of N_tot,s, M_s, S and the largest-bin mass. Add step doubling as an error estimate.
* **Diagnostics.** Log the conservation errors of elements, the number budget (nucleation − coagulation + fragments − sublimation losses) and the realizability violations corrected.

### 7.6 Verification suite (in order)

1. **Pure advection, constant ȧ > 0.** Translate a log-normal and a δ-burst. Compare to the exact n(a − ȧt). N and M should be exact, and the burst width should stay ≤ 1 bin.
2. **Pure sublimation, constant ȧ < 0.** Exact n(a + |ȧ|t), N(t) = ∫_{|ȧ|t}^∞ n₀ da, and returned gas mass. Also run size-dependent cases with ȧ = G/a (exact a² = a₀² + 2Gt) and a Kelvin-type ȧ with a fixed point (the pile-up test for two-moment vs one-moment).
3. **Nucleation + growth, constant J and ȧ.** Exact top-hat n = J/ȧ on [a*, a* + ȧt]. This is the class of test used by Kumar & Ramkrishna (1997).
4. **Closed-box condensation to saturation.** Element conservation to round-off and S → 1 (Kelvin-corrected). Compare with a monodisperse ODE reference.
5. **Ostwald ripening (LSW).** Use a near-saturated closed box with the Kelvin effect (Lifshitz & Slyozov 1961; Wagner 1961). For free-molecular, interface-limited growth the Wagner-type scaling ⟨a⟩² ∝ t should apply rather than the diffusion-limited ⟨a⟩³ ∝ t *(standard result; confirm against the papers)*.
6. **Smoluchowski analytic solutions.** Constant, additive and multiplicative kernels (Scott 1968; review Aldous 1999), in the Lombart & Laibe (2021) setup (L1 errors, moments M₀, M₁; multiplicative only up to gelation). Also check resolution sensitivity against the Ohtsuki et al. (1990) artificial-acceleration effect.
7. **Coagulation–fragmentation.**
   * Dubovskii et al. (1992) exact solutions.
   * Dohnanyi (1969) / Tanaka et al. (1996) steady-state cascades.
   * Birnstiel et al. (2011) steady-state distributions and fits.
   * Optionally DustPy or Coala runs as references (whether a single-annulus DustPy setup is practical was **not verified**).
8. **Published astrophysical benchmarks.**
   * Hirashita & Yan (2009) shattering/coagulation in WNM/WIM/CNM/MC (N = 32, 0.001–0.25 µm, α_f = 3.3).
   * Hirashita & Kobayashi (2013) parameter sensitivities (P₁/t scaling, f_M).
   * Hirashita & Kuo (2011) analytic accretion, and Hirashita (2012) accretion + coagulation. A low-diffusion scheme should preserve the peak that their upwind scheme smoothed.
   * NK13 non-steady nucleation: condensation efficiency f_con,∞ and average radius a_ave,∞ versus the cooling-time parameter, with gas depletion.
9. **Regression against the current one-moment pyCALIMA solver** on accretion-only and coagulation-only problems.

**Charging (phase 2e).** Each radius gets the RAMSES-CALIMA group solve, on groups extended to 10 keV and potentials to 30 kV.
* **AGN photon field:** the SKIRTOR SED gives n_γ(>13.6 eV) = 2.5e6 L45 (r/pc)⁻². WDB06 quote 2.0e6 and TIK20 2.6e6.
* **Coulomb explosion:** in TIK20's polar-dust gas (T = 1e4 K, n_e = 1e2) at L = 1e46, a_CE ≈ 0.07 µm for graphite at S_max = 1e9 and 0.02 µm for silicate at 1e10, at r = 3 L45^{1/2} pc. TIK20 find 0.01–0.1 µm. a_CE falls ∝ r^{−1/2}, far more slowly than the sublimation radius.
* **Field emission:** at S_max = 1e11 ion field emission caps the charge first and nothing explodes (WD00 s. 4.2).
* **Ion accretion:** positive grains at 10²–10³ V take up no ions (⟨J̃⟩ < 1e-37), so growth in photoionised gas runs on the neutral fraction alone.
* **Caveats:**
  * WDB06 covers only graphite and silicate; other materials are mapped onto them (`charging_material`).
  * Grain-grain Coulomb uses point charges.
  * The rim densities of 2f (n_H ≳ 1e8) are where electron capture should pull the potentials down. That needs checking there.

**Performance pass.** CPU times, single-threaded:

| Workload | Before | After | Change |
|---|---|---|---|
| WIM shattering + sticking, 10 Myr, 120 bins | 20 s | 7.3 s | 2.7× |
| Tabulated rate evaluation | 25 ms | 7.4 ms | 3.4× |
| Evolved-state rate evaluation | 24 ms | 14 ms | 1.7× |
| Coagulation step | 3.6 ms | 2.3 ms | 1.6× |
| Coulomb-explosion rates | 12.7 ms | 0.5 ms | 25× |

What changed:
* **Fused, compiled pair sum** with no 20 MB channel arrays.
* **Bin lookups and fragment powers:** O(1) bin lookups from logarithms the caller already has; fragment powers from ln a, with the fixed lower edge precomputed.
* **Single-bin remnant fast path.**
* **Negligible-pair pruning** at 1e-12 (results identical to 4e-11).
* **Profiles:** one `BinProfile` per evaluation, and an Illinois β solve (180 → 39 µs).
* **Explosion rates:** skip charges more than 40σ out.

Rejected:
* **RMS error norm:** equivalent to a looser max norm, no free gain.
* **numba `parallel=True`:** slower than serial here even with OpenMP enabled.

**Thread scaling** (idle machine, nogil Python threads on the performance cores):
* WIM shattering takes 5.9 s on 1 thread, 1.8 s on 4 and 1.07 s on 10 (5.5×), with no gain on 12–14 threads.
* Coagulation sums are small (~0.6 ms) and run best on 3–4 threads.
* Overall the WIM 10 Myr run went from 20 s to ~1.1 s of wall time (~17×).

**Status of the suite (`tests/test_gsd.py`).** Items 1-6, the steady-cascade part of item 7 and the NK13 and HK13 parts of item 8 pass.

* **Steady cascade (item 7).** Mass is fed at the top and removed below 0.01 µm, with constant v = 1 km/s and Q*_D = 1e9 erg/g. KT10 ejecta come from the larger grain, the projectile is shattered, and fragments are a^{−α_f} up to (f/2)^{1/3}a_t.
  * dN/da ∝ a^{−3.50±0.01} over 0.1–10 µm for α_f = 2.5, 3.0 and 3.5, independent of the fragment law as Tanaka et al. (1996) predict.
  * At v²/2Q*_D ≫ 1 (v = 10 km/s) the cut-off waves (Campo Bagatin et al. 1994) span the whole 4-decade grid. That this is physical is confirmed by shifting a_min (the pattern moves with it) and by doubling the resolution (no change), so the test stays at moderate v²/2Q*_D. Q*_D ∝ a^β runs show the predicted trend in slope, but not the K10 value, because v²/2Q*_D then varies by orders of magnitude across the grid.
* **HK13 (item 8).**
  * The KT10 law gives their worked example exactly (M_ej/m₁ = 0.18).
  * For MRN silicate in their WIM Model H, runs with equal P₁/t agree within 0.55–1.19 below 0.03 µm (HK13: factor 3), and JTH96/KT10 within 0.64–1.35 (HK13: ≲1.8).
* **Lessons:**
  * Remnants must be deposited spread over the parent node's width, not at a point. Point deposits switch bins discontinuously as a node crosses an edge, so cratering piles grains at the lower bin edges (in-bin β → −10 instead of −1.83) and the step control chatters (176k against 606 sub-steps for the cascade).
  * External sources belong inside the MPRK step. Adding them between long chunks sends pulses down the cascade.
  * Velocity laws must be continuous at the bin nodes (HK13's step at a_c: 300 s against 1.6 s for the same test). Physical runs therefore use `gsd.velocities.RelativeVelocity`:
    * the random rms speed is Brownian plus the full Ormel & Cuzzi (2007) form, as in the collapse module and RAMSES `fold/*`;
    * an optional drift, with the skewed-Maxwellian speed distribution (Guillet+20 App. C);
    * shattering in n_q equal-flux quantile channels above v_shat, and sticking as the flux below v_coag.

    The flux above a speed matches quadrature to 2e-9, and the quantiles invert it to ≤2e-3 in u. The summed rate reproduces the collapse coagulation kernel exactly, and the rates are smooth in T. A 50 Myr WIM run (v_turb = 10 km/s, L = 100 pc) takes 484 sub-steps.

* **Smoluchowski (item 6)**, exponential start, mass grid at 10 bins per mass decade:
  * **Constant kernel:** to τ = K N₀t/2 = 100, N matches N₀/(1+τ) to 2e-5. The L1 error of the binned distribution is 0.7%, falling to 0.16% at 20 bins/decade (second order).
  * **Additive kernel:** N matches N₀e^{−τ} to 4e-5. The mass distribution against Golovin's solution is off by 4.1% (10 bins/decade) and 1.4% (20) at τ = 4.
  * **Multiplicative kernel:** N matches N₀ − BM²t/2 to 5e-5 up to 0.95 of the gelation time.
  * Mass is conserved to ~1e-15 in every case.

* **LSW (item 5).** Linearising the Kelvin term gives da/dt = K(1/a* − 1/a), with K = V J_bulk c and c = θ(3V/4π)^{1/3}/T. This is Wagner's interface-limited law. Its self-similar solution has a*² = Kt/2, ā = (8/9)a*, r_max = 2a* and P(ρ) ∝ ρ(2−ρ)^{−5} exp[−3ρ/(2−ρ)]. Started on that distribution, the solver stays on it: a* is within 1% of √(Kt/2) and ā/a* within 1% of 8/9 (20 bins per mass decade). Over a factor 7 in a*, the error in ā² is 4.7% at 10 bins/decade and 1.4% at 20. From a narrow log-normal the coarsening rate approaches 32/81 K only logarithmically slowly (0.457, 0.435, 0.426 per decade of time), as expected for LSW.
* **NK13 (item 8).** NK13's homologous expansion, run through the solver (exchange steps plus dilution), with N03 carbon at s = 1:
  * **Kelvin term on** (grains injected past the barrier, below): a_ave,∞ matches their non-steady cluster model to 0.6%, 0.4% and 1.2% at c10 = 1e6, 1e8 and 1e9 cm⁻³, and eq. 64 to ≤4.5% for Λ_on ≥ 50. f_con = 1. Against their steady model the match is ≤3.5% over Λ_on = 0.04–3e6 at γ = 1.25 (the figure of `check_nucleation_nozawa_kozasa2013.py`).
  * **Kelvin term off:** matches their steady model to ≤1% up to Λ_on ~ 1e4. The remaining 3.5% at Λ_on ~ 1e6 is the first-order splitting of the dilution at Δt = 0.002t.
  * **Low Λ_on:** below ~30 the steady kernel keeps f_con = 1, as their steady model does (their Fig. 8, grey). Non-steady effects then need a cluster block.
* **Lessons from the NK13 run:**
  * The backward-Euler gas update must be sub-stepped while nucleation is active, because J is exponentially sensitive to S. With whole steps the error is first order in Δt, with a_ave 25% high at Λ ~ 1e5. `exchange_step` therefore halves steps until |Δ ln n| ≤ 0.005 whenever the grains nucleated in the step exceed 1e-3 of the population.
  * Grains must be injected past the nucleation barrier, at the Zeldovich width n* + 3n*^{2/3}/√μ. At n* itself ȧ = 0, the grains stall while J_s keeps counting, and a_ave comes out 9× too small at Λ ~ 1e6.
  * The driver's step must resolve the change of ln S from cooling, (A/T) Δln T, because T is held fixed within a step. At γ = 1.5 and Λ_on ~ 1e5–1e6, Δt = 0.002t puts a_ave 20–50% high, and 5e-4·t converges to ~1%. This rule carries over to any driver with a time-dependent T_d.
  * In cold gas the vapour pressures underflow, so S, the condensation α and the nucleation cap are evaluated in log space (`SublimationModel.log_flux`, `dust_sublimation.ln_supersaturation`).
  * The characteristic must be written x + f t·expm1(gt)/(gt). The form a₀ + (f e^{gt} − f₀)/g cancels when the velocity is nearly constant, and its 1e-8 relative noise in the dust mass became spurious gas, and so spurious nucleation, once the gas was depleted to ~1e-8 of the element.

---

## 8. Reference list with verification status

Verification key: **CR** = Crossref DOI metadata; **AX** = arXiv abstract page; **PDF** = full text read; **PUB** = publisher/HAL page; **WS** = web-search snippets of the publisher/ADS record; **GH** = GitHub API/LICENSE.

* Aldous D. J. 1999, Bernoulli 5, 3 — "Deterministic and stochastic models for coalescence…: a review of the mean-field theory for probabilists". CR.
* Anderl S., Guillet V., Pineau des Forêts G., Flower D. R. 2013, A&A 556, A69 — Shocks in dense clouds IV (note: Paper IV is Anderl et al., not Guillet et al. 2011). CR, AX 1408.0140.
* Aoyama S., Hirashita H., Nagamine K. 2020, MNRAS 491, 3844 — Galaxy simulation with the evolution of grain size distribution. AX 1906.01917, PDF, WS (volume/page; Crossref lacked them).
* Asano R. S., Takeuchi T. T., Hirashita H., Nozawa T. 2013, MNRAS 432, 637. AX 1303.5528, CR.
* Bardeen C. G. et al. 2008, JGR 113, D17202 (article number **[unverified]**; DOI 10.1029/2007JD009515). CR.
* Birnstiel T., Dullemond C. P., Brauer F. 2010, A&A 513, A79. AX 1002.0335, CR, PDF.
* Birnstiel T., Ormel C. W., Dullemond C. P. 2011, A&A 525, A11. AX 1009.3011, CR, PDF (partial).
* Birnstiel T., Klahr H., Ercolano B. 2012, A&A 539, A148. AX 1201.5781, CR.
* Bott A. 1989, Mon. Weather Rev. 117, 1006. CR.
* Bott A. 1998, J. Atmos. Sci. 55, 2284. CR.
* Brauer F., Dullemond C. P., Henning Th. 2008, A&A 480, 859. AX 0711.2192, CR.
* Burchard H., Deleersnijder E., Meister A. 2003, Appl. Numer. Math. 47, 1. CR.
* Chen J.-P., Lamb D. 1994, J. Atmos. Sci. 51, 2613. CR.
* Dohnanyi J. S. 1969, JGR 74, 2531. CR.
* Draine B. T., Salpeter E. E. 1979, ApJ 231, 438 (destruction mechanisms) and ApJ 231, 77 (hot gas). CR.
* Draine B. T., Sutin B. 1987, ApJ 320, 803. CR.
* Dubovskii P. B., Galkin V. A., Stewart I. W. 1992, J. Phys. A 25, 4737. CR.
* Filbet F., Laurençot Ph. 2004, SIAM J. Sci. Comput. 25, 2004. CR (abstract read).
* Frenklach M., Harris S. J. 1987, J. Colloid Interface Sci. 118, 252. CR.
* Gail H.-P., Keller R., Sedlmayr E. 1984, A&A 133, 320 — "Dust formation in stellar winds. I. A rapid computational method and application to graphite condensation". WS (Cambridge bibliography snippets; no DOI).
* Gail H.-P., Sedlmayr E. 2013, Physics and Chemistry of Circumstellar Dust Shells, CUP (doi:10.1017/CBO9780511985607). CR.
* Gelbard F., Seinfeld J. H. 1980, J. Colloid Interface Sci. 78, 485. CR.
* Gelbard F., Tambour Y., Seinfeld J. H. 1980, J. Colloid Interface Sci. 76, 541. CR.
* Gottlieb S., Shu C.-W., Tadmor E. 2001, SIAM Rev. 43, 89. CR.
* Guillet V., Jones A. P., Pineau des Forêts G. 2009, A&A 497, 145 — Shocks in dense clouds II (subtitle **[unverified]**). CR.
* Guillet V., Pineau des Forêts G., Jones A. P. 2011, A&A 527, A123 — Shocks in dense clouds III. Dust processing and feedback effects in C-type shocks. CR, PUB/WS.
* Hirashita H. 2010, MNRAS 407, L49. CR.
* Hirashita H. 2012, MNRAS 422, 1263. AX 1202.1345, CR, PDF.
* Hirashita H. 2015, MNRAS 447, 2937 (two-size approximation). CR.
* Hirashita H., Aoyama S. 2019, MNRAS 482, 2555 — Remodelling the evolution of grain size distribution in galaxies. CR.
* Hirashita H., Kobayashi H. 2013, Earth Planets Space 65, 1083. AX 1303.3958, CR, PDF.
* Hirashita H., Kuo T.-M. 2011, MNRAS 416, 1340. AX 1105.4930, CR.
* Hirashita H., Yan H. 2009, MNRAS 394, 1061. AX 0812.3451, CR, PDF.
* Hounslow M. J., Ryall R. L., Marshall V. R. 1988, AIChE J. 34, 1821. CR.
* Huang J., Shu C.-W. 2019, J. Sci. Comput. 78, 1811 (online 2018). CR.
* Jacobson M. Z. 1997a, Atmos. Environ. 31, 131 (aerosol module structure; moving-center). CR, WS.
* Jacobson M. Z. 1997b, Aerosol Sci. Technol. 27, 491 (APC). CR.
* Jacobson M. Z. 2002, JGR 107(D19), doi:10.1029/2001JD002044 (article number **[unverified]**). CR, PDF (author-hosted).
* Jacobson M. Z., Turco R. P. 1995, Aerosol Sci. Technol. 22, 73. CR (content **[unverified]**).
* Jacobson M. Z., Turco R. P., Jensen E. J., Toon O. B. 1994, Atmos. Environ. 28, 1327. CR.
* Jones A. P., Tielens A. G. G. M., Hollenbach D. J. 1996, ApJ 469, 740. CR.
* Kah D., Laurent F., Massot M., Jay S. 2012, J. Comput. Phys. 231, 394. CR.
* Kirchschlager F. et al. 2019, MNRAS 489, 4465. AX 1908.10875, CR, PDF.
* Kobayashi H., Tanaka H. 2010, Icarus 206, 735. CR, WS.
* Kopecz S., Meister A. 2018a, BIT Numer. Math. 58, 691; 2018b, Appl. Numer. Math. 123, 159. CR.
* Kovetz A., Olund B. 1969, J. Atmos. Sci. 26, 1060. CR.
* Kumar S., Ramkrishna D. 1996a, Chem. Eng. Sci. 51, 1311 (fixed pivot). CR.
* Kumar S., Ramkrishna D. 1996b, Chem. Eng. Sci. 51, 1333 (moving pivot). CR, PDF (author-hosted).
* Kumar S., Ramkrishna D. 1997, Chem. Eng. Sci. 52, 4659 (nucleation, growth, aggregation). CR, PDF (author-hosted).
* Kumar J., Peglow M., Warnecke G., Heinrich S., Mörl L. 2006, Chem. Eng. Sci. 61, 3327 (cell average). CR.
* Laurent F., Sibra A., Doisneau F. 2016, Commun. Comput. Phys. 20, 902. CR, PUB (HAL abstract).
* Lifshitz I. M., Slyozov V. V. 1961, J. Phys. Chem. Solids 19, 35. CR.
* Liu H., Gröpler R., Warnecke G. 2019, SIAM J. Sci. Comput. 41, B448. CR.
* Lombart M., Laibe G. 2021, MNRAS 501, 4298. AX 2011.12298, CR, PDF.
* Lombart M., Hutchison M., Lee Y.-N. 2022, MNRAS 517, 2012. AX 2208.02964, CR.
* Lombart M., Bréhier C.-E., Hutchison M., Lee Y.-N. 2024, arXiv:2404.11851 (journal **[unverified]**). AX.
* Marchand P., Guillet V., Lebreuilly U., Mac Low M.-M. 2021, A&A 649, A50 (Paper I); 2022, A&A 666, A27 (Paper II). AX 2103.00002, 2202.11625; CR; PDF (Paper I).
* Marchisio D. L., Fox R. O. 2005, J. Aerosol Sci. 36, 43. CR.
* Mattsson L. 2016, Planet. Space Sci. 133, 107. AX 1606.02272, CR, PDF.
* Mattsson L. 2020, MNRAS 499, 6035. AX 2010.01953, CR, PDF.
* McGraw R. 1997, Aerosol Sci. Technol. 27, 255. CR.
* McKinnon R., Vogelsberger M., Torrey P., Marinacci F., Kannan R. 2018, MNRAS 478, 2851. AX 1805.04521, CR, PDF.
* Nozawa T., Kozasa T., Habe A. 2006, ApJ 648, 435. AX astro-ph/0605193, CR.
* Nozawa T., Kozasa T. 2013, ApJ 776, 24 ("NK13"). AX 1308.1873, CR, PDF.
* Ohtsuki K., Nakagawa Y., Nakazawa K. 1990, Icarus 83, 205. CR.
* Okuzumi S. 2009, ApJ 698, 1122. CR.
* Ormel C. W., Cuzzi J. N. 2007, A&A 466, 413. AX astro-ph/0702303, CR.
* Rafikov R. R., Silsbee K., Booth R. A. 2020, ApJS 247, 65. CR (abstract read).
* Riemer N., West M., Zaveri R. A., Easter R. C. 2009, JGR 114, D09202 (article number **[unverified]**; DOI 10.1029/2008JD011073). CR.
* Robinson A., Booth R. A., Owen J. E. 2024, MNRAS 529, 1524 (cuDisc). AX 2402.18471, CR, PDF.
* Sandu A. 2001, J. Comput. Phys. 170, 589. CR.
* Scott W. T. 1968, J. Atmos. Sci. 25, 54. CR.
* Sportisse B. 2000, J. Comput. Phys. 161, 140. CR, WS.
* Stammler S. M., Birnstiel T. 2022, ApJ 935, 35 (DustPy). AX 2207.00322, CR, PDF.
* Tanaka H., Inaba S., Nakazawa K. 1996, Icarus 123, 450. CR.
* Tielens A. G. G. M., McKee C. F., Seab C. G., Hollenbach D. J. 1994, ApJ 431, 321. CR.
* Toon O. B., Turco R. P., Westphal D., Malone R., Liu M. S. 1988, J. Atmos. Sci. 45, 2123. CR.
* Tsai J. C., Mathews W. G. 1995, ApJ 448, 84 — "Interstellar grains in elliptical galaxies: grain evolution". CR (numerical scheme not checked).
* Turco R. P., Hamill P., Toon O. B., Whitten R. C., Kiang C. S. 1979, J. Atmos. Sci. 36, 699. CR.
* Tzivion S., Feingold G., Levin Z. 1987, J. Atmos. Sci. 44, 3139. CR.
* Wagner C. 1961, Z. Elektrochem. 65, 581. CR.
* Woitke P., Helling Ch. 2003, A&A 399, 297. CR, WS.
* Yan H., Lazarian A., Draine B. T. 2004, ApJ 616, 895. AX astro-ph/0408173, CR.
* Zaveri R. A., Easter R. C., Fast J. D., Peters L. K. 2008, JGR 113, D13204 (article number **[unverified]**; DOI 10.1029/2007JD008782). CR.
* Zhang Y., Seigneur C., Seinfeld J. H., Jacobson M. Z., Binkowski F. S. 1999, Aerosol Sci. Technol. 31, 487. CR, PDF (author-hosted).

**Not included / not found:** Golovin (1963) analytic sum-kernel solution (not located in Crossref; cite via Scott 1968 or Aldous 1999 instead); Smoluchowski (1916) (not checked); Jacobson (2005) textbook (not checked); "GRAINSIZE" code (not found); Hellyer (1970) fragment law (only cited via HY09).
