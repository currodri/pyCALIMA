# PAHchem against the PDRs4All methods: what we fit, what they fit (2026-10-21)

This page compares how the PDRs4All papers model the Orion Bar PAH emission
with how pahchem fits it. It lists every degree of freedom in our fit, and it
proposes how to produce profiles along the PDR (population, charge,
hydrogenation, spectra) that can be set against theirs.

Sources:
- Papers: read in full unless marked **[abs]** (abstract or snippet only).
- Data products: checked file by file on the PDRs4All site, CDS, MAST, Zenodo
  and GitHub.
- pahchem numbers: the current spectral fit (`fit_pdrs4all_spectra.py
  --fit-shape`, result `392313363fe8`: v4.00, IPs calibrated, lab CH-stretch
  scale, BT94 capture, Gaussian profiles). The M26 metrics were recomputed at
  our best cells for this page.

## 1. Summary

- **No PDRs4All paper does what pahchem does.** None computes the charge and
  dehydrogenation of each molecule from the local G$_0$, n$_{\rm H}$, n$_e$
  and T and then fits abundances. Their methods fall into four levels:
  1. empirical: band ratios, PAHFIT, blind signal separation;
  2. PAHdb fits with the charge free: pyPAHdb, {cite:t}`Maragkoudakis2026`
     (M26);
  3. one dust forward model with no charge states: THEMIS + DustEM + SOC,
     Elyajouri+24;
  4. Meudon PDR models, used only for gas tracers (H$_2$, HD, CH$^+$).
     Meudon is never used to predict the PAH spectrum in these papers. The
     PAH part of the pre-launch synthetic spectra came from PAHTAT.
- **Where we are ahead.**
  - The charge follows from the stated conditions, so γ is a test, not an
    output.
  - The 3.3 µm band is in the fit.
  - The continuum is fitted with the bands, by the same rule for data and
    model.
  - The band shape is fitted; it is not one fixed 15 cm$^{-1}$ Gaussian.
- **Where we are behind.**
  - Five templates only, no positions.
  - One unattenuated field shape at every depth.
  - Conditions on a coarse discrete grid, with no calibrated error model.
  - The band shape is empirical, not physical.
  - No geometry or line-of-sight integration, which Elyajouri+24 have.
  - Population sizes come out larger than everyone else's, with no gradient
    across the Bar.
- **The headline disagreement is size.**
  - M26, Knight+22 and Chown+24 all find PAHs getting smaller from the
    surface into the dissociation fronts:
    - M26: ⟨N$_{\rm C}$⟩ 71 → 62;
    - Knight+22: ≈ 90 → 60;
    - Chown+24: 3.3/11.2 about 40 % higher in DF3.
  - pahchem gives 83–87 in every template, with only the share above
    N$_{\rm C}$ = 70 falling (0.69 → 0.55).
- **Charge agrees; γ sits between the two published calibrations.**
  - Our cation share is 0.48–0.51 in every template; M26 find 0.47–0.60.
  - Our best γ ≈ 1.0 × 10$^4$ everywhere lies between Knight+22
    (5–8.5 × 10$^3$) and M26 (2.9–5.5 × 10$^4$).
  - Both of those γ are derived from band ratios through calibrations. Ours
    comes from a charge balance at sourced conditions.

## 2. What the PDRs4All papers do

### 2.1 Methods, ranked by how close they are to pahchem

**M26: pyPAHdb per pixel** ({cite:t}`Maragkoudakis2026`, arXiv:2601.23282;
numbered XIX in A&A, XVIII on arXiv). The closest analogue.
- Data: MIRI-MRS 5–15 µm, about 8 500 pixels and the five templates.
- Preprocessing: lines, baseline and continuum are removed first
  (pybaselines morphological, then imodpoly). No extinction correction.
- Library: PAHdb v4.00-α, pure PAHs with 20 < N$_{\rm C}$ < 384; anions,
  neutrals and cations, the cations including dications.
- Emission: the PAHdb "cascade" at one fixed 7 eV absorbed energy in every
  zone. Each line is a 15 cm$^{-1}$ FWHM Gaussian. Harmonic, with no shift.
- Fitted: only the non-negative molecule weights.
- Derived:
  - charge shares, three size bins and ⟨N$_{\rm C}$⟩;
  - γ = 2.66 (n$^+$/n$^0$) × 10$^4$, calibrated on circumcoronene. γ is
    never computed from G$_0$, T or n$_e$.
- 3.3 µm is excluded: it needs anharmonic spectra.
- pahchem has already rebuilt this pipeline from public pieces:
  `check_maragkoudakis2026.py`, and `explore_maragkoudakis2026.py`, which
  swaps their choices for ours one at a time (spec section "Maragkoudakis+26
  recovered").

**Elyajouri+24, PDRs4All V** (A&A 685, A76; arXiv:2401.01221). The only
forward model of the emission against position.
- Dust: THEMIS a-C(:H) nanograins plus coated silicates, through DustEM,
  then 3-D Monte-Carlo radiative transfer with SOC.
- Geometry: an edge-on slab with 0.001 pc cells.
- Illumination: a 38 000 K blackbody at 0.24 pc, G$_0$ = 2.6 × 10$^4$ at the
  edge.
- Density: n(z) = n$_0$(z/z$_0$)$^{2.5}$, flat beyond z$_0$.
- Parameters: 8, explored as a staged grid, not a fit: band gap E$_g$ (the
  hydrogenation), a-C mass per H, a$_{\min}$, slope α, large-grain size,
  z$_0$, n$_0$ and l$_{\rm PDR}$.
- Results:
  - against the NIRSpec templates: E$_g$ = 0.03 eV at the surface and
    0.07 eV in DF3, i.e. more hydrogenated deeper in;
  - from six NIRCam/MIRI imaging profiles: nanograins 15× depleted against
    the diffuse ISM, a$_{\min}$ = 0.4 nm, n$_0$ ≈ 9 × 10$^4$ cm$^{-3}$,
    l$_{\rm PDR}$ = 0.12 pc.
- No charge states; no 5–6 µm bands.

**Knight+22** (MNRAS 509, 3523; Spitzer + SOFIA, two slits). A pre-JWST γ
and size profile.
- Inputs: G$_0$ from the 10–13 µm continuum, scaled to 2.6 × 10$^4$ at the
  IF; n$_{\rm H}$ = 1.44 × 10$^4$ cm$^{-3}$; n$_e$ = 1.6 × 10$^{-4}$
  n$_{\rm H}$; T = 500 K.
- γ ≈ 5–8.5 × 10$^3$ in the PDR.
- ⟨N$_{\rm C}$⟩ from 3.3/11.2 against Ricca+12 at 9 eV: 90 at the IF, 70 at
  the PAH peak, 60 at the PDR front.

**Band ratios and empirical decompositions.**
- **Peeters+24, PDRs4All III** ({cite:t}`Peeters2024`). NIRSpec. It defines
  the templates, decomposes the 3.2–3.7 µm region (PAHFIT and Gaussians,
  Table 3) and fits Meudon to H$_2$.
- **Chown+24, PDRs4All IV** ({cite:t}`Chown2024`). MRS band inventory on one
  spline continuum. Two surprises:
  - 6.2/11.2 changes by only 12 % against a ≈ 40× change in ionising flux;
  - 3.4/3.3 doubles from the APDR to DF3.
- **Khan+26, PDRs4All XX** (arXiv:2510.06167). Band maps and a cut.
  - 6.2/11.2 is the best charge proxy; 6.2/8.6 and 7.7/8.6 are size proxies.
  - It gives synthetic-filter prescriptions.
- **Habart+24, PDRs4All II.** Six imaging cuts.
  - n$_{\rm H}$ = (5–10) × 10$^4$ cm$^{-3}$ from the decay of the AIB
    emission.
  - P ≈ 2 × 10$^8$ K cm$^{-3}$ from 2 n$_e$T$_e$.
- **Van De Putte+25, PDRs4All XVI.** The PAHFIT "PDR pack": 66 Drudes,
  widths tuned and then fixed, and a 13-temperature modified-blackbody
  continuum.
- **Schroetter+24 and Amiot+26.** Blind signal separation and NMF, rank 2:
  - "AIB-irradiated": large, compact cations;
  - "AIB-shielded": small, neutral, aliphatic.
  - Pasquini+24 **[abs]**: k-means clustering.

**Gas conditions, from papers that do not model PAHs.**
- Zannese+26, HD with the upgraded isobaric Meudon model: P = (3–9) ×
  10$^7$ K cm$^{-3}$.
- Joblin+18, Herschel: ≈ 2 × 10$^8$.
- Habart+24, from 2 n$_e$T$_e$: ≈ 2 × 10$^8$.
- Goicoechea+25 **[abs]**: ≈ 10$^8$ in the C$_2$H zone.
- Gaches+26 **[abs]**: 3-D shadowing by clumps explains overlapping fronts.

A pressure gradient is suggested. The card should carry this spread as
ranges, not single values.

### 2.2 Degrees of freedom of each method

| method | fitted | fixed | assumed |
|---|---|---|---|
| pyPAHdb (M26) | non-negative weights, per pixel | 7 eV; 15 cm$^{-1}$ Gaussian; harmonic; continuum removed first | charge free; γ from n$^+$/n$^0$ (circumcoronene factor); no conditions |
| THEMIS/DustEM/SOC (Elyajouri+24) | E$_g$, a$_{\min}$, α, abundance, z$_0$, n$_0$, l$_{\rm PDR}$ (grid) | G$_0$ = 2.6 × 10$^4$; 38 kK star; density slope | bulk a-C optics; no charge |
| Meudon (Peeters+24, Zannese+26, Habart+24) | G$_0$, n (isochoric) or P, d* (isobaric) | A$_V$ = 10; inclination 4° | 1-D steady state; H$_2$/HD only |
| PAHFIT PDR pack (Van De Putte+25) | Drude amplitudes, continuum weights, extinction | 66 centres and widths | descriptive |
| band ratios (Chown+24, Khan+26, Knight+22) | integrated intensities | spline anchors | ratio → charge/size calibrations |
| BSS/NMF (Schroetter+24, Amiot+26) | two spectra and their maps | rank 2 | read physically afterwards |

## 3. pahchem against M26, template by template

M26's Table 1 values are zone averages of per-pixel fits. Ours are fits to
the flux-weighted templates. Both N$_{\rm C}$ means are emission-weighted.
For pahchem, emission-weighted and carbon-weighted means agree to < 1: in
the single-photon regime a molecule's emitted power follows its UV cross
section, which scales with N$_{\rm C}$.

| | HII | APDR | DF1 | DF2 | DF3 |
|---|---|---|---|---|---|
| **f$_{\rm cat}$** (Z ≥ 1), M26 / pahchem | 0.51 / 0.48 | 0.60 / 0.51 | 0.58 / 0.50 | 0.48 / 0.49 | 0.47 / 0.51 |
| **f$_{\rm an}$**, M26 / pahchem | 0.03 / 0.01 | 0.11 / 0.01 | 0.12 / 0.01 | 0.16 / 0.01 | 0.17 / 0.01 |
| **⟨N$_{\rm C}$⟩**, M26 / pahchem | 70 / 84 | 71 / 87 | 69 / 84 | 63 / 85 | 62 / 83 |
| share N$_{\rm C}$ > 70, pahchem | 0.62 | 0.69 | 0.67 | 0.57 | 0.55 |
| **γ** [10$^4$], M26 / pahchem | 2.9 / 0.98 | 5.5 / 1.0 | 5.1 / 1.0 | 3.6 / 0.99 | 3.5 / 1.1 |
| pahchem best cell (G$_0$, n$_{\rm H}$, n$_e$) | 1.8e4, 7.1e4, 46 | 1.8e4, 1e5, 46 | 1e3, 1.9e4, 2.6 | 1e3, 1.9e4, 2.6 | 2e3, 3.5e4, 4.9 |
| pahchem rms, continuum share (5–15 µm) | 0.048, 0.71 | 0.102, 0.39 | 0.091, 0.44 | 0.064, 0.55 | 0.055, 0.63 |

Note: the G$_0$ in the best-cell row is the Habing-integral value, 1/1.236 of
the card's F$_\nu$(1030 Å) value.

Reading:
- **Cations.** Both are near half, but M26 peak at the APDR and fall into
  the DFs, while pahchem is flat.
- **Anions.** M26's 11–17 % anions in the DFs is a degeneracy of a
  charge-free fit; the spec's swap study R1–R6 shows anions act as spectral
  fillers. Our charge balance allows ≤ 1 %. This is the clearest gain from
  using physical charges.
- **γ.**
  - Ours is 0.99–1.07 × 10$^4$ in all five templates. That uniformity is
    suspicious.
  - In four templates the best G$_0$ sits at the low edge of the card range.
  - So the fit is selecting the γ that the band-shape and library
    combination prefers, more than reading the conditions.
  - The candidates are the same as in the spec: BT94 capture for
    N$_{\rm C}$ ~ 100 is unmeasured (×3 would put us at M26's γ), plus the
    IPs and the yields. The capture-×3 result of 2026-10-07 was withdrawn
    with finding C1 and has not been redone on the spectral fit.
- **Size.** Ours is ~15–25 larger and has no gradient. Candidates, none yet
  tested on the spectral fit:
  1. **3.3 µm in the fit.** The DFT CH stretch is 2× too strong per
     molecule, and the lab scale corrects that only on average. On the
     withdrawn band-ratio fits, removing 3.3 µm took DF3 from C90 to C52.
  2. **The band width absorbing the trend.** The fitted CH FWHM falls from
     43 (HII) to 34 cm$^{-1}$ (DF3), and the oop/rest widths change too.
  3. **The free continuum** takes 0.4–0.7 of the 5–15 µm power, so it can
     trade with the 11.2 and 12.7 µm wings.
  4. **The intact filter** (≥ 0.5) removing small molecules at the surface
     but not deeper in. That pushes the opposite way, so it would not
     explain the flat profile.

## 4. Every degree of freedom in pahchem's fit

### 4.1 Per template

| quantity | number | how chosen | what it absorbs |
|---|---|---|---|
| molecule abundances w$_s$ ≥ 0 | 1424 candidates; 55–93 non-zero | NNLS | everything spectral: band ratios, positions through isomer choice |
| continuum | 4 B-splines (3.1–3.65 µm) + ~10 (5–15 µm), ≥ 0, knots 1.5 µm | NNLS, with the w$_s$ | the plateaus, band wings, VSG emission |
| band shape | 5: CH shift, CH FWHM, CC shift, oop shift, rest FWHM | alternation with the scan | anharmonicity, emission temperature, carrier mismatch (CC blue shift) |
| profile type | Gaussian or Lorentzian | best sse | — |
| conditions | 1 of 48 cells (4 G$_0$ × 3 n$_{\rm H}$ × 4 n$_e$) inside the card ranges; T fixed at H$_2$ T$_1^*$ | best sse | the charge balance and the H-loss filter |

The conditions enter the spectrum only through two things:
- the charge fractions f$_{sz}$;
- the intact mask.

In the single-photon regime the emission per molecule is independent of
G$_0$, so G$_0$, n$_e$ and T act almost entirely through γ, plus a weak
n$_{\rm H}$ dependence through H-loss. The 48 cells are therefore close to a
one-parameter γ scan with a few duplicates. That is why every template lands
on the same γ grid point.

### 4.2 Fixed globally (calibrated or stated)

- **Library:** PAHdb v4.00, pure aromatic CH only (1424).
- **IP and EA shifts:** +0.37 / +0.60 / +0.64 eV.
- **Yields:** Jochims+96, and Wenzel+20 for large cations.
- **Electron capture:** BT94 at ×1, capped at the classical limit.
- **Lab CH-stretch scale:** 10$^{-0.29}$.
- **Field:** Kurucz 40 000 K, unattenuated, the same at all depths. G$_0$
  definitions per card entry.
- **Intact threshold:** 0.5.
- **Recombination heating:** on.
- **Not included:** extinction; PANHs, clusters and dehydrogenated states as
  emitters.

### 4.3 What the fit can and cannot tell us

- **Robust:**
  - the charge share of the emission, because the 6–9 µm against 11.2 µm
    contrast is strong;
  - the existence and rough size of the γ preference;
  - that one band shape fits all five templates.
- **Not robust:**
  - individual species: NNLS picks a sparse corner of a degenerate set;
  - ⟨N$_{\rm C}$⟩, which depends on 3.3 µm, the continuum and the band
    widths;
  - Δχ² ranges, which use the fit's own residual scale on structured
    residuals.
- **Not constrained at all:** G$_0$, n$_e$ and T separately; anything that
  depends on depth.

### 4.4 The 3.4 µm side-group fit (separate)

This fit runs after the main one, on 3.1–3.65 µm only:
- Free: x$_{\rm CH_2}$ and x$_{\rm CH_3}$ (≥ 0), a free aromatic scale and a
  linear continuum.
- Fixed: the profiles (CosmicPAH-IRDB lab laws on the cascade) and the main
  fit's population and conditions.
- Result: 2.3–4.5 % aliphatic carbon.
- Not fed back into the 5–15 µm fit or the charge balance.

## 5. What could be done better, to learn more physics

Ranked by information gained per effort.

1. **Bracket the size result (cheap).** Rerun the spectral fit:
   - without the 3 µm segment;
   - with the knot spacing at 1.0 and 2.5 µm;
   - with the band widths held at one template's values.

   If ⟨N$_{\rm C}$⟩ then drops toward 60–70 and gains a gradient, our size
   offset from M26 is the 3.3 µm treatment, not the population.
2. **A calibrated error model.** Use the template error columns plus a
   model-discrepancy floor fitted to the residual spectrum, and a block
   bootstrap over residual segments. Without this no range we quote means
   anything.
3. **Capture as a nuisance parameter.** What the spectra measure is roughly
   γ × (capture scale). Scan the BT94 scale (0.3–3, the lab trend) jointly
   with the cells, and report the degeneracy instead of one γ.
4. **Fewer population degrees of freedom.**
   - Replace 1424 free weights by a parametric distribution: a size
     distribution in N$_{\rm C}$ (two or three parameters) times a
     compactness class, or a small set of fixed sub-populations.
   - Then ⟨N$_{\rm C}$⟩ and its spread are fitted quantities with errors,
     not by-products of a sparse NNLS.
   - Validate by recovering mock spectra built at known conditions.
5. **Physical band shapes.**
   - Replace the 5 shape parameters with the cascade's emission-temperature
     distribution through the lab laws (CosmicPAH-IRDB, already validated on
     hot pyrene) or Mai+25 MLMD laws.
   - The widths then become a prediction, and their change across the Bar
     becomes a thermometer of the PAH excitation, set by the photon-energy
     distribution, not G$_0$.
   - Any residual shift that remains (the +18 cm$^{-1}$ CC blue shift) points
     to carriers, e.g. PANHs.
6. **A depth-dependent field shape.** Attenuate the 40 kK field with the
   extinction curve (R$_V$ = 5.5, as Zannese+26) at A$_V$(x). Deeper in, the
   field is softer, so there is less ionisation per G$_0$ and cooler
   emission. Today every template sees the surface spectrum.
7. **Hydrogenation states as emitters.**
   - Put dehydrogenated and side-group states in the main fit, with
     fractions from the H-loss chain, instead of the intact mask and the
     separate 3.4 µm fit.
   - This gives a hydrogenation profile to compare with Elyajouri's E$_g$
     (more hydrogenated deeper in) and Chown's 3.4/3.3 doubling.
8. **Extinction.** A foreground screen plus internal extinction, constrained
   by the H$_2$ and H I line ratios the templates already contain. It
   matters for 3.3 against 11.2, and for the 9.7 µm silicate feature under
   8.6 µm.
9. **A physical continuum.** A very-small-grain continuum from pyCALIMA's own
   dust models under the same field, or one continuum shape shared by all
   positions with a free scale. Its job would be to break the
   continuum–wing degeneracy.

## 6. Data for profiles along the Bar

| product | what | where | size | licence |
|---|---|---|---|---|
| NIRSpec IFU mosaic, stitched (Peeters+24) | 132 × 259 × 9648, 0.1″ pixels, 0.97–5.27 µm, WCS-corrected | PDRs4All SEPs, Google Drive `nirspec_cubes_v0.zip` | 3.0 GB zip, 5.3 GB cube | none stated |
| MIRI MRS cubes | MAST pipeline `s3d`, `jw01288-c1004_t010_miri_ch{1-4}-{short,medium,long}`; 9 × 1 strip along the cut; not the team reduction | MAST | not checked | public archive |
| templates | Van De Putte+25 stitched NIRSpec + MRS (corrected 2025-11-05; in use); Peeters+24 NIRSpec; Chown+24 MRS v0; MRS v1/v2 with errors | CDS J/A+A/701/A111, /685/A74, /685/A75; Drive | MB | CDS / none stated |
| aperture regions and extraction | `regions/aper_T_DF_extraction.reg`; `pdrs4all/postprocess/extract_templates.py` (any DS9 boxes → spectra) | github.com/PDRs4All/PDRs4All (GPL-3.0) | — | GPL-3.0 |
| synthetic spectra (Berné+22, PASP 134, 054301) | emergent spectra of 5 regions; Meudon + CLOUDY gas, **PAHTAT PAHs**, DustEM/THEMIS + SOC continuum; no physical profiles | pdrs4all.org | 0.9 MB | none stated |
| simulated NIRSpec cube (Canin+22) | G140H/F100LP only (0.97–1.89 µm): no PAH bands | Zenodo 5776707 | 110 MB | CC-BY-4.0 |
| 3.3 µm ML templates (Schroetter+24) | AIB-irradiated and AIB-shielded spectra | Zenodo 10776844 | 30 kB | CC-BY-4.0 |
| PAHFIT PDR pack | feature table, YAML | CDS A111 `tableb1`; drvdputt/pahfit `pdrpack` branch | — | — |

Not released anywhere:
- cut spectra;
- PAHFIT band and line maps (the SEP1 page has no link);
- the team's MRS cubes;
- M26's per-pixel maps.

Profiles have to be extracted by us. That is straightforward with the
PDRs4All extraction script and a row of boxes along the cut. The MRS strip
and the NIRSpec mosaic overlap along the same cut. At 414 pc,
1″ ≈ 0.002 pc, so the ~25″ from the IF to DF3 gives 25–50 boxes.

## 7. Proposal: profiles along the Bar

### 7.1 Observables

Extract spectra in boxes perpendicular to the Bar along the NIRSpec cut and
the MRS strip, with the template boxes as a check. The check is to reproduce
the CDS templates to the stitching accuracy. The steps:
- NIRSpec: memory-map the stitched cube.
- MRS: the MAST cubes, stitched additively as in v2.
- Remove lines and mask artefacts as in `spectral_fit`.

### 7.2 Conditions along the cut, as inputs

Following the rule that conditions are inputs, not predictions, these come
from a sourced **profile card**: two bracketing families, each with its
source.

A third source is planned: the self-consistent PDR models of the calima-pdr
code (`calima-pdr/docs/design.md`), whose profiles enter the card as a
labelled model source. The PAH–gas coupling it needs is specified in
{doc}`pahchem-gas-coupling`.

- **Empirical family.**
  - G$_0$(IF) from O I 1.317 µm (Peeters+24).
  - G$_0$(x) attenuated by A$_V$(x) from n$_{\rm H}$(x): the Habart+24 AIB
    decay, or Elyajouri's n(z) law.
  - T from the H$_2$ T$_1^*$ values at the templates.
  - n$_e$ = x$_e$ n$_{\rm H}$, with C$^+$ + S$^+$ in the molecular part and
    the recombination-line n$_e$ in the atomic part.
- **Isobaric family.** P = 5 × 10$^7$ and 2 × 10$^8$ K cm$^{-3}$, the spread
  in the literature, from Meudon runs. Meudon profiles are not released.
  Either we run the public Meudon code or ISMDB holds them (to be checked).
- The field shape at each depth is the attenuated 40 kK field (item 6 of
  section 5).

### 7.3 Fits, from most free to most constrained

- **P0, their method on our extraction.** Run the existing M26 pipeline
  (`check_maragkoudakis2026.py`) on the extracted spectra. This reproduces
  their charge and size trends and checks our extraction against their
  results.
- **P1, pahchem per position.** Run today's fit at each position. Scan
  conditions inside that position's profile-card range. Fix the band shape,
  or fit it physically (item 5).
  - Gives profiles of: charge shares, ⟨N$_{\rm C}$⟩ and the size
    distribution, γ preferred against γ stated, intact fraction, CH$_2$ and
    CH$_3$ fractions, continuum share, residuals.
  - Cost: the full template refit takes ~1 min on 14 cores; with the shape fixed, one position is a scan of NNLS fits, well under that (not yet timed).
- **P2, one parent population.**
  - One set of abundances w$_s$ shared by every position. Per position, only
    a column-density scale and the continuum are free.
  - Charge, H loss and excitation follow from the profile card.
  - The question is whether one population, processed only by the local
    physics, explains the whole cut, and where along the cut it fails.
  - The residual against depth then says what processing is missing:
    destruction of the smallest PAHs, hydrogenation, cluster evaporation.
  - The charge-free methods cannot ask this question. It is also the
    quantitative version of Chown's puzzle: a nearly constant 6.2/11.2
    against a 40× change in ionising flux.
- **P3, forward line-of-sight model (later).**
  - Emissivities per depth layer, integrated along the line of sight for an
    edge-on geometry with inclination, compared directly with the profiles,
    as Elyajouri+24 do for THEMIS.
  - This is also where the HII template's face-on background PDR belongs.

### 7.4 Comparisons at each step

| ours | against |
|---|---|
| charge-share and γ profile (P1, P2) | M26 (via P0 on the same boxes); Knight+22 γ; Khan+26 6.2/11.2 |
| ⟨N$_{\rm C}$⟩, size distribution | M26 size bins; Knight+22 3.3/11.2 sizes; Chown+24 3.3/11.2; Khan+26 6.2/8.6, 7.7/8.6 |
| hydrogenation, aliphatic fraction | Elyajouri+24 E$_g$; Peeters+24 3.4/3.29 (Table 3); Chown+24 3.4/3.3 |
| band widths and shifts (excitation) | Chown+24 FWHM trends; Van De Putte+25 3.3 µm blue/red fractions |
| two-component structure | Schroetter+24, Amiot+26 AIB-irradiated/AIB-shielded maps; our cation-large against neutral-small split |
| model spectra at the five regions | Berné+22 synthetic spectra (PAHTAT), model against model |
| synthetic filter profiles | Habart+24 and Elyajouri+24 imaging cuts, through the PDRs4All `bandpasses.py` |

## 8. Decisions needed

1. **Order.** Section 5 item 1 (cheap: does the size offset come from the
   3.3 µm treatment?) before any profile work, or straight to extraction?
2. **Data.** Download the 3 GB NIRSpec zip (no licence stated: fetched, never
   bundled, as a manual registry entry) and the MAST MRS cubes?
3. **Conditions profiles.** Empirical family only to start, or also run
   Meudon for the isobaric family?
4. **First fit level.** P1 (per position, closest to the papers) or P2 (one
   population, the new test)?
