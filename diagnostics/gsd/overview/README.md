# GSD status report: sources

The sources of the GSD status report (`gsd_overview.tex`, about 50 pages), kept here so that the report and the
to-do list (`../STATUS.md`) are versioned. The figures, the input data and the PDF are generated and live in the
git-ignored build directory `results/gsd_overview/`, which holds symlinks to these files.

| here | what |
|---|---|
| `gsd_overview.tex` | the report |
| `make_figures.py` | the figures it draws itself (PAH paths, scores, caps, tables, the PL2M scheme); writes `results/gsd_overview/figs/` |
| `make_infographic.py` | the carrier-network figure (`figs/carbon_network.html`, then a PNG) |
| `extract_paths.py` | `results/gsd_overview/data/paths.npz` from the PAH-bin cap runs (run where the run pickles are) |
| `lit/` | the literature notes behind the report's claims (each with what its sources actually support) |
| `variants/REPORT.md`, `variants/production_check.py` | the closure-variant study of 17-18 November (a dated study: its bench, `../bench_pl2m_variants.py`, predates the `GSDMethod` API) |
| `tables/` | the numbers behind the report as of 2026-11-22: the text outputs of the diagnostics (`figs/*.txt`, `variants/*.txt`, `grain_hydrogen_obs.txt`) |

## Building the report

The other figures come from the scripts in `diagnostics/gsd/` and `diagnostics/dust_charge/` (each writes to
`results/`; the report's tables name the script behind each number). From the repository root, with
`PYTHONPATH=src:diagnostics/gsd` and `CALIMA_DUSTEM_DIR` set:

```bash
python diagnostics/gsd/check_bin_convergence.py run --case all --schemes all --workers 2   # A0; ~13 min per reference
python diagnostics/gsd/check_bin_convergence.py report      # results/plots/gsd_bin_convergence.png
python diagnostics/gsd/check_mckinnon18_tests.py --out-dir results/gsd_overview/figs
python diagnostics/gsd/check_inbin_rates.py                 # figs/inbin_rates.pdf
python diagnostics/gsd/check_rates_convergence.py           # figs/rates_convergence.pdf (reads the A0 runs)
python diagnostics/gsd/check_rt_group_optics.py             # figs/rt_group_optics.txt (reads the A0 runs)
python diagnostics/gsd/check_pah_grain_seam.py --max-mol 60 # needs $CALIMA_PAHDB_DIR
python diagnostics/dust_charge/check_cm20_dustem.py --material ac
python diagnostics/gsd/overview/make_figures.py
cd results/gsd_overview && latexmk -pdf gsd_overview.tex
```

The A0 runs go to `$CALIMA_RESULTS/gsd_bin_convergence/`; the report reads them from `results/binconv_v4/` (PL2M,
one-moment schemes, references) and `results/gsd_overview/variants/a0_prod/` (EA, EAF).
