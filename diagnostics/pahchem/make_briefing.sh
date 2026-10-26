#!/usr/bin/env bash
# Build the briefing figure set described in docs/physics/pahchem-briefing.md.
#
# Writes <plots dir>/pahchem/briefing/ (A: overview, B: spectral fits, C: reproduced benchmarks, D: open problems,
# E: PAH-gas coupling, the V1-V16 validations of docs/physics/pahchem-gas-coupling.md).
# Run from the worktree root, with the PAHdb data and the main checkout's PAH mode files:
#   CALIMA_PAHDB_DIR=<main>/reference_data diagnostics/pahchem/make_briefing.sh <main>/model_data/PAH_states <West+18 ESI .xlsx>
# Steps 1-2 take ~30 min (the benchmark runners), step 2b ~2.5 h (the gas coupling: V4 and the phases ~40 min and
# ~1 h); SKIP_RUNS=1 reuses their figures already in the plots dir. BERNE_DIR (a checkout of Berné+22's
# photoelectric-heating code) adds their step-by-step comparison; WORKERS (default 4) the processes of step 2b.
# The figures of step 3 reuse cached results.
set -euo pipefail
STATES=${1:?usage: make_briefing.sh <PAH_states dir> <West+18 ESI xlsx>}
ESI=${2:?usage: make_briefing.sh <PAH_states dir> <West+18 ESI xlsx>}
PY=${PYTHON:-python}
export PYTHONPATH=src:diagnostics/pahchem${PYTHONPATH:+:$PYTHONPATH}
PLOTS=$($PY -c "from pycalima import _paths; print(_paths.get_plots_dir('pahchem'))")
OUT=$PLOTS/briefing
mkdir -p "$OUT"

# 1. benchmark figures with the current code (house style)
if [ -z "${SKIP_RUNS:-}" ]; then
$PY diagnostics/pahchem/plot_benchmarks.py west18 jochims94 montillaud13_fig3 andrews16_fig5 andrews16_tableB \
    montillaud13_fig10 andrews16_ngc7023 allain96_fig10 lepage01_fig10 murga2020_fig4 murga2022_charge \
    murga2022_model_b emission_cascade maragkoudakis2020_fig2 rf_borja2026 sensitivity \
    --states-dir "$STATES" --west-esi "$ESI" --out "$PLOTS"
# 2. cosmic-ray collisions (check_cr_impact.py writes cr_impact_*.png to the plots dir)
$PY diagnostics/pahchem/check_cr_impact.py --states-dir "$STATES"
# 2b. the PAH-gas coupling validations (each writes its pahchem_*.png to the plots dir), one at a time
W=${WORKERS:-4}
for s in check_pah_ion_rates check_wolfire08_cplus check_attachment_v6 check_thi20_recombination check_pah_h2_formation; do
    $PY diagnostics/pahchem/$s.py
done
$PY diagnostics/pahchem/check_diffuse_ionisation.py --workers "$W"
$PY diagnostics/pahchem/check_wd01b_alpha.py --workers "$W"
$PY diagnostics/pahchem/check_pe_efficiency.py --workers "$W"
$PY diagnostics/pahchem/check_tielens21_fz.py --workers "$W"
$PY diagnostics/pahchem/check_charge_phases.py --workers "$W"
$PY diagnostics/pahchem/check_dense_cloud_charge.py
if [ -n "${BERNE_DIR:-}" ]; then $PY diagnostics/pahchem/check_berne22_pe_heating.py --berne-dir "$BERNE_DIR"; fi
fi
# 3. the spectral fits (cached fit_pdrs4all_spectra.py --fit-shape result) and the extra figures
RESULTS=$($PY -c "from pycalima import _paths; print(_paths.get_results_dir())")
# FIT_CACHE: a fit_pdrs4all_spectra.py --fit-shape result; default the newest without aliphatics
FIT_CACHE=${FIT_CACHE:-$($PY -c "
import pickle, sys; from pathlib import Path
c = sorted(Path(sys.argv[1]).glob('pahchem_spectral_shape_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
print(next(p for p in c if pickle.load(open(p, 'rb'))['args'].get('aliphatic', 'none') == 'none'))" "$RESULTS")}
echo "spectral fits from $FIT_CACHE"
$PY diagnostics/pahchem/plot_briefing_fits.py --cache "$FIT_CACHE" --out "$OUT"
$PY diagnostics/pahchem/plot_briefing_extras.py --out "$OUT"

# 4. presentation order
mv -f "$OUT/bench_00_scorecard.png" "$OUT/A1_benchmark_scorecard.png"
mv -f "$OUT/fit_00_how_a_model_spectrum_is_built.png" "$OUT/B1_how_a_model_spectrum_is_built.png"
i=2
for t in HII Atomic DF1 DF2 DF3; do mv -f "$OUT/fit_01_$t.png" "$OUT/B${i}_fit_$t.png"; i=$((i + 1)); done
mv -f "$OUT/fit_02_summary.png" "$OUT/B7_fit_summary.png"
copy() { cp -f "$PLOTS/$1.png" "$OUT/$2.png"; }
copy west18 C01_west18_rrkm
copy montillaud13_fig3 C02_montillaud13_fig3_coronene_k
copy andrews16_fig5 C03_andrews16_fig5_rates
copy andrews16_tableB C04_andrews16_tableB_definition
copy allain96_fig10 C05_allain96_hydrogenation
copy lepage01_fig10 C06_lepage01_photodissociation
copy murga2022_charge C07_murga2022_charge
copy emission_cascade C08_emission_cascade_vs_amespahdb
copy maragkoudakis2020_fig2 C09_maragkoudakis2020_band_ratios
copy rf_borja2026 C10_recurrent_fluorescence_borja2026
copy cr_impact_deposition C11_cr_energy_deposition
copy cr_impact_lifetimes C12_cr_lifetimes
copy cr_impact_ir_spectra C13_cr_excited_ir
mv -f "$OUT/cr_02_secondary_electrons.png" "$OUT/C14_cr_secondary_electrons.png"
copy montillaud13_fig10 D01_montillaud13_fig10_dehydrogenation
copy andrews16_ngc7023 D02_andrews16_ngc7023
copy murga2020_fig4 D03_murga2020_destruction
copy murga2022_model_b D04_murga2022_superhydrogenation
copy jochims94 D05_west18_vs_jochims94_threshold
mv -f "$OUT/cr_01_eii_vs_beb.png" "$OUT/D06_gate_G2_electron_impact_ionisation.png"
copy sensitivity D07_sensitivity_ranking
mv -f "$OUT/gc_00_scorecard.png" "$OUT/E00_gas_coupling_scorecard.png"
copyif() { if [ -f "$PLOTS/$1.png" ]; then cp -f "$PLOTS/$1.png" "$OUT/$2.png"; else echo "missing $1.png"; fi; }
copyif pahchem_ion_rates E01_ion_electron_rates_V1_V5_V7_V8
copyif pahchem_v2_w08_cplus E02_wolfire08_cplus_V2
copyif pahchem_v3_w08_cicii E03_wolfire08_cicii_V3
copyif pahchem_v13_lepp88 E04_lepp88_metal_ions_V13
copyif pahchem_wd01b_alpha E05_wd01b_ion_recombination_V4
copyif pahchem_v4_wd01b_xe E06_wd01b_electron_fraction_V4
copyif pahchem_thi20_recombination E07_thi20_recombination_V10
copyif pahchem_attachment_v6 E08_electron_attachment_V6
copyif pahchem_pe_efficiency E09_pe_efficiency_V9
copyif pahchem_berne22_steps E10_berne22_step_by_step_V9
copyif pahchem_wd01_heating_cooling E11_heating_and_recombination_cooling_V16
copyif pahchem_wd01_mean_charge E12_mean_charge_vs_psi_V16
copyif pahchem_v14_tielens21_fz E13_tielens21_charge_V14
copyif pahchem_charge_phases E14_charge_across_phases
copyif pahchem_charge_phases_fz E15_charge_fractions_by_phase
copyif pahchem_dense_cloud_charge_od E16_dense_cloud_charge_V15
copyif pahchem_pah_h2_formation E17_h2_formation_V11
if [ -f "$PLOTS/archive_2026-10-08/aliphatic_34um_survey.png" ]; then
    # not regenerable without the Van De Putte+25 templates (fit_34um_profiles.py); kept in its original style
    cp -f "$PLOTS/archive_2026-10-08/aliphatic_34um_survey.png" "$OUT/D08_aliphatic_34um_survey.png"
fi
ls "$OUT"
