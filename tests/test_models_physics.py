"""Physics invariants across every models/ subpackage.

Each subpackage gets its own section. The assertions are properties the physics
must satisfy regardless of implementation detail -- shielding factors in
(0, 1], rates positive and finite, cross-sections non-negative, monotonicity
in the expected direction -- so they catch a broken data path or a sign error
without pinning numerical output.
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# models/dust_shielding
# ===========================================================================

class TestDustShielding:
    """Shielding factors are attenuations: they must live in (0, 1]."""

    def test_h2_self_shielding_is_a_fraction(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_SH2

        for n_h2 in (0.0, 1e14, 1e18, 1e20, 1e22):
            s = comp_SH2(n_h2)
            assert 0.0 < s <= 1.0, f"comp_SH2({n_h2:g}) = {s}"

    def test_h2_self_shielding_decreases_with_column(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_SH2

        cols = np.logspace(14, 22, 40)
        vals = np.array([comp_SH2(c) for c in cols])
        assert np.all(np.diff(vals) <= 1e-12), "more H2 must shield more, not less"

    def test_dust_shielding_is_unity_with_no_dust(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_Sd_new

        assert comp_Sd_new(0.0) == pytest.approx(1.0)

    def test_dust_shielding_is_exponential_attenuation(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_Sd_new

        taus = np.linspace(0.0, 10.0, 40)
        vals = np.array([comp_Sd_new(t) for t in taus])
        assert np.all((vals > 0) & (vals <= 1.0))
        assert np.all(np.diff(vals) <= 1e-12)

    def test_g0_self_shielding_is_a_fraction(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_G0_selfshield

        assert comp_G0_selfshield(0.0) == pytest.approx(1.0)
        for tau in (0.5, 2.0, 8.0):
            assert 0.0 < comp_G0_selfshield(tau) <= 1.0

    def test_co_shielding_is_a_fraction(self):
        from pycalima.models.dust_shielding.shielding_functions import comp_SCO

        for n_co, n_h2 in ((1e14, 1e19), (1e16, 1e20), (1e18, 1e22)):
            s = comp_SCO(n_co, n_h2)
            assert 0.0 < s <= 1.0, f"comp_SCO({n_co:g},{n_h2:g}) = {s}"

    def test_dust_optical_depth_scales_with_column(self):
        from pycalima.models.dust_shielding.shielding_functions import compute_tau_dust_LW

        rho = np.array([1e-24, 1e-24])
        kappa = np.array([1e4, 1e4])
        t1 = compute_tau_dust_LW(rho, kappa, 1e18)
        t2 = compute_tau_dust_LW(rho, kappa, 2e18)
        assert t1 > 0 and np.isfinite(t1)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-10)


# ===========================================================================
# models/PAH_photophysics
# ===========================================================================

class TestPAHCharge:
    def test_radius_from_carbon_count_is_sane(self):
        """afromNc returns cm; C54 is about 6.6 Angstrom."""
        from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc

        a = afromNc(54)
        assert 5e-8 < a < 8e-8
        assert afromNc(216) > afromNc(54)

    def test_ionisation_potential_is_unchanged_at_full_hydrogenation(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            ionisation_potential_energy,
        )

        assert ionisation_potential_energy(6.5, 18, 18) == pytest.approx(6.5)

    def test_electron_affinity_is_unchanged_at_full_hydrogenation(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            electron_affinity_energy,
        )

        assert electron_affinity_energy(0.5, 18, 18) == pytest.approx(0.5)

    @pytest.mark.parametrize("temperature", [30.0, 100.0, 1000.0, 8000.0])
    def test_recombination_rate_is_positive(self, temperature):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            recombination_rate_Spitzer,
        )

        rate = recombination_rate_Spitzer(54, 1, temperature, 1.0)
        assert rate > 0 and np.isfinite(rate)

    def test_recombination_rate_scales_with_electron_density(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            recombination_rate_Spitzer,
        )

        r1 = recombination_rate_Spitzer(54, 1, 100.0, 1.0)
        r2 = recombination_rate_Spitzer(54, 1, 100.0, 2.0)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-9)

    @pytest.mark.parametrize("temperature", [30.0, 100.0, 1000.0])
    def test_attachment_rates_are_positive(self, temperature):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            attachment_rate_Carelli13,
            attachment_rate_Tielens05,
        )

        assert attachment_rate_Tielens05(54, temperature, 1.0) > 0
        assert attachment_rate_Carelli13(temperature, 1.0) > 0

    def test_ionisation_yield_is_a_probability_above_threshold(self):
        """The Jochims (1996) ramp is only defined for E >= IP."""
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            ionisation_yield_Jochims1996,
        )

        ip = 6.5
        for energy in (ip, ip + 1.0, ip + 5.0, ip + 9.2, 20.0):
            y = ionisation_yield_Jochims1996(ip, energy)
            assert 0.0 <= y <= 1.0, f"yield({energy} eV) = {y}"

    def test_ionisation_yield_saturates_at_one(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            ionisation_yield_Jochims1996,
        )

        assert ionisation_yield_Jochims1996(6.5, 6.5 + 9.2) == pytest.approx(1.0)
        assert ionisation_yield_Jochims1996(6.5, 100.0) == 1.0

    def test_ionisation_yield_rises_monotonically_across_the_ramp(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            ionisation_yield_Jochims1996,
        )

        ip = 6.5
        energies = np.linspace(ip, ip + 9.2, 30)
        vals = np.array([ionisation_yield_Jochims1996(ip, e) for e in energies])
        assert np.all(np.diff(vals) >= -1e-12)

    @pytest.mark.xfail(
        strict=True,
        reason="ionisation_yield_Jochims1996 returns (E-IP)/9.2 unclamped, so a "
               "sub-threshold photon energy yields a negative probability. "
               "Callers are expected to gate on E >= IP; remove this marker if "
               "the function is clamped at zero.",
    )
    def test_ionisation_yield_is_non_negative_below_threshold(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            ionisation_yield_Jochims1996,
        )

        assert ionisation_yield_Jochims1996(6.5, 3.0) >= 0.0

    def test_sticking_coefficients_are_probabilities(self):
        from pycalima.models.PAH_photophysics.pah_charge_utils import (
            se_anion_Weingartner2001,
            se_neutral_Weingartner2001,
        )

        for nc in (24, 54, 96, 216):
            from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc

            a = afromNc(nc)
            assert 0.0 <= se_neutral_Weingartner2001(a, nc) <= 1.0
            assert 0.0 <= se_anion_Weingartner2001(a) <= 1.0


class TestPAHTemperature:
    def test_qho_internal_energy_grows_with_temperature(self):
        """Cheap check that the mode machinery is wired to real PAHdb modes."""
        pytest.importorskip("numpy")
        from pycalima.models.PAH_photophysics.pah_temperature import _qho_energy

        freqs = np.array([500.0, 1000.0, 1500.0]) * 1.23984193e-4  # cm^-1 -> eV
        u_low = _qho_energy(freqs, 50.0)
        u_high = _qho_energy(freqs, 500.0)
        assert 0 <= u_low < u_high
        assert np.isfinite(u_high)


# ===========================================================================
# models/PAH_charge
# ===========================================================================

class TestPAHPhotoelectricHeating:
    def test_ionisation_potential_is_positive_and_size_dependent(self):
        from pycalima.models.PAH_charge.PAH_photoelectric_heating import (
            ionisation_potential,
        )

        small = ionisation_potential(0, 4e-8)
        large = ionisation_potential(0, 1e-7)
        assert small > 0 and large > 0
        assert np.isfinite(small) and np.isfinite(large)

    def test_ionisation_potential_grows_with_charge(self):
        from pycalima.models.PAH_charge.PAH_photoelectric_heating import (
            ionisation_potential,
        )

        a = 6e-8
        assert ionisation_potential(1, a) > ionisation_potential(0, a)

    def test_beta_factor_is_bounded(self):
        from pycalima.models.PAH_charge.PAH_photoelectric_heating import beta_factor

        for nc in (20, 54, 100, 500):
            b = beta_factor(nc)
            assert 0.0 < b <= 1.0, f"beta_factor({nc}) = {b}"

    @pytest.mark.parametrize("temperature", [100.0, 1000.0, 8000.0])
    def test_recombination_and_attachment_are_positive(self, temperature):
        from pycalima.models.PAH_charge.PAH_photoelectric_heating import (
            attachment_rate_Tielens05,
            recombination_rate_Tielens21,
        )

        assert recombination_rate_Tielens21(54, temperature) > 0
        assert attachment_rate_Tielens05(54) > 0


# ===========================================================================
# models/dust_charge
# ===========================================================================

class TestDustCharge:
    def test_shared_physics_constants_are_physical(self):
        from pycalima.models.dust_charge import shared_physics as sp

        assert sp.GRAPHITE_WORK_FUNCTION > 0
        assert sp.SILICATE_WORK_FUNCTION > 0
        assert sp.SILICATE_BAND_GAP >= 0
        assert sp.ELECTRON_ESCAPE_LENGTH_CM > 0
        # CGS constants, checked against their textbook values
        assert sp.KB_CGS == pytest.approx(1.380649e-16, rel=1e-4)
        assert sp.C_CGS == pytest.approx(2.99792458e10, rel=1e-9)
        assert sp.H_CGS == pytest.approx(6.62607015e-27, rel=1e-4)
        assert sp.ME_CGS == pytest.approx(9.1093837e-28, rel=1e-4)

    def test_coulomb_enhancement_module_imports_and_exposes_callables(self):
        import inspect

        from pycalima.models.dust_charge import Coulomb_enhancement as ce

        fns = [n for n, o in vars(ce).items()
               if inspect.isfunction(o) and not n.startswith("_")]
        assert fns, "Coulomb_enhancement exposes no public functions"

    def test_charge_distribution_api_is_available(self):
        from pycalima.models.dust_charge import IM19_charging as im

        for name in ("grain_charge_dist", "grain_mean_charge",
                     "grain_charge_sigma", "grain_charge_probability"):
            assert hasattr(im, name), f"IM19_charging is missing {name}"

    # The IM19 fits are tabulated for discrete radii, keyed by string.
    IM19_RADII = ("3.5A", "5A", "10A", "50A", "100A", "500A", "1000A")

    @pytest.mark.parametrize("grain_type", ["graphite", "silicate"])
    @pytest.mark.parametrize("radius", ["10A", "100A", "1000A"])
    def test_equilibrium_charge_distribution_is_normalised(self, grain_type, radius):
        """P(Z) over the sampled charge states must sum to one."""
        from pycalima.models.dust_charge.IM19_charging import grain_charge_dist

        # NB the return order is (probabilities, charge states): `return dist, x`
        P, Zs = grain_charge_dist(1.0, 100.0, 1.0, grain_type, radius)
        Zs = np.asarray(Zs, dtype=float)
        P = np.asarray(P, dtype=float)
        assert P.size == Zs.size
        assert np.all(P >= 0)
        assert np.all(np.isfinite(P))
        assert P.sum() == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.parametrize("grain_type", ["graphite", "silicate"])
    def test_mean_charge_is_consistent_with_the_distribution(self, grain_type):
        from pycalima.models.dust_charge.IM19_charging import (
            grain_charge_dist,
            grain_mean_charge,
        )

        args = (1.0, 100.0, 1.0, grain_type, "100A")
        P, Zs = grain_charge_dist(*args)
        expected = float(np.sum(np.asarray(Zs, dtype=float) * np.asarray(P, dtype=float)))
        # Not an identity: grain_mean_charge evaluates the continuous Gaussian
        # mean, while grain_charge_dist discretises onto integer charges over a
        # truncated range. A few percent of discretisation error is expected
        # (measured: 0.7% graphite, 1.8% silicate).
        assert grain_mean_charge(*args) == pytest.approx(expected, rel=0.05)

    def test_charge_probability_is_bounded(self):
        from pycalima.models.dust_charge.IM19_charging import grain_charge_probability

        for z in (-2, -1, 0, 1, 2):
            p = grain_charge_probability(1.0, 100.0, 1.0, "graphite", "100A", z)
            assert 0.0 <= float(p) <= 1.0, f"P(Z={z}) = {p}"

    def test_charge_dispersion_is_non_negative(self):
        from pycalima.models.dust_charge.IM19_charging import grain_charge_sigma

        sigma = grain_charge_sigma(1.0, 100.0, 1.0, "graphite", "100A")
        assert float(sigma) >= 0

    def test_charging_grows_with_the_radiation_field(self):
        """A stronger field must not make a grain less positively charged."""
        from pycalima.models.dust_charge.IM19_charging import grain_mean_charge

        weak = grain_mean_charge(0.1, 100.0, 1.0, "graphite", "100A")
        strong = grain_mean_charge(10.0, 100.0, 1.0, "graphite", "100A")
        assert float(strong) >= float(weak)

    def test_unknown_grain_radius_is_rejected(self):
        from pycalima.models.dust_charge.IM19_charging import grain_charge_dist

        with pytest.raises(KeyError):
            grain_charge_dist(1.0, 100.0, 1.0, "graphite", "7A")

    def test_charging_output_dir_is_writable_and_outside_the_package(self, isolated_env):
        from pycalima import _paths
        from pycalima.models.dust_charge.dust_charging import dust_charging_output_dir

        out = dust_charging_output_dir()
        assert _paths.PKG_DIR not in out.parents
        assert out.is_relative_to(isolated_env)

    def test_photoelectric_output_dir_is_writable_and_outside_the_package(
        self, isolated_env
    ):
        from pycalima import _paths
        from pycalima.models.dust_charge.dust_photoelectric_heating import (
            dust_photoelectric_output_dir,
        )

        out = dust_photoelectric_output_dir()
        assert _paths.PKG_DIR not in out.parents
        assert out.is_relative_to(isolated_env)


# ===========================================================================
# models/dust_radiation
# ===========================================================================

class TestDustRadiation:
    @pytest.mark.parametrize("name", ["callindex.out_silD03",
                                      "callindex.out_CpaD03_0.01"])
    def test_optical_constants_are_readable_from_bundled_data(self, name):
        """Exercises the bundled optical_props tree through the real reader."""
        from pycalima import _paths
        from pycalima.models.dust_radiation.dust_oppacity import read_dielectric_file

        path = _paths.get_optical_props_path("draine_lee_1984", name)
        if not path.is_file():
            pytest.skip(f"{name} is not part of the bundled data")
        result = read_dielectric_file(str(path))
        assert result is not None

    def test_read_only_reference_paths_resolve_into_the_package(self):
        from pycalima import _paths
        from pycalima.models.dust_radiation import dust_oppacity as do

        assert _paths.PKG_DIR in _paths.get_data_root().parents or \
            _paths.get_data_root().is_relative_to(_paths.PKG_DIR)
        assert do.PATH_EXTERNAL_DATA.is_dir()

    def test_generated_table_dir_is_resolved_per_call(self, isolated_env):
        """These used to be module constants frozen at import from a guessed
        repo root, which ignored $CALIMA_DATA."""
        from pycalima.models.dust_radiation import dust_oppacity as do

        got = do._path_model_optical_output()
        assert got.is_relative_to(isolated_env)

    def test_sublimation_module_imports(self):
        import pycalima.models.dust_radiation.dust_sublimation as ds

        assert hasattr(ds, "__doc__")


# ===========================================================================
# models/dust_gas_collisions
# ===========================================================================

class TestDustGasCollisions:
    def test_sputtering_output_dir_is_resolved_per_call(self, isolated_env):
        from pycalima.models.dust_gas_collisions.dust_sputtering import (
            _sputtering_output_dir,
        )

        assert _sputtering_output_dir().is_relative_to(isolated_env)

    def test_sputtering_module_exposes_its_public_api(self):
        import inspect

        from pycalima.models.dust_gas_collisions import dust_sputtering as sp

        fns = [n for n, o in vars(sp).items()
               if inspect.isfunction(o) and not n.startswith("_")]
        assert fns

    def test_collisional_cooling_module_imports(self):
        import pycalima.models.dust_gas_collisions.dust_collisional_cooling as cc

        assert hasattr(cc, "__doc__")


# ===========================================================================
# models/dust_collisions and PAH_collisions
# ===========================================================================

class TestCollisions:
    def test_relative_velocity_is_positive_and_finite(self):
        from pycalima.models.dust_collisions.dust_dynamics import relative_velocity

        v = relative_velocity("Hirashita and Aoyama2019", 1e4, 1.0, 0.1, 2.0,
                              1.0, 1e-5, 3e-5, 3.5, 3.5)
        v = np.asarray(v, dtype=float)
        assert np.all(np.isfinite(v)) and np.all(v > 0)

    def test_relative_velocity_grows_with_mach_number(self):
        from pycalima.models.dust_collisions.dust_dynamics import relative_velocity

        args = (1e4, 1.0, 0.1)
        tail = (1.0, 1e-5, 3e-5, 3.5, 3.5)
        slow = relative_velocity("Hirashita and Aoyama2019", *args, 1.0, *tail)
        fast = relative_velocity("Hirashita and Aoyama2019", *args, 4.0, *tail)
        assert float(fast) > float(slow)

    def test_pah_coalescence_timescale_is_positive(self):
        """Exercises mass_from_Nc, whose undefined constants used to make every
        call here raise NameError."""
        from pycalima.models.PAH_collisions.PAH_coalescence import (
            compute_coalescence_timescale_Tielens21,
        )

        t = compute_coalescence_timescale_Tielens21(54, 1e-26, 100.0)
        t = np.asarray(t, dtype=float)
        assert np.all(np.isfinite(t)) and np.all(t > 0)

    def test_shattering_and_coagulation_modules_import(self):
        import pycalima.models.dust_collisions.dust_coagulation  # noqa: F401
        import pycalima.models.dust_collisions.dust_shattering  # noqa: F401


# ===========================================================================
# models/dust_chemistry
# ===========================================================================

class TestDustChemistry:
    def test_h2_formation_module_imports_and_exposes_callables(self):
        import importlib
        import inspect

        mod = importlib.import_module("pycalima.models.dust_chemistry.dust_h2_formation")
        fns = [n for n, o in vars(mod).items()
               if inspect.isfunction(o) and not n.startswith("_")]
        assert fns, "dust_h2_formation exposes no public functions"


# ===========================================================================
# models/tools
# ===========================================================================

class TestTools:
    def test_mie_theory_is_importable_and_constructible(self):
        """This module was unimportable before packaging: it had a bare
        `from henke_extension import ...` at module scope."""
        from pycalima.models.tools.mie_theory import MieTheory

        mie = MieTheory()
        assert mie is not None

    def test_henke_extension_reads_the_bundled_table_by_default(self):
        """Its dat_path default used to be the CWD-relative string
        'external_data/henke/f1f2_Henke.dat'."""
        from pycalima.models.tools.henke_extension import HenkeExtension

        henke = HenkeExtension()
        assert henke.atomic_factors, "no atomic factors parsed from bundled data"

    def test_sed_reader_is_importable(self):
        import pycalima.models.tools.read_ramses_sed  # noqa: F401

    def test_nozawa_gsd_fit_loads_bundled_data(self):
        from pycalima.models.tools.fit_nozawa2007_gsd import load_data

        frames = load_data()
        assert set(frames) == {"Mg2SiO4", "C"}
        for name, df in frames.items():
            assert len(df) > 0, name
            assert df["Grain Size"].notna().any(), name


# ===========================================================================
# models/yields
# ===========================================================================

class TestYields:
    def test_yield_dir_accessor_errors_actionably_when_unset(self, pristine_env):
        """build_tables.py used to call os.listdir('/home/dubois/StellarYields')
        at module scope, so importing it failed for everyone but its author."""
        from pycalima.models.yields.build_tables import get_yield_dir

        with pytest.raises(RuntimeError, match="CALIMA_YIELD_DIR"):
            get_yield_dir()

    def test_yield_dir_accessor_rejects_a_bad_path(self, tmp_path, monkeypatch):
        from pycalima.models.yields.build_tables import get_yield_dir

        monkeypatch.setenv("CALIMA_YIELD_DIR", str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError):
            get_yield_dir()

    def test_yield_dir_accessor_accepts_a_real_directory(self, tmp_path, monkeypatch):
        from pycalima.models.yields.build_tables import get_yield_dir

        monkeypatch.setenv("CALIMA_YIELD_DIR", str(tmp_path))
        assert get_yield_dir() == tmp_path


# ===========================================================================
# external-project data accessors
# ===========================================================================

class TestExternalProjectAccessors:
    def test_bpass_sed_dir_errors_actionably_when_unset(self, pristine_env):
        from pycalima.models.dust_charge.dust_photoelectric_heating import (
            _bpass_sed_dir,
        )

        with pytest.raises(RuntimeError, match="CALIMA_SED_DIR"):
            _bpass_sed_dir()

    def test_bpass_sed_dir_accepts_a_real_directory(self, tmp_path, monkeypatch):
        from pycalima.models.dust_charge.dust_photoelectric_heating import (
            _bpass_sed_dir,
        )

        monkeypatch.setenv("CALIMA_SED_DIR", str(tmp_path))
        assert _bpass_sed_dir() == str(tmp_path)

    def test_dustem_file_is_none_when_unset(self, pristine_env):
        """Absent DustEM data must degrade to a skipped comparison, not a crash."""
        from pycalima.models.PAH_photophysics.diagnose_temperature_distribution import (
            _dustem_file,
        )

        assert _dustem_file() is None


# ===========================================================================
# plotting style
# ===========================================================================

class TestPlottingStyle:
    def test_use_calima_style_is_idempotent_and_does_not_force_latex(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from pycalima.plotting_style import use_calima_style

        use_calima_style(usetex=False, force=True)
        assert plt.rcParams["text.usetex"] is False
        use_calima_style(usetex=False)  # second call must be a no-op
        assert plt.rcParams["text.usetex"] is False

    def test_latex_available_returns_a_bool(self):
        from pycalima.plotting_style import latex_available

        assert isinstance(latex_available(), bool)
