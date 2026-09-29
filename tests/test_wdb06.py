"""Weingartner, Draine & Barr (2006) photoemission model and its atomic inputs."""

from __future__ import annotations

import numpy as np
import pytest

# (element, shell, E [eV], sigma [Mb]) printed by the original phfit2.f v2
# (gfortran, single precision). Covers ph1 and ph2 branches and both sides of
# the inner-shell edges.
PHFIT2_REFERENCE = [
    ("C", "1s", 300.0, 8.806428e-01),
    ("C", "2s", 700.0, 4.191065e-03),
    ("C", "2p", 12.0, 1.761741e01),
    ("C", "2p", 150.0, 2.240805e-01),
    ("C", "2p", 300.0, 4.943237e-03),
    ("O", "1s", 2000.0, 1.746866e-02),
    ("O", "2s", 700.0, 1.188779e-02),
    ("O", "2p", 300.0, 1.348424e-01),
    ("Mg", "1s", 8000.0, 1.501333e-03),
    ("Mg", "3s", 60.0, 1.242418e-01),
    ("Si", "2p", 2000.0, 4.661793e-03),
    ("Si", "3p", 150.0, 7.277545e-02),
    ("Fe", "1s", 8000.0, 2.503198e-02),
    ("Fe", "3d", 150.0, 2.265848e00),
    ("Fe", "4s", 8.0, 5.658654e00),
    ("Fe", "4s", 300.0, 1.329219e-02),
]

# WDB06 Table 1, eV
WDB06_TABLE1 = {
    "C": {"1s": 291, "2s": 19.39, "2p": 11.26},
    "O": {"1s": 538, "2s": 28.48, "2p": 13.62},
    "Mg": {"1s": 1311, "2s": 94.0, "2p": 54.9, "3s": 7.646},
    "Si": {"1s": 1846, "2s": 156, "2p": 106, "3s": 15.17, "3p": 8.152},
    "Fe": {"1s": 7124, "2s": 857, "2p": 724, "3s": 104, "3p": 66,
           "3d": 14.7, "4s": 7.902},
}

# Dwek & Smith (1996) Table 3, "This work": mean number of Auger electrons
# per initial vacancy. Independent of the Table 4.x spectra we bundle.
DS96_TABLE3 = {
    ("C", "1s"): 0.999, ("O", "1s"): 0.996, ("Mg", "1s"): 2.035,
    ("Si", "1s"): 2.928, ("Fe", "1s"): 4.863,
    ("Mg", "2s"): 1.000, ("Si", "2s"): 1.974, ("Fe", "2s"): 4.309,
    ("Mg", "2p"): 0.997, ("Si", "2p"): 1.000, ("Fe", "2p"): 2.463,
    ("Fe", "3s"): 1.904, ("Fe", "3p"): 1.000,
}


class TestAtomicData:
    @pytest.mark.parametrize("element,shell,E,sigma_Mb", PHFIT2_REFERENCE)
    def test_cross_sections_match_phfit2(self, element, shell, E, sigma_Mb):
        from pycalima.models.dust_charge.wdb06_atomic import (
            photoionisation_cross_section,
        )

        got = float(photoionisation_cross_section(element, shell, E))
        assert got == pytest.approx(sigma_Mb * 1e-18, rel=1e-5)

    @pytest.mark.parametrize("element", sorted(WDB06_TABLE1))
    def test_thresholds_match_wdb06_table1(self, element):
        from pycalima.models.dust_charge.wdb06_atomic import (
            shell_threshold_eV,
            shells,
        )

        assert set(shells(element)) == set(WDB06_TABLE1[element])
        for shell, E in WDB06_TABLE1[element].items():
            assert shell_threshold_eV(element, shell) == pytest.approx(E)

    @pytest.mark.parametrize("element", sorted(WDB06_TABLE1))
    def test_cross_sections_vanish_below_threshold(self, element):
        from pycalima.models.dust_charge.wdb06_atomic import (
            photoionisation_cross_section,
            shell_threshold_eV,
            shells,
        )

        for shell in shells(element):
            Eth = shell_threshold_eV(element, shell)
            assert float(photoionisation_cross_section(element, shell, 0.999 * Eth)) == 0.0
            assert np.all(photoionisation_cross_section(
                element, shell, np.geomspace(Eth, 1e5, 50)) >= 0.0)

    @pytest.mark.parametrize("key,total", sorted(DS96_TABLE3.items()))
    def test_auger_spectra_sum_to_ds96_table3(self, key, total):
        from pycalima.models.dust_charge.wdb06_atomic import auger_spectrum

        E, p = auger_spectrum(*key)
        assert p.sum() == pytest.approx(total, abs=1.5e-3)
        # an Auger electron cannot carry more than the vacancy's binding energy
        from pycalima.models.dust_charge.wdb06_atomic import shell_threshold_eV
        assert np.all(E < shell_threshold_eV(*key))

    def test_outermost_shells_have_no_auger_emission(self):
        from pycalima.models.dust_charge.wdb06_atomic import auger_spectrum

        for key in (("C", "2p"), ("Mg", "3s"), ("Si", "3p"), ("Fe", "4s")):
            assert auger_spectrum(*key)[0].size == 0

    def test_secondary_parameters_are_the_ones_wdb06_adopts(self):
        from pycalima.models.dust_charge.wdb06_atomic import (
            secondary_emission_parameters,
        )

        assert secondary_emission_parameters("graphite")["eps_eV"] == 117.0
        assert secondary_emission_parameters("lunar_dust")["eps_eV"] == 155.0


MATERIALS = ["graphite", "silicate"]


class TestYields:
    @pytest.mark.parametrize("material", MATERIALS)
    def test_band_bulk_yield_is_wd01_below_20_eV(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        W = wy.MATERIALS[material]["W"]
        theta = np.linspace(0.0, 20.0 - W, 50)
        np.testing.assert_allclose(wy.band_bulk_yield(theta, material),
                                   wy.MATERIALS[material]["y0_wd01"](theta, W))

    @pytest.mark.parametrize("material", MATERIALS)
    def test_band_bulk_yield_is_continuous_at_the_blend_edges(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        W = wy.MATERIALS[material]["W"]
        for hnu in (20.0, 50.0):
            lo, hi = wy.band_bulk_yield(np.array([hnu - 1e-6, hnu + 1e-6]) - W, material)
            assert hi == pytest.approx(lo, rel=1e-4)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_band_bulk_yield_blends_geometrically(self, material):
        """log y0 is linear in log hnu across 20-50 eV (WDB06 Figs. 1-2), so at
        the geometric mean energy y0 is the geometric mean of the two yields."""
        from pycalima.models.dust_charge import wdb06_yields as wy

        m = wy.MATERIALS[material]
        W = m["W"]
        E = np.array([np.sqrt(20.0 * 50.0)])
        theta = E - W
        le = wy.electron_escape_length(theta, material)
        la = 1.0 / wy.absorption_coefficient(E, material)
        y_atom = wy._slab_escape(wy.absorption_coefficient(E, material, m["band"]), le, la)
        y_wd01 = m["y0_wd01"](theta, W)
        np.testing.assert_allclose(wy.band_bulk_yield(theta, material),
                                   np.sqrt(y_atom * y_wd01), rtol=1e-10)

    @pytest.mark.parametrize("material", MATERIALS)
    @pytest.mark.parametrize("Z", [-1, 0, 1, 5])
    def test_primary_yield_matches_wd01_below_20_eV(self, material, Z):
        """Without secondaries, WDB06 is WD01 for soft photons (Z >= -1, where
        the van Hoof E_min does not enter)."""
        from pycalima.models.dust_charge import dust_charging as dc
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import ELECTRON_ESCAPE_LENGTH_CM

        a = 5e-7
        E = np.linspace(5.0, 19.9, 120)
        la = np.full_like(E, 3e-6)
        W = wy.MATERIALS[material]["W"]
        got = wy.total_yield(E, [Z], a, material, la_wd01=la, include_secondaries=False)
        if material == "graphite":
            ref = dc.photoelectric_yield_graphite_vec(W, [Z], a, ELECTRON_ESCAPE_LENGTH_CM,
                                                      E, None, None, None, la=la)
        else:
            ref = dc.photoelectric_yield_silicate_vec(W, [Z], a, ELECTRON_ESCAPE_LENGTH_CM,
                                                      E, None, None, la=la)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-14)

    @pytest.mark.parametrize("material", MATERIALS)
    @pytest.mark.parametrize("a", [5e-8, 1e-6, 1e-5, 1e-4])
    def test_yields_are_bounded(self, material, a):
        from pycalima.models.dust_charge import wdb06_yields as wy

        from pycalima.models.dust_charge import shared_physics as sp

        E = np.geomspace(5.0, 1e4, 300)
        Zmin = int(getattr(sp, f"most_negative_allowed_charge_{material}")(a))
        Zs = np.unique(np.clip([-5, -1, 0, 1, 10, 50], Zmin, None))
        comps = wy.yield_components(E, Zs, a, material, la_wd01=np.full_like(E, 1e-5))
        for name, (Y, Em) in comps.items():
            assert np.all(np.isfinite(Y)) and np.all(Y >= 0.0), name
            assert np.all(np.isfinite(Em)) and np.all(Em >= 0.0), name
            if name == "band" or name.startswith("inner:"):
                assert np.all(Y <= 1.0), name  # at most one primary per photon
            if not name.startswith("sec:"):
                # escaping electrons cannot carry more than the photon brought
                assert np.all(Em[Y > 0] <= np.broadcast_to(E[:, None], Y.shape)[Y > 0] + 1e-9), name

    @pytest.mark.parametrize("material", MATERIALS)
    def test_band_secondaries_follow_eq18_with_the_escape_factor(self, material):
        """Y_sec/Y = (y2_sec/y2)(E_e/eps) l_e^-1/(a^-1 + 10 A^-1) (1 - e^{-a/l_e})."""
        from pycalima.models.dust_charge import wdb06_yields as wy

        W, eps = wy.MATERIALS[material]["W"], wy.MATERIALS[material]["eps"]
        E = np.array([30.0, 150.0, 800.0, 5000.0])
        for a in (1e-7, 1e-5):
            c = wy.yield_components(E, [-1], a, material, la_wd01=np.full_like(E, 1e-5))
            Y, Ys = c["band"][0][:, 0], c["sec:band"][0][:, 0]
            le = wy.electron_escape_length(E - W, material)
            # Z = -1: y2 = y2_sec = 1 and E_low = 0
            expect = (E - W) / eps / le / (1 / a + 1 / wy.L_E_LOW) * (1 - np.exp(-a / le))
            ok = Y > 0
            np.testing.assert_allclose(Ys[ok] / Y[ok], expect[ok], rtol=1e-6)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_secondaries_need_a_parent(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        E = np.geomspace(5.0, 1e4, 200)
        comps = wy.yield_components(E, [-3, 0, 4], 1e-6, material, la_wd01=np.full_like(E, 1e-5))
        for name, (Ys, _) in comps.items():
            if name.startswith("sec:"):
                parent = comps[name[4:]][0]
                assert np.all(Ys[parent == 0.0] == 0.0), name

    @pytest.mark.parametrize("material", MATERIALS)
    def test_positive_charge_suppresses_emission(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        E = np.geomspace(10.0, 1e4, 200)
        Y = wy.total_yield(E, np.arange(0, 400, 20), 1e-6, material,
                           la_wd01=np.full_like(E, 1e-5))
        assert np.all(np.diff(Y, axis=1) <= 1e-12)

    def test_vh04_min_energy(self):
        """E_min vanishes for Z >= -1, grows with |Z|, and tends to WD01 eq. 7
        for large grains and charges (both -> |Z+1| e^2/a)."""
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import min_energy_ejection_vec

        Zs = np.arange(-30, 3)
        for a in (5e-8, 1e-7, 1e-6):
            Em = wy.min_energy_vH04(Zs, a)
            assert np.all(Em[Zs >= -1] == 0.0)
            assert np.all(Em[Zs < -1] > 0.0)
            assert np.all(np.diff(Em[Zs <= -1]) < 0)  # more negative -> larger
        Z, a = -1000.0, 1e-5
        assert float(wy.min_energy_vH04(Z, a)) == pytest.approx(
            float(min_energy_ejection_vec(Z, a)), rel=0.05)


class TestGasSecondaries:
    def test_ds79_single_atom_limit(self):
        """DS79 eq. A12: for a = 1.28 A graphite at E0 = E_m = 250 eV,
        delta pi a^2 = 2.42 pi a0^2 (A12 drops O(a/lambda, a/R) terms)."""
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        a, a0 = 1.28e-8, 0.529177e-8
        delta = float(wy.gas_secondary_yield_at(250.0, 0.0, a, "graphite"))
        assert delta * a ** 2 / a0 ** 2 == pytest.approx(2.42, rel=0.05)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_no_secondaries_below_the_work_function(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        W = wy.MATERIALS[material]["W"]
        E = np.array([0.5 * W, 0.999 * W, 1.001 * W, 10 * W])
        d = wy.gas_secondary_yield_at(E, 0.0, 1e-6, material)
        assert np.all(d[:2] == 0.0) and np.all(d[2:] > 0.0)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_thermal_weights_integrate_the_flux_above_threshold(self, material):
        """sum(w) = int_max(L,W) E e^{-E/kT} / int_L E e^{-E/kT} in closed form."""
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        a, T = 1e-6, 3e4
        kT = KB_CGS * T / EV2ERG
        W = wy.MATERIALS[material]["W"]
        Zs = np.array([-10.0, 0.0, 5.0, 50.0])
        _, w = wy._thermal_nodes(Zs, a, T, material)
        L = np.maximum(Zs * wy._e2_over_a(a), 0.0)
        Lw = np.maximum(L, W)
        expect = np.exp(-(Lw - L) / kT) * (Lw + kT) / (L + kT)
        np.testing.assert_allclose(w.sum(axis=1), expect, rtol=1e-10)

    def test_thermal_mean_impact_energy_is_ds79_eq16(self):
        """Above threshold, the flux-weighted mean impact energy on an
        attracting grain is kT [2 + phi^2 / (1 + phi)], phi = eU/kT."""
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        a, T = 1e-6, 1e5
        kT = KB_CGS * T / EV2ERG
        Z = np.array([200.0, 1000.0])  # eU = 29, 144 eV > W
        E, w = wy._thermal_nodes(Z, a, T, "graphite")
        phi = Z * wy._e2_over_a(a) / kT
        np.testing.assert_allclose((w * E).sum(axis=1) / w.sum(axis=1),
                                   kT * (2.0 + phi ** 2 / (1.0 + phi)), rtol=1e-8)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_cold_gas_excites_almost_no_secondaries(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        assert float(wy.gas_secondary_yield(0, 1e-5, 6e3, material)) < 1e-3

    @pytest.mark.parametrize("material", MATERIALS)
    def test_gas_secondary_rate_properties(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        Zs = np.arange(-20, 21)
        for a in (1e-7, 1e-6, 1e-5):
            lo = wy.gas_secondary_rate(Zs, a, 1.0, 1e4, material)
            hi = wy.gas_secondary_rate(Zs, a, 1.0, 1e5, material)
            assert np.all(lo >= 0) and np.all(hi >= 0)
            # hotter gas puts more of the flux above the threshold; for Z > 0
            # focusing (1 + eU/kT) can favour cold gas instead
            assert np.all(hi[Zs <= 0] > lo[Zs <= 0])
            # the escape barrier eU traps the secondaries of a positive grain
            Zpos = 100.0 * a / 1.44e-7  # U = 100 V
            d = wy.gas_secondary_yield(np.array([0.0, Zpos]), a, 1e5, material)
            assert d[1] < d[0]
            assert np.all(wy.gas_secondary_mean_energy(Zs, a, 1e5, material) >= 0)


class TestSolverIntegration:
    @staticmethod
    def _solve(yield_params, G0=1.0, ne=0.03, T=6000.0, grain="graphite", a=1e-6):
        from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

        ions = [{'n': ne, 'T': T, 'm': 1.6726219e-27, 'z': 1}]
        return equilibrium_charge_for_grain(G0, ne, T, grain, a, ion_species=ions,
                                            yield_params=yield_params)

    @pytest.mark.parametrize("grain", MATERIALS)
    def test_wdb06_without_secondaries_is_wd01_for_soft_fields(self, grain):
        """Mathis field (< 13.6 eV), positive grain: only E_min (Z < -1),
        secondaries and Q_abs (D03 vs the bundled tables) could differ."""
        wd01 = self._solve({'yield_model': 'WD01'}, grain=grain)
        wdb = self._solve({'yield_model': 'WDB06', 'include_secondaries': False,
                           'include_gas_secondaries': False}, grain=grain)
        assert wd01[3] > 1
        assert wdb[3] == pytest.approx(wd01[3], rel=0.03)

    def test_gas_secondaries_raise_the_charge_and_can_be_switched_off(self):
        base = {'yield_model': 'WDB06', 'include_secondaries': False}
        off = self._solve({**base, 'include_gas_secondaries': False}, T=1e5, ne=1.0)
        on = self._solve(base, T=1e5, ne=1.0)
        assert on[3] > off[3]


class TestElectronTransmission:
    @pytest.mark.parametrize("material", MATERIALS)
    def test_repelling_grain_matches_ds79_eq13(self, material):
        """For phi < 0, DS79 eq. 13: <s_e> = 1 - (1 + psi) exp(-psi), psi = E_pen/kT,
        so the transmitted fraction is (1 + psi) exp(-psi)."""
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        for a, T in ((1e-7, 1e6), (1e-6, 1e7), (3e-7, 3e6)):
            psi = wy.penetration_energy(a, material) / (KB_CGS * T / EV2ERG)
            P, _ = wy.electron_transmission([-5.0], a, T, material)
            assert P[0] == pytest.approx((1 + psi) * np.exp(-psi), rel=1e-12)

    @pytest.mark.parametrize("Z", [-4.0, 0.0, 8.0])
    def test_cooling_factor_matches_direct_integration(self, Z):
        from scipy.integrate import quad

        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        a, T, mat = 1e-7, 2e6, "graphite"
        kT = KB_CGS * T / EV2ERG
        eU = Z * wy._e2_over_a(a)
        L = max(eU, 0.0)
        Ep = wy.penetration_energy(a, mat)
        Ec = max((Ep ** 1.5 + L ** 1.5) ** (2 / 3), L)
        w = lambda E: E * np.exp(-(E - L) / kT)
        E2 = lambda E: max(E ** 1.5 - Ep ** 1.5, 0.0) ** (2 / 3)
        top = L + 80 * kT
        e_all = quad(lambda E: w(E) * (E - eU), L, top)[0]
        e_stop = quad(lambda E: w(E) * (E - eU), L, Ec)[0]
        e_dep = quad(lambda E: w(E) * (E - E2(E)), Ec, top, limit=200)[0]
        P_ref = quad(w, Ec, top)[0] / quad(w, L, top)[0]
        P, f = wy.electron_transmission([Z], a, T, mat)
        assert P[0] == pytest.approx(P_ref, rel=1e-8)
        # E - E_2 has a (E - E_c)^(2/3) cusp at threshold, which limits the
        # Gauss-Laguerre sum to ~1e-4
        assert f[0] == pytest.approx((e_stop + e_dep) / e_all, rel=1e-3)

    @pytest.mark.parametrize("material", MATERIALS)
    def test_limits(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy

        Zs = np.arange(-5, 30)
        # H II gas: nothing passes through even the smallest grains
        P, f = wy.electron_transmission(Zs, 3.5e-8, 1e4, material)
        assert np.all(P < 1e-20) and np.allclose(f, 1.0)
        prev = np.zeros(Zs.size)
        for T in (1e5, 1e6, 1e7, 1e8):
            P, f = wy.electron_transmission(Zs, 1e-7, T, material)
            assert np.all(P >= prev) and np.all((f >= 0) & (f <= 1))
            prev = P
        assert np.all(prev > 0.9)

    def test_transmission_raises_the_charge_of_small_grains_in_hot_gas(self):
        from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

        T, ne = 1e6, 1.0
        ions = [{'n': ne, 'T': T, 'm': 1.6726219e-27, 'z': 1}]
        base = {'yield_model': 'WDB06', 'include_gas_secondaries': False}
        args = (1e-6, ne, T, 'graphite', 1e-7)
        off = equilibrium_charge_for_grain(*args, ion_species=ions,
                                           yield_params={**base, 'include_electron_transmission': False})
        on = equilibrium_charge_for_grain(*args, ion_species=ions, yield_params=base)
        assert on[3] > off[3]


class TestIonTransmission:
    def test_transfer_hook_scales_the_ion_current(self):
        from pycalima.models.dust_charge.shared_physics import (
            collisional_rates_ions_scalar,
            collisional_rates_ions_vector,
        )

        ion = {'n': 1.0, 'T': 1e4, 'm': 1.6726219e-27, 'z': 1}
        Zs = np.array([-3.0, 0.0, 4.0])
        ref = collisional_rates_ions_vector(np.array([1e-6]), Zs, [ion])
        half = collisional_rates_ions_vector(np.array([1e-6]), Zs,
                                             [{**ion, 'transfer': lambda Z: 0.5 + 0 * Z}])
        np.testing.assert_allclose(half, 0.5 * ref)
        assert collisional_rates_ions_scalar(1e-6, 2.0, [{**ion, 'transfer': lambda Z: 0.25 + 0 * Z}]) \
            == pytest.approx(0.25 * collisional_rates_ions_scalar(1e-6, 2.0, [ion]))

    @pytest.mark.parametrize("material", MATERIALS)
    def test_charge_transfer_bounds_and_limits(self, material):
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_charge.shared_physics import EV2ERG, KB_CGS

        Zs = np.array([-20.0, 0.0, 20.0])
        for a in (1e-7, 1e-5):
            prev = np.ones(Zs.size)
            for T in (1e4, 1e6, 1e7, 1e8, 1e9):
                p = wy.ion_charge_transfer(Zs, a, T, material, 1, 1, 1.008)
                assert np.all((p >= 0) & (p <= 1)) and np.all(p <= prev + 1e-15)
                prev = p
                # an ion only escapes ionised above E_B, so for Z = 0 the failure
                # is bounded by the Maxwellian flux above E_B
                x = (1.008 / wy.M_E_AMU * wy.I_H_EV) / (KB_CGS * T / EV2ERG)
                assert 1 - p[1] <= (1 + x) * np.exp(-x) + 1e-15
            # nothing escapes ionised in gas below 1e6 K
            assert np.all(wy.ion_charge_transfer(Zs, a, 1e6, material, 1, 1, 1.008) == 1.0)

    def test_ion_transmission_lowers_the_charge_in_very_hot_gas(self):
        from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

        T, ne = 1e8, 1.0
        ions = [{'n': ne, 'T': T, 'm': 1.6726219e-27, 'z': 1}]
        base = {'yield_model': 'WDB06', 'include_gas_secondaries': False,
                'include_electron_transmission': False}
        args = (1e-6, ne, T, 'graphite', 1e-6)
        off = equilibrium_charge_for_grain(*args, ion_species=ions,
                                           yield_params={**base, 'include_ion_transmission': False})
        on = equilibrium_charge_for_grain(*args, ion_species=ions, yield_params=base)
        assert on[3] < off[3]
