"""Kinetic Monte Carlo of single PAH molecules: an independent check of
``energy_me``.

Each molecule carries a continuous internal energy E. Events are drawn with
the Gillespie algorithm from the same physical rates the master equation
uses: IR emission in mode i, E -> E - h nu_i, at k_i(E); absorption of a
heating photon, E -> E + e, with e drawn from h(e); and dissociation at
k_c(E). The rates are linearly interpolated in E from a ``CascadeModel``
grid. Nothing is binned and no jump is split, so agreement with
``energy_me`` tests both its algebra and its discretisation.
"""

from __future__ import annotations

import numpy as np


class MonteCarlo:
    """Gillespie simulation on the rates of a ``CascadeModel``.

    Parameters
    ----------
    model : energy_me.CascadeModel
    channel_rates : dict
        Channel name -> k_c on ``model.energy_cm``.
    photon_energy_ev, heating_spectrum : array_like, optional
        h(e) [s^-1 eV^-1]; omit for a cascade without absorption.
    seed : int
    """

    def __init__(self, model, channel_rates, photon_energy_ev=None, heating_spectrum=None, seed=0):
        from pycalima.pahchem import dos

        self.model = model
        self.names = list(channel_rates)
        self.K = np.array([np.asarray(channel_rates[c], dtype=float) for c in self.names]).reshape(
            len(self.names), model.n)
        self.rng = np.random.default_rng(seed)
        self.r_abs = 0.0
        if heating_spectrum is not None:
            e_cm = np.asarray(photon_energy_ev, dtype=float) * dos.CM_PER_EV
            h = np.asarray(heating_spectrum, dtype=float)
            cdf = np.concatenate([[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(e_cm / dos.CM_PER_EV))])
            self.r_abs = float(cdf[-1])
            self._e_cm, self._cdf = e_cm, cdf / cdf[-1]

    def _rates(self, E):
        de = self.model.de_cm
        x = min(E / de, self.model.n - 1.000001)
        j = int(x)
        f = x - j
        k_modes = (1.0 - f) * self.model.k_modes[:, j] + f * self.model.k_modes[:, j + 1]
        k_modes = np.where(self.model.freqs_cm <= E, k_modes, 0.0)
        k_ch = (1.0 - f) * self.K[:, j] + f * self.K[:, j + 1]
        return k_modes, k_ch

    def _photon(self):
        return float(np.interp(self.rng.random(), self._cdf, self._e_cm))

    def photon_yields(self, start_cm, n_molecules):
        """Fraction of molecules started at ``start_cm`` that end in each channel."""
        counts = np.zeros(len(self.names))
        open_ = np.nonzero(self.K.sum(axis=0) > 0)[0]
        # Energy only falls in a cascade: below the first open bin's lower
        # neighbour (rates are interpolated) no channel can be reached.
        e_cut = (open_[0] - 1) * self.model.de_cm if open_.size else np.inf
        for _ in range(n_molecules):
            E = float(start_cm)
            while E > e_cut:
                k_modes, k_ch = self._rates(E)
                total = k_modes.sum() + k_ch.sum()
                if total <= 0.0:
                    break
                u = self.rng.random() * total
                if u < k_ch.sum():
                    counts[np.searchsorted(np.cumsum(k_ch), u)] += 1
                    break
                E -= self.model.freqs_cm[np.searchsorted(np.cumsum(k_modes), u - k_ch.sum())]
        return dict(zip(self.names, counts / n_molecules))

    def cascade_photons(self, start_cm, n_molecules, e_stop_cm=0.0):
        """Mean IR photons emitted per mode by molecules started at ``start_cm``.

        Each cascade runs until no mode can emit, a channel fires, or E falls
        below ``e_stop_cm``. This checks ``emission.cascade_photons``.
        """
        counts = np.zeros(self.model.freqs_cm.size)
        for _ in range(n_molecules):
            E = float(start_cm)
            while E > e_stop_cm:
                k_modes, k_ch = self._rates(E)
                total = k_modes.sum() + k_ch.sum()
                if total <= 0.0:
                    break
                u = self.rng.random() * total
                if u < k_ch.sum():
                    break
                i = np.searchsorted(np.cumsum(k_modes), u - k_ch.sum())
                counts[i] += 1
                E -= self.model.freqs_cm[i]
        return counts / n_molecules

    def steady_rates(self, n_photons, burn_in=200):
        """Dissociation rates [s^-1] of a molecule re-injected cold after each dissociation.

        The clock runs for ``n_photons`` absorptions after a ``burn_in``.
        """
        E, t, counted, n_abs = 0.0, 0.0, np.zeros(len(self.names)), 0
        while n_abs < n_photons + burn_in:
            k_modes, k_ch = self._rates(E)
            kir, kc = k_modes.sum(), k_ch.sum()
            total = kir + kc + self.r_abs
            dt = self.rng.exponential(1.0 / total)
            u = self.rng.random() * total
            if n_abs >= burn_in:
                t += dt
            if u < self.r_abs:
                E += self._photon()
                n_abs += 1
            elif u < self.r_abs + kc:
                if n_abs >= burn_in:
                    counted[np.searchsorted(np.cumsum(k_ch), u - self.r_abs)] += 1
                E = 0.0
            else:
                E -= self.model.freqs_cm[np.searchsorted(np.cumsum(k_modes), u - self.r_abs - kc)]
        return dict(zip(self.names, counted / t))
