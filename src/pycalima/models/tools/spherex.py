"""SPHEREx spectral channels and synthetic spectrophotometry.

SPHEREx observes 0.75-5.0 um in six linear-variable-filter bands of 17
channels each (Crill et al. 2020, SPIE 11443; Bock et al. 2025):

    band  range [um]   R
    1     0.75-1.11    41
    2     1.11-1.64    41
    3     1.64-2.42    41
    4     2.42-3.82    35
    5     3.82-4.42   110
    6     4.42-5.00   130

Here each channel is a top-hat in wavelength, the 17 channels of a band
log-spaced edge to edge. This is an approximation to the true LVF responses
(which are close to Gaussian with FWHM lambda/R and overlap slightly), good
for the shape of a smooth continuum at the band resolution; it is not a
substitute for the released response files.
"""

import numpy as np

SPHEREX_BANDS = (
    (0.75, 1.11, 41.0),
    (1.11, 1.64, 41.0),
    (1.64, 2.42, 41.0),
    (2.42, 3.82, 35.0),
    (3.82, 4.42, 110.0),
    (4.42, 5.00, 130.0),
)
CHANNELS_PER_BAND = 17


def spherex_channels():
    """Channel centres, lower and upper edges [um] and band resolving power.

    Returns a dict of arrays ``lam_um`` (geometric centre), ``lo_um``,
    ``hi_um``, ``R`` and ``band`` (1-6), 102 channels in increasing wavelength.
    """
    lo, hi, R, band = [], [], [], []
    for b, (l0, l1, Rb) in enumerate(SPHEREX_BANDS, start=1):
        edges = np.geomspace(l0, l1, CHANNELS_PER_BAND + 1)
        lo.append(edges[:-1])
        hi.append(edges[1:])
        R.append(np.full(CHANNELS_PER_BAND, Rb))
        band.append(np.full(CHANNELS_PER_BAND, b))
    lo, hi = np.concatenate(lo), np.concatenate(hi)
    return {'lam_um': np.sqrt(lo * hi), 'lo_um': lo, 'hi_um': hi,
            'R': np.concatenate(R), 'band': np.concatenate(band)}


def spherex_photometry(lam_um, F_lambda, n_sub=21):
    """Mean F_lambda in each SPHEREx channel (photon-weighted, top-hat).

    Parameters
    ----------
    lam_um : array
        Ascending wavelength grid [um] of the spectrum (observed frame).
    F_lambda : array
        Spectrum on lam_um, any per-wavelength unit; the last axis is
        wavelength, so several spectra can be passed at once.
    n_sub : int
        Sampling points per channel.

    Returns
    -------
    (channels, F_channel) with ``channels`` from ``spherex_channels`` and
    F_channel = int F_lambda lambda dlambda / int lambda dlambda per channel.
    """
    ch = spherex_channels()
    F = np.asarray(F_lambda, dtype=float)
    out = np.empty(F.shape[:-1] + (ch['lam_um'].size,))
    for k, (l0, l1) in enumerate(zip(ch['lo_um'], ch['hi_um'])):
        x = np.linspace(l0, l1, n_sub)
        Fx = np.stack([np.interp(x, lam_um, f) for f in F.reshape(-1, F.shape[-1])])
        val = np.trapezoid(Fx * x, x, axis=-1) / np.trapezoid(x, x)
        out[..., k] = val.reshape(F.shape[:-1])
    return ch, out
