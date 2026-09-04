"""Opt-in matplotlib styling for pyCALIMA figures.

Many physics modules used to call ``seaborn.set_theme()`` and
``plt.rcParams.update({"text.usetex": True, ...})`` at *module scope*. Once
pyCALIMA is an installed library that is a bug in two ways:

* it silently reconfigures matplotlib for the whole importing process,
  including any unrelated user code; and
* on a machine with no working LaTeX installation it makes every subsequent
  ``savefig`` raise, merely because something imported a physics module.

Call :func:`use_calima_style` explicitly from plotting code instead. LaTeX is
enabled only when matplotlib can actually find a usable installation, so the
same call is safe on a bare CI container.
"""

from __future__ import annotations

__all__ = ["use_calima_style", "latex_available"]

_APPLIED = False


def latex_available() -> bool:
    """True if matplotlib has a usable LaTeX toolchain.

    Falls back to a direct check for ``latex`` and ``dvipng`` on PATH when
    matplotlib offers no dependency-check helper.
    """
    try:
        import matplotlib
    except ModuleNotFoundError:
        return False

    checkdep = getattr(matplotlib, "checkdep_usetex", None)
    if callable(checkdep):
        try:
            return bool(checkdep(True))
        except Exception:  # noqa: BLE001 - never let a style probe raise
            return False

    import shutil

    return all(shutil.which(exe) for exe in ("latex", "dvipng"))


def use_calima_style(*, usetex: bool | None = None, force: bool = False) -> None:
    """Apply the CALIMA figure style to the current matplotlib session.

    Parameters
    ----------
    usetex
        Force LaTeX text rendering on or off. The default, None, enables it
        only when :func:`latex_available` says a usable installation exists.
    force
        Re-apply even if this function already ran in this process.

    Notes
    -----
    This mutates global matplotlib state, which is why it is opt-in. It is
    idempotent unless *force* is given.
    """
    global _APPLIED
    if _APPLIED and not force:
        return

    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="white")
    except ModuleNotFoundError:
        pass

    if usetex is None:
        usetex = latex_available()

    plt.rcParams.update(
        {
            "text.usetex": bool(usetex),
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        }
    )
    _APPLIED = True
