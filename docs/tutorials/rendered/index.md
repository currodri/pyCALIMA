(rendered-tutorials)=
# RAMSES post-processing notebooks

These two notebooks read RAMSES simulation snapshots, which are multi-gigabyte
and **not distributed with pyCALIMA**. They are therefore not executed when the
documentation is built — you are reading the code, not results computed from
your data.

To run them yourself, point `$CALIMA_SIM_DIR` at your own outputs and open them
from `notebooks/`. {doc}`/guide/post-processing` explains what they expect.

:::{note}
`CALIMA_model_explorer.ipynb` has section labels that do not ascend in cell
order — §5, §7, §6, §11, §8, §9 — and several code cells carry section numbers
in their comments that disagree with the surrounding headings. Read the section
titles rather than the numbers. Reordering the cells would change the execution
order, so it needs its author's attention rather than a mechanical fix.
:::

```{toctree}
:maxdepth: 1

ramses_equilibrium_tutorial
CALIMA_model_explorer
```
