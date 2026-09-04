(api-reference)=
# API reference

pyCALIMA has no curated top-level namespace: `pycalima/__init__.py`
deliberately re-exports nothing, because three circular dependencies between
the physics subpackages are resolved only by deferred, function-local imports.
Thirteen of the eighteen `__init__.py` files are empty for the same reason.

So this reference is organised by leaf module, and each page names its modules
explicitly rather than discovering them. That is deliberate: it keeps the 29
figure-reproduction scripts and the 12 batch exporters out of the API surface,
where they would otherwise dominate it. Those are indexed separately in
{doc}`/reference/scripts`, and the exporters are documented as commands in
{doc}`/cli/index`.

Three subpackages do export a curated `__all__` — `solvers`,
`models.dust_shielding` and `galaxysam` — and their pages lead with it.

```{toctree}
:maxdepth: 1

solvers
pycalima
models
models.dust_radiation
models.dust_charge
models.dust_gas_collisions
models.dust_collisions
models.dust_chemistry
models.dust_shielding
models.PAH_radiation
models.PAH_charge
models.PAH_gas_collisions
models.PAH_photophysics
models.PAH_collisions
models.tools
models.yields
galaxysam
```
