(data-provenance)=
# Data provenance

pyCALIMA ships or fetches fifteen reference datasets. The table below is
generated from the package's own registry at build time, so it cannot drift
from what is actually installed. `calima-fetch-data list` prints the same
information for your installation, including what is present locally.

:::{warning}
This repository did not originally record where most of its reference data came
from. Only one file carried a usable URL. Every entry marked `unconfirmed`
below is a *plausible* upstream that has **not** been verified, and several
datasets are locally derived and cannot be re-downloaded byte-for-byte.
Confirm before relying on `calima-fetch-data fetch`.
:::

## Registry

```{eval-rst}
.. calima-registry::
```

## Kinds

`bundled`
: Ships inside the wheel. Fetching is a presence check.

`fetch`
: Downloaded on demand from `url` and verified against a recorded SHA-256.

`manual`
: You must obtain it yourself, because the source requires registration or
  issues per-download filenames. `calima-fetch-data import` then registers
  your local copy. The Ames PAH database is the notable case: its download
  filenames carry a per-request token, so no stable URL exists.

## Redistribution

Not every bundled dataset has confirmed redistribution rights. In particular,
the registry records that rights for `optical_props/fromDanieleRogantini` are
**unconfirmed**, and that question gates any public release of the package.
Resolve it before publishing to an index.
