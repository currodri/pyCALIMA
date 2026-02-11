dust_charge_distributions/ — README

This folder contains running-median outputs produced by
`compute_charge_vs_gamma()` in `dust_charging.py`.

File naming convention
- New filename pattern used by `compute_charge_vs_gamma()` (since the recent update):
  - dust_charge_lowT_{size}_micron_{material}.dat
  - dust_charge_highT_{size}_micron_{material}.dat
  where:
    - {size} is the grain radius in microns formatted with 4 decimal places (e.g. 0.0004)
    - {material} is the short material code: `Gra` for graphite, `suvSil` for silicate

Example filenames:
  - `dust_charge_lowT_0.0004_micron_Gra.dat`
  - `dust_charge_highT_0.1000_micron_suvSil.dat`

File format (Fortran-friendly)
- First line: a single integer giving the number of data rows that follow.
- Remaining lines: three space-separated floating-point values per row.
  - Column 1: log10(gamma)  (dimensionless; gamma = G0 * sqrt(T) / n_e)
  - Column 2: median_Z (mean grain charge, dimensionless)
  - Column 3: median_sigma (distribution width, dimensionless)

How medians were computed
- Low-temperature subset: points with T < 1e2 K (100 K).
- High-temperature subset: points with T > 1e4 K (10,000 K).
- For each subset:
  1. select points with finite gamma and finite values;
  2. build a log-spaced gamma grid (200 points) spanning the subset's gamma range;
  3. for each grid point, compute the median of sample points whose log10(gamma)
     lies within ±half_width dex of the grid point (half_width defaults used in
     the code; low-T and high-T windows may differ);
  4. write the resulting running-median arrays to disk.

Notes
- Files contain only finite rows (no NaNs). The integer on the first line equals
  the number of data rows that follow.
- These files are intended for simple sequential text reading from Fortran90.

Regenerating the files
- The files are produced by `compute_charge_vs_gamma()` in `dust_charging.py`.
  Run the example script to recreate them:

```bash
python examples/scan_gamma.py
```

Fortran90 reading example (sequential list-directed reads)
```fortran
integer :: nrows, i
real(8), allocatable :: lg(:), medZ(:), medSig(:)
open(unit=10, file='graphite_a0.02um_lowT_medians.dat', status='old')
read(10, *) nrows
allocate(lg(nrows), medZ(nrows), medSig(nrows))
do i = 1, nrows
  read(10, *) lg(i), medZ(i), medSig(i)
end do
close(10)
! lg contains log10(gamma), medZ the median <Z>, medSig the median sigma
```

If you want additional metadata (timestamp, sample counts, gamma ranges) stored
for each file I can write a small companion JSON metadata file next to each
`.dat` file.
