(configuration)=
# Configuration model


Key output folders inside `model_data/`:

- `optical_properties/`: dust and PAH averaged cross-sections per configured bin.
- `collisional_cooling_data/`: cooling tables versus gas temperature and charge state.
- `thermal_sputtering_data/`: dust sputtering tables (temperature/phi grids).
- `pah_sputtering_data/`: PAH sputtering tables and quicklook plots.
- `dust_charging_data/`: charge moments and related quicklook products.
- `dust_photoelectric_heating_data/`: dust photoelectric heating/cooling tables.
- `PAH_photoelectric_heating_data/`: PAH heating efficiency and ionization-state tables.
- `PAH_dissociation_data/`: PAH dissociation tables and contour-style plots.

See `model_data/README.md` for a generation-time report with metadata and export summary.

# Configuration Model

The file `models/grain_size_distribution.json` defines:

- Active bins (`bins`) with composition, PAH flag, and rank.
- Distribution parameters for each bin (`basic`, `shattering`).
- Export sampling parameters (`export_parameters`) used by batch exporters.

Most workflows read this file through `models/grain_size_config.py`, so changing it is the
primary way to adapt the model setup.
