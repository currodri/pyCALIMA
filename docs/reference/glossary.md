(glossary)=
# Glossary

```{glossary}
DTM
  Dust-to-metal ratio: the fraction of refractory metal mass locked in grains.
  The solver's headline diagnostic.

G0
  Interstellar radiation field strength in Habing units, normalised so that
  `G0 = 1` is the solar-neighbourhood field.

ISRF
  Interstellar radiation field.

gamma (γ)
  The charging parameter `G0 sqrt(T) / n_e`, which sets the equilibrium grain
  charge distribution. The charging and photoelectric heating tables are
  tabulated against it.

phi (φ)
  Reduced grain potential, `Z e^2 / (a k T)` — the ratio of Coulomb to thermal
  energy that governs Coulomb-enhanced collision rates.

PEH
  Photoelectric heating: gas heating by electrons ejected from grains and PAHs
  by UV photons.

PAH
  Polycyclic aromatic hydrocarbon. Treated as the smallest grain bins, with
  their own charging, photodissociation and collisional physics.

N_C
  Number of carbon atoms in a PAH. Related to radius by
  `N_C = 418 (a / 10 Å)^3`, so a 10 Å PAH has 418 carbon atoms.

RATD
  Rotational disruption by radiative torques: destruction of grains spun to
  above their tensile limit by anisotropic radiation.

LW
  Lyman–Werner band (91.2–111 nm), which photodissociates molecular hydrogen.

f_small
  Mass fraction of the dust in the small-grain bins.

f_carb
  Carbonaceous mass fraction of the dust.

sputtering
  Erosion of grain material by impacting gas particles. *Thermal* sputtering is
  driven by the gas temperature; *inertial* sputtering by the grain's drift
  velocity.

coagulation
  Grain growth by sticking collisions between grains.

shattering
  Fragmentation of grains in high-velocity collisions.

accretion
  Growth of grains by condensation of gas-phase metals onto their surfaces.
```
