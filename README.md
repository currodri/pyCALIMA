The various routines have been written by Yohan Dubois with an
important contribution from Taysun Kimm for type Ia SNe. For
complaints, queries, etc., send an email to dubois@iap.fr. These
routines are not supposed to be distributed and should be taken with
caution since they have been tested by Y. Dubois alone so far.

This "ydroutines/" directory contains several pieces of code to
produce the chemical evolution from a single stellar population (SSP),
aka SSP yields, of several elements using a given set of individual
stellar yields. These SSP yields can then be used into Ramses or into
a simple one zone model with appropriate evolution histories, a
so-called galactic chemical evolution (GCE) model. 

So far, the elements that are included by default are:
H,He,C,N,O,F,Ne,Mg,Si,S,Fe

Routines are all in IDL (it does not work with GDL yet in case you
wanted to try).

Some useful jargon:
SSP: single stellar population
IMF: initial mass function
SNII: type II core-collapse supernova
SNIa: type Ia binary supernova
GCE: galactic chemical evolution
ZAS: zero age star (for the initial metallicity or rotation of stars)
LC18: Limongi & Chieffi 18
K10: Karakas 10
K13: Kobayashi+ 13
N97: Nomoto+ 97
HAGN: Horizon-AGN (what else?)

=================================================================
+ Produce SSP yield evolution from stellar yields

- cmp_yield_release.pro: This is the routine that computes the
chemical outputs from single intermediate (AGB) and massive (SNII)
stars of a 1 Msun SSP. It provides the total (not the net) mass
returned (in Msun) for each element as a function of time. It does not
take into account the contribution from type Ia, which are dealt with
separately in TaysunKimm_supernova/sn1a.pro. One must give an IMF, a
low- and high- mass cutoffs, a mass separatrix between intermediate
and massive stellar yields, the mass for failed SNII, a fraction of
HNe, etc., but must importantly, it must be provided a set of input
stellar yields. At the moment the available stellar yields for
intermediate stars are those from K10, and for SNII those from LC18
(recommended) or K13. The files produced by this routine are the input
files for the galactic chemical evolution model GCE or Ramses. This
routine returns in addition the energy released over time by massive
stars and the total dust mass. Automated execution of this routine to
produce the chemical SSP with some standard set of parameters can be
found here: command_idl_run_all_yield.pro. Once the SSP yields have
been produced, it is recommended to put them into the ResultingYields
directory before proceeding further. For more detailed explanations,
type in your IDL the following: cmp_yield_release,/help
^
| it calls
|
- run_all_yield_release.pro: This is a wrapper that loops over a
sample of ZAS metallicities and velocities to produce the desired
chemical SSP. It calls cmp_yield_release.pro. Some explanations are
provided is you type in your IDL the following naked command:
run_all_yield_release
^
| it calls
|
- command_idl_run_all_yield.pro: This is a just a series of IDL
command lines for sending a batch of calculations of chemical SSP. It
calls run_all_yield_release.pro.


- sn1a.pro: This routine computes the cumulative SNIa rate. Routine
from Taysun Kimm, adapted to have various IMF and low- and high-mass
cutoffs. Be careful to make it consistent with the parameters used in
cmp_yield_release.pro. At the moment the min and max values for
binaries are hardcoded (as well as in cmp_yield_release.pro), as well
as the delay time distribution. Feel free to modify this at your
convenience. Be careful, the fraction of SNIa progenitors (ASNIa) of
the IMF is 1 here. Hence the output file needs to be re-multiplied
appropriately by ASNIa, which exact value should be calibrated on the
data (essentially through the amount of Fe or the cosmic history of
SNIa rates) and might also depend on the IMF.
|
| feeds in
V
- cmp_typeia_yield_release.pro: This routine computes the evolution of
chemical elements released by type Ia SNe according to the type Ia
cumulative rates computed by snia.pro and the ratio of type Ia
progenitors ASNIa (<<1) of the IMF. N97 stellar yields are
assumed. Note that these evolved yields are completely independent of
initial metallicity or stellar rotation as opposed to the SSP yields
for SNII and AGB. Ramses HAGN does not require this step (this
step is done internaly), only the cumulative yields produced by
sn1a.pro, however the GCE model requires it. This routine also
produces the cumulative SNIa cumulative rate required by Ramses in the
Ramses HAGN ready format (cr_Ia.ramses).


+ From now on, if you have executed command_idl_run_all_yield.pro (or
made your own stuff with cmp_yield_release.pro), sn1a.pro and
cmp_typeia_yield_release.pro, you have a complete set of SSP yields
(SNII+SNIa+AGN) that can be used to make some predictions with a GCE
model. To turn the SSP yield (SNII+AGB) files into a Ramses
HAGN-ready format, it requires a bit of manipulations of the
various files through the following routines just below.

=================================================================
+ Convert SSP yields into the Ramses format

- convert_myyieldsintoramses.pro: This routine will just convert a set
of files produced by cmp_yield_release.pro for different ZAS
metallicities into the expected Ramses format as of HAGN. The
time binning MUST be the same for all the input files. This routine
also plots the SSP yield evolution (see also
plots_yieldevolejecta.pro)
^
| it calls
|
- execute_convertintoramses.pro: This routine contains one exemple of
how the convert_myyieldsintoramses.pro routine can be used for a full
set of SSP yields.

=================================================================
+ Convert raw stellar yields (extracted from appropriate database
elsewhere) into an appropriate format to feed in
cmp_yield_release.pro. There is no need to play with those routines
unless you want to add elements to the default set (and this
non-trivial: please, read below)

- crunch_karakas.pro: This routine turns the raw data of K10 yields
into a simplified format for only a bunch of selected elements. The
output files from this routine are used by cmp_yield_release.pro. If
you want to add more elements this is possible but you have to
consistently change it everywhere (in all routines that manipulate
those yields...): this last bit can be improved...
^
| it calls (and also interpolate_metallicity_karakas.pro)
|
- run_all_crunch_karakas.pro: This routines is wrapper that produces
all the necessary K10 yields in their simplified format.

- interpolate_metallicity_karakas.pro: This routine interpolates the
stellar yields of K10 in their simplified format for a given ZAS
metallicity.

- crunch_lc18.pro: This routine turns the raw data of LC18 yields into
a simplified format for only a bunch of selected elements. The output
files from this routine are used by cmp_yield_release.pro. If you want
to add more elements this is possible but you have to consistently
change it everywhere (in all routines that manipulate those
yields...): this last bit can be improved... Be careful, this is
non-trivial for LC18 because elements are given very soon after the
explosion by their isotopic abundances, which have radioactive decay
time sometimes much smaller than any relevant galactic timescale. This
is done consistently for the already computed elements, but this must
be checked carefully for any new element. LC18 models come in two
flavors: one where all massive stars explode into SNII (model M),
another where all stars more massive than 25 Msun are failing to go
into SNII (their recommended set). The advantage of having those two
sets is that one can freely chose how to mix failed SNe with
non-failed SNe (see cmp_yield_release.pro for details).
^
plots_yieldevolejecta.pro| it calls (and also rewrite_lc18_modelm.pro and crunch_lc18esn.pro)
|
- run_all_crunch_lc18.pro: This routine is wrapper that produces
all the necessary LC18 yields in their simplified format.

- rewrite_lc18_modelm.pro: This routine turns the raw data of LC18 
SNII model M yields into a simplified format for only a bunch of
selected elements. The output files from this routine are eventually
used by cmp_yield_release.pro.

- crunch_lc18esn.pro: This routine turns the raw data of LC18 SN
energies (the binding energy of the star) into a simplified format
used by cmp_yield_release.pro. This is purely experimental and should
not be used unless you know what you do.

- interpolate_lc18.pro: This routine interpolates the stellar yields
  of LC18 in their simplified format for a given ZAS metallicity.
^
| it calls
|
- run_all_interpolate_lc18.pro: This routine is a wrapper that
interpolate the stellar yields of LC18 for several ZAS metallicities.

- rewrite_kobayashi.pro: This routine turns the raw data of K13 AGB,
SNII and HN yields into a simplified format for only a bunch of
selected elements. The output files from this routine are eventually
used by cmp_yield_release.pro.

- crunch_ww.pro: This routine turns the raw data of WW95 yields
(extracted from Molla+15) into a simplified format for only a bunch of
selected elements. The output files from this routine are used by
cmp_yield_release.pro. The number of extra elements than can be added
is rather thin, thus, it is not recommended to use this set of yields.

=================================================================
+ Predict a galactic chemical evolution from a one-zone model. It is
very useful to check whether some galactic abundances are not
completely off before running an expensive Ramses solution.

- galactic_chemical_evolution.pro: This is the GCE model that returns
chemical tracks (alpha versus [Fe/H]) for a given galaxy history. This
routine requires files produced by cmp_yield_release.pro and
cmp_typeiayield_release.pro, i.e. chemical SSP for different
metallicities. A GCE basically replaces a hydrodynamical simulations
with a single zone model for which the accretion rate, star formation
rate, and outflow rates needs to be prescribed.
Exemples can be found in here: exemples_command_idl_gce.pro
^
| it calls
|
- command_idl_gce.pro: This is a wrapper for the GCE model. You should
  sniff into this to change the input stellar yields.
^
| it calls
|
- exemples_command_idl_gce.pro: Just a few exemples of command lines
  for command_idl_gce.pro that are ssupposed to work for a Milky Way
  galaxy.

=================================================================
- plot_allimf.pro: This routine just plots a serie of IMF (Salpeter,
  Kroupa, Chabrier, metallicity-dependent Varying IMF)

- plots_yieldevolejecta.pro: This routine selects the set of SSP
  yields to be plotted (IMF, rotational velocity) by calling
  convert_myyieldsintoramses.pro

- plot_wiersma.pro: This routine reproduces the figures A2 and A4 from
  Wiersma+09, i.e. the integrated AGB or SNII ejecta abundances.

- plot_yields.pro: This routine plots the individual stellar yields.

- plot_yields_ratio.pro: This routine plots the individual stellar
yields as a ratio of the initial individual stellar mass

- plot_yieldsnia.pro: This routine plots the yields for SNIa from N97
as a function of the atomic mass (it can be multi-valued because of
isotopes).

=================================================================

So the typical flow chart is:
- Obtain a grid of individual stellar yields for different ZAS
  metallicities.
- Compute the mass release as a function of time for a given SSP from
  intermediate and massive stars
- Compute the mass release as a function of time from SNIa
(- Test them in the GCE model)
(- Plug them into Ramses)


