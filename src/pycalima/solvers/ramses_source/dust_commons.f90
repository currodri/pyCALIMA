! Dust commons.
! For details, see Dubois et al. 2022
! By: Yohan Dubois (Original: 1 Feb 2022)
! Changes:
!       - Curro Rodriguez (Cleaning into external 
!                           module: 21 Feb 2022)

module dust_commons
    use amr_parameters, only:dp,metal
    use hydro_parameters, only:n_elements,ndust,ndchemtype,npah,idust,ipah
    use constants, only: amu2g
    use dust_utils
    use dustbin_types

    implicit none

    ! ==== Flags and logicals (read from nml) ====
    logical, parameter ::dust=(ndust>0)          ! CALIMA includes dust if ndust > 0
    logical ::dust_log=.false.                   ! Activate dust logging (ODE solver stats: accept/reject/substep counts)
    logical ::dust_debug=.false.                 ! Activate detailed dust mass-budget logging (per-process dM/dt; heavier overhead)
    integer ::dust_solver_type=1                 ! Solver type: 1 = RK4, 2 = Anninos, 3 = RK54
    logical ::solver_substepped=.true.           ! Whether the dust solver uses adaptive substepping (e.g. RK4, RK54)
    logical ::dust_only_rtadv=.false.            ! Activate dust chemistry only when RT is on
    logical ::dust_eq_test=.false.               ! Activate dust equilibrium test parameters
    logical ::dust_SNdest=.false.                ! Dust destruction in SN explosions
    logical ::dust_inSN=.false.                  ! Inject dust in SNII explosions
    logical ::dust_inSNIa=.false.                ! Inject dust in SNIa explosions
    logical ::dust_inSW=.false.                  ! Inject dust in AGB winds
    logical ::dust_coagulation=.false.           ! Activate grain coagulation
    logical ::dust_coagulation_boost=.false.     ! Activate the boost of coagulation in dense gas
    logical ::dust_shattering=.false.            ! Activate grain shattering
    logical ::dust_shattering_all=.false.        ! Activate the shattering caused by the collision of all grain sizes
    logical ::dust_shattering_dest=.false.       ! Activate the destruction of dust mass via shattering fragmentation
    logical ::dust_shattering_SN=.false.         ! Activate the redistribution of grain mass in SN(II and Ia) shocks due to inertial sputtering
    logical ::dust_accretion=.false.             ! Activate grain growth by accretion
    logical ::dust_sputtering=.false.            ! Activate grain destruction by thermal sputtering
    logical ::dust_sputtering_charge=.false.     ! Activate the dependence of thermal sputtering on grain and ion charge
    logical ::dust_sublimation=.false.           ! Activate grain destruction by thermal sublimation
    logical ::dust_acc_coulomb=.false.           ! Compute on-the-fly Coulomb enhancement of refractory material accretion
    logical ::dust_ratd=.false.                  ! Activate destruction of dust grains by RATD
    logical ::dust_coll_cooling=.false.          ! Activate dust collisional cooling
    logical ::dust_coll_lowT=.false.             ! Activate low-temperature dust collisional heating (Hollenbach & McKee 1980)
    logical ::dust_coll_charge=.false.           ! Activate the dependence of dust collisional cooling on grain and ion charge
    logical ::dust_pe_heating=.false.            ! Activate photo-electric heating by dust grains
    logical ::dust_pe_heating_isrf=.false.       ! PE heating from the uniform-ISRF tables at the local G0 (models other than WDB06rt)
    logical ::ratd_only_rtadv=.false.            ! Only allow for RATD if the rt_advect=.true.
    logical ::poppe_ice_enhancement=.false.      ! Whether to use the empirical enhancement in coagulation threshold due to ice mantel
    logical ::H2ondust=.false.                   ! Activate H2 formation on dust
    logical, parameter ::dust_pahs=(npah>0)       ! CALIMA includes PAHs if npah > 0
    logical ::dust_turbulent_model=.false.       ! Activate the subgrid model of turbulent shattering and coagulation
    logical ::pah_accretion=.false.              ! Activate the simple growth of PAH mass by accretion of gas phase C atoms
    logical ::pah_acc_spu=.false.                ! Activate the destruction of PAHs by accretion of C+
    logical ::pah_coalescence=.false.            ! Activate coalescence of PAHs into small carbonaceous grains
    logical ::pah_freezing=.false.               ! Activate the freezing of PAHs onto carbonaceous grains
    logical ::pah_desorption=.false.             ! Activate the desorption of freezed-out PAHs from carbonaceous grains
    logical ::pah_photolysis=.false.             ! Activate photolysis of PAHs by high energy photons
    logical ::pah_sn_destruction=.false.         ! Inertial and non-thermal destruction of PAHs by SN shocks
    logical ::pah_cluster_evaporation=.false.    ! Activate the evaporation of PAH clusters to form small PAHs due to UV photon absorption
    logical ::pah_AGBwinds=.false.               ! Inject PAHs during AGB winds
    logical ::pah_sputtering=.false.             ! Ion and electron destruction of PAHs
    logical ::pah_pe_heating=.false.             ! Photo-electric heating by PAHs
    logical ::pah_pe_heating_isrf=.false.        ! Activate the simple PAH PE heating based on an averaged ISRF G0
    logical ::pah_pe_nolyman=.false.             ! Deactivate the 13.6 eV limit for PAH PE heating
    logical ::H2onpah=.false.                    ! Formation of H2 molecules on PAHs

    ! ==== Dust dynamics (read from nml) ====
    logical ::dust_tva=.false.                   ! Activate the dust dynamics using the Terminal Velocity Approximation (TVA)
    logical ::dust_radpressure=.false.           ! Activate the dust dynamics using the radiation pressure force
    logical ::use_w_drift_test=.false.           ! Override the drift velocity with a constant value for testing
    real(dp),dimension(1:3)::w_drift_test=0.0_dp ! Constant drift velocity for each dimension (X, Y, Z)
    real(dp),dimension(1:ndust)::drag_coefficient=1d0 ! Constant drag coefficient for testing
    real(dp) ::epstein_coef=1d0                  ! sqrt(pi*gamma/8), set once in init_CALIMA_dust.
                                                 ! Epstein drag uses the MEAN thermal speed
                                                 ! v_th = sqrt(8kT/(pi mu mH)); written with the
                                                 ! ADIABATIC c_s used in the solver that is
                                                 ! v_th = sqrt(8/(pi*gamma))*c_s, hence
                                                 ! t_s = sqrt(pi*gamma/8)*s*a/(rho_g*c_s).
                                                 ! (Laibe & Price 2012; Lebreuilly+2019.)
    character(LEN=20)::drag_model='epstein'      ! Gas-grain drag law for the TVA stopping time:
                                                 ! 'epstein'    -- linear Epstein drag (above);
                                                 ! 'draine2011' -- Draine (2011) eq. (24)-(25), after
                                                 !   Draine & Salpeter (1979): collisional + Coulomb
                                                 !   drag with the grain charge from charging_model.
                                                 !   Nonlinear in the drift; needs dust_radpressure.
                                                 ! 'draine2011_pz' -- the same, with the Coulomb term
                                                 !   averaged over the charge distribution P(Z) of
                                                 !   charging_model instead of taken at |<Z>|.
    real(dp) ::tva_wmax_cs=1d0                   ! Cap on |w_drift| in units of the local sound speed.
                                                 ! TVA assumes Stokes << 1, which fails once the drift
                                                 ! approaches c_s (typically in hot/diffuse cells, where
                                                 ! t_s ~ 1/rho_gas blows up). <=0 disables the cap.
    ! condinit_kind resolved to an integer once in init_CALIMA_dust, so the innermost
    ! solver loops carry an integer compare instead of a character(60) one -- and so an
    ! innocently named IC cannot silently switch the physics to a test branch.
    integer,parameter ::TVA_TEST_NONE=0, TVA_TEST_DIFFUSE=1, TVA_TEST_SHOCK=2, &
                      & TVA_TEST_BLAST1D=3, TVA_TEST_SPRESS=4, TVA_TEST_GAUSS=5, &
                      & TVA_TEST_IRTRAP=6
    integer ::tva_test_mode=TVA_TEST_NONE
    ! ==== Dust dynamics diagnostics (not in nml) ====
    integer(kind=8) ::tva_nclip=0                ! Faces/cells whose drift was clipped since last report
    real(dp) ::tva_wmax_seen=0d0                 ! Largest |w_drift|/c_s seen since last report
    integer ::tva_last_warn_step=-1              ! Coarse step of the last "TVA sets dt" message
    ! ==== Dust standalone testing (read from nml) ====
    logical ::dust_test=.false.                  ! Activate accretion solver test
    real(dp) ::test_nH=1.0d2                     ! Target test gas hydrogen density [cm-3]
    real(dp) ::test_Tk=1.0d1                     ! Target test gas temperature [K]
    integer ::test_nsteps=100                    ! Target test solver steps
    real(dp) ::test_dt=1.0d4                     ! Target test solver step size [yr]
    real(dp) ::test_ne=1.0d-2                    ! Target test gas electron density [cm-3]
    real(dp) ::test_mu=1.0d0                     ! Target test gas mean molecular weight

    ! ==== Dust modelling options (read from nml) ====
    character(LEN=30)::sputtering_model='Tsai1998'    ! Thermal sputtering law (Tsai&Matthews 1995)
    character(LEN=30)::accretion_model='Chaabouni2012'    ! Accretion model
    character(LEN=30)::shattering_model='Granato2021' ! Model for the shattering dispersion velocity
    character(LEN=30)::coagulation_model='Aoyama2017' ! Model for the coagulation dispersion velocity
    character(LEN=30)::dust_velocity_model='Ormel2007' ! Model for the relative velocity of grains
#ifdef RT
    character(LEN=30)::charging_model='WDB06rt'        ! Model for the grain charge distribution: WDB06rt (RT photon
                                                       !   groups), WDB06isrf (uniform ISRF), Ibanez2019
#else
    character(LEN=30)::charging_model='WDB06isrf'      ! Model for the grain charge distribution: WDB06isrf (uniform ISRF)
                                                       !   or Ibanez2019 (WDB06rt needs RT photon groups)
#endif
    character(LEN=30)::ice_model='Hollenbach2009'     ! Model for ice formation on dust grains
    integer :: nZmix=3                                  ! Number of representative charge points (1: mean, 2: two-point, 3: three-point)
    logical :: dust_rtgroups_debug=.false.             ! WDB06rt: dump per-call inputs/outputs (dust_rtgroups_debug_XXXXX.dat)
    integer :: dust_rtgroups_debug_max=2000            ! WDB06rt: maximum number of dumped calls per rank
    logical :: dust_rtgroups_verify=.false.            ! WDB06rt: dump (as dust_rtgroups_debug) and stop after dust_rtgroups_debug_max calls
    logical :: dust_charging_timer=.false.             ! wall-clock time of the grain charging and PE heating (all models), printed at the end
    logical :: dust_ion_recombination=.false.          ! WDB06rt/WDB06isrf: grain-assisted recombination of the singly charged
                                                       ! ions from the charge balance, used by the RTZ chemistry instead of its
                                                       ! fit; off: WDB06rt skips it and WDB06isrf does not load that part
    real(dp) :: rtg_predict_tol=0.2d0                  ! WDB06rt: first-order prediction from the cell's last full solve while
    real(dp) :: rtg_predict_tol_shape=0.3d0            ! |d ln| of the inputs <= tol and of each group's spectrum shape <= tol_shape
                                                       ! (0: always a full solve)

    ! ==== PAH modelling options (read from nml) ====
    character(LEN=30)::photolysis_model='RM2026'         ! Model for UV sublimation of PAHs
    character(LEN=30)::peh_attach_model='Berne'             ! Photo-electric model assumptions
    character(LEN=30)::coalescence_model='Totton2012'  ! PAH coalescence model
    character(LEN=30)::pah_h2_model='RM2026'           ! Model for the formation of H2 by PAHs
    character(LEN=30)::pah_growth_model='subgrid' ! Model for PAH growth by accretion of gas phase C atoms
    character(LEN=30)::pah_sputtering_model='RM2026' ! Model for the sputtering of PAHs by ions and electrons
    character(LEN=30)::cluster_evaporation_model='Montillaud2014' ! Model for the evaporation of PAH clusters into small PAHs

    ! ==== Rates and efficiency parameters (read from nml)====
    real(dp)::Sconstant=1.0d0   ! Sticking coefficient constant
    real(dp)::max_accretion_rate=1d-7 ! Maximum accretion rate coefficient [s-1] to prevent stiff ODE integration
    real(dp),dimension(1:ndchemtype)::nh_coa=0.1d0            ! Gas density above which dust coagulation is allowed (H/cm3)
    real(dp),dimension(1:ndchemtype)::nhmax_acc=1d4           ! Max gas density for accretion subgrid model
    real(dp),dimension(1:ndchemtype)::nhmax_coa=1d6           ! Max gas density for coagulation subgrid model
    real(dp),dimension(1:ndchemtype)::nhmax_sha=1d3           ! Max gas density for shattering subgrid model
    real(dp),dimension(1:ndchemtype)::dust_SNdest_eff=0.1d0   ! Dust SN destruction efficiency
    real(dp),dimension(1:ndchemtype)::dust_SNsha_eff=0.1d0    ! Shattering efficiency of large grains into small (and PAHs)
    real(dp),dimension(1:ndchemtype)::dust_SNII_cond_eff=0.1d0 ! SNII condensation efficiency
    real(dp),dimension(1:ndchemtype)::dust_SNIa_cond_eff=0.1d0 ! SNIa condensation efficiency
    real(dp),dimension(1:ndchemtype)::dust_AGB_cond_eff=0.1d0 ! AGB condensation efficiency
    real(dp),dimension(1:ndchemtype)::Coulomb_enhance=1d0     ! Enhancement of ion accretion due to dust grain charge (basic model)
    real(dp),dimension(1:ndchemtype)::tensile_strength=1d7    ! Dust tensile strength (erg/cm3)
    real(dp),dimension(1:ndchemtype)::Youngs_modulus=1d10   ! Dust Young's modulus (erg/cm3)
    real(dp),dimension(1:ndchemtype)::Poisson_ratio=0.25d0  ! Dust Poisson's ratio
    real(dp),dimension(1:ndchemtype)::surf_energy=25d0    ! Dust surface energy (erg/cm2)
    real(dp),dimension(1:ndchemtype)::work_function=8.0d0 ! Dust work function (eV)
    real(dp),dimension(1:ndchemtype)::band_gap=5.0d0      ! Dust band gap (eV)
    real(dp),dimension(1:ndchemtype)::e_escape_length=1.0d-7 ! Dust electron escape length (in cm)
    logical,dimension(1:ndchemtype) ::separate_refractive_index=.false. ! Whether the dust bin has separate refractive index tables for parallel and perpendicular waves
    real(dp)::slope_frag_func=1.3D0/3D0                  ! Fragment distribution (shattering and RATD) power-law slope
    real(dp)::errmax=0.1d0            ! Criterion for convergence in dust chemistry solver
    real(dp)::countmax=10000          ! Maximum number of iterations in dust chemistry solver
    real(dp)::GDinit=162d0            ! Initial gas-to-dust ratio (Def: 162 as given by Zubko et al. 2004)
    real(dp)::DTMinit=-1d0            ! Initial dust-to-metal ratio (Def: 1d-3)
    real(dp)::fpah_ini=0.1d0          ! Initial fraction of C locked in PAHs
    real(dp)::smallr_dust=1d-12       ! Minimum dust mass for a cell to be considered as a dust cell


    ! ==== Dust grain and PAHs bin properties (read from nml)====
    ! Dust composition is now provided per chemical type (not per dust bin).
    real(dp),dimension(1:ndchemtype,1:n_elements) ::dust_composition=0d0
    integer,dimension(1:ndchemtype) ::dustbins_per_chemtype=0
    integer,dimension(1:ndchemtype) ::istart_chemtype=0
    real(dp),dimension(1:ndust):: asize=0.1d0                           ! Grain size (in microns)
    real(dp),dimension(1:ndust):: sgrain=1d0                            ! Grain material density (in g/cm^3) divided by 3 g/cm^3
    real(dp),dimension(1:ndust):: amin=1d-2                             ! Minimum grain size of underlying distribution (in microns)
    real(dp),dimension(1:ndust):: amax=1d0                              ! Maximum grain size of underlying distribution (in microns)
    real(dp),dimension(1:ndust)::fmass_ej=0.0d0                         ! Fraction of the total dust mass in SN ejecta that is injected in each dust bin
    integer,dimension(1:npah)::pah_nc=0                             ! Number of carbon atoms in the PAH molecule
    integer,dimension(1:npah)::pah_nc_min=0                         ! Minimum number of carbon atoms in the PAH molecule
    integer,dimension(1:npah)::pah_nc_max=0                         ! Maximum number of carbon atoms in the PAH molecule
    real(dp),dimension(1:npah)::spah=2d0      ! PAH density (in g/cm^3) (Def: 2 g/cm^3 more appropiate for hydrocarbon)
    real(dp),dimension(1:npah)::pah_SNdest_eff=0.1d0   ! PAH SN destruction efficiency
    real(dp),dimension(1:npah)::fpah_inwind=0.5d0 ! Fraction of AGB wind PAH mass in each PAH size bin
    integer,dimension(1:npah)::pah_ncharge_states=4 ! Number of charge states for PAHs in charging calculations
    logical,dimension(1:npah)::pah_is_cluster=.false. ! Whether the PAH bin corresponds to a cluster of PAHs (Def: false, i.e. all bins correspond to single PAH molecules)
    
    ! ==== ISM depletion factors on dust (read from nml) ====
    ! These values are used for starting isolated sims and tests
    ! following the fractional contributions of the BARE-GR-S model
    ! from Zubko et al. (2004) - see Table 6
    ! (https://ui.adsabs.harvard.edu/abs/2004ApJS..152..211Z/abstract)
    ! and Dopita et al. (2000) - see Table 1 (N,Fe,Si,C)
    ! (https://ui.adsabs.harvard.edu/abs/2000ApJ...539..742D/abstract)
    real(dp),dimension(1:n_elements)::fDust_depletions=(/0d0,0d0,0d0,0d0,0d0,4.9881d-1,3.9744d-1,2.72d-1,&
                                                        0d0,0d0,0d0,8.37d-1,0d0,9.d-1,0d0,3.9744d-1,0d0,0d0,&
                                                        0d0,0d0,0d0,0d0,0d0,0d0,0d0,9.9d-1,0d0/)
    real(dp)::fCDust_inPAH=1.342d-1
    real(dp)::GD_solar=162d0 ! Gas-to-dust ratio in the solar neighbourhood (Def: 162 as given by Zubko et al. 2004)
    real(dp)::DTM_solar=0.458d0 ! Dust-to-metal ratio in the solar neighbourhood (Def: 0.458 as given by Zubko et al. 2004)
    real(dp),dimension(1:ndust)::fdustmass_ini=1d0/max(dble(ndust),1d0) ! Initial dust mass fraction in each dust bin (Def: same for every bin)
    real(dp),dimension(1:npah)::fpahmass_ini=1d0/max(dble(npah),1d0) ! Initial PAH mass fraction in each PAH bin (Def: same for every bin)


    ! ==== Radiation parameters (read from nml)====
    real(dp)::fixed_rad_ani=-1d0                                        ! Fixed radiation file anisotropy for tests of RATD
    real(dp)::fixed_lambda_mean=-1d0                                    ! Fixed mean radiation wavelength for tests of RATD (in microns)

    ! ==== Element parameters in the case of no RTZ module ====
#ifndef RTZ
    real(dp),dimension(1:n_elements),parameter :: el_atomic_masses_amu = (/1.00794d0, 4.002602d0, 6.941d0, 9.012182d0, &
                                                                10.811d0, 12.0107d0, 14.0067d0, 15.9994d0, 18.9984032d0, &
                                                                20.1797d0, 22.98976928d0, 24.3050d0, 26.9815386d0, &
                                                                28.0855d0, 30.973762d0, 32.065d0, 35.453d0, &
                                                                39.948d0, 39.0983d0, 40.078d0, 44.955910d0, &
                                                                47.867d0, 50.9415d0, 51.9961d0, 54.938044d0, &
                                                                55.845d0, 58.933195d0/)
    real(dp),dimension(1:n_elements),parameter :: el_atomic_masses_g = el_atomic_masses_amu * amu2g
    character(LEN=2),dimension(1:n_elements),parameter :: el_names = (/'H ','He','Li','Be','B ', &
                                                                'C ','N ','O ','F ','Ne', &
                                                                'Na','Mg','Al','Si', &
                                                                'P ','S ','Cl','Ar', &
                                                                'K ','Ca','Sc','Ti', &
                                                                'V ','Cr','Mn','Fe', &
                                                                'Co'/)
#endif

    ! ==== Global dust and PAH bin properties ====
    type(DustBin),dimension(1:ndust) ::dustbins_props
    type(PAHBin),dimension(1:npah) ::pahbins_props
    real(dp),dimension(:,:),allocatable ::group_csa_dust,group_css_dust,group_csr_dust
    real(dp),dimension(:,:),allocatable ::group_csrat_dust
    real(dp),dimension(:,:),allocatable ::group_csa_pah,group_css_pah,group_csr_pah
    real(dp),dimension(:,:),allocatable ::sigca_dust,sigcs_dust,sigcr_dust,sigcrat_dust
    real(dp),dimension(:,:),allocatable ::sigca_pah,sigcs_pah,sigcr_pah
    real(dp),dimension(:,:),allocatable ::att_len_dust


    ! ==== Coefficients of the polynomial fit from Hu+19 ====
    real(dp),dimension(1:6)::aCth=(/-2.34333937d2,1.38485732d2,-3.39021615d1,&
                                    & 4.17705353d0,-2.58281473d-1,6.38827523d-3/)
    real(dp),dimension(1:6)::aSith=(/-2.34790500d2,1.33208637d2,-3.13027448d1,&
                                    & 3.71345730d0,-2.21823668d-1,5.31746427d-3/)


    ! ==== Global counters ====
    ! SN destruction/seeding mass changes (accumulated in dust_dynamics.f90)
    real(dp),dimension(1:ndust+npah):: dM_SNIId = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIId_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIad = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIad_all = 0.0d0
    ! Per-process ODE mass change [g cm-3] per bin, accumulated over all cells per coarse step.
    ! Indexed as dM_ode_dust(ispecies, iprocess) and dM_ode_pah(ispecies, jpahprocess).
    ! Allocated after init_dust_processes is called (see dust_init.f90).
    real(dp), dimension(:,:), allocatable :: dM_ode_dust, dM_ode_dust_all
    real(dp), dimension(:,:), allocatable :: dM_ode_pah,  dM_ode_pah_all
    ! We also track the cell count for the dust chemistry solver
    integer*8::ndust_cells=0,ndust_cells_all=0
    ! Track the total masses
    real(dp)::total_gas_mass=0d0,total_gas_mass_all=0d0
    real(dp)::total_dust_mass=0d0,total_dust_mass_all=0d0
    real(dp)::total_mass_test=0d0,total_mass_test_all=0d0
    real(dp)::total_CO_mass=0d0,total_CO_mass_all=0d0
    real(dp),dimension(1:n_elements)::total_metal_mass=0d0,total_metal_mass_all=0d0
    real(dp),dimension(1:ndust+npah)::total_dust_mass_species=0d0,total_dust_mass_species_all=0d0

    ! ==== Internal flags and variables ====
    integer::ncharge_pah_max=0                      ! Maximum number of PAH charge states across all PAH bins (for charging calculations)
    type(DustChemistryInfo)::dust_helper  ! Reusable per-rank dust chemistry workspace
    type(DustProcess),dimension(:),allocatable::dust_processes_list ! List of the DustProcess types to use in the dust chemistry solver
    type(DustProcess),dimension(:),allocatable::pah_processes_list ! List of the DustProcess types to use in the PAH chemistry solver
    integer::ndust_processes=0                     ! Number of dust processes activated (length of dust_processes_list)
    integer::npah_processes=0                      ! Number of PAH processes activated (length of pah_processes_list)
    logical::Coulomb_precompute=.false.   ! whether to precompute the Coulomb focusing factor at beginning of dust_fine
    logical::comp_sigma_turb=.false.            ! Activate the computation of turbulent velocity dispersion
    logical::carry_gas_ions=.false.       ! Whether to carry the individual ion densities for gas species


    ! ==== Some internal constants ====
    ! Mathis et al. (1983) ISRF energy density in erg/cm3
    ! This is obtained using the CALIMA python library
    ! using the parametrisation of the ISRF from Mathis et al.
    ! (1983) as described in Eq. 31 of Weingartner & Draine (2001)
    ! and integrated from 0.1-13.6 eV
    real(dp),parameter::u_Mathis1983=8.635471d-13 ! [erg/cm3]
    real(dp),parameter::Td_max = 1d5 ! [K]

    ! ==== External dust files ====
    character(LEN=256)::dust_tables_dir='../lib/dust_tables/'    ! Name of folder holding pre-computed dust tables (extinction, charging, etc.)

    integer*8 :: tdust_solver_calls=0
    integer*8 :: tdust_solver_iter_sum=0
    integer*8 :: tdust_solver_iter_min=huge(0_8)
    integer*8 :: tdust_solver_iter_max=0
    integer*8 :: tdust_solver_brent_calls=0
    integer*8 :: tdust_solver_calls_all=0
    integer*8 :: tdust_solver_iter_sum_all=0
    integer*8 :: tdust_solver_iter_min_all=huge(0_8)
    integer*8 :: tdust_solver_iter_max_all=0
    integer*8 :: tdust_solver_brent_calls_all=0

    ! Per-dust-bin temperature statistics for tdust_solver log
    real(dp), dimension(1:ndust) :: tdust_min = huge(0d0)
    real(dp), dimension(1:ndust) :: tdust_max = -huge(0d0)
    real(dp), dimension(1:ndust) :: tdust_sum = 0d0
    integer*8, dimension(1:ndust) :: tdust_bins_calls = 0_8

    real(dp), dimension(1:ndust) :: tdust_min_all = huge(0d0)
    real(dp), dimension(1:ndust) :: tdust_max_all = -huge(0d0)
    real(dp), dimension(1:ndust) :: tdust_sum_all = 0d0
    integer*8, dimension(1:ndust) :: tdust_bins_calls_all = 0_8
    ! ODE driver acceptance/rejection statistics (summed over all cells per coarse step)
    integer*8 :: ode_naccepted=0, ode_nrejected=0, ode_nreduced=0
    integer*8 :: ode_naccepted_all=0, ode_nrejected_all=0, ode_nreduced_all=0
    ! ODE per-cell substep counts: min/max/sum of naccepted substeps per cell integration
    integer*8 :: ode_substeps_sum=0, ode_substeps_min=huge(0_8), ode_substeps_max=0
    integer*8 :: ode_substeps_sum_all=0, ode_substeps_min_all=huge(0_8), ode_substeps_max_all=0
    ! Per-process ODE timestep reduction attributions (indexed by process in dust/pah lists)
    integer*8, dimension(:), allocatable :: ode_reduction_count_dust
    integer*8, dimension(:), allocatable :: ode_reduction_count_pah
    integer*8, dimension(:), allocatable :: ode_reduction_count_dust_all
    integer*8, dimension(:), allocatable :: ode_reduction_count_pah_all

    real(dp), parameter :: y_min = 1.0d-60
    !! minimum of mass density of gas, dust, and PAH

    contains

    logical function draine_drag()
        ! drag_model is one of the Draine (2011) drag laws
        draine_drag = trim(drag_model) == 'draine2011' .or. trim(drag_model) == 'draine2011_pz'
    end function draine_drag

    subroutine dust_log_tdust_solver_update(i_dust, T, n_iter, used_brent)
        implicit none
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: T
        integer, intent(in) :: n_iter
        logical, intent(in) :: used_brent
        integer*8 :: n_iter_i8

        if (.not. dust_log) return

        n_iter_i8 = int(max(n_iter,0), kind=8)

        tdust_solver_calls = tdust_solver_calls + 1_8
        tdust_solver_iter_sum = tdust_solver_iter_sum + n_iter_i8
        tdust_solver_iter_min = min(tdust_solver_iter_min, n_iter_i8)
        tdust_solver_iter_max = max(tdust_solver_iter_max, n_iter_i8)
        if (used_brent) tdust_solver_brent_calls = tdust_solver_brent_calls + 1_8

        if (i_dust >= 1 .and. i_dust <= ndust) then
            tdust_min(i_dust) = min(tdust_min(i_dust), T)
            tdust_max(i_dust) = max(tdust_max(i_dust), T)
            tdust_sum(i_dust) = tdust_sum(i_dust) + T
            tdust_bins_calls(i_dust) = tdust_bins_calls(i_dust) + 1_8
        end if
    end subroutine dust_log_tdust_solver_update

    subroutine dust_log_tdust_solver_print_reset
        implicit none
        real(dp) :: avg_iter
        integer :: ii

        if (.not. dust_log) return

        if (tdust_solver_calls > 0_8) then
            avg_iter = real(tdust_solver_iter_sum, dp) / real(tdust_solver_calls, dp)
            write(*,'(A,I0,A,I0,A,F10.3,A,I0)') 'Tdust solver stats: min_iter=', &
                int(tdust_solver_iter_min), ', max_iter=', int(tdust_solver_iter_max), &
                ', avg_iter=', avg_iter, ', brent_calls=', int(tdust_solver_brent_calls)
            write(*,*) 'Tdust solver stats per dust bin [K]:'
            do ii = 1, ndust
                if (tdust_bins_calls(ii) > 0_8) then
                    write(*,'(A,I0,A,ES13.6,A,ES13.6,A,ES13.6)') &
                        '  bin ', ii, ': min=', tdust_min(ii), &
                        ', max=', tdust_max(ii), &
                        ', avg=', tdust_sum(ii)/real(tdust_bins_calls(ii), dp)
                else
                    write(*,'(A,I0,A)') '  bin ', ii, ': no calls'
                end if
            end do
        else
            write(*,'(A)') 'Tdust solver stats: no calls in this equilibrium iteration.'
        end if

        tdust_solver_calls = 0_8
        tdust_solver_iter_sum = 0_8
        tdust_solver_iter_min = huge(0_8)
        tdust_solver_iter_max = 0_8
        tdust_solver_brent_calls = 0_8

        tdust_min = huge(0d0)
        tdust_max = -huge(0d0)
        tdust_sum = 0d0
        tdust_bins_calls = 0_8
    end subroutine dust_log_tdust_solver_print_reset

    subroutine add_total_masses
        use amr_commons
        use hydro_commons
#ifdef RTZ
        use rtz_module, only: elements
#endif
        implicit none
        integer::ilevel
        integer::i,ivar,ind,iskip
        integer::ii,icell,igrid,e_counter
        integer::nx_loc,ncache,ngrid
        integer,dimension(1:nvector)::ind_grid,ind_cell
        real(dp)::dx,dx_loc,scale

        nx_loc=(icoarse_max-icoarse_min+1)
        scale=boxlen/dble(nx_loc)

        do ilevel=1,nlevelmax
            dx = 0.5d0**ilevel
            dx_loc=dx*scale
            ncache=active(ilevel)%ngrid
            ! Loop over active grids by vector sweeps
            do igrid=1,ncache,nvector
               ngrid=MIN(nvector,ncache-igrid+1)
               do i=1,ngrid
                  ind_grid(i)=active(ilevel)%igrid(igrid+i-1)
               end do
               ! Loop over cells
               do ind=1,twotondim
                  ! Gather cell indices
                  iskip=ncoarse+(ind-1)*ngridmax
                  do i=1,ngrid
                     ind_cell(i)=iskip+ind_grid(i)
                  end do
                  ! Check if cell is a leaf cell
                  do i=1,ngrid
                     if (son(ind_cell(i))==0) then
                        ! Add total gas mass
                        total_gas_mass = total_gas_mass + (uold(ind_cell(i),1) * dx_loc**3)
                        ! Add total dust mass
                        total_dust_mass = total_dust_mass + (sum(uold(ind_cell(i),idust:idust+ndust-1)) * dx_loc**3)
                        ! Add individual dust species
                        do ii=1,npah
                            total_dust_mass_species(ii) = total_dust_mass_species(ii) + (uold(ind_cell(i),ipah+ii-1) * dx_loc**3)
                        end do
                        do ii=1,ndust
                            total_dust_mass_species(npah+ii) = total_dust_mass_species(npah+ii) + (uold(ind_cell(i),idust+ii-1) * dx_loc**3)
                        end do
                        ! Add total metal mass. total_metal_mass is dimensioned over
                        ! n_elements, but the uold block at imetal is PACKED over the
                        ! elements(:) entries with atomic_number > 0, so the two must be
                        ! walked with separate counters.
                        if (metal) then
#ifdef RTZ
                            e_counter = 0
                            do ii=1,n_elements
                                if (elements(ii)%atomic_number > 0) then
                                    total_metal_mass(ii) = total_metal_mass(ii) + &
                                        (uold(ind_cell(i),imetal+e_counter) * dx_loc**3)
                                    e_counter = e_counter + 1
                                end if
                            end do
#else
                            do ii=1,nmetals
                                total_metal_mass(ii) = total_metal_mass(ii) + (uold(ind_cell(i),imetal+ii-1) * dx_loc**3)
                            end do
#endif
                        end if
                        ! Add total CO mass
#ifdef CO
                        total_CO_mass = total_CO_mass + (uold(ind_cell(i),ico) * dx_loc**3)
#endif
                    end if
                  end do
               end do
            end do
         end do
        
    end subroutine add_total_masses

    subroutine print_total_masses(myid,tcurrent,scale_factor)
        use constants
        use amr_parameters, only:cosmo
        use hydro_parameters, only:nmetals
        use mpi_mod
#ifdef RTZ
        use rtz_module, only: elements
#endif
        implicit none
        integer,intent(in) :: myid
        real(dp),intent(in) :: tcurrent,scale_factor
        real(dp) :: scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2,scale_msun
        integer :: ii
        character(len=2048) :: line
        character(len=32) :: buf
#ifndef WITHOUTMPI
        integer ::mpi_err
#endif
        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
        scale_msun = scale_l**3*scale_d/M_sun
#ifndef WITHOUTMPI
        ! 1. Add up all the masses from all CPUs
        call MPI_ALLREDUCE(total_gas_mass,total_gas_mass_all,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        call MPI_ALLREDUCE(total_dust_mass,total_dust_mass_all,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        call MPI_ALLREDUCE(total_dust_mass_species,total_dust_mass_species_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        call MPI_ALLREDUCE(total_metal_mass,total_metal_mass_all,N_ELEMENTS,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        call MPI_ALLREDUCE(total_CO_mass,total_CO_mass_all,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        total_gas_mass = total_gas_mass_all * scale_msun
        total_dust_mass = total_dust_mass_all * scale_msun
        total_dust_mass_species = total_dust_mass_species_all * scale_msun
        total_metal_mass = total_metal_mass_all * scale_msun
        total_CO_mass = total_CO_mass_all * scale_msun
#endif
        ! 2. Print the total masses. The line is assembled from elements(:) and from the
        ! actual bin counts, so it adapts to whatever element and dust/PAH set is
        ! compiled in rather than assuming a fixed 7-element, 6-bin layout.
        if(myid==1)then
            if (cosmo) then
                write(line,'("aexp:",e13.6,", Gas=",e13.6)') scale_factor, total_gas_mass
            else
                write(line,'("t:",e13.6,", Gas=",e13.6)') tcurrent*scale_t / Myr2sec, total_gas_mass
            end if
            ! Gas-phase metals. total_metal_mass is indexed by atomic number.
#ifdef RTZ
            do ii = 1, n_elements
                if (elements(ii)%atomic_number > 0) then
                    write(buf,'(1x,A,"=",e13.6)') trim(elements(ii)%symbol), total_metal_mass(ii)
                    line = trim(line)//trim(buf)
                end if
            end do
#else
            ! Without RTZ the metal block carries no element identity, so label by slot.
            do ii = 1, nmetals
                write(buf,'(1x,"Z",i0,"=",e13.6)') ii, total_metal_mass(ii)
                line = trim(line)//trim(buf)
            end do
#endif
            ! PAH bins, then dust bins, in the order they are stored
            do ii = 1, npah
                write(buf,'(1x,"PAH",i0,"=",e13.6)') ii, total_dust_mass_species(ii)
                line = trim(line)//trim(buf)
            end do
            do ii = 1, ndust
                write(buf,'(1x,"Dust",i0,"=",e13.6)') ii, total_dust_mass_species(npah+ii)
                line = trim(line)//trim(buf)
            end do
            write(buf,'(1x,"CO=",e13.6)') total_CO_mass
            write(*,'(A)') trim(line)//trim(buf)
        end if

        ! 3. Set the total masses to zero
        total_gas_mass = 0d0; total_gas_mass_all = 0d0
        total_dust_mass = 0d0; total_dust_mass_all = 0d0
        total_dust_mass_species = 0d0; total_dust_mass_species_all = 0d0
        total_metal_mass = 0d0; total_metal_mass_all = 0d0
        total_mass_test = 0d0; total_mass_test_all = 0d0
        total_CO_mass = 0d0; total_CO_mass_all = 0d0

    end subroutine print_total_masses

    subroutine print_dust_log(myid,dt,tcurrent,scale_factor)
        use constants
        use amr_parameters, only:cosmo
        use mpi_mod
        implicit none
        integer,intent(in) :: myid
        real(dp),intent(in) :: dt,tcurrent,scale_factor
        real(dp) :: scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2
        character(len=30) :: format_str
        real(dp) :: tdust_avg_iter
        integer :: ii
#ifndef WITHOUTMPI
        integer ::mpi_err
#endif
        ! 1. Get the code units
        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)

        ! 2. If MPI, get the cell count from all CPUs
#ifndef WITHOUTMPI
        call MPI_ALLREDUCE(ndust_cells, ndust_cells_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        ndust_cells = ndust_cells_all
#endif
#ifndef WITHOUTMPI
        ! 3. If MPI, reduce SN mass changes and (when dust_debug) ODE per-process mass changes
        call MPI_ALLREDUCE(dM_SNIId,dM_SNIId_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_SNIId=dM_SNIId_all
        call MPI_ALLREDUCE(dM_SNIad,dM_SNIad_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_SNIad=dM_SNIad_all
        if (dust_debug) then
            if (ndust_processes > 0 .and. allocated(dM_ode_dust)) then
                if (.not. allocated(dM_ode_dust_all)) &
                    allocate(dM_ode_dust_all(ndust+npah, ndust_processes))
                call MPI_ALLREDUCE(dM_ode_dust, dM_ode_dust_all, (ndust+npah)*ndust_processes, &
                    MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
                dM_ode_dust = dM_ode_dust_all
            end if
            if (npah_processes > 0 .and. allocated(dM_ode_pah)) then
                if (.not. allocated(dM_ode_pah_all)) &
                    allocate(dM_ode_pah_all(ndust+npah, npah_processes))
                call MPI_ALLREDUCE(dM_ode_pah, dM_ode_pah_all, (ndust+npah)*npah_processes, &
                    MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
                dM_ode_pah = dM_ode_pah_all
            end if
        end if
#endif
#ifndef WITHOUTMPI
        ! 4. Reduce Tdust solver and ODE acceptance/rejection stats across all CPUs
        call MPI_ALLREDUCE(tdust_solver_calls, tdust_solver_calls_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_solver_iter_sum, tdust_solver_iter_sum_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_solver_iter_min, tdust_solver_iter_min_all, 1, MPI_INTEGER8, MPI_MIN, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_solver_iter_max, tdust_solver_iter_max_all, 1, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_solver_brent_calls, tdust_solver_brent_calls_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        tdust_solver_calls = tdust_solver_calls_all
        tdust_solver_iter_sum = tdust_solver_iter_sum_all
        tdust_solver_iter_min = tdust_solver_iter_min_all
        tdust_solver_iter_max = tdust_solver_iter_max_all
        tdust_solver_brent_calls = tdust_solver_brent_calls_all
        call MPI_ALLREDUCE(ode_naccepted, ode_naccepted_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ode_nrejected, ode_nrejected_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ode_nreduced, ode_nreduced_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        ode_naccepted = ode_naccepted_all
        ode_nrejected = ode_nrejected_all
        ode_nreduced  = ode_nreduced_all
        call MPI_ALLREDUCE(ode_substeps_sum, ode_substeps_sum_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ode_substeps_min, ode_substeps_min_all, 1, MPI_INTEGER8, MPI_MIN, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ode_substeps_max, ode_substeps_max_all, 1, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, mpi_err)
        ode_substeps_sum = ode_substeps_sum_all
        ode_substeps_min = ode_substeps_min_all
        ode_substeps_max = ode_substeps_max_all
        if (ndust_processes > 0 .and. allocated(ode_reduction_count_dust)) then
            if (.not. allocated(ode_reduction_count_dust_all)) &
                allocate(ode_reduction_count_dust_all(ndust_processes))
            call MPI_ALLREDUCE(ode_reduction_count_dust, ode_reduction_count_dust_all, &
                ndust_processes, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
            ode_reduction_count_dust = ode_reduction_count_dust_all
        end if
        if (npah_processes > 0 .and. allocated(ode_reduction_count_pah)) then
            if (.not. allocated(ode_reduction_count_pah_all)) &
                allocate(ode_reduction_count_pah_all(npah_processes))
            call MPI_ALLREDUCE(ode_reduction_count_pah, ode_reduction_count_pah_all, &
                npah_processes, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
            ode_reduction_count_pah = ode_reduction_count_pah_all
        end if
        call MPI_ALLREDUCE(tdust_min, tdust_min_all, ndust, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_max, tdust_max_all, ndust, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_sum, tdust_sum_all, ndust, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(tdust_bins_calls, tdust_bins_calls_all, ndust, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        tdust_min = tdust_min_all
        tdust_max = tdust_max_all
        tdust_sum = tdust_sum_all
        tdust_bins_calls = tdust_bins_calls_all
#endif
        ! 5. Construct the format string
        write(format_str, '(A, I0, A)') '(A,', ndust + npah, 'ES14.6)'
        if (myid==1) then
            ! 5. Print header line with time information
            if (cosmo) then
                write(*,'(A,ES13.6,A,ES13.6)') &
                    ' === CALIMA dust step === aexp=', scale_factor, &
                    ', dt[Myr]=', dt*scale_t/Myr2sec
            else
                write(*,'(A,ES13.6,A,ES13.6)') &
                    ' === CALIMA dust step === t[Myr]=', tcurrent*scale_t/Myr2sec, &
                    ', dt[Myr]=', dt*scale_t/Myr2sec
            end if
            ! 6. Print ODE solver statistics
            write(*,*) ' --- ODE solver ---'
            write(*,'(A,I12,A,I12,A,I12,A,I12)') &
                '  Cells=', ndust_cells, &
                '  Acc=',   ode_naccepted, &
                '  Rej=',   ode_nrejected, &
                '  Red=',   ode_nreduced
            if (ode_naccepted + ode_nrejected > 0_8) then
                write(*,'(A,F7.2,A,F8.2)') &
                    '  Rejection rate=', &
                    1d2*dble(ode_nrejected)/dble(ode_naccepted+ode_nrejected), &
                    '%, avg substeps/cell=', &
                    dble(ode_naccepted)/dble(max(1_8,ndust_cells))
            end if
            if (ndust_cells > 0_8) then
                write(*,'(A,I12,A,F8.2,A,I12)') &
                    '  Substeps per cell: min=', ode_substeps_min, &
                    ', avg=', dble(ode_substeps_sum)/dble(ndust_cells), &
                    ', max=', ode_substeps_max
            end if
            if (ode_nreduced > 0_8) then
                write(*,*) '  Timestep reduction fraction per process:'
                if (ndust_processes > 0 .and. allocated(ode_reduction_count_dust)) then
                    do ii = 1, ndust_processes
                        write(*,'(A,A,A,F7.2,A,I12,A)') '    dust: ', &
                            trim(dust_processes_list(ii)%name), ' = ', &
                            1d2*dble(ode_reduction_count_dust(ii))/dble(ode_nreduced), &
                            '% (', ode_reduction_count_dust(ii), ')'
                    end do
                end if
                if (npah_processes > 0 .and. allocated(ode_reduction_count_pah)) then
                    do ii = 1, npah_processes
                        write(*,'(A,A,A,F7.2,A,I12,A)') '    pah:  ', &
                            trim(pah_processes_list(ii)%name), ' = ', &
                            1d2*dble(ode_reduction_count_pah(ii))/dble(ode_nreduced), &
                            '% (', ode_reduction_count_pah(ii), ')'
                    end do
                end if
            end if
            ! 7. Print Tdust solver statistics
            write(*,*) ' --- Tdust solver ---'
            if (tdust_solver_calls > 0_8) then
                tdust_avg_iter = real(tdust_solver_iter_sum, dp) / real(tdust_solver_calls, dp)
                write(*,'(A,F8.3,A,I0,A,I0,A,I0,A,F5.1,A)') &
                    'avg_iter=', tdust_avg_iter, &
                    ', min=', tdust_solver_iter_min, &
                    ', max=', tdust_solver_iter_max, &
                    ', brent=', tdust_solver_brent_calls, &
                    ' (', 1d2*dble(tdust_solver_brent_calls)/dble(tdust_solver_calls), '%)'
                write(*,*) '  Per-bin Tdust stats [K]:'
                do ii = 1, ndust
                    if (tdust_bins_calls(ii) > 0_8) then
                        write(*,'(A,I0,A,ES13.6,A,ES13.6,A,ES13.6)') &
                            '    bin ', ii, ': min=', tdust_min(ii), &
                            ', max=', tdust_max(ii), &
                            ', avg=', tdust_sum(ii)/real(tdust_bins_calls(ii), dp)
                    else
                        write(*,'(A,I0,A)') '    bin ', ii, ': no calls'
                    end if
                end do
            else
                write(*,*) '  No Tdust solver calls this step.'
            end if
            ! 8. Print per-process ODE mass change rates [Msun/yr per bin] (dust_debug only)
            if (dust_debug) then
                write(*,*) ' --- ODE process dM/dt [Msun/yr per bin] ---'
                if (ndust_processes > 0 .and. allocated(dM_ode_dust)) then
                    do ii = 1, ndust_processes
                        write(*,format_str) 'dust '//trim(dust_processes_list(ii)%name)//' =', &
                            dM_ode_dust(:, ii) / (dt*scale_t) / M_sun * yr2sec
                    end do
                end if
                if (npah_processes > 0 .and. allocated(dM_ode_pah)) then
                    do ii = 1, npah_processes
                        write(*,format_str) 'pah  '//trim(pah_processes_list(ii)%name)//' =', &
                            dM_ode_pah(:, ii) / (dt*scale_t) / M_sun * yr2sec
                    end do
                end if
            end if
            ! 9. Print SN destruction statistics
            if (dust_SNdest) then
                write(*,*) ' --- SN dust statistics ---'
                write(*,format_str) 'dM SNd  (II)  =', dM_SNIId/(dt*scale_t) / M_sun * yr2sec
                write(*,format_str) 'dM SNd  (Ia)  =', dM_SNIad/(dt*scale_t) / M_sun * yr2sec
            end if
        endif

        ! Print total dust and PAH masses in simulation box
        call print_box_dust_masses()

        dM_SNIId          = 0.0d0; dM_SNIId_all          = 0.0d0
        dM_SNIad          = 0.0d0; dM_SNIad_all          = 0.0d0
        if (dust_debug) then
            if (allocated(dM_ode_dust))     dM_ode_dust     = 0.0d0
            if (allocated(dM_ode_dust_all)) dM_ode_dust_all = 0.0d0
            if (allocated(dM_ode_pah))      dM_ode_pah      = 0.0d0
            if (allocated(dM_ode_pah_all))  dM_ode_pah_all  = 0.0d0
        end if
        ndust_cells       = 0;     ndust_cells_all       = 0
        tdust_solver_calls = 0_8;       tdust_solver_calls_all = 0_8
        tdust_solver_iter_sum = 0_8;    tdust_solver_iter_sum_all = 0_8
        tdust_solver_iter_min = huge(0_8); tdust_solver_iter_min_all = huge(0_8)
        tdust_solver_iter_max = 0_8;    tdust_solver_iter_max_all = 0_8
        tdust_solver_brent_calls = 0_8; tdust_solver_brent_calls_all = 0_8
        ode_naccepted = 0_8; ode_naccepted_all = 0_8
        ode_nrejected = 0_8; ode_nrejected_all = 0_8
        ode_nreduced  = 0_8; ode_nreduced_all  = 0_8
        ode_substeps_sum = 0_8;       ode_substeps_sum_all = 0_8
        ode_substeps_min = huge(0_8); ode_substeps_min_all = huge(0_8)
        ode_substeps_max = 0_8;       ode_substeps_max_all = 0_8
        if (allocated(ode_reduction_count_dust))     ode_reduction_count_dust     = 0_8
        if (allocated(ode_reduction_count_dust_all)) ode_reduction_count_dust_all = 0_8
        if (allocated(ode_reduction_count_pah))      ode_reduction_count_pah      = 0_8
        if (allocated(ode_reduction_count_pah_all))  ode_reduction_count_pah_all  = 0_8
        tdust_min = huge(0d0); tdust_min_all = huge(0d0)
        tdust_max = -huge(0d0); tdust_max_all = -huge(0d0)
        tdust_sum = 0d0; tdust_sum_all = 0d0
        tdust_bins_calls = 0_8; tdust_bins_calls_all = 0_8
    end subroutine print_dust_log

    subroutine print_box_dust_masses()
        use amr_commons
        use hydro_commons, only: uold, imetal,nmetals
        use constants, only: M_sun
#ifdef RTZ
        use rtz_module, only: elements
#endif
#ifndef WITHOUTMPI
        use mpi_mod
#endif
        implicit none
        integer :: ilevel, i, ind, iskip, ii, icell, igrid, nx_loc, ncache, ngrid, counter
        integer, dimension(1:nvector) :: ind_grid, ind_cell
        real(dp) :: dx, dx_loc, scale, scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2, scale_msun
        real(dp), dimension(1:ndust+npah) :: local_bin_masses, global_bin_masses
        real(dp) :: local_gas_mass, local_metal_mass, global_gas_mass, global_metal_mass
        logical, dimension(1:nmetals) :: metal_slot
#ifndef WITHOUTMPI
        integer :: mpi_err
#endif

        ! 1. Initialize local bin masses
        local_bin_masses = 0d0
        global_bin_masses = 0d0
        local_gas_mass = 0d0
        local_metal_mass = 0d0
        global_gas_mass = 0d0
        global_metal_mass = 0d0

        ! The element block at imetal is packed in the order of elements(:) with
        ! atomic_number > 0, so it starts with H (and He, when helium is tracked).
        ! Those are not metals and must be left out of the metal mass.
#ifdef RTZ
        metal_slot = .false.
        counter = 0
        do ii = 1, n_elements
            if (elements(ii)%atomic_number > 0) then
                counter = counter + 1
                if (counter > nmetals) exit
                metal_slot(counter) = (elements(ii)%atomic_number > 2)
            end if
        end do
#else
        ! Without RTZ, imetal is the plain metallicity scalar block: all of it is metal.
        metal_slot = .true.
#endif

        ! 2. Compute local grid leaf cell contributions
        nx_loc = (icoarse_max - icoarse_min + 1)
        scale = boxlen / dble(nx_loc)

        do ilevel = 1, nlevelmax
            dx = 0.5d0**ilevel
            dx_loc = dx * scale
            ncache = active(ilevel)%ngrid
            do igrid = 1, ncache, nvector
                ngrid = min(nvector, ncache - igrid + 1)
                do i = 1, ngrid
                    ind_grid(i) = active(ilevel)%igrid(igrid + i - 1)
                end do
                do ind = 1, twotondim
                    iskip = ncoarse + (ind - 1) * ngridmax
                    do i = 1, ngrid
                        ind_cell(i) = iskip + ind_grid(i)
                    end do
                    do i = 1, ngrid
                        if (son(ind_cell(i)) == 0) then
                            local_gas_mass = local_gas_mass + (uold(ind_cell(i), 1) * dx_loc**3)
                            ! Accumulate metal mass (H and He are skipped, see metal_slot)
                            do ii = 1, nmetals
                                if (metal_slot(ii)) local_metal_mass = local_metal_mass + &
                                    (uold(ind_cell(i), imetal + ii - 1) * dx_loc**3)
                            end do
                            ! Accumulate PAH bin masses
                            do ii = 1, npah
                                local_bin_masses(ii) = local_bin_masses(ii) + &
                                    (uold(ind_cell(i), ipah + ii - 1) * dx_loc**3)
                            end do
                            ! Accumulate dust bin masses
                            do ii = 1, ndust
                                local_bin_masses(npah + ii) = local_bin_masses(npah + ii) + &
                                    (uold(ind_cell(i), idust + ii - 1) * dx_loc**3)
                            end do
                        end if
                    end do
                end do
            end do
        end do

        ! 3. Reduce across MPI processors and convert to solar masses
        call units(scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)
        scale_msun = (scale_l**3 * scale_d) / M_sun

#ifndef WITHOUTMPI
        call MPI_ALLREDUCE(local_bin_masses, global_bin_masses, ndust + npah, &
            MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(local_gas_mass, global_gas_mass, 1, &
            MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(local_metal_mass, global_metal_mass, 1, &
            MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, mpi_err)
#else
        global_bin_masses = local_bin_masses
#endif
        global_bin_masses = global_bin_masses * scale_msun
        global_gas_mass = global_gas_mass * scale_msun
        global_metal_mass = global_metal_mass * scale_msun

        ! 4. Print the diagnostics on task 1
        if (myid == 1) then
            write(*,*) ' --- Total dust / PAH bin masses in box [Msun] ---'
            do ii = 1, npah
                write(*, '(A, I2, A, ES14.6)') '  PAH bin ', ii, ': ', global_bin_masses(ii)
            end do
            do ii = 1, ndust
                write(*, '(A, I2, A, ES14.6)') '  Dust bin ', ii, ': ', global_bin_masses(npah + ii)
            end do
            write(*, '(A, ES14.6)') '  Total dust+PAH mass in box [Msun]: ', sum(global_bin_masses)
            write(*, '(A, ES14.6)') '  Total gas mass in box [Msun]: ', global_gas_mass
            write(*, '(A, ES14.6)') '  Total gas-phase metal mass in box [Msun]: ', global_metal_mass
            write(*, '(A, ES14.6)') '  Total dust-to-gas ratio in the box: ', sum(global_bin_masses) / global_gas_mass
            ! Metals locked in grains have left the gas phase, so they belong in the
            ! denominator: this is the DTM convention DTM_solar = 0.458 refers to.
            write(*, '(A, ES14.6)') '  Total dust-to-metal ratio in the box: ', &
                sum(global_bin_masses) / max(global_metal_mass + sum(global_bin_masses), tiny(1d0))
        end if
    end subroutine print_box_dust_masses

end module