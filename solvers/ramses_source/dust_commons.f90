! Dust commons.
! For details, see Dubois et al. 2022
! By: Yohan Dubois (Original: 1 Feb 2022)
! Changes:
!       - Curro Rodriguez (Cleaning into external 
!                           module: 21 Feb 2022)

module dust_commons
    use amr_parameters, only:dp
    use hydro_parameters, only:n_elements,ndust,ndchemtype,npah,idust,ipah
    use dust_utils
    use dustbin_types

    implicit none

    ! ==== Flags and logicals (read from nml) ====
    logical, parameter ::dust=.true.             ! CALIMA always includes dust
    logical ::dust_log=.false.                   ! Activate dust logging
    logical ::dust_10percent=.true.              ! Activate the 10% rule for the chemistry solver
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
    logical ::dust_acc_coulomb=.false.           ! Compute on-the-fly Coulomb enhancement of refractory material accretion
    logical ::dust_ratd=.false.                  ! Activate destruction of dust grains by RATD
    logical ::dust_coll_cooling=.false.          ! Activate dust collisional cooling
    logical ::dust_coll_lowT=.false.             ! Activate low-temperature dust collisional heating (Hollenbach & McKee 1980)
    logical ::dust_coll_charge=.false.           ! Activate the dependence of dust collisional cooling on grain and ion charge
    logical ::dust_pe_heating=.false.            ! Activate photo-electric heating by dust grains
    logical ::dust_pe_heating_isrf=.false.       ! Activate the simple dust PE heating based on an averaged ISRF G0
    logical ::ratd_only_rtadv=.false.            ! Only allow for RATD if the rt_advect=.true.
    logical ::poppe_ice_enhancement=.false.      ! Whether to use the empirical enhancement in coagulation threshold due to ice mantel
    logical ::H2ondust=.false.                   ! Activate H2 formation on dust
    logical, parameter ::dust_pahs=.true.       ! CALIMA always includes PAHs
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

    ! ==== Dust modelling options (read from nml) ====
    character(LEN=30)::sputtering_model='Tsai1998'    ! Thermal sputtering law (Tsai&Matthews 1995)
    character(LEN=30)::accretion_model='Chaabouni2012'    ! Accretion model
    character(LEN=30)::shattering_model='Granato2021' ! Model for the shattering dispersion velocity
    character(LEN=30)::coagulation_model='Aoyama2017' ! Model for the coagulation dispersion velocity
    character(LEN=30)::dust_velocity_model='Ormel2007' ! Model for the relative velocity of grains
    character(LEN=30)::charging_model='Ibanez2019'     ! Model for the grain charge distribution
    integer :: nZmix=3                                  ! Number of representative charge points (1: mean, 2: two-point, 3: three-point)

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
    real(dp),dimension(1:n_elements) :: el_atomic_masses_amu = (/1.00794d0, 4.002602d0, 6.941d0, 9.012182d0, &
                                                                10.811d0, 12.0107d0, 14.0067d0, 15.9994d0, 18.9984032d0, &
                                                                20.1797d0, 22.98976928d0, 24.3050d0, 26.9815386d0, &
                                                                28.0855d0, 30.973762d0, 32.065d0, 35.453d0, &
                                                                39.948d0, 39.0983d0, 40.078d0, 44.955910d0, &
                                                                47.867d0, 50.9415d0, 51.9961d0, 54.938044d0, &
                                                                55.845d0, 58.933195d0/)
    real(dp),dimension(1:n_elements) :: el_atomic_masses_g = el_atomic_masses_amu * amu2g
    character(LEN=2),dimension(1:n_elements) :: el_names = (/'H','He','Li','Be','B', &
                                                                'C','N','O','F','Ne', &
                                                                'Na','Mg','Al','Si', &
                                                                'P','S','Cl','Ar', &
                                                                'K','Ca','Sc','Ti', &
                                                                'V','Cr','Mn','Fe', &
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
    ! This counters hold the global contribution of each dust process
    ! to the evolution of dust mass
    real(dp),dimension(1:ndust+npah):: dM_acc = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_acc_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_spu = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_spu_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_coa = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_coa_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_sha = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_sha_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_sha_dest = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_sha_dest_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SNII = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SNII_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SNIa = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SNIa_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SW = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_prod_SW_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIId = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIId_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIad = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_SNIad_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ast = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ast_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_subl = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_subl_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ratd = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ratd_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ratd_dest = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_ratd_dest_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_coal = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_coal_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_fre = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_fre_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_deso = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_deso_all = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_evap = 0.0d0
    real(dp),dimension(1:ndust+npah):: dM_evap_all = 0.0d0
    ! We also track the statistics of the dust chemistry solver
    integer*8::ndust_cells=0,ndust_cells_all=0
    integer*8,dimension(1:ndchemtype)::ntot_dust_loopcnt=0,nmax_dust_loopcnt=0,nmin_dust_loopcnt=0
    integer*8,dimension(1:ndchemtype)::ntot_dust_loopcnt_all=0,nmax_dust_loopcnt_all=0,nmin_dust_loopcnt_all=0
    integer*8,dimension(1:ndust+npah,1:14)::nt0_limiter=0,nt0_limiter_all=0
    integer*8,dimension(1:ndust+npah,1:14)::ncells_skipped=0,ncells_skipped_all=0
    real(dp),dimension(1:ndust+npah,1:14)::mdust_skipped=0d0,mdust_skipped_all=0d0
    real(dp),dimension(1:ndust+npah)::total_mdust=0d0,total_mdust_all=0d0
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

    ! ==== External dust files ====
    character(LEN=256)::dust_tables_dir='../lib/dust_tables'    ! Name of folder holding optical properties files
    
    logical :: first_time_call=.true.
    integer :: icell_call=0
    real(dp) :: debug_nH=0d0,debug_T=0d0
    real(dp),dimension(1:ndust) :: debug_rho_dust=0d0
    real(dp),dimension(1:ndust) :: debug_acc_rate=0d0
    real(dp) :: h2_prime_before=0d0,h2_prime_after=0d0
    integer*8 :: tdust_solver_calls=0
    integer*8 :: tdust_solver_iter_sum=0
    integer*8 :: tdust_solver_iter_min=huge(0_8)
    integer*8 :: tdust_solver_iter_max=0
    integer*8 :: tdust_solver_brent_calls=0
    contains

    subroutine dust_log_tdust_solver_update(n_iter, used_brent)
        implicit none
        integer, intent(in) :: n_iter
        logical, intent(in) :: used_brent
        integer*8 :: n_iter_i8

        n_iter_i8 = int(max(n_iter,0), kind=8)

        tdust_solver_calls = tdust_solver_calls + 1_8
        tdust_solver_iter_sum = tdust_solver_iter_sum + n_iter_i8
        tdust_solver_iter_min = min(tdust_solver_iter_min, n_iter_i8)
        tdust_solver_iter_max = max(tdust_solver_iter_max, n_iter_i8)
        if (used_brent) tdust_solver_brent_calls = tdust_solver_brent_calls + 1_8
    end subroutine dust_log_tdust_solver_update

    subroutine dust_log_tdust_solver_print_reset
        implicit none
        real(dp) :: avg_iter

        if (.not. dust_log) return

        if (tdust_solver_calls > 0_8) then
            avg_iter = real(tdust_solver_iter_sum, dp) / real(tdust_solver_calls, dp)
            write(*,'(A,I0,A,I0,A,F10.3,A,I0)') 'Tdust solver stats: min_iter=', &
                int(tdust_solver_iter_min), ', max_iter=', int(tdust_solver_iter_max), &
                ', avg_iter=', avg_iter, ', brent_calls=', int(tdust_solver_brent_calls)
        else
            write(*,'(A)') 'Tdust solver stats: no calls in this equilibrium iteration.'
        end if

        tdust_solver_calls = 0_8
        tdust_solver_iter_sum = 0_8
        tdust_solver_iter_min = huge(0_8)
        tdust_solver_iter_max = 0_8
        tdust_solver_brent_calls = 0_8
    end subroutine dust_log_tdust_solver_print_reset

    subroutine add_total_masses
        use amr_commons
        use hydro_commons
        implicit none
        integer::ilevel
        integer::i,ivar,ind,iskip
        integer::ii,icell,igrid
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
                        do ii=npah,ndust+npah
                            total_dust_mass_species(ii) = total_dust_mass_species(ii) + (uold(ind_cell(i),idust+ii-npah-1) * dx_loc**3)
                        end do
                        ! Add total metal mass
                        do ii=1,n_elements
                            total_metal_mass(ii) = total_metal_mass(ii) + (uold(ind_cell(i),imetal+ii-1) * dx_loc**3)
                        end do
                        ! Add total CO mass
                        total_CO_mass = total_CO_mass + (uold(ind_cell(i),ico) * dx_loc**3)
                    end if
                  end do
               end do
            end do
         end do
        
    end subroutine add_total_masses

    subroutine print_total_masses(myid,tcurrent,scale_factor)
        use constants
        use amr_parameters, only:cosmo
        use mpi_mod
        implicit none
        integer,intent(in) :: myid
        real(dp),intent(in) :: tcurrent,scale_factor
        real(dp) :: scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2,scale_msun
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
        ! 2. Print the total masses
        if(myid==1)then
            if (cosmo) then
222             format('aexp:'e13.6,' Gas='e13.6,' Fe=',e13.6,&
                & ' O=',e13.6,' N=',e13.6,' Mg=',e13.6,' Si=',e13.6,' C=',e13.6,' S=',e13.6,&
                & ' PAHSmall=',e13.6,' PAHLarge=',e13.6,&
                & ' CSmall=',e13.6,' CLarge=',e13.6,' SilSmall=',e13.6,' SilLarge=',e13.6,' CO=',e13.6)
                write(*,222)scale_factor,total_gas_mass,total_metal_mass(1),total_metal_mass(2),&
                    &total_metal_mass(3),total_metal_mass(4),total_metal_mass(5),total_metal_mass(6),&
                    &total_metal_mass(7),total_dust_mass_species(1),total_dust_mass_species(2),&
                    &total_dust_mass_species(3),total_dust_mass_species(4),total_dust_mass_species(5),&
                    &total_dust_mass_species(6),total_CO_mass
            else
223             format('t:'e13.6,' Gas='e13.6,' Fe=',e13.6,&
                & ' O=',e13.6,' N=',e13.6,' Mg=',e13.6,' Si=',e13.6,' C=',e13.6,' S=',e13.6,&
                & ' PAHSmall=',e13.6,' PAHLarge=',e13.6,&
                & ' CSmall=',e13.6,' CLarge=',e13.6,' SilSmall=',e13.6,' SilLarge=',e13.6,' CO=',e13.6)
                write(*,223)tcurrent*scale_t / Myr2sec,total_gas_mass,total_metal_mass(1),total_metal_mass(2),&
                    &total_metal_mass(3),total_metal_mass(4),total_metal_mass(5),total_metal_mass(6),&
                    &total_metal_mass(7),total_dust_mass_species(1),total_dust_mass_species(2),&
                    &total_dust_mass_species(3),total_dust_mass_species(4),total_dust_mass_species(5),&
                    &total_dust_mass_species(6),total_CO_mass
            end if
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
        real(dp),dimension(1:ndust+npah,1:14) :: nt0_limiter_real, ncells_skipped_real
        character(len=30) :: format_str
        integer :: ii
#ifndef WITHOUTMPI
        integer ::mpi_err
#endif
        ! 1. Get the code units
        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)
        
        ! 2. If MPI, get the counts from all CPUs
#ifndef WITHOUTMPI
        call MPI_ALLREDUCE(ndust_cells, ndust_cells_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ntot_dust_loopcnt, ntot_dust_loopcnt_all, ndchemtype, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(nmax_dust_loopcnt, nmax_dust_loopcnt_all, ndchemtype, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(nmin_dust_loopcnt, nmin_dust_loopcnt_all, ndchemtype, MPI_INTEGER8, MPI_MIN, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(nt0_limiter, nt0_limiter_all, (ndust+npah)*14, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        call MPI_ALLREDUCE(ncells_skipped, ncells_skipped_all, (ndust+npah)*14, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, mpi_err)
        ndust_cells = ndust_cells_all ; ntot_dust_loopcnt = ntot_dust_loopcnt_all
        nmax_dust_loopcnt = nmax_dust_loopcnt_all ; nmin_dust_loopcnt = nmin_dust_loopcnt_all
        nt0_limiter = nt0_limiter_all ; ncells_skipped = ncells_skipped_all
#endif
#ifndef WITHOUTMPI
        ! 3. If MPI, get the mass from all CPUs
        call MPI_ALLREDUCE(dM_acc,dM_acc_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_acc=dM_acc_all
        call MPI_ALLREDUCE(dM_spu,dM_spu_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_spu=dM_spu_all
        call MPI_ALLREDUCE(dM_coa,dM_coa_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_coa=dM_coa_all
        call MPI_ALLREDUCE(dM_sha,dM_sha_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_sha=dM_sha_all
        call MPI_ALLREDUCE(dM_SNIId,dM_SNIId_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_SNIId=dM_SNIId_all
        call MPI_ALLREDUCE(dM_SNIad,dM_SNIad_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_SNIad=dM_SNIad_all
        call MPI_ALLREDUCE(dM_prod_SNII,dM_prod_SNII_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_prod_SNII=dM_prod_SNII_all
        call MPI_ALLREDUCE(dM_prod_SNIa,dM_prod_SNIa_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_prod_SNIa=dM_prod_SNIa_all
        call MPI_ALLREDUCE(dM_prod_SW,dM_prod_SW_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_prod_SW=dM_prod_SW_all
        call MPI_ALLREDUCE(dM_ast,dM_ast_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_ast=dM_ast_all
        call MPI_ALLREDUCE(dM_subl,dM_subl_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_subl=dM_subl_all
        call MPI_ALLREDUCE(dM_ratd,dM_ratd_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_ratd=dM_ratd_all
        call MPI_ALLREDUCE(dM_ratd_dest,dM_ratd_dest_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_ratd_dest=dM_ratd_dest_all
        call MPI_ALLREDUCE(dM_sha_dest,dM_sha_dest_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_sha_dest=dM_sha_dest_all
        call MPI_ALLREDUCE(dM_fre,dM_fre_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_fre=dM_fre_all
        call MPI_ALLREDUCE(dM_coal,dM_coal_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_coal=dM_coal_all
        call MPI_ALLREDUCE(dM_evap,dM_evap_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        dM_evap=dM_evap_all
        call MPI_ALLREDUCE(mdust_skipped,mdust_skipped_all,(NDUST+NPAH)*14,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        mdust_skipped=mdust_skipped_all
        call MPI_ALLREDUCE(total_mdust,total_mdust_all,NDUST+NPAH,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,mpi_err)
        total_mdust=total_mdust_all
#endif
        ! 4. Construct the format string
        write(format_str, '(A, I0, A)') '(A,', ndust + npah, 'ES14.6)'
        if (myid==1) then
            ! 4. If it's the CPU 1, we print the log
            write(*,format_str) 'dM Acc        =', dM_acc/(dt*scale_t)
            if (dust_sputtering.or.pah_sputtering) write(*,format_str) 'dM Spu        =', dM_spu/(dt*scale_t)
            if (dust_coagulation) write(*,format_str) 'dM Coa        =', dM_coa/(dt*scale_t)
            if (dust_shattering) write(*,format_str) 'dM Sha        =', dM_sha/(dt*scale_t)
            if (dust_turbulent_model) write(*,format_str) 'dM Sha Dest   =', dM_sha_dest/(dt*scale_t)
            if (pah_freezing) write(*,format_str) 'dM Fre        =', dM_fre/(dt*scale_t)
            if (pah_coalescence) write(*,format_str) 'dM Coal       =', dM_coal/(dt*scale_t)
            if (dust_SNdest) write(*,format_str) 'dM SNd  (II)  =', dM_SNIId*(scale_d*scale_l**3)/(dt*scale_t)
            if (dust_SNdest) write(*,format_str) 'dM SNd  (Ia)  =', dM_SNIad*(scale_d*scale_l**3)/(dt*scale_t)
            if (dust_inSN) write(*,format_str) 'dM Prod (II)  =', dM_prod_SNII*(scale_d*scale_l**3)/(dt*scale_t)
            if (dust_inSNIa) write(*,format_str) 'dM Prod (Ia)  =', dM_prod_SNIa*(scale_d*scale_l**3)/(dt*scale_t)
            if (dust_inSW) write(*,format_str) 'dM Prod (SW)  =', dM_prod_SW*(scale_d*scale_l**3)/(dt*scale_t)
            write(*,format_str) 'dM Ast        =', dM_ast*(scale_d*scale_l**3)/(dt*scale_t)
            if (pah_photolysis) write(*,format_str) 'dM Subl       =', dM_subl/(dt*scale_t)
            if (pah_cluster_evaporation) write(*,format_str) 'dM Evap       =', dM_evap/(dt*scale_t)
            if (dust_ratd) write(*,format_str) 'dM RATD       =', dM_ratd/(dt*scale_t)
            if (dust_ratd) write(*,format_str) 'dM RATD Dest  =', dM_ratd_dest/(dt*scale_t)
            if (cosmo)  then
                write(*,*) 'aexp       :', scale_factor
            else
                write(*,*) 'time [Myr] :', tcurrent*scale_t / Myr2sec
            end if
            write(*,*) 'dt   [Myr] :', dt*scale_t / Myr2sec
            ! 5. Compute the average dust loop count
            if (any(ntot_dust_loopcnt > 0)) then
                if (ndchemtype==1) then
                    write(*,124)ntot_dust_loopcnt,dble(ntot_dust_loopcnt) / dble(ndust_cells), nmax_dust_loopcnt, nmin_dust_loopcnt
                elseif (ndchemtype==2) then
                    write(*,125) ntot_dust_loopcnt
                    write(*,126) dble(ntot_dust_loopcnt) / dble(ndust_cells)
                    write(*,127) nmax_dust_loopcnt
                    write(*,128) nmin_dust_loopcnt
                end if
                nt0_limiter_real(:,:) = 1d2 * nt0_limiter(:,:) / dble(sum(ntot_dust_loopcnt(:)))
                write(format_str, '(A, I0, A)') '(A,', ndust + npah, 'ES10.2)'
                write(*,*) 'Avg % of times the dust processes was the limiting factor'
                write(*,format_str) 'Accretion        = ', nt0_limiter_real(1:ndust+npah,1)
                write(*,format_str) 'Sputtering       = ', nt0_limiter_real(1:ndust+npah,2)
                write(*,format_str) 'Shattering       = ', nt0_limiter_real(1:ndust+npah,3)
                write(*,format_str) 'Coagulation      = ', nt0_limiter_real(1:ndust+npah,4)
                write(*,format_str) 'Shatt Dest       = ', nt0_limiter_real(1:ndust+npah,7)
                write(*,format_str) 'Shatt Small      = ', nt0_limiter_real(1:ndust+npah,8)
                write(*,format_str) 'Shatt Small Dest = ', nt0_limiter_real(1:ndust+npah,9)
                write(*,format_str) 'RATD             = ', nt0_limiter_real(1:ndust+npah,5)
                write(*,format_str) 'RATD Dest        = ', nt0_limiter_real(1:ndust+npah,6)
                write(*,format_str) 'Sublimation      = ', nt0_limiter_real(1:ndust+npah,10)
                write(*,format_str) 'Coalescence      = ', nt0_limiter_real(1:ndust+npah,11)
                write(*,format_str) 'Freezing         = ', nt0_limiter_real(1:ndust+npah,13)
                write(*,format_str) 'Evaporation      = ', nt0_limiter_real(1:ndust+npah,14)
                if (any(nt0_limiter_real(:,:) < 0d0)) then
                    write(*,*) 'WARNING: Some dust processes show negative time fractions!'
                    write(*,*) 'nt0_limiter(:,:): ', nt0_limiter(:,:)
                    write(*,*) 'ntot_dust_loopcnt(:): ', ntot_dust_loopcnt(:)
                    write(*,*) 'sum(ntot_dust_loopcnt(:)): ', sum(ntot_dust_loopcnt(:))
                    write(*,*) 'nt0_limiter_real(:,:): ', nt0_limiter_real(:,:) 
                    stop
                end if
                ncells_skipped_real(:,:) = 1d2 * ncells_skipped(:,:) / dble(ndust_cells)
                do ii = 1, ndust+npah
                    mdust_skipped(ii,:) = mdust_skipped(ii,:) / total_mdust(ii)
                end do
                write(*,*) 'Avg % of cells where the dust processes was skipped'
                write(*,format_str) 'Sputtering       = ', ncells_skipped_real(1:ndust+npah,2)
                write(*,format_str) 'RATD             = ', nt0_limiter_real(1:ndust+npah,5)   
                write(*,format_str) 'Evaporation      = ', nt0_limiter_real(1:ndust+npah,14)
                write(*,*) 'Avg % of dust mass skipped'
                write(*,format_str) 'Sputtering       = ', mdust_skipped(1:ndust+npah,2)
                write(*,format_str) 'RATD             = ', mdust_skipped(1:ndust+npah,5)
                write(*,format_str) 'Evaporation      = ', mdust_skipped(1:ndust+npah,14)    
            end if
        endif     
        dM_acc            = 0.0d0; dM_acc_all            = 0.0d0
        dM_spu            = 0.0d0; dM_spu_all            = 0.0d0
        dM_coa            = 0.0d0; dM_coa_all            = 0.0d0
        dM_sha            = 0.0d0; dM_sha_all            = 0.0d0
        dM_SNIId          = 0.0d0; dM_SNIId_all          = 0.0d0
        dM_SNIad          = 0.0d0; dM_SNIad_all          = 0.0d0
        dM_prod_SNII      = 0.0d0; dM_prod_SNII_all      = 0.0d0
        dM_prod_SNIa      = 0.0d0; dM_prod_SNIa_all      = 0.0d0
        dM_prod_SW        = 0.0d0; dM_prod_SW_all        = 0.0d0
        dM_ast            = 0.0d0; dM_ast_all            = 0.0d0
        dM_subl           = 0.0d0; dM_subl_all           = 0.0d0
        dM_ratd           = 0.0d0; dM_ratd_all           = 0.0d0
        dM_ratd_dest      = 0.0d0; dM_ratd_dest_all      = 0.0d0
        dM_fre            = 0.0d0; dM_fre_all            = 0.0d0
        dM_sha_dest       = 0.0d0; dM_sha_dest_all       = 0.0d0
        dM_coal           = 0.0d0; dM_coal_all           = 0.0d0
        dM_evap           = 0.0d0; dM_evap_all           = 0.0d0
        ndust_cells       = 0;     ndust_cells_all       = 0
        ntot_dust_loopcnt = 0;     ntot_dust_loopcnt_all = 0
        nmax_dust_loopcnt = 0;     nmax_dust_loopcnt_all = 0
        nmin_dust_loopcnt = 0;     nmin_dust_loopcnt_all = 0
        nt0_limiter       = 0;     nt0_limiter_all       = 0
        ncells_skipped    = 0;     ncells_skipped_all    = 0
        mdust_skipped     = 0.0d0; mdust_skipped_all     = 0.0d0
        total_mdust       = 0.0d0; total_mdust_all       = 0.0d0
        if (ndchemtype==1) then
124 format(' Duststats: Tot # loops = ',I20,', Avg. # loops = ', ES14.6, ', max. # loops = ', I20, ', min. # loops = ', I10)
        elseif (ndchemtype==2) then
125 format(' Duststats: Tot. # loops = ',2I20)
126 format('            Avg. # loops = ',2ES14.6)
127 format('            Max. # loops = ',2I20)
128 format('            Min. # loops = ',2I20)
        end if
    end subroutine print_dust_log

end module