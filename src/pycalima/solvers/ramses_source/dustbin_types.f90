module dustbin_types
    use amr_parameters, only:dp
    use hydro_parameters, only:n_elements
    use dust_utils, only: interpolate1D_eqw, interpolate1D_noeqw, &
                          interpolate2D_eqw, interpolate2D_noeqw, &
                          interpolate3D_eqw, interpolate3D_noeqw, locate, locate_eqw

    implicit none

    ! ==== Dust table derived type ====
    type DustTable
        character(len=64) :: name='noname' ! Name of the dust table
        logical :: initialised = .false. ! Flag indicating whether table has been initialised
        logical :: eqw = .false. ! Flag indicating whether the table is equally spaced
        integer :: ndim = 0 ! Number of dimension of the dust table (1 for thermal sputtering, 2 for collisional tables, etc)
        integer, dimension(:), allocatable :: ipos_zero ! Position of the zero value along each dimension (for interpolation purposes)
        integer, dimension(:), allocatable :: npts ! Number of points along each dimension
        real(dp), dimension(:,:), allocatable :: tab1d ! 1D table values
        real(dp), dimension(:,:,:), allocatable :: tab2d ! 2D table values
        real(dp), dimension(:,:,:,:), allocatable :: tab3d ! 3D table values
        real(dp), dimension(:), allocatable :: dx ! Grid spacing along each dimension
        real(dp), dimension(:), allocatable :: inv_dx ! Inverse grid spacing along each dimension
    contains
        procedure, private :: interpolate_1d
        procedure, private :: interpolate_2d
        procedure, private :: interpolate_3d
        generic :: interpolate => interpolate_1d, interpolate_2d, interpolate_3d
        procedure :: slope_1d, slope_2d
        procedure :: init => initialise_dust_table
    end type DustTable

    ! ==== Dust Chemistry Info ====
    type DustChemistryInfo
        logical :: initialised = .false.
        logical :: use_precomp = .false. ! Whether to return the cooling/heating rates already saved
        integer :: ndust = 0 ! Number of dust bins
        integer :: npah = 0 ! Number of PAH bins
        integer :: nGroups = 0 ! Number of radiation groups
        integer :: nion_charges = 0 ! Number of charge states for Coulomb effects
        integer :: icell = 0 ! Index of the cell in the RTZ cell vector (WDB06rt warm states; 0: none)
        integer :: ncharge_pah_max = 0 ! Maximum number of charge states for PAHs
        real(dp) :: H2_formation_rate = 0d0 ! H2 formation rate on dust
        real(dp),dimension(1:n_elements)  :: rec_ion_rate = 0d0 ! dust_ion_recombination: grain-assisted X+ -> X rate per X+ ion [s^-1], by atomic number
        real(dp) :: G0_background = 0d0 ! Background radiation field in units of Habing field
        real(dp) :: local_c = 0d0 ! Local reduced speed of light (cm/s)
        real(dp) :: local_mu = 0d0 ! Local mean molecular weight (in units of H mass)
        real(dp) :: local_sigma = 0d0 ! Local gas velocity dispersion (cm/s)
        real(dp) :: local_Tk = 0d0 ! Local gas temperature (K)
        real(dp) :: local_nH = 0d0 ! Local hydrogen density (in H/cm3)
        real(dp) :: local_rho = 0d0 ! Local total cell mass density (in g/cm3)
        real(dp) :: local_Jeans = 0d0 ! Local Jeans length (in cm)
        real(dp) :: local_dx = 0d0 ! Local cell size (in cm)
        real(dp) :: local_vol = 0d0 ! Local cell volume (in cm^3)
        real(dp) :: local_G0 = 0d0 ! Local radiation field in units of Habing field
        real(dp) :: local_ne = 0d0 ! Local electron density (in cm-3)
        real(dp) :: local_nCO = 0d0 ! Local CO density (in cm-3)
        real(dp) :: smallNp = 0d0 ! Threshold for small photon number density below which we consider the radiation field to be negligible for processes like photoelectric heating and radiation pressure
        real(dp),dimension(1:n_elements)  :: el_atomic_mass_g ! Element atomic mass [g]
        real(dp),dimension(1:n_elements)  :: inv_atomic_mass ! Inverse element atomic mass [1/g]
        real(dp),dimension(:),allocatable :: local_rad_ani ! Local radiation anisotropy factor
        real(dp),dimension(:),allocatable :: local_solid_angle ! Local solid angle subtended by radiation sources
        real(dp),dimension(:),allocatable :: group_eV ! Energy of each radiation group in eV
        real(dp),dimension(:),allocatable :: rho_dust ! Dust mass density
        real(dp),dimension(:),allocatable :: n_dust ! Dust number density
        real(dp),dimension(:),allocatable :: rho_pah ! PAH mass density
        real(dp),dimension(:),allocatable :: n_pah ! PAH number density
        real(dp),dimension(:),allocatable :: Z_dust ! Dust median charge
        real(dp),dimension(:),allocatable :: Z_sigma ! Dust charge distribution width
        real(dp),dimension(:,:),allocatable :: fcharge_pah ! PAH charge distribution function
        real(dp),dimension(:),allocatable :: T_dust ! Dust temperature
        real(dp),dimension(:,:),allocatable :: Pabs_dust ! Dust absorption power
        real(dp),dimension(:),allocatable :: Pinj_dust ! Dust thermal injection power
        real(dp),dimension(:),allocatable :: Prad_dust ! Dust radiative power
        real(dp),dimension(:),allocatable :: Prec_dust ! Dust recombination power
        real(dp),dimension(:),allocatable :: Pcoll_dust ! Dust collisional power
        real(dp),dimension(:,:),allocatable :: Pabs_pah ! PAH absorption power
        real(dp),dimension(:),allocatable :: Pinj_pah ! PAH thermal injection power
        real(dp),dimension(:),allocatable :: Prad_pah ! PAH radiative power
        real(dp),dimension(:),allocatable :: Prec_pah ! PAH recombination power
        real(dp),dimension(:),allocatable :: Pcoll_pah ! PAH collisional power
        real(dp),dimension(:,:),allocatable :: l_a ! Attenuation length
        real(dp),dimension(:,:),allocatable :: csa_dust ! Dust absorption cross section
        real(dp),dimension(:,:),allocatable :: css_dust ! Dust scattering cross section
        real(dp),dimension(:,:),allocatable :: csr_dust ! Dust radiation pressure cross section
        real(dp),dimension(:,:),allocatable :: csrat_dust ! Dust RAT-D cross section
        real(dp),dimension(:,:),allocatable :: csa_pah ! PAH absorption cross section
        real(dp),dimension(:,:),allocatable :: css_pah ! PAH scattering cross section
        real(dp),dimension(:,:),allocatable :: csr_pah ! PAH radiation pressure cross section
        real(dp),dimension(:),allocatable :: dustAbs ! Dust absorption rate [1/s]
        real(dp),dimension(:),allocatable :: pahAbs ! PAH absorption rate [1/s]
        real(dp),dimension(:,:),allocatable :: Coulomb_factor ! Coulomb enhancement factor
        real(dp),dimension(:),allocatable :: rat_torque ! Radiative torque on dust grains
        real(dp),dimension(:),allocatable :: IR_damp_factor ! Infrared damping factor for grain rotation
    contains
        procedure :: init => init_dust_chemistry_info
        procedure :: reset => reset_dust_chemistry_info
    end type DustChemistryInfo

    ! ==== Dust Process type ====
    abstract interface
        subroutine comp_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
            import :: DustChemistryInfo, dp
            implicit none

            ! ---- Input DustChemistryInfo with local variables----
            class(DustChemistryInfo),intent(in) :: dust_info
            real(dp),intent(in) :: y_gas(:,:), y_dust(:)
            real(dp),intent(inout) :: dydt_gas(:,:), dydt_dust(:, :)
            real(dp),intent(inout),optional :: kmax

            ! Do something to compute the rate
        end subroutine comp_rate
    end interface
    type DustProcess
        character(len=30) :: name='accretion' ! Name of the dust process (e.g. "sputtering", "coagulation", etc)
        integer :: processID=1 ! Integer ID for the dust process (e.g. 1 for sputtering, 2 for coagulation, etc)
        logical :: source=.false., sink=.false. ! Whether the process is a source or a sink for dust mass
        procedure(comp_rate), pointer, nopass :: comp_rate => null() ! Pointer to the subroutine that computes the rate of the process
    end type DustProcess

    ! ==== Dust bin derived type ====
    type DustBin
        integer  :: dust_index              ! Index of the dust bin
        integer  :: u_hydro_idx             ! Variable index in uold
        integer  :: nelements               ! Number of elements in the dust composition
        integer  :: interact_group          ! Index of the grain interaction group
        logical  :: interact_pah=.false.    ! Whether the dust bin interacts with PAHs
        logical  :: separate_refractive_index=.false. ! Whether the dust bin has separate refractive index tables for parallel and perpendicular waves
        real(dp) :: asize                   ! Grain size (in microns)
        real(dp) :: asize_cm                ! Grain size (in cm)
        real(dp) :: asize_nm                ! Grain size (in nm)
        real(dp) :: sgrain                  ! Grain material density (in g/cm^3)
        real(dp) :: mgrain                  ! Grain mass (in g)
        real(dp) :: Youngs_modulus          ! Young's modulus (erg/cm3)
        real(dp) :: Poisson_ratio           ! Poisson's ratio
        real(dp) :: catastrophic_spec_energy! Catastrophic impact specific energy (erg/g)
        real(dp) :: tensile_strength        ! Tensile strength (erg/cm3)
        real(dp) :: shear_modulus           ! Shear modulus (erg/cm3)
        real(dp) :: surf_energy             ! Surface energy (erg/cm2)
        real(dp) :: work_function           ! Work function (eV)
        real(dp) :: band_gap                ! Band gap (eV)
        real(dp) :: e_escape_length         ! Electron escape length (in cm)
        real(dp) :: Zmin                    ! Minimum grain charge allowed for this dust bin
        real(dp) :: amin                    ! Minimum grain size (in microns)
        real(dp) :: amax                    ! Maximum grain size (in microns)
        real(dp) :: mgrain_min              ! Minimum grain mass (in g)
        real(dp) :: mgrain_max              ! Maximum grain mass (in g)
        real(dp) :: SNII_cond_eff           ! SNII condensation efficiency
        real(dp) :: SNIa_cond_eff           ! SNIa condensation efficiency
        real(dp) :: AGB_cond_eff            ! AGB condensation efficiency
        real(dp) :: w_disr                  ! Rotation rate at which grain disruption occurs (rad/s)
        real(dp) :: grain_inertia           ! Grain moment of inertia (g cm2)
        real(dp) :: tau_gas_0               ! Reference gas damping time (s)
        real(dp) :: RAT_torque_0            ! Reference radiative torque (erg) for a radiation field of 1 Habing
        real(dp) :: SNsha_eff               ! SN shattering efficiency for grain size
        real(dp) :: k0_spu                  ! Reference rate for sputtering (s-1)
        real(dp) :: k0_acc                  ! Reference rate for accretion (s-1)
        real(dp) :: nh_coa                  ! Gas density above which dust coagulation is allowed
        real(dp) :: nhmax_coa               ! Max gas density for coagulation subgrid model
        real(dp) :: nhmax_acc               ! Max gas density for accretion subgrid model
        real(dp) :: nhmax_sha               ! Max gas density for shattering subgrid model
        real(dp) :: SNdest_eff              ! SN dust destruction efficiency
        real(dp) :: Coulomb_enhance         ! Coulomb enhancement factor for ion accretion
        real(dp) :: T_sublimation_min = 0d0 ! Minimum sublimation temperature in Kelvin
        real(dp) :: Pabs_isrf               ! Mathis ISRF-averaged absorption rate (in erg/s)
        real(dp) :: Psc_isrf                ! Mathis ISRF-averaged scattering rate (in erg/s)
        real(dp) :: Prp_isrf                ! Mathis ISRF-averaged radiation pressure rate (in erg/s)
        real(dp),dimension(:),allocatable :: k0_coa                ! Reference rate for coagulation (s-1)
        real(dp),dimension(:),allocatable :: k0_sha                ! Reference rate for shattering (s-1)
        integer ,dimension(:),allocatable :: el_index              ! Index of the elements used in the full element list
        integer ,dimension(:),allocatable :: el_atomic_number      ! Atomic number of the elements used in the full element list
        real(dp),dimension(:),allocatable :: stoichiometry         ! Stoichiometry of the dust bin
        real(dp),dimension(:),allocatable :: el_mfractions         ! Element mass fractions
        real(dp),dimension(:),allocatable :: el_atomic_masses_amu  ! Element atomic masses (in amu)
        real(dp),dimension(:),allocatable :: el_atomic_masses_g    ! Element masses (in g)
        real(dp),dimension(:),allocatable :: el_conv_factors       ! Element conversion factors
        real(dp),dimension(:),allocatable :: el_lim_factors        ! Element limiting factors
        real(dp),dimension(:),allocatable :: chi_frag_ratd         ! Fragment distribution for RATD
        integer,dimension(:),allocatable  :: idend_coag            ! Index of the dust bin that is the destination of coagulation
        real(dp),dimension(:),allocatable :: vthresh_coag          ! Threshold velocity for coagulation
        real(dp),dimension(:),allocatable :: phi_prefact           ! Prefactors to quickly compute phi=Z_dust*q/a
#ifdef RTZ
        integer,dimension(:),allocatable  :: el_nions              ! Number of ions followed for each element
#endif
        character(len=2),dimension(:),allocatable :: el_names      ! Element names

        ! Tables for dust processes
        type(DustTable),dimension(1:n_elements) :: sputtering_tab ! Sputtering tables
        type(DustTable) :: sublimation_tab ! Thermal sublimation erosion rate table (function of dust temperature)
        type(DustTable),dimension(0:n_elements) :: collisional_tab ! Collisional tables (0 is for electrons)
        type(DustTable) :: cs_abs_tab, cs_scat_tab, cs_ext_tab ! Absorption, scattering and extinction cross-section tables
        type(DustTable) :: Rosseland_tab, Planck_tab ! Rosseland and Planck mean opacity tables
        type(DustTable),dimension(:),allocatable :: Im_n ! Imaginary part of the refractive index tables for each element
        type(DustTable) :: Tdust_tab ! Dust temperature table
        type(DustTable) :: Planck_power_tab ! Planck power tables for dust temperature calculation
        type(DustTable) :: Planckderiv_tab  ! Derivative of the Planck power tables for dust temperature calculation
        type(DustTable),dimension(:),allocatable :: IRemission_tab ! Infrared band emission tables
    end type DustBin

    ! ==== PAH bin derived type ====
    type PAHBin
        logical  :: is_cluster = .false.    ! Whether the PAH bin corresponds to a cluster of PAHs
        integer  :: pah_index               ! Index of the PAH bin
        integer  :: u_hydro_idx             ! Variable index in uold
        integer  :: nc                      ! Number of carbon atoms in the PAH
        integer  :: nc_min                  ! Minimum number of carbon atoms in the PAH
        integer  :: nc_max                  ! Maximum number of carbon atoms in the PAH
        integer  :: n                       ! Number of hydrogen atoms in the PAH
        integer  :: C_index                 ! Index of the carbon element in the full element list
        integer  :: dust_index_interact     ! Index of the starting dust bin that interacts with this PAH bin
        integer  :: nd_bins                 ! Number of carbonaceous grain bins that interact with this PAH bin
        integer  :: ncharge_states          ! Number of charge states followed for the PAH bin
        integer  :: cation_start_idx        ! First index in fcharge_pahs corresponding to cation states (>0)
        real(dp) :: apah                    ! PAH size (in microns)
        real(dp) :: apah_cm                 ! PAH size (in cm)
        real(dp) :: spah                    ! PAH material density (in g/cm^3)
        real(dp) :: mpah                    ! PAH mass (in g)
        real(dp) :: mpah_min                ! Minimum PAH mass (in g)
        real(dp) :: mpah_max                ! Maximum PAH mass (in g)
        real(dp) :: AGB_cond_eff            ! AGB condensation efficiency
        real(dp) :: SNdest_eff              ! SN dust destruction efficiency
        real(dp) :: t0_acc                  ! Reference time for accretion (s)
        real(dp) :: nhmax_acc               ! Max gas density for accretion subgrid
        real(dp) :: nhmmax_clus             ! Max gas density for clustering subgrid
        real(dp),dimension(1:2) :: Pabs_isrf               ! Mathis ISRF-averaged absorption rate (in erg/s)
        real(dp),dimension(1:2) :: Psc_isrf                ! Mathis ISRF-averaged scattering rate (in erg/s)
        real(dp),dimension(1:2) :: Prp_isrf                ! Mathis ISRF-averaged radiation pressure rate (in erg/s)
        real(dp),dimension(:),allocatable :: charge_states ! Charge states of the PAH bin

        ! Tables for PAH processes
        type(DustTable),dimension(0:n_elements) :: sputtering_tab ! PAH tables for each element
        type(DustTable) :: peh_eff_tab, peh_pabs_tab ! Photoelectric efficiency and absorption cross-section tables for PAHs
        type(DustTable),dimension(:),allocatable :: fcharge_tab ! Charging states distribution tables for PAHs
        type(DustTable) :: cs_abs_tab_n, cs_scat_tab_n, cs_ext_tab_n ! Absorption, scattering and extinction cross-section tables for neutral PAHs
        type(DustTable) :: cs_abs_tab_i, cs_scat_tab_i, cs_ext_tab_i ! Absorption, scattering and extinction cross-section tables for ionised PAHs
        ! Rosseland (flux) and Planck (absorption) mean cross sections vs dust
        ! temperature, per charge state. Used for the IR photon group, where the
        ! band-weighted group_cs*_pah values are the wrong spectral weight.
        type(DustTable) :: Rosseland_tab_n, Planck_tab_n ! Mean cross-section tables for neutral PAHs
        type(DustTable) :: Rosseland_tab_i, Planck_tab_i ! Mean cross-section tables for ionised PAHs
        type(DustTable) :: dissociation_tab ! PAH dissociation tables
    end type PAHBin

contains

    subroutine interpolate_1d(this, xi, val, idx_x)
        class(DustTable), intent(in) :: this
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: val
        integer, intent(inout), optional :: idx_x

        integer :: n1
        n1 = this%npts(1)

        select case (this%ndim)
        case (1)
            if (this%eqw) then
                call interpolate1D_eqw(this%tab1d(1:n1, 1), this%tab1d(1:n1, 2), n1, &
                                       this%inv_dx(1), xi, val, idx_x)
            else
                call interpolate1D_noeqw(this%tab1d(1:n1, 1), this%tab1d(1:n1, 2), n1, &
                                         xi, val, idx_x)
            end if
        case (2)
            ! 2D table sliced at the zero value of the 2nd dimension (e.g. phi = 0)
            if (this%eqw) then
                call interpolate1D_eqw(this%tab1d(1:n1, 1), this%tab2d(1:n1, this%ipos_zero(2), 1), n1, &
                                       this%inv_dx(1), xi, val, idx_x)
            else
                call interpolate1D_noeqw(this%tab1d(1:n1, 1), this%tab2d(1:n1, this%ipos_zero(2), 1), n1, &
                                         xi, val, idx_x)
            end if
        case (3)
            ! 3D table sliced at the zero values of the 2nd & 3rd dimensions
            if (this%eqw) then
                call interpolate1D_eqw(this%tab1d(1:n1, 1), this%tab3d(1:n1, this%ipos_zero(2), this%ipos_zero(3), 1), n1, &
                                       this%inv_dx(1), xi, val, idx_x)
            else
                call interpolate1D_noeqw(this%tab1d(1:n1, 1), this%tab3d(1:n1, this%ipos_zero(2), this%ipos_zero(3), 1), n1, &
                                         xi, val, idx_x)
            end if
        end select
    end subroutine interpolate_1d

    subroutine slope_1d(this, xi, dydx)
        ! d/dx of interpolate_1d at xi (the same slice, linear in the same cell); 0 where xi is
        ! clamped at the edges of the table. At a node, the cell to the right (a forward difference)
        class(DustTable), intent(in) :: this
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: dydx
        integer :: n1, i
        real(dp) :: y0, y1

        n1 = this%npts(1)
        dydx = 0d0
        if (xi <= this%tab1d(1,1) .or. xi >= this%tab1d(n1,1)) return
        if (this%eqw) then
            i = locate_eqw(this%tab1d(1:n1,1), n1, xi, this%inv_dx(1))
        else
            i = locate(this%tab1d(1:n1,1), n1, xi)
        end if
        if (i < n1 - 1 .and. xi >= this%tab1d(i+1,1)) i = i + 1
        select case (this%ndim)
        case (1)
            y0 = this%tab1d(i,2)
            y1 = this%tab1d(i+1,2)
        case (2)
            y0 = this%tab2d(i,this%ipos_zero(2),1)
            y1 = this%tab2d(i+1,this%ipos_zero(2),1)
        case default
            y0 = this%tab3d(i,this%ipos_zero(2),this%ipos_zero(3),1)
            y1 = this%tab3d(i+1,this%ipos_zero(2),this%ipos_zero(3),1)
        end select
        dydx = (y1 - y0) / (this%tab1d(i+1,1) - this%tab1d(i,1))
    end subroutine slope_1d

    subroutine slope_2d(this, xi, yi, dydx, dydy)
        ! d/dx and d/dy of interpolate_2d at (xi, yi) (bilinear in the same cell); 0 along a
        ! coordinate clamped at the edges of the table. At a node, the cell to the right
        class(DustTable), intent(in) :: this
        real(dp), intent(in) :: xi, yi
        real(dp), intent(out) :: dydx, dydy
        integer :: n1, n2, i, j
        real(dp) :: xc, yc, t, u, r00, r10, r01, r11, dx, dy

        n1 = this%npts(1)
        n2 = this%npts(2)
        xc = min(max(xi, this%tab1d(1,1)), this%tab1d(n1,1))
        yc = min(max(yi, this%tab1d(1,2)), this%tab1d(n2,2))
        if (this%eqw) then
            i = locate_eqw(this%tab1d(1:n1,1), n1, xc, this%inv_dx(1))
            j = locate_eqw(this%tab1d(1:n2,2), n2, yc, this%inv_dx(2))
        else
            i = locate(this%tab1d(1:n1,1), n1, xc)
            j = locate(this%tab1d(1:n2,2), n2, yc)
        end if
        if (i < n1 - 1 .and. xc >= this%tab1d(i+1,1)) i = i + 1
        if (j < n2 - 1 .and. yc >= this%tab1d(j+1,2)) j = j + 1
        if (this%ndim == 2) then
            r00 = this%tab2d(i,j,1)
            r10 = this%tab2d(i+1,j,1)
            r01 = this%tab2d(i,j+1,1)
            r11 = this%tab2d(i+1,j+1,1)
        else
            r00 = this%tab3d(i,j,this%ipos_zero(3),1)
            r10 = this%tab3d(i+1,j,this%ipos_zero(3),1)
            r01 = this%tab3d(i,j+1,this%ipos_zero(3),1)
            r11 = this%tab3d(i+1,j+1,this%ipos_zero(3),1)
        end if
        dx = this%tab1d(i+1,1) - this%tab1d(i,1)
        dy = this%tab1d(j+1,2) - this%tab1d(j,2)
        t = (xc - this%tab1d(i,1)) / dx
        u = (yc - this%tab1d(j,2)) / dy
        dydx = 0d0
        dydy = 0d0
        if (xi > this%tab1d(1,1) .and. xi < this%tab1d(n1,1)) dydx = ((1d0 - u)*(r10 - r00) + u*(r11 - r01)) / dx
        if (yi > this%tab1d(1,2) .and. yi < this%tab1d(n2,2)) dydy = ((1d0 - t)*(r01 - r00) + t*(r11 - r10)) / dy
    end subroutine slope_2d

    subroutine interpolate_2d(this, xi, yi, val, idx_x, idx_y)
        class(DustTable), intent(in) :: this
        real(dp), intent(in) :: xi, yi
        real(dp), intent(out) :: val
        integer, intent(inout), optional :: idx_x, idx_y

        integer :: n1, n2
        n1 = this%npts(1)
        n2 = this%npts(2)

        select case (this%ndim)
        case (2)
            if (this%eqw) then
                call interpolate2D_eqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), &
                                       this%tab2d(:, :, 1), n1, n2, &
                                       this%inv_dx(1), this%inv_dx(2), xi, yi, &
                                       val, idx_x, idx_y)
            else
                call interpolate2D_noeqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), &
                                         this%tab2d(:, :, 1), n1, n2, xi, yi, &
                                         val, idx_x, idx_y)
            end if
        case (3)
            ! 3D table sliced at the zero value of the 3rd dimension
            if (this%eqw) then
                call interpolate2D_eqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), &
                                       this%tab3d(:, :, this%ipos_zero(3), 1), n1, n2, &
                                       this%inv_dx(1), this%inv_dx(2), xi, yi, &
                                       val, idx_x, idx_y)
            else
                call interpolate2D_noeqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), &
                                         this%tab3d(:, :, this%ipos_zero(3), 1), n1, n2, xi, yi, &
                                         val, idx_x, idx_y)
            end if
        end select
    end subroutine interpolate_2d

    subroutine interpolate_3d(this, xi, yi, zi, val, idx_x, idx_y, idx_z)
        implicit none
        class(DustTable), intent(in) :: this
        real(dp), intent(in) :: xi, yi, zi
        real(dp), intent(out) :: val
        integer, intent(inout), optional :: idx_x, idx_y, idx_z

        integer :: n1, n2, n3
        n1 = this%npts(1)
        n2 = this%npts(2)
        n3 = this%npts(3)

        if (this%eqw) then
            call interpolate3D_eqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), this%tab1d(1:n3, 3), &
                                     this%tab3d(:, :, :, 1), n1, n2, n3, &
                                     this%inv_dx(1), this%inv_dx(2), this%inv_dx(3), &
                                     xi, yi, zi, val, idx_x, idx_y, idx_z)
        else
            call interpolate3D_noeqw(this%tab1d(1:n1, 1), this%tab1d(1:n2, 2), this%tab1d(1:n3, 3), &
                                     this%tab3d(:, :, :, 1), n1, n2, n3, &
                                     xi, yi, zi, val, idx_x, idx_y, idx_z)
        end if
    end subroutine interpolate_3d

    subroutine initialise_dust_table(this)
        implicit none
        class(DustTable), intent(inout) :: this
        integer :: d, i
        real(dp) :: dx_first, dx_curr, diff, thresh

        ! 1. Allocate the dx and inv_dx arrays
        allocate(this%dx(1:this%ndim))
        allocate(this%inv_dx(1:this%ndim))
        
        ! 2. Compute the grid spacings
        do d = 1, this%ndim
            this%dx(d) = (this%tab1d(this%npts(d), d) - this%tab1d(1, d)) / dble(this%npts(d) - 1)
            if (this%dx(d) .eq. 0d0) then
                write(*,*) 'Error: table%dx(d) is zero for ',trim(this%name)
                call clean_stop
            end if
            this%inv_dx(d) = 1d0 / this%dx(d)
        end do

        ! 3. Check for equal spacing
        this%eqw = .true.
        do d = 1, this%ndim
            if (this%npts(d) < 2) cycle
            dx_first = this%tab1d(2, d) - this%tab1d(1, d)
            if (dx_first == 0d0) then
                write(*,*) 'ERROR: Grid ', this%name, ' has zero spacing between first two elements!'
                call clean_stop
            end if
            do i = 2, this%npts(d) - 1
                dx_curr = this%tab1d(i+1, d) - this%tab1d(i, d)
                diff = abs(dx_curr - this%dx(d))
                thresh = 1d-4 * abs(this%dx(d)) + 1d-7
                if (diff > thresh) then
                    this%eqw = .false.
                    exit
                end if
            end do
        end do
        if (.not. this%eqw) then
            write(*,*) 'WARNING: Grid ', this%name, ' is not equally spaced!'
            write(*,*) 'Expected spacing: ', this%dx(1)
            write(*,*) 'Found spacing between element 1 and 2: ', this%tab1d(2,1) - this%tab1d(1,1)
        end if

        ! 4. Set the table as initialised
        this%initialised = .true.
    end subroutine initialise_dust_table

    subroutine init_dust_chemistry_info(this, ndust, npah, nGroups, ncharge_pah_max, nion_charges)
        ! Initializes the DustChemistryInfo derived type by allocating arrays and setting default values.
        ! This is done at RAMSES initialisation, and re-used for each threat
        ! this --> the DustChemistryInfo instance to initialize
        ! ndust --> number of dust bins
        ! npah --> number of PAH bins
        ! nGroups --> number of radiation groups
        ! ncharge_pah_max --> maximum number of charge states for PAHs
        ! nion_charges --> number of charge states for Coulomb effects
        class(DustChemistryInfo), intent(inout) :: this
        integer, intent(in) :: ndust, npah, nGroups, ncharge_pah_max, nion_charges

        this%initialised = .false.
        this%ndust = max(ndust, 0)
        this%npah = max(npah, 0)
        this%nGroups = max(nGroups, 0)
        this%nion_charges = max(nion_charges, 0)
        this%ncharge_pah_max = max(ncharge_pah_max, 0)
        this%H2_formation_rate = 0d0
        this%G0_background = 0d0
        this%local_c = 0d0
        this%local_mu = 0d0
        this%local_sigma = 0d0
        this%local_Tk = 0d0
        this%local_nH = 0d0
        this%local_rho = 0d0
        this%local_Jeans = 0d0
        this%local_dx = 0d0
        this%local_vol = 0d0
        this%local_ne = 0d0
        this%local_nCO = 0d0
        this%smallNp = 0d0

        if (allocated(this%rho_dust)) deallocate(this%rho_dust)
        allocate(this%rho_dust(1:this%ndust))
        this%rho_dust = 0d0

        if (allocated(this%rho_pah)) deallocate(this%rho_pah)
        allocate(this%rho_pah(1:this%npah))
        this%rho_pah = 0d0

        if (allocated(this%n_dust)) deallocate(this%n_dust)
        allocate(this%n_dust(1:this%ndust))
        this%n_dust = 0d0

        if (allocated(this%n_pah)) deallocate(this%n_pah)
        allocate(this%n_pah(1:this%npah))
        this%n_pah = 0d0

        if (allocated(this%Z_dust)) deallocate(this%Z_dust)
        allocate(this%Z_dust(1:this%ndust))
        this%Z_dust = 0d0

        if (allocated(this%Z_sigma)) deallocate(this%Z_sigma)
        allocate(this%Z_sigma(1:this%ndust))
        this%Z_sigma = 0d0

        if (allocated(this%fcharge_pah)) deallocate(this%fcharge_pah)
        allocate(this%fcharge_pah(1:this%ncharge_pah_max,1:this%npah))
        this%fcharge_pah = 0d0
        if (this%npah > 0) then
           this%fcharge_pah(1,:) = 1d0 ! Assume all PAHs are neutral at beginning
        end if

        if (allocated(this%T_dust)) deallocate(this%T_dust)
        allocate(this%T_dust(1:this%ndust))
        this%T_dust = 0d0

        if (allocated(this%Pabs_dust)) deallocate(this%Pabs_dust)
        allocate(this%Pabs_dust(1:this%ndust,1:this%nGroups))
        this%Pabs_dust = 0d0

        if (allocated(this%Pinj_dust)) deallocate(this%Pinj_dust)
        allocate(this%Pinj_dust(1:this%ndust))
        this%Pinj_dust = 0d0

        if (allocated(this%Prad_dust)) deallocate(this%Prad_dust)
        allocate(this%Prad_dust(1:this%ndust))
        this%Prad_dust = 0d0

        if (allocated(this%Prec_dust)) deallocate(this%Prec_dust)
        allocate(this%Prec_dust(1:this%ndust))
        this%Prec_dust = 0d0

        if (allocated(this%Pcoll_dust)) deallocate(this%Pcoll_dust)
        allocate(this%Pcoll_dust(1:this%ndust))
        this%Pcoll_dust = 0d0

        if (allocated(this%Pabs_pah)) deallocate(this%Pabs_pah)
        allocate(this%Pabs_pah(1:this%npah,1:this%nGroups))
        this%Pabs_pah = 0d0

        if (allocated(this%Pinj_pah)) deallocate(this%Pinj_pah)
        allocate(this%Pinj_pah(1:this%npah))
        this%Pinj_pah = 0d0

        if (allocated(this%Prad_pah)) deallocate(this%Prad_pah)
        allocate(this%Prad_pah(1:this%npah))
        this%Prad_pah = 0d0

        if (allocated(this%Prec_pah)) deallocate(this%Prec_pah)
        allocate(this%Prec_pah(1:this%npah))
        this%Prec_pah = 0d0

        if (allocated(this%Pcoll_pah)) deallocate(this%Pcoll_pah)
        allocate(this%Pcoll_pah(1:this%npah))
        this%Pcoll_pah = 0d0

        if (allocated(this%csa_dust)) deallocate(this%csa_dust)
        allocate(this%csa_dust(1:this%ndust,1:this%nGroups))
        this%csa_dust = 0d0

        if (allocated(this%css_dust)) deallocate(this%css_dust)
        allocate(this%css_dust(1:this%ndust,1:this%nGroups))
        this%css_dust = 0d0

        if (allocated(this%csr_dust)) deallocate(this%csr_dust)
        allocate(this%csr_dust(1:this%ndust,1:this%nGroups))
        this%csr_dust = 0d0

        if (allocated(this%csrat_dust)) deallocate(this%csrat_dust)
        allocate(this%csrat_dust(1:this%ndust,1:this%nGroups))
        this%csrat_dust = 0d0

        if (allocated(this%csa_pah)) deallocate(this%csa_pah)
        allocate(this%csa_pah(1:2*this%npah,1:this%nGroups))
        this%csa_pah = 0d0

        if (allocated(this%css_pah)) deallocate(this%css_pah)
        allocate(this%css_pah(1:2*this%npah,1:this%nGroups))
        this%css_pah = 0d0
        
        if (allocated(this%csr_pah)) deallocate(this%csr_pah)
        allocate(this%csr_pah(1:2*this%npah,1:this%nGroups))
        this%csr_pah = 0d0

        if (allocated(this%dustAbs)) deallocate(this%dustAbs)
        allocate(this%dustAbs(1:this%nGroups))
        this%dustAbs = 0d0

        if (allocated(this%l_a)) deallocate(this%l_a)
        allocate(this%l_a(1:this%ndust,1:this%nGroups))
        this%l_a = 0d0

        if (allocated(this%pahAbs)) deallocate(this%pahAbs)
        allocate(this%pahAbs(1:this%nGroups))
        this%pahAbs = 0d0

        if (allocated(this%Coulomb_factor)) deallocate(this%Coulomb_factor)
        allocate(this%Coulomb_factor(1:this%ndust,-1:this%nion_charges))
        this%Coulomb_factor = 1d0

        if (allocated(this%local_rad_ani)) deallocate(this%local_rad_ani)
        allocate(this%local_rad_ani(1:this%nGroups))
        this%local_rad_ani = 0d0

        if (allocated(this%local_solid_angle)) deallocate(this%local_solid_angle)
        allocate(this%local_solid_angle(1:this%nGroups))
        this%local_solid_angle = 0d0

        if (allocated(this%group_eV)) deallocate(this%group_eV)
        allocate(this%group_eV(1:this%nGroups))
        this%group_eV = 0d0

        if (allocated(this%rat_torque)) deallocate(this%rat_torque)
        allocate(this%rat_torque(1:this%ndust))
        this%rat_torque = 0d0

        if (allocated(this%IR_damp_factor)) deallocate(this%IR_damp_factor)
        allocate(this%IR_damp_factor(1:this%ndust))
        this%IR_damp_factor = 0d0
    
        this%initialised = .true.
    end subroutine init_dust_chemistry_info

    subroutine reset_dust_chemistry_info(this)
        ! Resets the values in the DustChemistryInfo derived type to 
        ! default values (e.g. at the beginning of each timestep)
        ! This needs to be done for each chemistry step
        ! this --> the DustChemistryInfo instance to reset
        class(DustChemistryInfo), intent(inout) :: this

        if (allocated(this%rho_dust)) this%rho_dust = 0d0
        if (allocated(this%n_dust)) this%n_dust = 0d0
        if (allocated(this%rho_pah)) this%rho_pah = 0d0
        if (allocated(this%n_pah)) this%n_pah = 0d0
        if (allocated(this%Z_dust)) this%Z_dust = 0d0
        if (allocated(this%Z_sigma)) this%Z_sigma = 0d0
        if (allocated(this%fcharge_pah)) this%fcharge_pah = 0d0
        if (allocated(this%T_dust)) this%T_dust = 0d0
        if (allocated(this%Pabs_dust)) this%Pabs_dust = 0d0
        if (allocated(this%Pinj_dust)) this%Pinj_dust = 0d0
        if (allocated(this%Prad_dust)) this%Prad_dust = 0d0
        if (allocated(this%Prec_dust)) this%Prec_dust = 0d0
        if (allocated(this%Pcoll_dust)) this%Pcoll_dust = 0d0
        if (allocated(this%Pabs_pah)) this%Pabs_pah = 0d0
        if (allocated(this%Pinj_pah)) this%Pinj_pah = 0d0
        if (allocated(this%Prad_pah)) this%Prad_pah = 0d0
        if (allocated(this%Prec_pah)) this%Prec_pah = 0d0
        if (allocated(this%Pcoll_pah)) this%Pcoll_pah = 0d0
        if (allocated(this%csa_dust)) this%csa_dust = 0d0
        if (allocated(this%css_dust)) this%css_dust = 0d0
        if (allocated(this%csr_dust)) this%csr_dust = 0d0
        if (allocated(this%csrat_dust)) this%csrat_dust = 0d0
        if (allocated(this%csa_pah)) this%csa_pah = 0d0
        if (allocated(this%css_pah)) this%css_pah = 0d0
        if (allocated(this%csr_pah)) this%csr_pah = 0d0
        if (allocated(this%dustAbs)) this%dustAbs = 0d0
        if (allocated(this%pahAbs)) this%pahAbs = 0d0
        if (allocated(this%Coulomb_factor)) this%Coulomb_factor = 1d0
        if (allocated(this%local_rad_ani)) this%local_rad_ani = 0d0
        if (allocated(this%local_solid_angle)) this%local_solid_angle = 0d0
        if (allocated(this%group_eV)) this%group_eV = 0d0
        if (allocated(this%rat_torque)) this%rat_torque = 0d0
        if (allocated(this%IR_damp_factor)) this%IR_damp_factor = 0d0
        this%H2_formation_rate = 0d0
    end subroutine reset_dust_chemistry_info
end module dustbin_types
