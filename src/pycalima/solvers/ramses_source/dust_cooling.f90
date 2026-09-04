module dust_cooling
    use constants, only: amu2g, kB, mC_amu, mO_amu,pi, mH
    use dust_commons
    use dust_charging
#ifdef RTZ
    use rtz_module, only:elements
#endif

    implicit none

    private

    public :: compute_dust_coll_heating_BH80
    public :: compute_dust_coll_heating
    public :: init_dust_coll_heating_BH80_cache

    logical, save :: bh80_cache_ready = .false.
    real(dp), allocatable, save :: bh80_h2_prefactor(:)
    real(dp), allocatable, save :: bh80_co_prefactor(:)
    real(dp), allocatable, save :: bh80_h2_accomm_zero(:)
    real(dp), allocatable, save :: bh80_h_accomm_zero(:)
    real(dp), allocatable, save :: bh80_species_prefactor(:,:)

    contains

    subroutine init_dust_coll_heating_BH80_cache
        implicit none

        integer :: iel, i
        real(dp) :: agrain, mproj

        if (bh80_cache_ready) return

        if (allocated(bh80_h2_prefactor)) deallocate(bh80_h2_prefactor)
        if (allocated(bh80_co_prefactor)) deallocate(bh80_co_prefactor)
        if (allocated(bh80_h2_accomm_zero)) deallocate(bh80_h2_accomm_zero)
        if (allocated(bh80_h_accomm_zero)) deallocate(bh80_h_accomm_zero)
        if (allocated(bh80_species_prefactor)) deallocate(bh80_species_prefactor)

        allocate(bh80_h2_prefactor(1:ndust))
        allocate(bh80_co_prefactor(1:ndust))
        allocate(bh80_h2_accomm_zero(1:ndust))
        allocate(bh80_h_accomm_zero(1:ndust))
        allocate(bh80_species_prefactor(1:ndust,1:n_elements))

        bh80_h2_prefactor(:) = 0d0
        bh80_co_prefactor(:) = 0d0
        bh80_h2_accomm_zero(:) = 0d0
        bh80_h_accomm_zero(:) = 0d0
        bh80_species_prefactor(:,:) = 0d0

        do i = 1, ndust
            agrain = dustbins_props(i)%asize_cm
            bh80_h2_prefactor(i) = sqrt(8d0*kB/(pi*2d0*mH)) * pi * agrain**2d0
            bh80_co_prefactor(i) = sqrt(8d0*kB/(pi*(mC_amu+mO_amu)*amu2g)) * pi * agrain**2d0
            bh80_h2_accomm_zero(i) = 4d0 * mH * dustbins_props(i)%mgrain / (2d0*mH + dustbins_props(i)%mgrain)**2d0
            bh80_h_accomm_zero(i) = 2d0 * mH * dustbins_props(i)%mgrain / (mH + dustbins_props(i)%mgrain)**2d0

            do iel = 1, n_elements
#ifdef RTZ
                mproj = elements(iel)%atomic_mass * amu2g
#else
                mproj = el_atomic_masses_amu(iel) * amu2g
#endif
                bh80_species_prefactor(i,iel) = sqrt(8d0*kB/(pi*mproj)) * pi * agrain**2d0
            end do
        end do

        bh80_cache_ready = .true.
    end subroutine init_dust_coll_heating_BH80_cache

    subroutine compute_dust_coll_heating_BH80(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,Td,Hcoll)
        implicit none

        integer, intent(in) :: i_dust
        real(dp), intent(in) :: nH2,nCO
        real(dp), dimension(1:n_elements), intent(in) :: nElement
        real(dp), dimension(1:n_elements,1:n_elements), intent(in) :: xelem_ions
        real(dp), intent(in) :: Tgas
        real(dp), intent(in) :: Td

        real(dp), intent(inout) :: Hcoll

        integer :: j,iel,nions_loc
        real(dp) :: accomm_factor,prefactor,accomm_factor_zero,xion

        if (.not. bh80_cache_ready) call init_dust_coll_heating_BH80_cache

        Hcoll = 0d0

        ! 1. Compute common prefactor
        prefactor = 2d0 * kB * (Tgas - Td) * sqrt(Tgas)

        ! 2. Compute contribution from H2 molecules
        ! Accommodation factor based on Burke & Hollenbach (1980)
        ! (No enhancement factor due to grain charge)
        accomm_factor_zero = bh80_h2_accomm_zero(i_dust)
        accomm_factor = (1d0 - accomm_factor_zero) * exp(-sqrt((Td+Tgas)/250d0)) + accomm_factor_zero
        Hcoll = Hcoll + nH2 * bh80_h2_prefactor(i_dust) * accomm_factor

        ! 2b. CO molecules use accommodation coefficient unity
        Hcoll = Hcoll + nCO * bh80_co_prefactor(i_dust)

        ! 3. Compute contribution from all elements and ionisation states.
        ! For H neutral use BH80 accommodation; for He and heavier ions/neutrals use unity.
        do iel = 1, n_elements
#ifdef RTZ
            nions_loc = max(1, elements(iel)%n_ions)
#else
            nions_loc = 1
#endif
            if (nElement(iel) <= 1d-20) cycle

            do j = 1, nions_loc
                xion = xelem_ions(iel,j)
                if (xion <= 1d-20) cycle

                if (iel == 1 .and. j == 1) then
                    accomm_factor_zero = bh80_h_accomm_zero(i_dust)
                    accomm_factor = (1d0 - accomm_factor_zero) * exp(-sqrt((Td+Tgas)/250d0)) + accomm_factor_zero
                else
                    accomm_factor = 1d0
                end if

                Hcoll = Hcoll + nElement(iel) * xion * bh80_species_prefactor(i_dust,iel) * accomm_factor
            end do
        end do

        Hcoll = prefactor * Hcoll

    end subroutine compute_dust_coll_heating_BH80

    subroutine compute_dust_coll_heating(i_dust,ne,nElement,xelem_ions,&
                                        Coulomb_factor,nH2,nCO,Tgas,Td,&
                                        dust_charge,Hcoll)
        implicit none

        integer, intent(in) :: i_dust
        real(dp), intent(in) :: ne
        real(dp), dimension(1:n_elements), intent(in) :: nElement
        real(dp), dimension(1:n_elements,1:n_elements), intent(in) :: xelem_ions
        real(dp), dimension(-1:n_elements),intent(in) :: Coulomb_factor
        real(dp), intent(in) :: nH2,nCO
        real(dp), intent(in) :: Tgas
        real(dp), intent(in) :: Td,dust_charge

        real(dp), intent(inout) :: Hcoll

        integer :: j,iel,iphi0,nT,nphi,nions_loc,izion,jj
                real(dp) :: lT,cooling_rate
        real(dp) :: agrain,D
        real(dp) :: dT
        real(dp) :: xion,phi_charge
        real(dp) :: supp_factor,supp_factor_inv
        real(dp):: Hcoll_HM80
        
        if (Tgas > 1d3) then
            Hcoll    = 0d0
            lT = log10(Tgas)
            dT = Tgas - Td

            agrain = dustbins_props(i_dust)%asize_cm

            ! 1. Do first the contribution from electron collisions
            nT = dustbins_props(i_dust)%collisional_tab(0)%npts(1)
            nphi = dustbins_props(i_dust)%collisional_tab(0)%npts(2)
            if (dust_coll_charge) then
                phi_charge = dust_charge * dustbins_props(i_dust)%phi_prefact(-1) ! [eV]
                call interpolate2D(dustbins_props(i_dust)%collisional_tab(0)%tab1d(1:nT,1), &
                                    dustbins_props(i_dust)%collisional_tab(0)%tab1d(1:nphi,2), &
                                    dustbins_props(i_dust)%collisional_tab(0)%tab2d(1:nT,1:nphi,1), &
                                    nT, nphi, lT, phi_charge, cooling_rate)
                cooling_rate = 10d0**cooling_rate
                Hcoll = Hcoll + Coulomb_factor(-1) * ne * cooling_rate * dT
            else
                ! 1D interpolation: use stored phi=0 index
                iphi0 = dustbins_props(i_dust)%collisional_tab(0)%ipos_zero(2)
                call interpolate1D(dustbins_props(i_dust)%collisional_tab(0)%tab1d(1:nT,1), &
                                    dustbins_props(i_dust)%collisional_tab(0)%tab2d(1:nT,iphi0,1), &
                                    nT, lT, cooling_rate)
                cooling_rate = 10d0**cooling_rate
                Hcoll = Hcoll + ne * cooling_rate * dT
            end if

            ! 2. Loop over all elements
            species_loop: do iel = 1, n_elements
                
                ! Skip if tables not initialized
                if (.not. dustbins_props(i_dust)%collisional_tab(iel)%initialised) cycle
                
                nT = dustbins_props(i_dust)%collisional_tab(iel)%npts(1)
                nphi = dustbins_props(i_dust)%collisional_tab(iel)%npts(2)

                if (dust_coll_charge) then
                    ! Add contributions for all charge states of this element
                    nions_loc = n_elements
#ifdef RTZ
                    nions_loc = max(1, elements(iel)%n_ions)
#endif
                    do j = 1, nions_loc
                        xion = xelem_ions(iel, j)
                        if (xion <= 1d-20) cycle
                        phi_charge = dust_charge * dustbins_props(i_dust)%phi_prefact(j-1) ! [eV]
                        call interpolate2D(dustbins_props(i_dust)%collisional_tab(iel)%tab1d(1:nT,1), &
                                            dustbins_props(i_dust)%collisional_tab(iel)%tab1d(1:nphi,2), &
                                            dustbins_props(i_dust)%collisional_tab(iel)%tab2d(1:nT,1:nphi,1), &
                                            nT, nphi, lT, phi_charge, cooling_rate)
                        cooling_rate = 10d0**cooling_rate
                        Hcoll = Hcoll + Coulomb_factor(j-1) * nElement(iel) * xion * cooling_rate * dT
                    end do
                else
                    ! No charge dependence: just add contribution from the total abundance of this element
                    iphi0 = dustbins_props(i_dust)%collisional_tab(iel)%ipos_zero(2)
                    call interpolate1D(dustbins_props(i_dust)%collisional_tab(iel)%tab1d(1:nT,1), &
                                        dustbins_props(i_dust)%collisional_tab(iel)%tab2d(1:nT,iphi0,1), &
                                        nT, lT, cooling_rate)
                    cooling_rate = 10d0**cooling_rate
                    Hcoll = Hcoll + nElement(iel) * cooling_rate * dT
                end if
            
            end do species_loop

            ! 3. Now compute the low-temperature soft-cube collisional heating from
            ! Hollenbach and McKee (1980) for Tgas < 1e4 K
            if (Tgas .lt. 1d5 .and. (dust_coll_lowT)) then
                supp_factor = 1d0 - 1d0/(1d0+exp(-1d1*(log10(Tgas)-4d0)))
                supp_factor_inv = 1d0 / (1d0 + exp(-1d1*(log10(Tgas) - 4d0)))
                call compute_dust_coll_heating_BH80(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,Td,Hcoll_HM80)
                Hcoll = supp_factor_inv * Hcoll + supp_factor * Hcoll_HM80
            end if
        else
            call compute_dust_coll_heating_BH80(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,Td,Hcoll)
        end if
    end subroutine compute_dust_coll_heating

end module dust_cooling