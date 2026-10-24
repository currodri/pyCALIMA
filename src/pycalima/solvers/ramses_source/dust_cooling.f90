module dust_cooling
    use constants, only: amu2g, kB, mC_amu, mO_amu,pi, mH, ln10
    use dust_commons
    use dust_charging
    use dust_utils
#ifdef RTZ
    use rtz_module, only:elements
#endif

    implicit none

    private

    public :: compute_dust_coll_heating_BH80
    public :: compute_dust_coll_heating
    public :: init_dust_coll_heating_BH80_cache
    public :: CollHeatPre, coll_heating_prepare, coll_heating_eval, coll_heating_dTg, coll_heating_dZ

    ! compute_dust_coll_heating split at the dust temperature: everything but the T_d dependence,
    ! for one bin and one gas state (coll_heating_prepare), then H_coll at any T_d
    ! (coll_heating_eval), with the operations of compute_dust_coll_heating in the same order
    type CollHeatPre
        logical :: hot = .false., blend = .false., hneutral = .false.
        real(dp) :: Tgas = 0d0, sum_rate = 0d0, supp = 0d0, supp_inv = 0d0   ! T > 1e3 K: tables, blend with BH80
        real(dp) :: sqrtTg = 0d0, s_const = 0d0, nh2_pf = 0d0, a0_h2 = 0d0   ! BH80: species with unit accommodation,
        real(dp) :: nco = 0d0, co_pf = 0d0, nhi_pf = 0d0, a0_h = 0d0         ! H2, CO and neutral H
        ! with deriv: d sum_rate / d T_gas, d sum_rate / d Z (dust_coll_charge), d supp_inv / d T_gas
        real(dp) :: dsum_dT = 0d0, dsum_dZ = 0d0, dsupp_inv = 0d0
    end type CollHeatPre

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
            ! BH83 accommodation coefficient: alpha_0 = 4*m_proj*m_grain/(m_proj+m_grain)^2
            bh80_h2_accomm_zero(i) = 8d0 * mH * dustbins_props(i)%mgrain / (2d0*mH + dustbins_props(i)%mgrain)**2d0
            bh80_h_accomm_zero(i) = 4d0 * mH * dustbins_props(i)%mgrain / (mH + dustbins_props(i)%mgrain)**2d0

            do iel = 1, n_elements
#ifdef RTZ
                mproj = elements(iel)%atomic_mass * amu2g
#else
                mproj = el_atomic_masses_amu(iel) * amu2g
#endif
                if (mproj <= 0d0) cycle
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

        type(CollHeatPre) :: pre

        call bh80_prepare(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,pre)
        Hcoll = bh80_eval(pre,Tgas,Td)

    end subroutine compute_dust_coll_heating_BH80

    subroutine bh80_prepare(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,pre)
        ! the T_d-independent part of the BH80 rate
        implicit none

        integer, intent(in) :: i_dust
        real(dp), intent(in) :: nH2,nCO
        real(dp), dimension(1:n_elements), intent(in) :: nElement
        real(dp), dimension(1:n_elements,1:n_elements), intent(in) :: xelem_ions
        real(dp), intent(in) :: Tgas
        type(CollHeatPre), intent(inout) :: pre

        integer :: j,iel,nions_loc
        real(dp) :: xion,Hcoll

        if (.not. bh80_cache_ready) call init_dust_coll_heating_BH80_cache

        ! 1. Common prefactor: 2 kB (Tgas - Td) sqrt(Tgas), finished in bh80_eval
        pre%sqrtTg = sqrt(Tgas)

        ! 2. Compute contribution from species whose accommodation factor is unity
        ! (everything except H2 and H neutral)
        Hcoll = 0d0
        do iel = 1, n_elements
#ifdef RTZ
            nions_loc = max(1, elements(iel)%n_ions)
#else
            nions_loc = 1
#endif
            if (nElement(iel) <= 1d-10) cycle

            do j = 1, nions_loc
                xion = xelem_ions(iel,j)
                if (xion <= 1d-10) cycle

                ! Skip H neutral here as it depends on Td
                if (iel == 1 .and. j == 1) cycle

                Hcoll = Hcoll + nElement(iel) * xion * bh80_species_prefactor(i_dust,iel)
            end do
        end do
        pre%s_const = Hcoll

        ! 3. H2 (accommodation depends on Td), CO (unity) and H neutral (depends on Td)
        pre%nh2_pf = nH2 * bh80_h2_prefactor(i_dust)
        pre%a0_h2 = bh80_h2_accomm_zero(i_dust)
        pre%nco = nCO                ! kept apart: Hcoll + nCO*prefactor is one fused multiply-add
        pre%co_pf = bh80_co_prefactor(i_dust)
        pre%hneutral = nElement(1) > 1d-20 .and. xelem_ions(1,1) > 1d-20
        if (pre%hneutral) then
            pre%nhi_pf = nElement(1) * xelem_ions(1,1) * bh80_species_prefactor(i_dust,1)
            pre%a0_h = bh80_h_accomm_zero(i_dust)
        end if

    end subroutine bh80_prepare

    real(dp) function bh80_eval(pre,Tgas,Td)
        ! the BH80 rate at dust temperature Td from bh80_prepare
        implicit none

        type(CollHeatPre), intent(in) :: pre
        real(dp), intent(in) :: Tgas,Td

        real(dp) :: accomm_factor,prefactor,e,Hcoll

        prefactor = 2d0 * kB * (Tgas - Td) * pre%sqrtTg
        e = exp(-sqrt((Td+Tgas)/250d0))
        accomm_factor = (1d0 - pre%a0_h2) * e + pre%a0_h2
        Hcoll = pre%s_const + pre%nh2_pf * accomm_factor
        Hcoll = Hcoll + pre%nco * pre%co_pf
        if (pre%hneutral) then
            accomm_factor = (1d0 - pre%a0_h) * e + pre%a0_h
            Hcoll = Hcoll + pre%nhi_pf * accomm_factor
        end if
        bh80_eval = prefactor * Hcoll

    end function bh80_eval

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

        type(CollHeatPre) :: pre

        call coll_heating_prepare(i_dust,ne,nElement,xelem_ions,nH2,nCO,Tgas,dust_charge,pre)
        Hcoll = coll_heating_eval(pre,Td)

    end subroutine compute_dust_coll_heating

    subroutine coll_heating_prepare(i_dust,ne,nElement,xelem_ions,nH2,nCO,Tgas,dust_charge,pre,deriv)
        ! compute_dust_coll_heating but its dependence on the dust temperature (the tables of the
        ! electrons and ions, and the BH80 terms that do not depend on Td). deriv (optional): also
        ! what coll_heating_dTg and coll_heating_dZ need (the slopes of the tables)
        implicit none

        integer, intent(in) :: i_dust
        real(dp), intent(in) :: ne
        real(dp), dimension(1:n_elements), intent(in) :: nElement
        real(dp), dimension(1:n_elements,1:n_elements), intent(in) :: xelem_ions
        real(dp), intent(in) :: nH2,nCO
        real(dp), intent(in) :: Tgas
        real(dp), intent(in) :: dust_charge
        type(CollHeatPre), intent(out) :: pre
        logical, intent(in), optional :: deriv

        integer :: j,iel,nT,nphi,nions_loc
        real(dp) :: lT,cooling_rate
        real(dp) :: xion,phi_charge

        real(dp) :: sum_rate, sum_element
        real(dp) :: dsl, dsz, sx, sy
        integer :: idx_T
        logical :: want_d

        want_d = .false.
        if (present(deriv)) want_d = deriv
        dsl = 0d0
        dsz = 0d0
        pre%Tgas = Tgas
        pre%hot = Tgas > 1d3
        if (pre%hot) then
            lT = log10(Tgas)

            sum_rate = 0d0

            ! 1. Do first the contribution from electron collisions
            nT = dustbins_props(i_dust)%collisional_tab(0)%npts(1)
            nphi = dustbins_props(i_dust)%collisional_tab(0)%npts(2)
            idx_T = -1
            if (dust_coll_charge) then
                phi_charge = dust_charge * dustbins_props(i_dust)%phi_prefact(-1) ! [eV]
                call dustbins_props(i_dust)%collisional_tab(0)%interpolate(lT, phi_charge, cooling_rate, idx_x=idx_T)
                cooling_rate = exp(cooling_rate * ln10)
                sum_rate = sum_rate + ne * cooling_rate
                if (want_d) then
                    call dustbins_props(i_dust)%collisional_tab(0)%slope_2d(lT, phi_charge, sx, sy)
                    dsl = dsl + ne * cooling_rate * sx
                    dsz = dsz + ne * cooling_rate * sy * dustbins_props(i_dust)%phi_prefact(-1)
                end if
            else
                ! 1D interpolation: use stored phi=0 index
                call dustbins_props(i_dust)%collisional_tab(0)%interpolate(lT, cooling_rate, idx_x=idx_T)
                cooling_rate = exp(cooling_rate * ln10)
                sum_rate = sum_rate + ne * cooling_rate
                if (want_d) then
                    call dustbins_props(i_dust)%collisional_tab(0)%slope_1d(lT, sx)
                    dsl = dsl + ne * cooling_rate * sx
                end if
            end if

            ! 2. Loop over all elements
            species_loop: do iel = 1, n_elements

                ! Skip if tables not initialized or element abundance is negligible
                if (nElement(iel) <= 1d-10) cycle
                if (.not. dustbins_props(i_dust)%collisional_tab(iel)%initialised) cycle

                nT = dustbins_props(i_dust)%collisional_tab(iel)%npts(1)
                nphi = dustbins_props(i_dust)%collisional_tab(iel)%npts(2)

                if (dust_coll_charge) then
                    ! Add contributions for all charge states of this element
                    nions_loc = n_elements
#ifdef RTZ
                    nions_loc = max(1, elements(iel)%n_ions)
#endif
                    sum_element = 0d0
                    do j = 1, nions_loc
                        xion = xelem_ions(iel, j)
                        if (xion <= 1d-5) cycle
                        phi_charge = dust_charge * dustbins_props(i_dust)%phi_prefact(j-1) ! [eV]
                        call dustbins_props(i_dust)%collisional_tab(iel)%interpolate(lT, phi_charge, cooling_rate, idx_x=idx_T)
                        if (cooling_rate < -30d0) cycle
                        sum_element = sum_element + xion * exp(cooling_rate * ln10)
                        if (want_d) then
                            call dustbins_props(i_dust)%collisional_tab(iel)%slope_2d(lT, phi_charge, sx, sy)
                            dsl = dsl + nElement(iel) * xion * exp(cooling_rate * ln10) * sx
                            dsz = dsz + nElement(iel) * xion * exp(cooling_rate * ln10) * sy * &
                                  dustbins_props(i_dust)%phi_prefact(j-1)
                        end if
                    end do
                    sum_rate = sum_rate + nElement(iel) * sum_element
                else
                    ! No charge dependence: just add contribution from the total abundance of this element
                    call dustbins_props(i_dust)%collisional_tab(iel)%interpolate(lT, cooling_rate, idx_x=idx_T)
                    if (cooling_rate >= -30d0) then
                        sum_rate = sum_rate + nElement(iel) * exp(cooling_rate * ln10)
                        if (want_d) then
                            call dustbins_props(i_dust)%collisional_tab(iel)%slope_1d(lT, sx)
                            dsl = dsl + nElement(iel) * exp(cooling_rate * ln10) * sx
                        end if
                    end if
                end if

            end do species_loop
            pre%sum_rate = sum_rate
            ! d 10**y / d T_gas = 10**y (dy / dlog10 T) / T_gas; d 10**y / d Z = 10**y ln10 (dy / dphi) dphi/dZ
            pre%dsum_dT = dsl / Tgas
            pre%dsum_dZ = ln10 * dsz

            ! 3. Always blend with BH80 for smooth transition at low T
            pre%blend = Tgas < 1d5
            if (pre%blend) then
                pre%supp = 1d0 - 1d0/(1d0+exp(-1d1*(log10(Tgas)-4d0)))
                pre%supp_inv = 1d0 / (1d0 + exp(-1d1*(log10(Tgas) - 4d0)))
                pre%dsupp_inv = pre%supp_inv * (1d0 - pre%supp_inv) * 1d1 / (Tgas * ln10)
                call bh80_prepare(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,pre)
            end if
        else
            call bh80_prepare(i_dust,nElement,xelem_ions,nH2,nCO,Tgas,pre)
        end if

    end subroutine coll_heating_prepare

    real(dp) function coll_heating_dTg(pre,Td)
        ! d H_coll / d T_gas at dust temperature Td, fixed Td and charge (pre prepared with deriv)
        implicit none
        type(CollHeatPre), intent(in) :: pre
        real(dp), intent(in) :: Td
        real(dp) :: dH

        if (pre%hot) then
            dH = pre%sum_rate + (pre%Tgas - Td) * pre%dsum_dT
            if (pre%blend) dH = pre%supp_inv * dH + pre%dsupp_inv * pre%sum_rate * (pre%Tgas - Td) &
                                + pre%supp * bh80_dTg(pre,Td) - pre%dsupp_inv * bh80_eval(pre,pre%Tgas,Td)
        else
            dH = bh80_dTg(pre,Td)
        end if
        coll_heating_dTg = dH
    end function coll_heating_dTg

    real(dp) function coll_heating_dZ(pre,Td)
        ! d H_coll / d Z (the grain charge, dust_coll_charge) at dust temperature Td, fixed T_gas and Td
        implicit none
        type(CollHeatPre), intent(in) :: pre
        real(dp), intent(in) :: Td
        real(dp) :: dH

        dH = 0d0
        if (pre%hot) then
            dH = (pre%Tgas - Td) * pre%dsum_dZ
            if (pre%blend) dH = pre%supp_inv * dH
        end if
        coll_heating_dZ = dH
    end function coll_heating_dZ

    real(dp) function bh80_dTg(pre,Td)
        ! d/d T_gas of bh80_eval at fixed Td
        implicit none
        type(CollHeatPre), intent(in) :: pre
        real(dp), intent(in) :: Td
        real(dp) :: Tg, x, e, de, P, dP, S, dS

        Tg = pre%Tgas
        P = 2d0 * kB * (Tg - Td) * pre%sqrtTg
        dP = 2d0 * kB * (pre%sqrtTg + (Tg - Td) / (2d0 * pre%sqrtTg))
        x = (Td + Tg) / 250d0
        e = exp(-sqrt(x))
        de = -e / (500d0 * sqrt(x))
        S = pre%s_const + pre%nh2_pf * ((1d0 - pre%a0_h2) * e + pre%a0_h2) + pre%nco * pre%co_pf
        dS = pre%nh2_pf * (1d0 - pre%a0_h2) * de
        if (pre%hneutral) then
            S = S + pre%nhi_pf * ((1d0 - pre%a0_h) * e + pre%a0_h)
            dS = dS + pre%nhi_pf * (1d0 - pre%a0_h) * de
        end if
        bh80_dTg = dP * S + P * dS
    end function bh80_dTg

    real(dp) function coll_heating_eval(pre,Td)
        ! H_coll [erg/s] at dust temperature Td from coll_heating_prepare
        implicit none

        type(CollHeatPre), intent(in) :: pre
        real(dp), intent(in) :: Td

        real(dp) :: Hcoll

        if (pre%hot) then
            Hcoll = pre%sum_rate * (pre%Tgas - Td)
            if (pre%blend) Hcoll = pre%supp_inv * Hcoll + pre%supp * bh80_eval(pre,pre%Tgas,Td)
        else
            Hcoll = bh80_eval(pre,pre%Tgas,Td)
        end if
        coll_heating_eval = Hcoll

    end function coll_heating_eval

end module dust_cooling