module dust_rates
    use amr_parameters, only: dp
    use constants
    use dust_commons
    use dust_utils
#ifdef RTZ
    use rtz_module, only: elements
#endif

    implicit none

    public :: compute_rate_caches

    ! Cached relative velocities and rate prefactors to avoid redundant calculations and RNG noise
    real(dp), allocatable, save :: cached_v_rel_dust_dust(:,:)
    real(dp), allocatable, save :: cached_v_rel_pah_dust(:,:)
    real(dp), allocatable, save :: cached_p_stick_dust_dust(:,:)
    real(dp), allocatable, save :: cached_p_stick_pah_dust(:,:)
    real(dp), allocatable, save :: cached_chi_frag_dust(:,:,:)
    real(dp), allocatable, save :: cached_chi_frag_pah(:,:,:)
    real(dp), allocatable, save :: cached_chi_frag_dest(:,:)

    contains

    subroutine ensure_rate_caches(dust_info)
        ! Ensure that the rate cache arrays are allocated with the correct dimensions.
        ! dust_info --> DustChemistryInfo type with the current number of dust and PAH bins.
        class(DustChemistryInfo), intent(in) :: dust_info
        integer :: ndust, npah
        ndust = dust_info%ndust
        npah = dust_info%npah
        
        if (.not. allocated(cached_v_rel_dust_dust)) then
            allocate(cached_v_rel_dust_dust(ndust, ndust))
            allocate(cached_p_stick_dust_dust(ndust, ndust))
            allocate(cached_chi_frag_dust(ndust, ndust, ndust))
            allocate(cached_chi_frag_pah(ndust, ndust, npah))
            allocate(cached_chi_frag_dest(ndust, ndust))
        else if (size(cached_v_rel_dust_dust,1) /= ndust) then
            deallocate(cached_v_rel_dust_dust, cached_p_stick_dust_dust, &
                       cached_chi_frag_dust, cached_chi_frag_pah, cached_chi_frag_dest)
            allocate(cached_v_rel_dust_dust(ndust, ndust))
            allocate(cached_p_stick_dust_dust(ndust, ndust))
            allocate(cached_chi_frag_dust(ndust, ndust, ndust))
            allocate(cached_chi_frag_pah(ndust, ndust, npah))
            allocate(cached_chi_frag_dest(ndust, ndust))
        end if
        
        if (.not. allocated(cached_v_rel_pah_dust)) then
            allocate(cached_v_rel_pah_dust(npah, ndust))
            allocate(cached_p_stick_pah_dust(npah, ndust))
        else if (size(cached_v_rel_pah_dust,1) /= npah .or. size(cached_v_rel_pah_dust,2) /= ndust) then
            deallocate(cached_v_rel_pah_dust, cached_p_stick_pah_dust)
            allocate(cached_v_rel_pah_dust(npah, ndust))
            allocate(cached_p_stick_pah_dust(npah, ndust))
        end if
    end subroutine ensure_rate_caches

    subroutine compute_rate_caches(dust_info, nElement)
        ! Compute and cache relative velocities, sticking probabilities, and shattered
        ! fragment distributions for all grain pairs to avoid redundant computations inside
        ! the ODE solver RHS evaluations.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        use dust_dynamics, only: grain_relative_velocity
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: nElement(:)
        integer :: ii, kk, jj, pp, C_index, ii1, ii2, kk_loc, dust_start, dust_end
        real(dp) :: temp_sigma, temp_L
        real(dp) :: v_rel, v_coag, p_stick
        real(dp) :: reduced_mass, v_stick_thresh
        logical :: interact_pah_flag
        
        real(dp) :: nO, vth, N_mono, f_ice_ii, enhan_factor_pair
        real(dp), dimension(1:max(1, dust_info%ndust)) :: enhan_factor
        
        call ensure_rate_caches(dust_info)
        
        if (dust_eq_test) then
            temp_sigma = 5.67d5 * (dust_info%local_nH/1d2)**(-0.25d0)
            temp_L = 10d0 * pc2cm * (dust_info%local_nH/1d2)**(-1d0/3d0)
        else
            temp_sigma = dust_info%local_sigma
            temp_L = dust_info%local_dx
        end if

        if (poppe_ice_enhancement) then
            if (size(nElement) >= 8) then
                nO = nElement(8)
            else
                nO = 0d0
            end if
            vth = 3624.65d0 * sqrt(dust_info%local_Tk)
            N_mono = 2d0 * nO * vth / (max(dust_info%local_G0, 1d-5) * 3d5)
            do ii = 1, dust_info%ndust
                if (dust_info%T_dust(ii) >= 100d0) then
                    f_ice_ii = 0d0
                else
                    f_ice_ii = 1d0 - exp(-N_mono)
                end if
                enhan_factor(ii) = 1d0 + 3d0 * f_ice_ii
            end do
        end if

        ! 1. Cache dust-dust relative velocities and sticking probabilities
        do jj = 1, ndchemtype
            ii1 = istart_chemtype(jj)
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            do ii = ii1, ii2
                do kk = ii, ii2
                    kk_loc = kk - ii1 + 1
                    v_rel = grain_relative_velocity(dust_velocity_model,dust_info%local_Tk,&
                                                    dust_info%local_rho,dust_info%local_nH,&
                                                    temp_sigma,dust_info%local_mu,temp_L,&
                                                    dustbins_props(ii)%asize_cm,&
                                                    dustbins_props(kk)%asize_cm,&
                                                    dustbins_props(ii)%sgrain,&
                                                    dustbins_props(kk)%sgrain,&
                                                    dustbins_props(ii)%mgrain,&
                                                    dustbins_props(kk)%mgrain)
                    cached_v_rel_dust_dust(ii,kk) = v_rel
                    cached_v_rel_dust_dust(kk,ii) = v_rel
                    
                    if (allocated(dustbins_props(ii)%vthresh_coag)) then
                        if (kk_loc <= size(dustbins_props(ii)%vthresh_coag)) then
                            v_coag = dustbins_props(ii)%vthresh_coag(kk_loc)
                            if (poppe_ice_enhancement) then
                                enhan_factor_pair = 0.5d0 * (enhan_factor(ii) + enhan_factor(kk))
                                v_coag = enhan_factor_pair * v_coag
                            end if
                            p_stick = sticking_probability_from_velocity(v_rel, v_coag)
                        else
                            p_stick = 0d0
                        end if
                    else
                        p_stick = 0d0
                    end if
                    cached_p_stick_dust_dust(ii,kk) = p_stick
                    cached_p_stick_dust_dust(kk,ii) = p_stick
                end do
            end do
        end do

        ! 2. Cache PAH-dust relative velocities and sticking probabilities
#if NPAH > 0
        do pp = 1, dust_info%npah
            dust_start = pahbins_props(pp)%dust_index_interact
            if (dust_start <= 0) cycle
            dust_end = min(dust_info%ndust, dust_start + pahbins_props(pp)%nd_bins - 1)
            
            do kk = dust_start, dust_end
                if (.not. dustbins_props(kk)%interact_pah) cycle
                v_rel = grain_relative_velocity(dust_velocity_model,dust_info%local_Tk,dust_info%local_rho,&
                                               dust_info%local_nH,temp_sigma,dust_info%local_mu,temp_L,&
                                               dustbins_props(kk)%asize_cm,pahbins_props(pp)%apah_cm,&
                                               dustbins_props(kk)%sgrain,pahbins_props(pp)%spah,&
                                               dustbins_props(kk)%mgrain,pahbins_props(pp)%mpah)
                cached_v_rel_pah_dust(pp,kk) = v_rel
                
                reduced_mass = 5d-1 * (pahbins_props(pp)%mpah * dustbins_props(kk)%mgrain) / &
                               (pahbins_props(pp)%mpah + dustbins_props(kk)%mgrain)
                v_stick_thresh = sqrt(2d0 * eV2erg / max(reduced_mass, 1d-99))
                cached_p_stick_pah_dust(pp,kk) = sticking_probability_from_velocity(v_rel, v_stick_thresh)
            end do
        end do
#endif

        ! 3. Cache shattering fragments
        do jj = 1, ndchemtype
            ii1 = istart_chemtype(jj)
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1
            interact_pah_flag = dustbins_props(ii1)%interact_pah .and. dust_info%npah > 0
            
            do ii = ii1, ii2
                do kk = ii1, ii2
                    call compute_shattered_fragments_direct(dust_info,ii,kk,temp_sigma,temp_L,&
                                                         cached_chi_frag_dest(ii,kk),&
                                                         cached_chi_frag_dust(ii,kk,ii1:ii2),&
                                                         cached_chi_frag_pah(ii,kk,1:dust_info%npah),&
                                                         interact_pah_flag)
                end do
            end do
        end do
    end subroutine compute_rate_caches

    subroutine compute_shattered_fragments_direct(dust_info,id1,id2,local_sigma,local_L,chi_frag_dest_out,chi_frag_out,chi_frag_pah_out,interact_pah)
        ! Compute the fragmented mass distribution resulting from the shattering collision
        ! between target grain id1 and impactor grain id2.
        ! dust_info          --> DustChemistryInfo type with target/impactor properties.
        ! id1                --> Target grain bin index.
        ! id2                --> Impactor grain bin index.
        ! local_sigma        --> Local velocity dispersion [cm s-1].
        ! local_L            --> Local spatial scale [cm].
        ! chi_frag_dest_out  <-- Fraction of shattered mass destroyed (sputtered) to gas phase.
        ! chi_frag_out       <-- Mass distribution of shattered fragments in dust bins.
        ! chi_frag_pah_out   <-- Mass distribution of shattered fragments in PAH bins.
        ! interact_pah       --> Logical flag indicating whether PAH-dust interaction is active.
        implicit none

        class(DustChemistryInfo), intent(in) :: dust_info
        integer, intent(in) :: id1, id2
        real(dp), intent(in) :: local_sigma, local_L
        logical, intent(in) :: interact_pah
        real(dp), intent(out) :: chi_frag_dest_out
        real(dp), dimension(:), intent(out) :: chi_frag_out
        real(dp), dimension(:), intent(out) :: chi_frag_pah_out

        integer :: pp_local, ll_local, ii1, ii2, nearest_idx
        integer :: global_ii1, global_ii2, ll_global
        real(dp) :: E_imp, phi, m_ej, m_remnant, m_max, m_min
        real(dp) :: prefactor, m_tot, denom, logdist, min_logdist
        logical :: remnant_assigned
        real(dp) :: m_max_pow, m_min_pow
        real(dp) :: v_rel_out

        ii1 = lbound(chi_frag_out, 1)
        ii2 = ubound(chi_frag_out, 1)

        global_ii1 = istart_chemtype(dustbins_props(id1)%interact_group)
        global_ii2 = global_ii1 + dustbins_per_chemtype(dustbins_props(id1)%interact_group) - 1

        v_rel_out = cached_v_rel_dust_dust(id1, id2)

        E_imp = 5d-1 * (dustbins_props(id1)%mgrain*dustbins_props(id2)%mgrain) / &
                (dustbins_props(id1)%mgrain+dustbins_props(id2)%mgrain) * v_rel_out**2d0
        phi = E_imp / (dustbins_props(id1)%mgrain*dustbins_props(id1)%catastrophic_spec_energy)
        m_ej = phi / (1d0 + phi) * dustbins_props(id1)%mgrain

        m_remnant = dustbins_props(id1)%mgrain - m_ej
        m_max = 2d-2*m_ej
        m_min = 1d-6*m_max

        if (m_ej > tiny(m_ej)) then
            m_max_pow = m_max**slope_frag_func
            m_min_pow = m_min**slope_frag_func
            denom = m_max_pow - m_min_pow
            if (abs(denom) > tiny(denom)) then
                prefactor = m_ej / denom
            else
                prefactor = 0d0
            end if
        else
            m_max_pow = 0d0
            m_min_pow = 0d0
            prefactor = 0d0
        end if

        chi_frag_dest_out = 0d0
        chi_frag_out(:) = 0d0
        chi_frag_pah_out(:) = 0d0
        if (prefactor > 0d0) then
#if NPAH > 0
            if (interact_pah) then
                if (m_min < pahbins_props(1)%mpah_min) then
                    chi_frag_dest_out = prefactor * (min(pahbins_props(1)%mpah_min,m_max)**slope_frag_func - m_min_pow)
                end if
            else
                if (m_min < dustbins_props(global_ii1)%mgrain_min) then
                    chi_frag_dest_out = prefactor * (min(dustbins_props(global_ii1)%mgrain_min,m_max)**slope_frag_func - m_min_pow)
                end if
            end if
#else
            if (m_min < dustbins_props(global_ii1)%mgrain_min) then
                chi_frag_dest_out = prefactor * (min(dustbins_props(global_ii1)%mgrain_min,m_max)**slope_frag_func - m_min_pow)
            end if
#endif
        end if

#if NPAH > 0
        if (interact_pah .and. prefactor > 0d0) then
            do pp_local = 1, dust_info%npah
                if ((m_min.ge.pahbins_props(pp_local)%mpah_max).or.(m_max<pahbins_props(pp_local)%mpah_min)) then
                    chi_frag_pah_out(pp_local) = 0d0
                else
                    chi_frag_pah_out(pp_local) = prefactor * (min(pahbins_props(pp_local)%mpah_max,m_max)**slope_frag_func - &
                                                    max(pahbins_props(pp_local)%mpah_min,m_min)**slope_frag_func)
                end if
            end do
        end if
#endif

        if (prefactor > 0d0) then
            do ll_local = ii1, ii2
                ll_global = ll_local + global_ii1 - ii1
                if ((m_min.ge.dustbins_props(ll_global)%mgrain_max).or.(m_max<dustbins_props(ll_global)%mgrain_min)) then
                    chi_frag_out(ll_local) = 0d0
                else
                    chi_frag_out(ll_local) = prefactor * (min(dustbins_props(ll_global)%mgrain_max,m_max)**slope_frag_func - &
                                                max(dustbins_props(ll_global)%mgrain_min,m_min)**slope_frag_func)
                end if
            end do
        end if

        remnant_assigned = .false.
        if (m_remnant > 0d0) then
            if (interact_pah) then
                do pp_local = 1, dust_info%npah
                    if ((pahbins_props(pp_local)%mpah_min.le.m_remnant).and.(m_remnant<pahbins_props(pp_local)%mpah_max)) then
                        chi_frag_pah_out(pp_local) = chi_frag_pah_out(pp_local) + m_remnant
                        remnant_assigned = .true.
                        exit
                    end if
                end do
            end if

            if (.not. remnant_assigned) then
                if ((global_ii1 <= id1) .and. (id1 <= global_ii2)) then
                    if ((dustbins_props(id1)%mgrain_min <= m_remnant) .and. (m_remnant <= dustbins_props(id1)%mgrain_max)) then
                        chi_frag_out(id1 - global_ii1 + ii1) = chi_frag_out(id1 - global_ii1 + ii1) + m_remnant
                        remnant_assigned = .true.
                    end if
                end if
            end if

            if (.not. remnant_assigned) then
                do ll_local = ii1, ii2
                    ll_global = ll_local + global_ii1 - ii1
                    if ((dustbins_props(ll_global)%mgrain_min.le.m_remnant).and.(m_remnant<dustbins_props(ll_global)%mgrain_max)) then
                        chi_frag_out(ll_local) = chi_frag_out(ll_local) + m_remnant
                        remnant_assigned = .true.
                        exit
                    end if
                end do
            end if

            if (.not. remnant_assigned) then
                nearest_idx = ii1
                min_logdist = abs(log(max(m_remnant,tiny(m_remnant)) / dustbins_props(global_ii1)%mgrain))
                do ll_local = ii1 + 1, ii2
                    ll_global = ll_local + global_ii1 - ii1
                    logdist = abs(log(max(m_remnant,tiny(m_remnant)) / dustbins_props(ll_global)%mgrain))
                    if (logdist < min_logdist) then
                        min_logdist = logdist
                        nearest_idx = ll_local
                    end if
                end do
                chi_frag_out(nearest_idx) = chi_frag_out(nearest_idx) + m_remnant
                remnant_assigned = .true.
            end if

            if (.not. remnant_assigned) then
                chi_frag_dest_out = chi_frag_dest_out + m_remnant
            end if
        end if

        m_tot = chi_frag_dest_out + sum(chi_frag_out(:)) + sum(chi_frag_pah_out(:))
        if (m_tot > tiny(m_tot)) then
            chi_frag_dest_out = chi_frag_dest_out / m_tot
            chi_frag_out(:) = chi_frag_out(:) / m_tot
            chi_frag_pah_out(:) = chi_frag_pah_out(:) / m_tot
        end if
    end subroutine compute_shattered_fragments_direct

    subroutine LeBourlot2012_accretion_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the accretion rate of dust grains in the unrestricted case, following Le Bourlot et al. (2012).
        ! dust_info --> DustChemistryInfo type with all the necessary information to compute the accretion rate.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process (optional output)
        use dust_surface_chemistry, only: ice_sticking_coefficient
        implicit none
        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, kk, e_index
        integer :: n_el
        real(dp) :: pseudo_rate, rate, prefactor, Tk_loc, limit_rate
        real(dp) :: total_rate_type, sticking_ice,nO

        Tk_loc = dust_info%local_Tk
        prefactor = sqrt(Tk_loc) / (1d0 + 1d-4*Tk_loc**1.5d0)
#ifdef RTZ
        nO = y_gas(8,1) / elements(8)%atomic_mass_g
#else
        nO = y_gas(8,1) / el_atomic_masses_g(8)
#endif

        speciesloop: do jj = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(jj) 
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            associate(bin => dustbins_props(ii1))
                n_el = bin%nelements

                if (n_el == 1) then
                    ! 2. A single-element chemistry type has a limiter.
                    e_index = bin%el_index(1)
                    limit_rate = y_gas(e_index,1) / (bin%el_mfractions(1) * sqrt(bin%el_atomic_masses_g(1)))
                else
                    ! 3. Find the limiting element in a single pass, without a temporary array.
                    e_index = bin%el_index(1)
                    limit_rate = y_gas(e_index,1) / (bin%el_mfractions(1) * sqrt(bin%el_atomic_masses_g(1)))
                    do kk = 2, n_el
                        e_index = bin%el_index(kk)
                        pseudo_rate = y_gas(e_index,1) / (bin%el_mfractions(kk) * sqrt(bin%el_atomic_masses_g(kk)))
                        if (pseudo_rate < limit_rate) then
                            limit_rate = pseudo_rate
                        end if
                    end do
                end if

                ! 4. Apply the same limiting rate to every dust bin in the chemical type.
                total_rate_type = 0d0
                do ii = ii1, ii2
                    sticking_ice = ice_sticking_coefficient(dust_info%local_G0,nO,dust_info%local_Tk,dust_info%T_dust(ii))
                    rate = limit_rate * dustbins_props(ii)%k0_acc * prefactor * sticking_ice ! [s-1]
                    rate = min(rate, max_accretion_rate)
                    ! 5. Get the maximum rate computed here, if requested.
                    if (present(kmax)) then
                        kmax = max(kmax, abs(rate))
                    end if
                    rate = rate * y_dust(ii+dust_info%npah) ! [g cm-3 s-1]

                    if (rate > 0.0) then
                        dydt_dust(ii+dust_info%npah, 1) = dydt_dust(ii+dust_info%npah, 1) + rate  ! [g cm-3 s-1]
                    else
                        dydt_dust(ii+dust_info%npah, 2) = dydt_dust(ii+dust_info%npah, 2) - rate  ! [g cm-3 s-1]
                    end if

                    total_rate_type = total_rate_type + rate
                end do
            ! 6. Update gas phase once per chemical type
            do kk = 1, n_el
                e_index = bin%el_index(kk)
                dydt_gas(e_index,1) = dydt_gas(e_index,1) - total_rate_type * bin%el_mfractions(kk) ! [g cm-3 s-1]
            end do
            end associate
        end do speciesloop
    end subroutine LeBourlot2012_accretion_rate

    subroutine coulomb_accretion_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the accretion rate of dust grains in the unrestricted case, incorporating the effect of 
        ! grain charge interaction with ions. We use the sticking coefficient from Le Bourlot et al. (2012)
        ! dust_info --> DustChemistryInfo type with all the necessary information to compute the accretion rate.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process (optional output)
        use dust_surface_chemistry, only: ice_sticking_coefficient
        implicit none
        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, kk, e_index, iion, izion, nions_loc
        integer :: n_el
        real(dp) :: pseudo_rate, rate, prefactor, Tk_loc, bin_prefactor
        real(dp) :: sticking_ice, nO, min_growth_rate, element_growth_rate
        real(dp),dimension(:),allocatable :: element_pot_rate
        real(dp),dimension(:,:),allocatable :: ion_pot_contribution
        real(dp) :: min_abundance_density,depletion,ion_flux
        real(dp) :: element_pot_rate_1

        Tk_loc = dust_info%local_Tk
        min_abundance_density = 1.d-10 * dust_info%local_rho
        prefactor = sqrt(Tk_loc) / (1d0 + 1d-4*Tk_loc**1.5d0)
#ifdef RTZ
        nO = y_gas(8,1) / elements(8)%atomic_mass_g
#else
        nO = y_gas(8,1) / el_atomic_masses_g(8)
#endif

        speciesloop: do jj = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(jj) 
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            ! 2. Loop over all bins in a given chemical species
            do ii = ii1, ii2
                associate(bin => dustbins_props(ii))
                    n_el = bin%nelements
                    sticking_ice = ice_sticking_coefficient(dust_info%local_G0,nO,dust_info%local_Tk,dust_info%T_dust(ii))
                    bin_prefactor = bin%k0_acc * prefactor * sticking_ice
                    if (n_el == 1) then
                        e_index = bin%el_index(1)
                        element_pot_rate_1 = 0d0
                        nions_loc = n_elements
#ifdef RTZ
                        nions_loc = max(1, elements(e_index)%n_ions)
#endif
                        do iion = 1, nions_loc
                            izion = iion - 1
                            if (y_gas(e_index, iion+1) <= min_abundance_density) cycle
                            
                            ! Raw kinetic arrival rate for this specific ion stage
                            ion_flux = y_gas(e_index, iion+1) * dust_info%Coulomb_factor(ii,izion)
                            element_pot_rate_1 = element_pot_rate_1 + ion_flux
                        end do
                        
                        ! Normalize by stoichiometry and mass fraction to find the element's growth limit
                        min_growth_rate = element_pot_rate_1 * bin_prefactor / (bin%el_mfractions(1) * sqrt(bin%el_atomic_masses_g(1)))
                        
                        ! Final mass growth rate for this dust bin [g cm-3 s-1]. max_accretion_rate
                        ! caps the specific rate [s-1], as in LeBourlot2012_accretion_rate, so it
                        ! applies before multiplying by the dust density.
                        min_growth_rate = min(min_growth_rate, max_accretion_rate)
                        rate = min_growth_rate * y_dust(ii+dust_info%npah)

                        if (rate > 0.0) then
                            dydt_dust(ii+dust_info%npah, 1) = dydt_dust(ii+dust_info%npah, 1) + rate
                        else
                            dydt_dust(ii+dust_info%npah, 2) = dydt_dust(ii+dust_info%npah, 2) - rate
                        end if

                        if (present(kmax)) then
                            kmax = max(kmax, abs(min_growth_rate))
                        end if

                        if (element_pot_rate_1 > 0d0) then
                            do iion = 1, nions_loc
                                izion = iion - 1
                                if (y_gas(e_index, iion+1) <= min_abundance_density) cycle
                                ion_flux = y_gas(e_index, iion+1) * bin_prefactor * dust_info%Coulomb_factor(ii,izion)
                                depletion = rate * bin%el_mfractions(1) * (ion_flux / element_pot_rate_1)
                                dydt_gas(e_index, iion+1) = dydt_gas(e_index, iion+1) - depletion
                                dydt_gas(e_index, 1)      = dydt_gas(e_index, 1)      - depletion
                            end do
                        end if
                    else
                        allocate(element_pot_rate(1:n_el))
                        allocate(ion_pot_contribution(1:n_el,1:n_elements))
                        min_growth_rate = huge(1d0)
                        bin_prefactor = bin%k0_acc * prefactor * sticking_ice

                        ! --- STEP 1: Compute total potential arrival rate for each required element ---
                        do kk = 1, n_el
                            e_index = bin%el_index(kk)
                            element_pot_rate(kk) = 0d0
                            ion_pot_contribution(kk,:) = 0d0
                            nions_loc = n_elements
#ifdef RTZ
                            nions_loc = max(1, elements(e_index)%n_ions)
#endif
                            do iion = 1, nions_loc
                                izion = iion - 1
                                if (y_gas(e_index, iion+1) <= min_abundance_density) cycle
                                
                                ! Raw kinetic arrival rate for this specific ion stage
                                ion_flux = y_gas(e_index, iion+1) * dust_info%Coulomb_factor(ii,izion)
                                
                                ! Store this ion's individual potential contribution for Step 3
                                ion_pot_contribution(kk, iion) = ion_flux
                                element_pot_rate(kk) = element_pot_rate(kk) + ion_flux
                            end do
                            
                            ! Normalize by stoichiometry and mass fraction to find the element's growth limit
                            element_growth_rate = element_pot_rate(kk) * bin_prefactor / (bin%el_mfractions(kk) * sqrt(bin%el_atomic_masses_g(kk)))
                            
                            ! Find the absolute limiting element for this mineral composition
                            if (element_growth_rate < min_growth_rate) then
                                min_growth_rate = element_growth_rate
                            end if
                        end do
                        
                        ! --- STEP 2: Scale actual growth and deplete the gas phase ---
                        ! Final mass growth rate for this dust bin [g cm-3 s-1]. max_accretion_rate
                        ! caps the specific rate [s-1], as in LeBourlot2012_accretion_rate, so it
                        ! applies before multiplying by the dust density.
                        min_growth_rate = min(min_growth_rate, max_accretion_rate)
                        rate = min_growth_rate * y_dust(ii+dust_info%npah)

                        if (rate > 0.0) then
                            dydt_dust(ii+dust_info%npah, 1) = dydt_dust(ii+dust_info%npah, 1) + rate
                        else
                            dydt_dust(ii+dust_info%npah, 2) = dydt_dust(ii+dust_info%npah, 2) - rate
                        end if

                        if (present(kmax)) then
                            kmax = max(kmax, abs(min_growth_rate))
                        end if

                        do kk = 1, n_el
                            e_index = bin%el_index(kk)
                            
                            if (element_pot_rate(kk) > 0d0) then
                                nions_loc = n_elements
#ifdef RTZ
                                nions_loc = max(1, elements(e_index)%n_ions)
#endif
                                do iion = 1, nions_loc
                                    ! Actual depletion of this specific ion stage
                                    depletion = rate * bin%el_mfractions(kk) * (ion_pot_contribution(kk, iion) / element_pot_rate(kk))
                                    dydt_gas(e_index, iion+1) = dydt_gas(e_index, iion+1) - depletion
                                    dydt_gas(e_index, 1)      = dydt_gas(e_index, 1)      - depletion
                                end do
                            end if
                        end do
                        deallocate(element_pot_rate)
                        deallocate(ion_pot_contribution)
                    end if
                end associate
            end do
        end do speciesloop
    end subroutine coulomb_accretion_rate

    subroutine Aoyama2017_coagulation_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the coagulation rate of dust grains following Aoyama et al. (2017).
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, index
        real(dp) :: rate1, rate2, inv_nH

        inv_nH = 1.0_dp / max(dust_info%local_nH, tiny(1.0_dp))

        speciesloop: do jj = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(jj) 
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            do ii = ii1, ii2-1
                index = ii+dust_info%npah
                if ((dust_info%local_Tk .gt.1d4) .or. &
                    (dust_info%local_nH .lt. dustbins_props(ii)%nH_coa) .or. &
                    (dust_info%local_Jeans .gt. 4d0*dust_info%local_dx)) then
                    cycle
                end if
                rate1 = dustbins_props(ii)%k0_coa(1) * y_dust(index) * inv_nH ! [s-1]
                if (present(kmax)) then
                    kmax = max(kmax, rate1)
                end if
                rate2 = rate1 * y_dust(index) ! [g cm-3 s-1]
                dydt_dust(index, 2) = dydt_dust(index, 2) + rate2 ! [g cm-3 s-1]
                dydt_dust(index+1, 1) = dydt_dust(index+1, 1) + rate2 ! [g cm-3 s-1]
            end do
        end do speciesloop
    end subroutine Aoyama2017_coagulation_rate

    subroutine turbulent_coagulation_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the turbulent coagulation rate of dust grains.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, index
        real(dp) :: rate1, rate2
        real(dp) :: v_rel, p_stick

        speciesloop: do jj = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(jj) 
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            do ii = ii1, ii2-1
                index = ii+dust_info%npah
                
                v_rel = cached_v_rel_dust_dust(ii, ii)
                p_stick = cached_p_stick_dust_dust(ii, ii)
                if (p_stick <= 1d-20) cycle

                ! 4. Collision rate calculation
                ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                ! Marchand et al. (2021) and considers that grain velocities along the x-,
                ! y-, and z-axes are Gaussian distributed
                rate1 = dustbins_props(ii)%k0_coa(1) * v_rel * y_dust(index) * p_stick
                if (present(kmax)) then
                    kmax = max(kmax, rate1)
                end if
                rate2 = 0.5d0 * rate1 * y_dust(index) ! [g cm-3 s-1]; factor 1/2 for identical-grain self-collisions
                dydt_dust(index, 2) = dydt_dust(index, 2) + rate2 ! [g cm-3 s-1]
                dydt_dust(index+1, 1) = dydt_dust(index+1, 1) + rate2 ! [g cm-3 s-1]
            end do
        end do speciesloop

    end subroutine turbulent_coagulation_rate

    subroutine turbulent_all_coagulation_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the turbulent coagulation rate for all grain size bin pairs.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, index, kk, kk1, kk2, kk_loc, idest
        real(dp) :: rate1, rate2
        real(dp) :: v_rel, p_stick
        real(dp) :: mi, mk, msum, loss_ii, loss_kk

        speciesloop: do jj = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(jj) 
            ii2 = ii1 + dustbins_per_chemtype(jj) - 1

            do ii = ii1, ii2-1
                index = ii+dust_info%npah
                kk1 = ii
                kk2 = ii2
                do kk = kk1, kk2
                    kk_loc = kk - ii1 + 1
                    
                    v_rel = cached_v_rel_dust_dust(ii,kk)
                    p_stick = cached_p_stick_dust_dust(ii,kk)
                    if (p_stick <= 1d-20) cycle

                    ! 4. Collision rate calculation
                    ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                    ! Marchand et al. (2021) and considers that grain velocities along the x-,
                    ! y-, and z-axes are Gaussian distributed
                    rate1 = dustbins_props(ii)%k0_coa(kk_loc) * v_rel * y_dust(index) * p_stick

                    if (present(kmax)) then
                        kmax = max(kmax, rate1)
                    end if
                    rate2 = rate1 * y_dust(kk+dust_info%npah) ! [g cm-3 s-1]
                    if (ii == kk) then
                        ! Identical-grain collisions must be halved to avoid double counting.
                        rate2 = 0.5d0 * rate2
                    end if

                    mi = dustbins_props(ii)%mgrain
                    mk = dustbins_props(kk)%mgrain
                    msum = mi + mk
                    idest = dustbins_props(ii)%idend_coag(kk_loc)
                    loss_ii = rate2 * (mi / msum)
                    loss_kk = rate2 * (mk / msum)

                    ! Three-term update for each pair: two sinks (ii, kk) and one gain (idest).
                    dydt_dust(index, 2) = dydt_dust(index, 2) + loss_ii
                    dydt_dust(kk+dust_info%npah, 2) = dydt_dust(kk+dust_info%npah, 2) + loss_kk
                    dydt_dust(idest+dust_info%npah, 1) = dydt_dust(idest+dust_info%npah, 1) + (loss_ii + loss_kk)
                end do
            end do
        end do speciesloop

    end subroutine turbulent_all_coagulation_rate

    subroutine sputtering_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the sputtering erosion rate of neutral dust grains.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: ii, index, iel, nT_loc, iphi0, idx_T
        real(dp) :: rate1, rate2, lT, rate_total, irate, mass_loss

        lT = log10(dust_info%local_Tk)

        idx_T = -1
        ! 1. Loop over the dust bins
        binloop: do ii = 1, dust_info%ndust
            rate_total = 0d0
            index = ii + dust_info%npah

            ! 2. Loop over elements in the gas phase
            do iel = 1, n_elements
                if (y_gas(iel,1) < 1d-40) cycle
                if (.not. dustbins_props(ii)%sputtering_tab(iel)%initialised) cycle
                nT_loc = dustbins_props(ii)%sputtering_tab(iel)%npts(1)
                iphi0 = dustbins_props(ii)%sputtering_tab(iel)%ipos_zero(2)
                call dustbins_props(ii)%sputtering_tab(iel)%interpolate(lT, irate, idx_T)
                rate_total = rate_total + exp(irate * ln10) * y_gas(iel,1) * dust_info%inv_atomic_mass(iel) ! [micron / yr]
            end do

            ! 3. Convert to the real erosion rate in [s-1]
            rate1 = 3d0 * rate_total / dustbins_props(ii)%asize / yr2sec ! [s-1]
            if (present(kmax)) then
                kmax = max(kmax, abs(rate1))
            end if

            ! 4. Now compute the mass rates [g cm-3 s-1]
            mass_loss = rate1 * y_dust(index)
            dydt_dust(index, 2) = dydt_dust(index, 2) + mass_loss ! [g cm-3 s-1]
            do iel = 1, dustbins_props(ii)%nelements
                dydt_gas(dustbins_props(ii)%el_index(iel),1) = dydt_gas(dustbins_props(ii)%el_index(iel),1) + &
                    mass_loss * dustbins_props(ii)%el_mfractions(iel) ! [g cm-3 s-1]
            end do
        end do binloop
    end subroutine sputtering_rate

    subroutine sublimation_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the thermal sublimation erosion of dust grains using the
        ! pre-computed sublimation rate tables (epsilon = |da/dt|/a in [s-1]),
        ! interpolated at the local dust temperature stored in dust_info%T_dust.
        ! The eroded dust mass is returned to the gas phase following the dust
        ! bin elemental mass fractions, analogous to thermal sputtering.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: ii, index, iel, nT_loc
        real(dp) :: rate1, lTd, irate, Td_loc, mass_loss

        ! 1. Loop over the dust bins
        binloop: do ii = 1, dust_info%ndust
            index = ii + dust_info%npah
            if (.not. dustbins_props(ii)%sublimation_tab%initialised) cycle

            ! 2. Interpolate the fractional erosion rate (in log10) at the local dust temperature.
            !    The table axis is stored as log10(T_d), so we interpolate against log10(T_d).
            !    Tables only cover temperatures hot enough for sublimation to matter
            !    (timescale < 10 x age of the universe), so below the first tabulated
            !    temperature the rate is taken to be zero (skip the bin).
            Td_loc = dust_info%T_dust(ii)
            if (Td_loc <= dustbins_props(ii)%T_sublimation_min) cycle
            lTd = log10(Td_loc)
            nT_loc = dustbins_props(ii)%sublimation_tab%npts(1)
            call dustbins_props(ii)%sublimation_tab%interpolate(lTd, irate)

            ! 3. Convert the size erosion rate (|da/dt|/a) into the mass loss rate.
            !    Since m ~ a^3, the fractional mass loss rate is 3 * (|da/dt|/a) [s-1].
            rate1 = 3d0 * exp(irate * ln10) ! [s-1]
            if (present(kmax)) then
                kmax = max(kmax, abs(rate1))
            end if

            ! 4. Now compute the mass rates [g cm-3 s-1]
            mass_loss = rate1 * y_dust(index)
            dydt_dust(index,2) = dydt_dust(index,2) + mass_loss ! [g cm-3 s-1]
            do iel = 1, dustbins_props(ii)%nelements
                dydt_gas(dustbins_props(ii)%el_index(iel),1) = dydt_gas(dustbins_props(ii)%el_index(iel),1) + &
                    mass_loss * dustbins_props(ii)%el_mfractions(iel) ! [g cm-3 s-1]
            end do
        end do binloop
    end subroutine sublimation_rate

    subroutine charged_sputtering_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the sputtering erosion rate of charged dust grains, accounting for Coulomb effects.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: ii, index, iel, nT_loc, nphi_loc, iion, izion, idx_T, nions_loc
        real(dp) :: rate1, rate2, lT, rate_total, irate, rate_element, phi_charge, mass_loss
        real(dp) :: min_abundance_density

        lT = log10(dust_info%local_Tk)
        idx_T = -1
        min_abundance_density = 1d-10 * dust_info%local_rho

        ! 1. Loop over the dust bins
        binloop: do ii = 1, dust_info%ndust
            rate_total = 0d0
            index = ii + dust_info%npah

            ! 2. Loop over elements in the gas phase
            do iel = 1, n_elements
                rate_element = 0d0
                if (y_gas(iel,1) < min_abundance_density) cycle
                if (.not. dustbins_props(ii)%sputtering_tab(iel)%initialised) cycle

                nions_loc = n_elements
#ifdef RTZ
                nions_loc = max(1, elements(iel)%n_ions)
#endif

                nT_loc = dustbins_props(ii)%sputtering_tab(iel)%npts(1)
                nphi_loc = dustbins_props(ii)%sputtering_tab(iel)%npts(2)
                ! 3. Loop over the ions of the element
                do iion = 1, nions_loc
                    if (y_gas(iel,iion+1) < min_abundance_density) cycle
                    izion = iion - 1
                    phi_charge = dust_info%Z_dust(ii) * dustbins_props(ii)%phi_prefact(izion)
                    call dustbins_props(ii)%sputtering_tab(iel)%interpolate(lT, phi_charge, irate, idx_x=idx_T)
                    if (irate < -30d0) cycle
                    rate_element = rate_element + exp(irate * ln10) * y_gas(iel,iion+1) ! [micron / yr]
                end do
                rate_total = rate_total + rate_element * dust_info%inv_atomic_mass(iel) ! [micron / yr]
            end do
            ! 3. Convert to the real erosion rate in [s-1]
            rate1 = 3d0 * rate_total / dustbins_props(ii)%asize / yr2sec ! [s-1]
            if (present(kmax)) then
                kmax = max(kmax, abs(rate1))
            end if

            ! 4. Now compute the mass rates [g cm-3 s-1]
            mass_loss = rate1 * y_dust(index) ! [g cm-3 s-1]
            dydt_dust(index, 2) = dydt_dust(index, 2) + mass_loss ! [g cm-3 s-1]
            do iel = 1, dustbins_props(ii)%nelements
                dydt_gas(dustbins_props(ii)%el_index(iel),1) = dydt_gas(dustbins_props(ii)%el_index(iel),1) + &
                    mass_loss * dustbins_props(ii)%el_mfractions(iel) ! [g cm-3 s-1]
            end do
        end do binloop
    end subroutine charged_sputtering_rate

    subroutine turbulent_shattering_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the turbulent shattering rate of dust grains.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: jj, ii, ii1, ii2, index, pp, ll, iel
        real(dp) :: rate1, rate_dest
        real(dp) :: coll_factor, v_rel
        real(dp) :: chi_frag_dest
        real(dp), dimension(:), allocatable :: chi_frag
        real(dp), dimension(:), allocatable :: chi_frag_pah
        logical :: interact_pah_flag

        speciesloop: do ii = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(ii)
            ii2 = ii1 + dustbins_per_chemtype(ii) - 1

            allocate(chi_frag(ii1:ii2))
            allocate(chi_frag_pah(1:dust_info%npah))

            ! Cache PAH interaction flag outside grain loop for efficiency
            interact_pah_flag = dustbins_props(ii1)%interact_pah .and. dust_info%npah > 0

            do jj = ii2, ii1, -1
                index = jj + dust_info%npah

                v_rel = cached_v_rel_dust_dust(jj, jj)
                chi_frag_dest = cached_chi_frag_dest(jj, jj)
                chi_frag(ii1:ii2) = cached_chi_frag_dust(jj, jj, ii1:ii2)
                if (interact_pah_flag) then
                    chi_frag_pah(1:dust_info%npah) = cached_chi_frag_pah(jj, jj, 1:dust_info%npah)
                end if

                ! 3. Compute the collision rate factor
                ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                ! Marchand et al. (2021) and considers that grain velocities along the x-,
                ! y-, and z-axes are Gaussian distributed.
                coll_factor = sqrt(8d0/(3d0*pi)) * 4d0 * pi * (dustbins_props(jj)%asize_cm)**2d0 * v_rel

                ! 4. Compute the mass loss rate from shattering collisions [g cm-3 s-1]
                rate1 = coll_factor * y_dust(index) / dustbins_props(jj)%mgrain
                                
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if

                ! Factor of 1/2 for identical-grain self-collisions (avoids double-counting).
                ! kmax is kept un-halved above for conservative step-size control.
                rate1 = 0.5d0 * rate1

                ! 5. Update the dust derivatives for all fragments
                dydt_dust(index, 2) = dydt_dust(index, 2) + rate1 * y_dust(index) ! [g cm-3 s-1]

                ! 5a. Combined update for fragment dust bins and PAH bins
                ! Loop over all destination bins in a single pass for better cache efficiency
                do ll = ii1, ii2
                    dydt_dust(ll + dust_info%npah, 1) = dydt_dust(ll + dust_info%npah, 1) + rate1 * chi_frag(ll) * y_dust(index)
                end do

                if (interact_pah_flag) then
                    do pp = 1, dust_info%npah
                        dydt_dust(pp, 1) = dydt_dust(pp, 1) + rate1 * chi_frag_pah(pp) * y_dust(index)
                    end do
                end if

                ! 5c. Return destroyed material to the gas phase
                ! Return destroyed material to the gas phase based on the elemental composition of the dust grain
                if (chi_frag_dest > 1d-10) then
                    rate_dest = rate1 * chi_frag_dest * y_dust(index)
                    do iel = 1, dustbins_props(jj)%nelements
                        dydt_gas(dustbins_props(jj)%el_index(iel),1) = dydt_gas(dustbins_props(jj)%el_index(iel),1) + &
                            rate_dest * dustbins_props(jj)%el_mfractions(iel) ! [g cm-3 s-1]
                    end do
                end if
            end do

            deallocate(chi_frag)
            deallocate(chi_frag_pah)
        end do speciesloop

    end subroutine turbulent_shattering_rate

    subroutine turbulent_all_shattering_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the turbulent shattering rate for all grain size bin pairs.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: ichem, ii, jj, ii1, ii2, index_i, index_j, pp, ll, iel
        real(dp) :: rate_dest, mass_rate
        real(dp) :: coll_factor, v_rel
        real(dp) :: chi_frag_dest
        real(dp), dimension(:), allocatable :: chi_frag
        real(dp), dimension(:), allocatable :: chi_frag_pah
        logical :: interact_pah_flag

        speciesloop: do ichem = 1, ndchemtype
            ! 1. Loop over the dust chemical species.
            ii1 = istart_chemtype(ichem)
            ii2 = ii1 + dustbins_per_chemtype(ichem) - 1

            allocate(chi_frag(ii1:ii2))
            allocate(chi_frag_pah(1:dust_info%npah))
            
            ! Cache PAH interaction flag outside grain loop for efficiency
            interact_pah_flag = dustbins_props(ii1)%interact_pah .and. dust_info%npah > 0

            do ii = ii1, ii2
                index_i = ii + dust_info%npah
                if (y_dust(index_i) < 1d-40) cycle

                ! Use jj >= ii so each pair is processed once (no ii-jj / jj-ii double counting).
                do jj = ii, ii2
                    index_j = jj + dust_info%npah
                    if (y_dust(index_j) < 1d-40) cycle

                    ! 2. Compute shattered fragments for target ii impacted by jj.
                    v_rel = cached_v_rel_dust_dust(ii, jj)
                    chi_frag_dest = cached_chi_frag_dest(ii, jj)
                    chi_frag(ii1:ii2) = cached_chi_frag_dust(ii, jj, ii1:ii2)
                    if (interact_pah_flag) then
                        chi_frag_pah(1:dust_info%npah) = cached_chi_frag_pah(ii, jj, 1:dust_info%npah)
                    end if

                    ! 3. Compute the collision rate factor for pair (ii,jj).
                    ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                    ! Marchand et al. (2021) and considers that grain velocities along the x-,
                    ! y-, and z-axes are Gaussian distributed.
                    coll_factor = sqrt(8d0/(3d0*pi)) * pi * (dustbins_props(ii)%asize_cm + dustbins_props(jj)%asize_cm)**2d0 * v_rel

                    ! 4. Collision event rate (events cm^-3 s^-1) × target mass:
                    !    mass_rate_ii = R * m_ii  = coll_factor * n_jj * rho_ii
                    !                             = coll_factor * (rho_jj/m_jj) * rho_ii
                    !    For ii==jj apply 0.5 to count each identical-grain collision once,
                    !    matching the convention in turbulent_all_coagulation_rate.
                    if (ii == jj) then
                        mass_rate = 0.5d0 * coll_factor * y_dust(index_j) / dustbins_props(jj)%mgrain * y_dust(index_i)
                    else
                        mass_rate = coll_factor * y_dust(index_j) / dustbins_props(jj)%mgrain * y_dust(index_i)
                    end if

                    if (present(kmax)) then
                        kmax = max(kmax, abs(mass_rate / max(y_dust(index_i), 1d-99)))
                    end if

                    ! 5a. Apply destruction and fragmentation for target ii.
                    dydt_dust(index_i, 2) = dydt_dust(index_i, 2) + mass_rate ! [g cm-3 s-1]

                    do ll = ii1, ii2
                        dydt_dust(ll + dust_info%npah, 1) = dydt_dust(ll + dust_info%npah, 1) + mass_rate * chi_frag(ll)
                    end do

                    if (interact_pah_flag) then
                        do pp = 1, dust_info%npah
                            dydt_dust(pp, 1) = dydt_dust(pp, 1) + mass_rate * chi_frag_pah(pp)
                        end do
                    end if

                    if (chi_frag_dest > 1d-10) then
                        rate_dest = mass_rate * chi_frag_dest
                        do iel = 1, dustbins_props(ii)%nelements
                            dydt_gas(dustbins_props(ii)%el_index(iel),1) = dydt_gas(dustbins_props(ii)%el_index(iel),1) + &
                                rate_dest * dustbins_props(ii)%el_mfractions(iel) ! [g cm-3 s-1]
                        end do
                    end if

                    ! 5b. Apply destruction and fragmentation for target jj (same collision event).
                    !    mass_rate_jj = R * m_jj = coll_factor * (rho_ii/m_ii) * rho_jj
                    !    For ii/=jj: chi_frag is recomputed for target jj hit by ii.
                    !    For ii==jj: chi_frag is identical by symmetry; reuse directly.
                    if (ii /= jj) then
                        v_rel = cached_v_rel_dust_dust(jj, ii)
                        chi_frag_dest = cached_chi_frag_dest(jj, ii)
                        chi_frag(ii1:ii2) = cached_chi_frag_dust(jj, ii, ii1:ii2)
                        if (interact_pah_flag) then
                            chi_frag_pah(1:dust_info%npah) = cached_chi_frag_pah(jj, ii, 1:dust_info%npah)
                        end if
                        mass_rate = coll_factor * y_dust(index_i) / dustbins_props(ii)%mgrain * y_dust(index_j)
                    else
                        ! ii==jj: chi_frag unchanged; m_ii==m_jj so formula is symmetric.
                        mass_rate = 0.5d0 * coll_factor * y_dust(index_i) / dustbins_props(ii)%mgrain * y_dust(index_j)
                    end if

                    dydt_dust(index_j, 2) = dydt_dust(index_j, 2) + mass_rate ! [g cm-3 s-1]

                    do ll = ii1, ii2
                        dydt_dust(ll + dust_info%npah, 1) = dydt_dust(ll + dust_info%npah, 1) + mass_rate * chi_frag(ll)
                    end do

                    if (interact_pah_flag) then
                        do pp = 1, dust_info%npah
                            dydt_dust(pp, 1) = dydt_dust(pp, 1) + mass_rate * chi_frag_pah(pp)
                        end do
                    end if

                    if (chi_frag_dest > 1d-10) then
                        rate_dest = mass_rate * chi_frag_dest
                        do iel = 1, dustbins_props(jj)%nelements
                            dydt_gas(dustbins_props(jj)%el_index(iel),1) = dydt_gas(dustbins_props(jj)%el_index(iel),1) + &
                                rate_dest * dustbins_props(jj)%el_mfractions(iel) ! [g cm-3 s-1]
                        end do
                    end if
                end do
            end do

            deallocate(chi_frag)
            deallocate(chi_frag_pah)
        end do speciesloop

    end subroutine turbulent_all_shattering_rate

    subroutine compute_shattered_fragments(dust_info,id1,id2,local_sigma,local_L,v_rel_out,chi_frag_dest_out,chi_frag_out,chi_frag_pah_out,interact_pah)
        ! Compute the fragmented mass distribution resulting from the shattering collision
        ! between target grain id1 and impactor grain id2.
        ! dust_info          --> DustChemistryInfo type with target/impactor properties.
        ! id1                --> Target grain bin index.
        ! id2                --> Impactor grain bin index.
        ! local_sigma        --> Local velocity dispersion [cm s-1].
        ! local_L            --> Local spatial scale [cm].
        ! v_rel_out          <-- Computed relative velocity of the colliding grains [cm s-1].
        ! chi_frag_dest_out  <-- Fraction of shattered mass destroyed (sputtered) to gas phase.
        ! chi_frag_out       <-- Mass distribution of shattered fragments in dust bins.
        ! chi_frag_pah_out   <-- Mass distribution of shattered fragments in PAH bins.
        ! interact_pah       --> Logical flag indicating whether PAH-dust interaction is active.
        use dust_dynamics, only: grain_relative_velocity
        implicit none

        class(DustChemistryInfo), intent(in) :: dust_info
        integer, intent(in) :: id1, id2
        real(dp), intent(in) :: local_sigma, local_L
        logical, intent(in) :: interact_pah
        real(dp), intent(out) :: v_rel_out, chi_frag_dest_out
        real(dp), dimension(:), intent(out) :: chi_frag_out
        real(dp), dimension(:), intent(out) :: chi_frag_pah_out

        integer :: pp_local, ll_local, ii1, ii2, nearest_idx
        integer :: global_ii1, global_ii2, ll_global
        real(dp) :: E_imp, phi, m_ej, m_remnant, m_max, m_min
        real(dp) :: prefactor, m_tot, denom, logdist, min_logdist
        logical :: remnant_assigned
        real(dp) :: m_max_pow, m_min_pow  ! OPTIMIZATION: Precomputed powers

        ! Infer ii1 and ii2 from the bounds of the output arrays
        ii1 = lbound(chi_frag_out, 1)
        ii2 = ubound(chi_frag_out, 1)

        global_ii1 = istart_chemtype(dustbins_props(id1)%interact_group)
        global_ii2 = global_ii1 + dustbins_per_chemtype(dustbins_props(id1)%interact_group) - 1

        ! 1. Compute the relative velocity of two grains
        v_rel_out = grain_relative_velocity(dust_velocity_model,dust_info%local_Tk,dust_info%local_rho,&
                                        dust_info%local_nH,local_sigma,dust_info%local_mu,local_L,&
                                        dustbins_props(id1)%asize_cm,dustbins_props(id2)%asize_cm,&
                                        dustbins_props(id1)%sgrain,dustbins_props(id2)%sgrain,&
                                        dustbins_props(id1)%mgrain,dustbins_props(id2)%mgrain)

        ! 2. Disrupted mass computation (Eqs. 20-22 of Hirashita & Aoyama 2019)
        E_imp = 5d-1 * (dustbins_props(id1)%mgrain*dustbins_props(id2)%mgrain) / &
                (dustbins_props(id1)%mgrain+dustbins_props(id2)%mgrain) * v_rel_out**2d0
        phi = E_imp / (dustbins_props(id1)%mgrain*dustbins_props(id1)%catastrophic_spec_energy)
        m_ej = phi / (1d0 + phi) * dustbins_props(id1)%mgrain

        ! 3. Compute the maximum and minimum masses of the fragment distribution
        m_remnant = dustbins_props(id1)%mgrain - m_ej
        m_max = 2d-2*m_ej
        m_min = 1d-6*m_max

        ! 4. Compute the distribution prefactor for ejecta fragments only
        if (m_ej > tiny(m_ej)) then
            ! OPTIMIZATION: Precompute powers of slope_frag_func to avoid repeated expensive pow() calls
            m_max_pow = m_max**slope_frag_func
            m_min_pow = m_min**slope_frag_func
            denom = m_max_pow - m_min_pow
            if (abs(denom) > tiny(denom)) then
                prefactor = m_ej / denom
            else
                prefactor = 0d0
            end if
        else
            m_max_pow = 0d0
            m_min_pow = 0d0
            prefactor = 0d0
        end if

        ! 5. Ejecta contribution below the minimum tracked size goes to gas
        chi_frag_dest_out = 0d0
        chi_frag_out(:) = 0d0
        chi_frag_pah_out(:) = 0d0
        if (prefactor > 0d0) then
#if NPAH > 0
            if (interact_pah) then
                if (m_min < pahbins_props(1)%mpah_min) then
                    chi_frag_dest_out = prefactor * (min(pahbins_props(1)%mpah_min,m_max)**slope_frag_func - m_min_pow)
                end if
            else
                if (m_min < dustbins_props(global_ii1)%mgrain_min) then
                    chi_frag_dest_out = prefactor * (min(dustbins_props(global_ii1)%mgrain_min,m_max)**slope_frag_func - m_min_pow)
                end if
            end if
#else
            if (m_min < dustbins_props(global_ii1)%mgrain_min) then
                chi_frag_dest_out = prefactor * (min(dustbins_props(global_ii1)%mgrain_min,m_max)**slope_frag_func - m_min_pow)
            end if
#endif
        end if

        ! 6. Ejecta contribution in PAH bins
#if NPAH > 0
        if (interact_pah .and. prefactor > 0d0) then
            do pp_local = 1, dust_info%npah
                if ((m_min.ge.pahbins_props(pp_local)%mpah_max).or.(m_max<pahbins_props(pp_local)%mpah_min)) then
                    chi_frag_pah_out(pp_local) = 0d0
                else
                    chi_frag_pah_out(pp_local) = prefactor * (min(pahbins_props(pp_local)%mpah_max,m_max)**slope_frag_func - &
                                                    max(pahbins_props(pp_local)%mpah_min,m_min)**slope_frag_func)
                end if
            end do
        end if
#endif

        ! 7. Ejecta contribution in dust bins of the chemical type (can include id1/id2 bins)
        if (prefactor > 0d0) then
            do ll_local = ii1, ii2
                ll_global = ll_local + global_ii1 - ii1
                if ((m_min.ge.dustbins_props(ll_global)%mgrain_max).or.(m_max<dustbins_props(ll_global)%mgrain_min)) then
                    chi_frag_out(ll_local) = 0d0
                else
                    chi_frag_out(ll_local) = prefactor * (min(dustbins_props(ll_global)%mgrain_max,m_max)**slope_frag_func - &
                                                max(dustbins_props(ll_global)%mgrain_min,m_min)**slope_frag_func)
                end if
            end do
        end if

        ! 8. Add remnant as a point-mass contribution to whichever tracked bin contains it.
        !    This makes source-bin deposition explicit (e.g. remnant in id1/id2 bin).
        remnant_assigned = .false.
        if (m_remnant > 0d0) then
            if (interact_pah) then
                do pp_local = 1, dust_info%npah
                    if ((pahbins_props(pp_local)%mpah_min.le.m_remnant).and.(m_remnant<pahbins_props(pp_local)%mpah_max)) then
                        chi_frag_pah_out(pp_local) = chi_frag_pah_out(pp_local) + m_remnant
                        remnant_assigned = .true.
                        exit
                    end if
                end do
            end if

            if (.not. remnant_assigned) then
                if ((global_ii1 <= id1) .and. (id1 <= global_ii2)) then
                    if ((dustbins_props(id1)%mgrain_min <= m_remnant) .and. (m_remnant <= dustbins_props(id1)%mgrain_max)) then
                        chi_frag_out(id1 - global_ii1 + ii1) = chi_frag_out(id1 - global_ii1 + ii1) + m_remnant
                        remnant_assigned = .true.
                    end if
                end if
            end if

            if (.not. remnant_assigned) then
                do ll_local = ii1, ii2
                    ll_global = ll_local + global_ii1 - ii1
                    if ((dustbins_props(ll_global)%mgrain_min.le.m_remnant).and.(m_remnant<dustbins_props(ll_global)%mgrain_max)) then
                        chi_frag_out(ll_local) = chi_frag_out(ll_local) + m_remnant
                        remnant_assigned = .true.
                        exit
                    end if
                end do
            end if

            if (.not. remnant_assigned) then
                nearest_idx = ii1
                min_logdist = abs(log(max(m_remnant,tiny(m_remnant)) / dustbins_props(global_ii1)%mgrain))
                do ll_local = ii1 + 1, ii2
                    ll_global = ll_local + global_ii1 - ii1
                    logdist = abs(log(max(m_remnant,tiny(m_remnant)) / dustbins_props(ll_global)%mgrain))
                    if (logdist < min_logdist) then
                        min_logdist = logdist
                        nearest_idx = ll_local
                    end if
                end do
                chi_frag_out(nearest_idx) = chi_frag_out(nearest_idx) + m_remnant
                remnant_assigned = .true.
            end if

            if (.not. remnant_assigned) then
                chi_frag_dest_out = chi_frag_dest_out + m_remnant
            end if
        end if

        ! 9. Normalise the fragmentation mass fractions
        m_tot = chi_frag_dest_out + sum(chi_frag_out(:)) + sum(chi_frag_pah_out(:))
        if (m_tot > tiny(m_tot)) then
            chi_frag_dest_out = chi_frag_dest_out / m_tot
            chi_frag_out(:) = chi_frag_out(:) / m_tot
            chi_frag_pah_out(:) = chi_frag_pah_out(:) / m_tot
        end if
    end subroutine compute_shattered_fragments

    subroutine pah_sputtering_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the sputtering erosion rate of PAH molecules.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)
        
        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp, iel, nT_loc, idx_T
        real(dp) :: lT, R_total, rate1, rate2, J_rate
        real(dp), dimension(:), allocatable :: J_rate_all

        lT = log10(dust_info%local_Tk)
        idx_T = -1

        ! 1. Loop over PAHs
        pahloop: do pp = 1, dust_info%npah
            R_total = 0d0

            if (allocated(J_rate_all)) deallocate(J_rate_all)
            allocate(J_rate_all(0:n_elements))

            ! 2. Interpolate sputtering rates from tables
            J_rate_all(:) = -100d0
            do iel = 0, n_elements
                if (.not. pahbins_props(pp)%sputtering_tab(iel)%initialised) cycle
                nT_loc = pahbins_props(pp)%sputtering_tab(iel)%npts(1)
                call pahbins_props(pp)%sputtering_tab(iel)%interpolate(lT, J_rate_all(iel), idx_T)
            end do
            J_rate_all(:) = exp(J_rate_all(:) * ln10)

            ! 3. Electron contribution to sputtering rate
            if (pahbins_props(pp)%sputtering_tab(0)%initialised) then
                R_total = R_total + dust_info%local_ne * J_rate_all(0)
            end if

            ! 4. Ion contributions to sputtering rate
            do iel = 1, n_elements
                if (.not. pahbins_props(pp)%sputtering_tab(iel)%initialised) cycle
                if (y_gas(iel, 1) < 1d-40) cycle
                R_total = R_total + y_gas(iel, 1) / dust_info%el_atomic_mass_g(iel) * J_rate_all(iel)
            end do

            ! 5. Convert rate to mass loss [atoms/s-1] and compute mass loss rate [g cm-3 s-1]
            if (R_total > 0d0) then
                rate1 = R_total ! [s-1]
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if
                rate2 = rate1 * y_dust(pp) / pahbins_props(pp)%mpah * dust_info%el_atomic_mass_g(pahbins_props(pp)%C_index) ! [g cm-3 s-1]
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2 ! [g cm-3 s-1]

                ! 6. Return sputtered material (carbon) to the gas phase
                dydt_gas(pahbins_props(pp)%C_index, 1) = dydt_gas(pahbins_props(pp)%C_index, 1) + rate2 ! [g cm-3 s-1]
            end if

            if (allocated(J_rate_all)) deallocate(J_rate_all)
        end do pahloop

    end subroutine pah_sputtering_rate

    subroutine pah_photolysis_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the photolysis (dissociation) rate of PAH molecules.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)
        
        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp, idx_G0, idx_nH
        real(dp):: log_nH, log_G0
        real(dp):: rate1, rate2

        log_nH = log10(dust_info%local_nH)
        log_G0 = log10(dust_info%local_G0)

        idx_G0 = -1
        idx_nH = -1

        ! 1. Loop over PAH sizes
        pahloop: do pp = 1, dust_info%npah
            if (pahbins_props(pp)%is_cluster) cycle ! TODO: Photolysis only implemented for non-cluster PAHs for now
            if (.not. pahbins_props(pp)%dissociation_tab%initialised) cycle

            call pahbins_props(pp)%dissociation_tab%interpolate(log_G0, log_nH, rate1, idx_G0, idx_nH)

            rate1 = exp(rate1 * ln10) ! [s-1]

            if (rate1 > 0d0) then
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if
                rate2 = rate1 * y_dust(pp) / pahbins_props(pp)%mpah * (2d0*dust_info%el_atomic_mass_g(pahbins_props(pp)%C_index)) ! [g cm-3 s-1]
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2 ! [g cm-3 s-1]

                ! 2. Return photolysed material (carbon) to the gas phase
                dydt_gas(pahbins_props(pp)%C_index, 1) = dydt_gas(pahbins_props(pp)%C_index, 1) + rate2 ! [g cm-3 s-1]
            end if
        end do pahloop

    end subroutine pah_photolysis_rate

    subroutine pah_cluster_evaporation_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the evaporation rate of PAH clusters.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)
        
        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp
        real(dp):: log_T, log_nH
        real(dp):: rate1, rate2
        real(dp):: k_single, k_multi

        log_T = log10(dust_info%local_Tk)
        log_nH = log10(dust_info%local_nH)

        ! 1. Loop over PAH sizes; start at 2 because the product bin is pp-1
        pahloop: do pp = 2, dust_info%npah
            if (.not. pahbins_props(pp)%is_cluster) cycle ! Cluster evaporation is only for PAH clusters
            k_single = dust_info%local_G0 / 0.19306d0
            k_multi = 1d0 / exp((-3.1692061d0 * log10(dust_info%local_G0) + 13.5642486d0) * ln10)
            rate1 = min(k_single, k_multi) / yr2sec ! [s-1]

            if (rate1 > 0d0) then
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if
                rate2 = rate1 * y_dust(pp) / pahbins_props(pp)%mpah * pahbins_props(pp-1)%mpah ! [g cm-3 s-1]
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2 ! [g cm-3 s-1]

                ! 2. Return evaporated molecule to the PAH bin below (pp-1)
                dydt_dust(pp-1, 1) = dydt_dust(pp-1, 1) + rate2 ! [g cm-3 s-1]
            end if
        end do pahloop
    end subroutine pah_cluster_evaporation_rate

    subroutine Totton2012_pah_coalescence_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the coalescence rate of PAH molecules following Totton et al. (2012).
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)
        
        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp
        real(dp) :: reduced_mass, dV_thermal, C_eff, coll_section
        real(dp) :: rate1, rate2

        ! Following compute_t_pah_coalescence (Totton2012):
        ! t_coal = mpah / (4*pi*a^2 * dV_thermal * C_eff * rho)
        ! so k_coal = 1/t_coal.
        pahloop: do pp = 1, dust_info%npah - 1
            if (pahbins_props(pp)%is_cluster) cycle

            reduced_mass = 5d-1 * pahbins_props(pp)%mpah
            C_eff = 1d0 / (1d0 + 9.92807181d-7 * (log10(dust_info%local_Tk))**1.37933821d1)
            dV_thermal = sqrt(8d0 * kB * dust_info%local_Tk / reduced_mass)
            coll_section = 4d0 * pi * (pahbins_props(pp)%apah_cm)**2d0
            rate1 = coll_section * dV_thermal * C_eff * y_dust(pp) / pahbins_props(pp)%mpah

            if (rate1 > 0d0) then
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if
                rate2 = rate1 * y_dust(pp)
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2
                dydt_dust(pp+1, 1) = dydt_dust(pp+1, 1) + rate2
            end if
        end do pahloop
    end subroutine Totton2012_pah_coalescence_rate

    subroutine Tielens2021_pah_coalescence_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the coalescence rate of PAH molecules following Tielens (2021).
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp, nstates, cation_start
        real(dp) :: reduced_mass, pah_ion_fraction
        real(dp) :: R1, R2
        real(dp) :: rate1, rate2

        if (dust_info%local_Tk <= 0d0) return
        if (dust_info%local_rho <= 0d0) return
        if (.not. allocated(dust_info%fcharge_pah)) return

        ! Following compute_t_pah_coalescence (Tielens2021):
        ! k_coal = ((R1*(1-fion) + R2*fion) * rho) / mpah.
        pahloop: do pp = 1, dust_info%npah - 1
            if (pahbins_props(pp)%is_cluster) cycle

            nstates = pahbins_props(pp)%ncharge_states
            cation_start = pahbins_props(pp)%cation_start_idx
            pah_ion_fraction = 0d0
            if (cation_start <= nstates) then
                pah_ion_fraction = sum(dust_info%fcharge_pah(cation_start:nstates, pp))
            end if
            pah_ion_fraction = max(0d0, min(1d0, pah_ion_fraction))

            R1 = 4d-11 * sqrt(dust_info%local_Tk / 10d0) * sqrt(pahbins_props(pp)%nc / 50d0)
            reduced_mass = 5d-1 * pahbins_props(pp)%mpah
            R2 = 6d-9 * sqrt(pahbins_props(pp)%nc / 50d0) * sqrt((12d0 * amu2g) / reduced_mass)
            rate1 = (R1 * (1d0 - pah_ion_fraction) + R2 * pah_ion_fraction) * y_dust(pp) / pahbins_props(pp)%mpah

            if (rate1 > 0d0) then
                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if
                rate2 = rate1 * y_dust(pp)
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2
                dydt_dust(pp+1, 1) = dydt_dust(pp+1, 1) + rate2
            end if
        end do pahloop
    end subroutine Tielens2021_pah_coalescence_rate

    subroutine pah_freezing_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax)
        ! Compute the freezing rate of PAH molecules onto larger dust grains.
        ! dust_info --> DustChemistryInfo type with the current cell's physical properties.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      --> Maximum allowed rate for the process [s-1] (optional output)

        implicit none

        ! ---- Input/Output variables ----
        class(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:,:)
        real(dp), intent(inout), optional :: kmax

        ! ---- Local variables ----
        integer :: pp, kk, ii, izion
        integer :: dust_start, dust_end, index_dust
        real(dp) :: v_rel, D_av, Z_single
        real(dp) :: coll_factor, rate1, rate2, weight, p_stick

#if NPAH > 0
        pahloop: do pp = 1, dust_info%npah
            dust_start = pahbins_props(pp)%dust_index_interact
            if (dust_start <= 0) cycle
            dust_end = min(dust_info%ndust, dust_start + pahbins_props(pp)%nd_bins - 1)

            do kk = dust_start, dust_end
                if (.not. dustbins_props(kk)%interact_pah) cycle
                index_dust = kk + dust_info%npah
                if (y_dust(index_dust) < 1d-40) cycle

                ! 1. Relative velocity between PAH pp and dust bin kk.
                v_rel = cached_v_rel_pah_dust(pp, kk)

                ! 2. Coulomb focusing averaged over PAH charge distribution,
                !    using precomputed dust_info%Coulomb_factor for this grain.
                D_av = 0d0
                do ii = 1, pahbins_props(pp)%ncharge_states
                    Z_single = pahbins_props(pp)%charge_states(ii)
                    izion = nint(Z_single)
                    izion = max(lbound(dust_info%Coulomb_factor,2), min(ubound(dust_info%Coulomb_factor,2), izion))

                    weight = 0d0
                    if (allocated(dust_info%fcharge_pah)) then
                        if (size(dust_info%fcharge_pah,1) >= ii .and. size(dust_info%fcharge_pah,2) >= pp) then
                            weight = dust_info%fcharge_pah(ii,pp)
                        end if
                    else if (ii == 1) then
                        weight = 1d0
                    end if
                    D_av = D_av + weight * dust_info%Coulomb_factor(kk,izion)
                end do
                D_av = max(D_av,1d-10)

                ! 3. Pair collision factor.
                coll_factor = pi * (pahbins_props(pp)%apah_cm + dustbins_props(kk)%asize_cm)**2d0 * v_rel * D_av

                ! 4. Collision rate from PAH-dust encounters, analogous to all-bin coagulation.
                rate1 = coll_factor * y_dust(index_dust) / dustbins_props(kk)%mgrain
                if (rate1 <= 0d0) cycle

                ! 5. Maxwellian sticking probability using a threshold equivalent to E_col = 1 eV.
                p_stick = cached_p_stick_pah_dust(pp, kk)
                rate1 = rate1 * p_stick
                if (rate1 <= 0d0) cycle

                if (present(kmax)) then
                    kmax = max(kmax, abs(rate1))
                end if

                ! 6. Transfer PAH mass to the interacting carbonaceous dust bin.
                rate2 = rate1 * y_dust(pp)
                dydt_dust(pp, 2) = dydt_dust(pp, 2) + rate2
                dydt_dust(index_dust, 1) = dydt_dust(index_dust, 1) + rate2
            end do
        end do pahloop
#endif

    end subroutine pah_freezing_rate
    
end module dust_rates