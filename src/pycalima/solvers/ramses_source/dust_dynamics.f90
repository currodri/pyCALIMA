module dust_dynamics
    use amr_parameters
    use constants, only: twopi, pi, e2instatC, kB, mH
    use hydro_parameters, only: n_elements,nvar
    use dust_commons

    implicit none

#ifdef RT
    ! Radiation-pressure and drag state of the level cells of one stencil sweep
    ! (get_dust_courant_dt, dust_push_fine; radpressure_acc_once). It depends only on
    ! the cell's uold and rtuold, which a sweep does not change, yet every cell is in
    ! the 6^ndim stencil of 3^ndim grids. Direct-mapped on (grid, son), at most
    ! RP_NMAX entries; an entry is valid for the sweep that wrote it (rp_stamp).
    integer, parameter :: RP_NMAX = 131072
    integer, save :: rp_n = 0, rp_sweep = 0
    integer, allocatable, save :: rp_key(:), rp_stamp(:)
    real(dp), allocatable, save :: rp_val(:,:)
#endif

    contains

    pure integer function ifind_ilo(idim, lo, f)
        integer, intent(in) :: idim, lo, f
        if (idim == 1) then; ifind_ilo = f; else; ifind_ilo = lo; endif
    end function ifind_ilo
    pure integer function ifind_ihi(idim, hi, f)
        integer, intent(in) :: idim, hi, f
        if (idim == 1) then; ifind_ihi = f; else; ifind_ihi = hi; endif
    end function ifind_ihi

    pure integer function ifind_jlo(idim, lo, f)
        integer, intent(in) :: idim, lo, f
        if (idim == 2) then; ifind_jlo = f; else; ifind_jlo = lo; endif
    end function ifind_jlo
    pure integer function ifind_jhi(idim, hi, f)
        integer, intent(in) :: idim, hi, f
        if (idim == 2) then; ifind_jhi = f; else; ifind_jhi = hi; endif
    end function ifind_jhi

    pure integer function ifind_klo(idim, lo, f)
        integer, intent(in) :: idim, lo, f
        if (idim == 3) then; ifind_klo = f; else; ifind_klo = lo; endif
    end function ifind_klo
    pure integer function ifind_khi(idim, hi, f)
        integer, intent(in) :: idim, hi, f
        if (idim == 3) then; ifind_khi = f; else; ifind_khi = hi; endif
    end function ifind_khi

    function draine2011_stopping_time(a, rho_s, D, nH, T, coul) result(t_s)
        ! Effective stopping time t_s = |w|/|D| [s] of a grain under the drag of
        ! Draine (2011) eq. (24)-(25), after Draine & Salpeter (1979): the drift
        ! w at which F = 2 pi a^2 n_H kT G(s) balances m_d |D|, s = w/(2kT/m_H)^1/2,
        !   G(s) = 8s/(3 sqrt(pi)) (1 + 9 pi s^2/64)^1/2 + coul s/(3 sqrt(pi)/4 + s^3),
        ! coul = (n_H+/n_H) phi^2 ln(Lambda), or its average over the charge
        ! distribution (drag_model='draine2011_pz'). With coul = 0 and s << 1 this is
        ! Epstein drag on the hydrogen alone. G is not monotonic for large coul, so
        ! take the smallest root, the branch a grain reaches from rest: past the local
        ! maximum of G it jumps to the collisional branch (Draine's runaway). cgs.
        real(dp), intent(in) :: a, rho_s, D, nH, T, coul
        real(dp) :: t_s
        real(dp) :: G_target, g0, s_lin, s_lo, s_hi, s_mid
        integer  :: it

        g0       = (8d0 + 4d0*coul) / (3d0*sqrt(pi))          ! dG/ds at s = 0
        G_target = (2d0/3d0) * a * rho_s * abs(D) / (max(nH, 1d-30) * kB * T)
        s_lin    = G_target / g0
        if (s_lin < 1d-3) then
            ! Linear regime, G = g0*s up to O(s^2)
            t_s = (2d0/3d0) * a * rho_s * sqrt(2d0*kB*T/mH) / (max(nH, 1d-30) * kB * T * g0)
            return
        end if
        ! Bracket the smallest root by stepping out from deep in the linear regime
        s_lo = 1d-3 * s_lin
        s_hi = s_lo
        do it = 1, 400
            s_hi = 1.2d0 * s_lo
            if (G_draine(s_hi) >= G_target) exit
            s_lo = s_hi
        end do
        do it = 1, 50
            s_mid = 0.5d0 * (s_lo + s_hi)
            if (G_draine(s_mid) >= G_target) then
                s_hi = s_mid
            else
                s_lo = s_mid
            end if
        end do
        t_s = 0.5d0 * (s_lo + s_hi) * sqrt(2d0*kB*T/mH) / abs(D)
    contains
        real(dp) function G_draine(s)
            real(dp), intent(in) :: s
            G_draine = 8d0*s/(3d0*sqrt(pi)) * sqrt(1d0 + 9d0*pi*s**2/64d0) &
                     + coul*s/(0.75d0*sqrt(pi) + s**3)
        end function G_draine
    end function draine2011_stopping_time

    function grain_relative_velocity(model,T,rho_gas,nH,v_turb&
                                    &,local_mu,inject_L&
                                    &,target_a,projectile_a&
                                    &,target_s,projectile_s&
                                    &,target_m,projectile_m)
        use pm_commons, only: localseed
        use random, only: ranf
        ! This function returns the relative collision velocity
        ! of two grains (target and projectile) based on a particular
        ! collision model. For further details see Section 2.3.1
        ! in Rodriguez Montero et al. (2023).

        ! model        => name of collision model to use
        ! T            => gas temperature [K]
        ! rho_gas      => gas density [g/cm**3]
        ! nH           => Hydrogen number density [1/cm**3]
        ! v_turb       => turbulent velocity [cm]
        ! local_mu     => local mean molecular weight
        ! inject_L     => turbulent injection scale [cm]
        ! target_a     => target grain radius [cm]
        ! projectile_a => projectile grain radius [cm]
        ! target_s     => target grain material density [g/cm**3]
        ! projectile_s => projectile grain material density [g/cm**3]
        ! target_m     => target grain mass [g]
        ! projectile_m => projectile grain mass [g]
        implicit none
        character(len=30), intent(in) :: model
        real(dp), intent(in) :: T,rho_gas,nH,v_turb,local_mu,inject_L
        real(dp), intent(in) :: target_a,target_s,target_m
        real(dp), intent(in) :: projectile_a,projectile_s,projectile_m

        real(dp) :: grain_relative_velocity
        real(dp) :: dV_thermal,cs_gas,v_th
        real(dp) :: mfp,tau_L,Re,tau_eta,rc
        real(dp) :: ts_target,ts_projectile
        real(dp) :: St_target,St_projectile
        real(dp) :: Stmin,dV_turb
        real(kind=8) :: RandNum
        real(dp) :: Mach,v_target,v_projectile,rand_costheta

        ! Cache for gas-phase invariants to avoid redundant calculations within a cell
        real(dp),save :: last_T=-1d0, last_rho_gas=-1d0, last_nH=-1d0, last_v_turb=-1d0, last_mu=-1d0, last_L=-1d0
        character(len=30),save :: last_model=''
        real(dp),save :: cs_gas_save, v_th_save, tau_L_save, Re_save, tau_eta_save, mfp_save
        if (T /= last_T .or. rho_gas /= last_rho_gas .or. nH /= last_nH .or. &
            v_turb /= last_v_turb .or. local_mu /= last_mu .or. inject_L /= last_L .or. &
            model /= last_model) then
            
            last_T = T; last_rho_gas = rho_gas; last_nH = nH
            last_v_turb = v_turb; last_mu = local_mu; last_L = inject_L
            last_model = model

            ! Gas sound speed (assumed gas with adiabatic constant of 5d0/3d0)
            cs_gas_save = sqrt(5d0/3d0 * kB * T / (mH * local_mu))
            ! Thermal velocity (Maxwelian distribution)
            v_th_save = sqrt(8d0/pi) * cs_gas_save

            if (trim(model) == 'Ormel2007') then
                ! Distance of closest particle approach (ionised)
                rc = e2instatC / (kB * T)
                ! Particle mean free path
                mfp_save = 1d0 / (nH * rc**2d0)
                ! Eddie injection timescale
                tau_L_save = inject_L / v_turb
                ! Reynolds number (ratio of inertial to viscous forces)
                Re_save = 3d0 * v_turb * inject_L / (cs_gas_save * mfp_save)
                ! Disipation timescale
                tau_eta_save = tau_L_save / sqrt(Re_save)
            end if
        end if
        
        cs_gas = cs_gas_save
        v_th = v_th_save

        if (trim(model).eq.'Ormel2007') then
            ! This is based on the formulation presented in Kawasaki & Machida (2023)
            ! which is basically the analytical model of Ormel & Cuzzi (2007)

            ! 1. Contribution to relative velocity from thermal (Brownian) motion
            dV_thermal = sqrt(8.d0 * kB * T * (target_m + projectile_m)/(target_m * projectile_m))

            ! 2. Assume that the injection scale of turbulence is a cell size of inject_L length and
            ! the velocity is given by the largest size eddie velocity
            tau_L = tau_L_save
            Re = Re_save
            tau_eta = tau_eta_save
            
            ! 3. Stopping time computation (we are always in the Epstein regime for large particles)
            ts_target = target_s * target_a / (rho_gas * v_th)
            ts_projectile = projectile_s * projectile_a / (rho_gas * v_th)

            ! 4. Stokes' number for both particles
            St_target = ts_target / tau_L
            St_projectile = ts_projectile / tau_L

            ! 5. Finally compute the relative velocity between the particles
            Stmin = tau_eta / tau_L
            if (ts_target < tau_eta) then
                dV_turb = sqrt(3d0/2d0) * v_turb * sqrt((St_target-St_projectile)/(St_target+St_projectile)) &
                            * sqrt((St_target**2d0/(St_target+Stmin))-(St_projectile**2d0/(St_projectile+Stmin)))
            else if ((tau_eta.le.ts_target).and.(ts_target<tau_L)) then
                dV_turb = sqrt(3d0/2d0) * v_turb * sqrt(OC07_function(St_projectile/St_target)*St_target)
            else if (ts_target.ge.tau_L) then
                dV_turb = sqrt(3d0/2d0) * v_turb * sqrt(1d0/(1d0+St_target) + 1d0/(1d0+St_projectile))
            end if
            grain_relative_velocity = sqrt(dV_thermal**2d0 + dV_turb**2d0)
        else if (trim(model).eq.'Hirashita2019') then
            ! Velocity scaling with the Mach number as given by the model of Hirashita & Aoyama (2019)
            ! which is a further approximation from the full Ormel & Cuzzi (2007) model (see Appendix C)

            Mach = v_turb / cs_gas
            v_target = 1.1d5 * (Mach**(3d0/2d0)) * sqrt(target_a/1d-5) * ((T/1d4)**(1d0/4d0)) * (nH**(-1d0/4d0)) * sqrt(target_s/3.5d0)
            v_projectile = 1.1d5 * (Mach**(3d0/2d0)) * sqrt(projectile_a/1d-5) * ((T/1d4)**(1d0/4d0)) * (nH**(-1d0/4d0)) * sqrt(projectile_s/3.5d0)
            call ranf(localseed,RandNum)
            ! Guard against occasional RNG roundoff/implementation excursions outside [0,1].
            RandNum = max(0d0, min(1d0, RandNum))
            rand_costheta = max(-1d0, min(1d0, 2d0 * RandNum - 1d0))
            grain_relative_velocity = sqrt(v_target**2d0 + v_projectile**2d0 - 2d0 * v_target * v_projectile * rand_costheta)
        else
            ! Just assume that the relative velocity is given by the turbulent velocity
            grain_relative_velocity = v_turb
        end if
    end function grain_relative_velocity

    function OC07_function(x)

        ! Limiting function for the intermediate case in Ormel & Cuzzi (2007)
        ! x => Stokes' number ratio between target and projectiles
        implicit none

        real(dp), intent(in) :: x
        
        real(dp) :: OC07_function

        OC07_function = 3.2d0 - (1d0 + x) + 2d0/(1d0 + x) * (1d0/2.6d0 + x**3d0/(1.6d0 + x))
    end function OC07_function

    subroutine dust_shock_destruction(tempvar,shocked_mass,metal_load,numofSN, &
                                        &SN_type,cell_vol,fraction_loadSN)
        ! This subroutines encapsulated the whole computation of
        ! destruction of dust in the gas shocked above 100 km/s
        ! tempvar         => uold for the cell chosen
        ! shocked_mass    => gas mass shocked above 100 km/s
        ! metal_load      => metal mass loaded in ejecta
        ! numofSN         => number SN events taking place of star particle
        ! SN_type         => type of SN event (Ia or II)
        ! cell_vol        => cell volume
        ! fraction_loadSN => fraction_loadSN
        implicit none

        real(dp),dimension(1:nvar),intent(inout)        :: tempvar
        real(dp),intent(in)                             :: shocked_mass
        real(dp),dimension(1:n_elements),intent(inout)  :: metal_load
        real(dp),intent(in)                             :: numofSN,cell_vol
        real(dp),intent(in)                             :: fraction_loadSN
        character(len=2),intent(in)                     :: SN_type

        integer                                         :: ii,jj,jj1,jj2,kk
        real(dp)                                        :: Mgas,Mdust
        real(dp)                                        :: dMdust,newMdust
        real(dp)                                        :: mmet
        real(dp),dimension(1:ndust)                     :: Mdust_sha
        real(dp),dimension(1:npah)                      :: Mpah_sha

        Mgas = tempvar(1)
        Mdust_sha(:) = 0.0d0

        ! 1. First do regular dust grains
        if (dust_SNdest) then
            do ii=1,ndchemtype
                jj1 = istart_chemtype(ii)
                jj2 = jj1 + dustbins_per_chemtype(ii) - 1
                do jj=jj1,jj2
                    Mdust = tempvar(idust-1+jj)
                    ! Yohan's model
                    ! (eqn 13 Granato,2021, and eqn 19 in Aoyama et al. 2017)
                    ! +size dependance like thermal sputtering
                    dMdust = -(1d0-(1d0-MIN(1d0-exp(-dustbins_props(jj)%SNdest_eff*0.1d0/dustbins_props(jj)%asize),1.0d0)*&
                                & MIN(shocked_mass/Mgas,1.0d0))**numofSN)*Mdust
                    newMdust = MAX(Mdust+dMdust,0d0)
                    dMdust = max(Mdust - newMdust,0d0)
                    tempvar(idust-1+jj) = newMdust

                    ! TODO: Code the production of smaller grains via shattering in shocks
                    
                    ! Move destroyed dust mass to the metal variables
                    if (dMdust.lt.0d0) then
                        print*,'NEGATIVE DUST IN SHOCK DESTRUCTION!!'
                        PRINT*, 'SN type:',SN_type
                        print*, 'dMdust:',dMdust
                        print*, 'newMdust:',newMdust
                        stop
                    end if
                    mmet = dMdust * cell_vol * fraction_loadSN
                    do kk = 1, dustbins_props(jj)%nelements
                        metal_load(dustbins_props(jj)%el_index(kk)) = metal_load(dustbins_props(jj)%el_index(kk)) + &
                            dustbins_props(jj)%el_mfractions(kk) * mmet
                    end do
                    if (dust_log) then
                        if (SN_type == 'II') then
                            dM_SNIId(npah+jj) = dM_SNIId(npah+jj) - dMdust*cell_vol
                        else
                            dM_SNIad(npah+jj) = dM_SNIad(npah+jj) - dMdust*cell_vol
                        end if
                    end if
                end do
            end do
        end if

#if NPAH>0
        ! 2. Now do PAHs
        if (dust_pahs .and. pah_sn_destruction) then
            do ii=1,npah
                Mdust = tempvar(ipah+ii-1)
                dMdust = -(1d0-(1d0-pahbins_props(ii)%SNdest_eff*MIN(shocked_mass/Mgas,1.0d0))**numofSN)*Mdust
                newMdust = MAX(Mdust+dMdust,0d0)
                dMdust = max(Mdust - newMdust,0d0)
                tempvar(ipah+ii-1) = newMdust
                ! Update carbon density with destroyed PAHs
                metal_load(pahbins_props(ii)%C_index) = metal_load(pahbins_props(ii)%C_index) + dMdust*cell_vol*fraction_loadSN ! carbon
                if (dust_log) then
                    if (SN_type == 'II') then
                        dM_SNIId(ii) = dM_SNIId(ii) - dMdust*cell_vol
                    else
                        dM_SNIad(ii) = dM_SNIad(ii) - dMdust*cell_vol
                    end if
                end if
            end do
        end if
#endif

        ! 3. Check that the metals and dust/pahs are not negative
        if (any(metal_load .lt. 0d0)) then
            print*,'NEGATIVE METALS AFTER SHOCK DESTRUCTION!!'
            PRINT*, 'SN type:',SN_type
            print*,'metals:',metal_load
            print*,'dust:',tempvar(idust:idust+ndust-1)
            if (dust_pahs) print*,'pahs:',tempvar(ipah:ipah+npah-1)
            print*,'tempvar: ',tempvar
            stop
        end if
        if (any(tempvar(idust:idust+ndust-1) .lt. 0d0)) then
            print*,'NEGATIVE DUST AFTER SHOCK DESTRUCTION!!'
            print*, 'SN type:',SN_type
            print*,'dust:',tempvar(idust:idust+ndust-1)
            stop
        end if
        if (dust_pahs .and. any(tempvar(ipah:ipah+npah-1) .lt. 0d0)) then
            print*,'NEGATIVE PAHS AFTER SHOCK DESTRUCTION!!'
            print*, 'SN type:',SN_type
            print*,'pahs:',tempvar(ipah:ipah+npah-1)
            stop
        end if
    end subroutine dust_shock_destruction

#ifdef RT
    subroutine rp_new_sweep()
        ! start a sweep: the cache entries of the previous one become invalid
        use amr_commons, only: ngridmax
        use hydro_parameters, only: ndust, npah
        implicit none
        if (.not. allocated(rp_key)) then
            rp_n = min(RP_NMAX, twotondim*ngridmax)
            allocate(rp_key(rp_n), rp_stamp(rp_n))
            allocate(rp_val((1 + max(1, ndust) + max(1, npah))*ndim + max(1, ndust) + ndust + 2, rp_n))
            rp_key = 0
            rp_stamp = 0
        end if
        rp_sweep = rp_sweep + 1
    end subroutine rp_new_sweep

    subroutine radpressure_acc_once(icell, cell_state, cell_rt_state, ilevel, dx, gas_acc, dust_acc, pah_acc, &
                                    irtrap_share, drag_state)
        ! compute_gas_dust_radpressure_acc of the level cell icell (uold/rtuold index), computed
        ! once per sweep and copied for the other stencils that hold the cell; a cache collision
        ! recomputes, so the outputs are always those of the direct call
        use amr_commons, only: ncoarse, ngridmax
        use hydro_parameters, only: ndust, npah
        use dust_radpressure_module, only: compute_gas_dust_radpressure_acc
        implicit none
        integer, intent(in) :: icell, ilevel
        real(dp), dimension(:), intent(in) :: cell_state, cell_rt_state
        real(dp), intent(in) :: dx
        real(dp), dimension(1:ndim), intent(out) :: gas_acc
        real(dp), dimension(max(1, ndust), 1:ndim), intent(out) :: dust_acc
        real(dp), dimension(max(1, npah), 1:ndim), intent(out) :: pah_acc
        real(dp), dimension(max(1, ndust)), intent(out) :: irtrap_share
        real(dp), dimension(1:ndust+2), intent(out) :: drag_state
        integer :: s, n1, n2, n3, n4
        ! slot of (grid, son): (grid - 1) 2^ndim + son - 1, modulo the cache size
        s = mod(mod(icell - ncoarse - 1, ngridmax)*twotondim + (icell - ncoarse - 1)/ngridmax, rp_n) + 1
        n1 = ndim
        n2 = n1 + max(1, ndust)*ndim
        n3 = n2 + max(1, npah)*ndim
        n4 = n3 + max(1, ndust)
        if (rp_key(s) == icell .and. rp_stamp(s) == rp_sweep) then
            gas_acc = rp_val(1:n1, s)
            dust_acc = reshape(rp_val(n1+1:n2, s), (/ max(1, ndust), ndim /))
            pah_acc = reshape(rp_val(n2+1:n3, s), (/ max(1, npah), ndim /))
            irtrap_share = rp_val(n3+1:n4, s)
            drag_state = rp_val(n4+1:n4+ndust+2, s)
            return
        end if
        call compute_gas_dust_radpressure_acc(cell_state, cell_rt_state, ilevel, dx, gas_acc, dust_acc, pah_acc, &
                                              irtrap_share, drag_state)
        rp_key(s) = icell
        rp_stamp(s) = rp_sweep
        rp_val(1:n1, s) = gas_acc
        rp_val(n1+1:n2, s) = reshape(dust_acc, (/ n2 - n1 /))
        rp_val(n2+1:n3, s) = reshape(pah_acc, (/ n3 - n2 /))
        rp_val(n3+1:n4, s) = irtrap_share
        rp_val(n4+1:n4+ndust+2, s) = drag_state
    end subroutine radpressure_acc_once
#endif

    subroutine get_dust_courant_dt(ilevel)
        ! ====================================================================
        ! Compute the dust-drift CFL timestep constraint by accessing uold
        ! directly and evaluating the true face pressure gradient.
        !
        ! The subroutine reuses the full AMR stencil-building infrastructure
        ! from dust_upwind_correct1 and mirrors the primitive-extraction step
        ! from calculate_pure_drag_fluxes so that the CFL bound is tight and
        ! consistent with the numerical scheme being stabilised.
        !
        !   (1) Advection (hyperbolic) CFL:
        !         dt <= courant_factor * dx / |w_drift_face|
        !       where w_drift_face is the TVA drift velocity evaluated using
        !       the true face pressure gradient (P_R - P_L)/dx.
        !
        ! MPI reduces to the global minimum and updates dtnew(ilevel).
        ! ====================================================================
        use amr_commons
        use const
        use hydro_commons
        use hydro_parameters, only: courant_factor, smallr, smallc, gamma, neul, npah
#ifdef RT
        use rt_hydro_commons, only: rtuold, nrtvar
        use dust_radpressure_module, only: compute_gas_dust_radpressure_acc
        use rt_parameters, only: rt_isIR, rt_isIRtrap, iIRtrapVar
#endif
        use mpi_mod
        implicit none
        integer, intent(in) :: ilevel

        ! ----------------------------------------------------------------
        ! Stencil workspace (same dimensions / SAVE as dust_upwind_correct1)
        ! ----------------------------------------------------------------
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:nvar_all), save :: uloc

        integer, dimension(1:nvector), save :: ind_grid, ind_cell, ind_father
        integer, dimension(1:nvector), save :: igrid_nbor, ind_exist, ind_nexist, ind_buffer
        integer, dimension(1:nvector, 1:threetondim) :: nbors_father_cells
        integer, dimension(1:nvector, 0:twondim)     :: ibuffer_father
        real(dp), dimension(1:nvector, 0:twondim, 1:nvar_all) :: u1
        real(dp), dimension(1:nvector, 1:twotondim,  1:nvar_all) :: u2

        ! ----------------------------------------------------------------
        ! Cell-centred primitive fields (same structure as calc_pure_drag)
        ! ----------------------------------------------------------------
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2),          save :: Pg, rho_mix, c_s, eps_tot_arr
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust), save :: eps_arr

        ! Radiation acceleration arrays
#ifdef RT
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndim), save :: a_rad_g
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust, 1:ndim), save :: a_rad_d
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, max(1,npah), 1:ndim), save :: a_rad_pah

        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:nrtvar), save :: rt_uloc
        real(dp), dimension(1:nvector, 0:twondim, 1:nrtvar) :: rt_u1
        real(dp), dimension(1:nvector, 1:twotondim, 1:nrtvar) :: rt_u2
#endif

        ! ----------------------------------------------------------------
        ! Loop indices, scalars
        ! ----------------------------------------------------------------
        integer  :: i, j, k, l, jbin, idim, ivar, ind_son, iskip
        integer  :: ncache, ngrid, igrid, nexist, nbuffer, ind_father_idx
        integer  :: i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3
        integer  :: i1min, i1max, j1min, j1max, k1min, k1max
        integer  :: i2min, i2max, j2min, j2max, k2min, k2max
        integer  :: i3min, i3max, j3min, j3max, k3min, k3max
        integer  :: ilo, ihi, jlo, jhi, klo, khi

        real(dp) :: dx, scale, dt_loc, dt_all, dtcell
        real(dp) :: scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2
        real(dp), dimension(1:ndust) :: agrain_code, sgrain_code, t_s_face_arr

        ! Face-centred quantities
        real(dp) :: Pg_L, Pg_R, Pg_face
        real(dp) :: rho_L, rho_R, rho_face, rho_g_face
        real(dp) :: eps_tot_L, eps_tot_R, eps_tot_face
        real(dp) :: eps_face_bin, c_s_face
        real(dp) :: grad_P, t_s_face, avg_ts_face, u_drift, D_i
        real(dp) :: eken, rho_gas_cell, erad_cell
        integer  :: irad
        ! Trapped-IR radiation pressure and per-bin Rosseland opacity share.
        ! Declared unconditionally: the drift driver below references them
        ! outside the #ifdef RT guards (they stay zero without RT).
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: Ptrap
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust), save :: s_IRtrap
        ! n_H, T and per-bin Coulomb coefficient for drag_model='draine2011'
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust+2), save :: drag_state
        real(dp), dimension(1:ndust+2) :: drag_state_face
        real(dp) :: Ptrap_L, Ptrap_R, grad_Ptrap
        real(dp), dimension(1:ndust) :: s_IRtrap_face
        logical  :: do_irtrap
        real(dp) :: a_rad_g_face, a_rad_mix, w_g, w_d_val, sum_eps_ts_D
        real(dp) :: w_cap, wmax_all
        integer(kind=8) :: nclip_all
        real(dp), dimension(1:ndust) :: a_rad_d_face, D_bin

#ifndef WITHOUTMPI
        integer  :: info
#endif

        if (numbtot(1,ilevel) == 0) return

        call units(scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)

        do i = 1, ndust
            agrain_code(i) = dustbins_props(i)%asize_cm / scale_l
            sgrain_code(i) = dustbins_props(i)%sgrain   / scale_d
        end do

        scale  = boxlen / dble(icoarse_max - icoarse_min + 1)
        dx     = 0.5d0**ilevel * scale
        dt_loc = dtnew(ilevel)      ! upper bound: current level timestep

        ! ----------------------------------------------------------------
        ! PART A: Simple analytic CFL for the drift-test mode
        ! ----------------------------------------------------------------
        if (use_w_drift_test) then
            do i = 1, ndim
                if (abs(w_drift_test(i)) > 0.0_dp) &
                    dt_loc = min(dt_loc, courant_factor * dx / abs(w_drift_test(i)))
            end do
        end if

        ! ----------------------------------------------------------------
        ! PART B: Stencil-based CFL using the true face pressure gradient
        ! ----------------------------------------------------------------
        if (.not. use_w_drift_test) then
#ifdef RT
            if (dust_radpressure) call rp_new_sweep()
#endif
            ! Index bounds — identical to dust_upwind_correct1
            i1min=0; i1max=0; i2min=0; i2max=0; i3min=1; i3max=1
            j1min=0; j1max=0; j2min=0; j2max=0; j3min=1; j3max=1
            k1min=0; k1max=0; k2min=0; k2max=0; k3min=1; k3max=1
            if (ndim>0) then; i1max=2; i2max=1; i3max=2; end if
            if (ndim>1) then; j1max=2; j2max=1; j3max=2; end if
            if (ndim>2) then; k1max=2; k2max=1; k3max=2; end if

            ilo = min(1,iu1+1); ihi = max(1,iu2-1)
            jlo = min(1,ju1+1); jhi = max(1,ju2-1)
            klo = min(1,ku1+1); khi = max(1,ku2-1)

            ncache = active(ilevel)%ngrid
            do igrid = 1, ncache, nvector
                ngrid = min(nvector, ncache - igrid + 1)
                do i = 1, ngrid
                    ind_grid(i) = active(ilevel)%igrid(igrid+i-1)
                end do

                ! Get the father cells for this batch of grids
                do i = 1, ngrid
                    ind_cell(i) = father(ind_grid(i))
                end do
                call get3cubefather(ind_cell, nbors_father_cells, ngrid, ilevel)

                ! ============================================================
                ! Fill uloc stencil — identical to dust_upwind_correct1
                ! ============================================================
                do k1=k1min,k1max; do j1=j1min,j1max; do i1=i1min,i1max
                    nbuffer=0; nexist=0
                    ind_father_idx = 1+i1+3*j1+9*k1
                    do i = 1, ngrid
                        igrid_nbor(i) = son(nbors_father_cells(i, ind_father_idx))
                        if (igrid_nbor(i) > 0) then
                            nexist = nexist+1; ind_exist(nexist) = i
                        else
                            nbuffer = nbuffer+1; ind_nexist(nbuffer) = i
                            ind_buffer(nbuffer) = nbors_father_cells(i, ind_father_idx)
                        end if
                    end do

                    if (nbuffer > 0) then
                        call getnborfather(ind_buffer, ibuffer_father, nbuffer, ilevel)
                        do j=0,twondim; do ivar=1,nvar_all; do i=1,nbuffer
                            u1(i,j,ivar) = uold(ibuffer_father(i,j), ivar)
                        end do; end do; end do
                        call interpol_hydro(u1, u2, nbuffer)

#ifdef RT
                        if (dust_radpressure) then
                            do j=0,twondim; do ivar=1,nrtvar; do i=1,nbuffer
                                rt_u1(i,j,ivar) = rtuold(ibuffer_father(i,j), ivar)
                            end do; end do; end do
                            call rt_interpol_hydro(rt_u1, rt_u2, nbuffer)
                        end if
#endif
                    end if

                    do k2=k2min,k2max; do j2=j2min,j2max; do i2=i2min,i2max
                        ind_son = 1+i2+2*j2+4*k2
                        iskip   = ncoarse + (ind_son-1)*ngridmax
                        do i=1,nexist; ind_cell(i) = iskip + igrid_nbor(ind_exist(i)); end do
                        i3=1; j3=1; k3=1
                        if (ndim>0) i3 = 1+2*(i1-1)+i2
                        if (ndim>1) j3 = 1+2*(j1-1)+j2
                        if (ndim>2) k3 = 1+2*(k1-1)+k2
                        do ivar=1,nvar_all
                            do i=1,nexist;  uloc(ind_exist(i), i3,j3,k3,ivar) = uold(ind_cell(i),ivar); end do
                            do i=1,nbuffer; uloc(ind_nexist(i),i3,j3,k3,ivar) = u2(i,ind_son,ivar);     end do
                        end do

#ifdef RT
                        if (dust_radpressure) then
                            do ivar=1,nrtvar
                                do i=1,nexist;  rt_uloc(ind_exist(i), i3,j3,k3,ivar) = rtuold(ind_cell(i),ivar); end do
                                do i=1,nbuffer; rt_uloc(ind_nexist(i),i3,j3,k3,ivar) = rt_u2(i,ind_son,ivar);     end do
                            end do

                            ! Compute the radiation pressures for each cell (level cells: once per sweep)
                            do i=1,nexist
                                call radpressure_acc_once(ind_cell(i), uloc(ind_exist(i),i3,j3,k3,:),&
                                                                        rt_uloc(ind_exist(i),i3,j3,k3,:),& 
                                                                        ilevel,dx,& 
                                                                        a_rad_g(ind_exist(i),i3,j3,k3,:),&
                                                                        a_rad_d(ind_exist(i),i3,j3,k3,:,:),&
                                                                        a_rad_pah(ind_exist(i),i3,j3,k3,:,:),&
                                                                        s_IRtrap(ind_exist(i),i3,j3,k3,:),&
                                                                        drag_state(ind_exist(i),i3,j3,k3,:))
                            end do
                            do i=1,nbuffer
                                call compute_gas_dust_radpressure_acc(uloc(ind_nexist(i),i3,j3,k3,:),&
                                                                        rt_uloc(ind_nexist(i),i3,j3,k3,:),& 
                                                                        ilevel,dx,& 
                                                                        a_rad_g(ind_nexist(i),i3,j3,k3,:),&
                                                                        a_rad_d(ind_nexist(i),i3,j3,k3,:,:),&
                                                                        a_rad_pah(ind_nexist(i),i3,j3,k3,:,:),&
                                                                        s_IRtrap(ind_nexist(i),i3,j3,k3,:),&
                                                                        drag_state(ind_nexist(i),i3,j3,k3,:))
                            end do
                        end if
#endif
                    end do; end do; end do
                end do; end do; end do

                ! ============================================================
                ! STEP 1: Cell-centred primitive extraction
                ! ============================================================
                do k=ku1,ku2; do j=ju1,ju2; do i=iu1,iu2; do l=1,ngrid
                    if (tva_test_mode == TVA_TEST_DIFFUSE) then
                        rho_mix(l,i,j,k) = 1.0_dp
                    else
                        rho_mix(l,i,j,k) = max(uloc(l,i,j,k,1), smallr)
                    end if
                    eps_tot_arr(l,i,j,k) = 0.0_dp
                    do jbin = 1, ndust
                        eps_arr(l,i,j,k,jbin) = uloc(l,i,j,k,idust+jbin-1) / rho_mix(l,i,j,k)
                        eps_tot_arr(l,i,j,k)  = eps_tot_arr(l,i,j,k) + eps_arr(l,i,j,k,jbin)
                    end do
                    eps_tot_arr(l,i,j,k) = min(max(eps_tot_arr(l,i,j,k), 0.0_dp), 1.0_dp-smallr)
                    rho_gas_cell = rho_mix(l,i,j,k) * (1.0_dp - eps_tot_arr(l,i,j,k))

                    eken = 0.0_dp
                    do idim = 1, ndim
                        eken = eken + uloc(l,i,j,k,idim+1)**2
                    end do
                    eken = half * eken / rho_mix(l,i,j,k)**2

                    ! Same NENER subtraction as the flux routines (see comment there).
                    erad_cell = 0.0_dp
#if NENER>0
                    do irad = 1, nener
                        erad_cell = erad_cell + uloc(l,i,j,k,nhydro+irad) / rho_mix(l,i,j,k)
                    end do
#endif

                    if (tva_test_mode == TVA_TEST_DIFFUSE) then
                        Pg(l,i,j,k)  = (1.0_dp - eps_tot_arr(l,i,j,k)) * rho_mix(l,i,j,k)
                        c_s(l,i,j,k) = 1.0_dp
                    else
                        Pg(l,i,j,k)  = max((gamma-1.0_dp) * rho_gas_cell * &
                            max(uloc(l,i,j,k,neul)/rho_mix(l,i,j,k) - eken - erad_cell, &
                                smallc**2/gamma/(gamma-1.0_dp)), &
                            smallr * smallc**2)
                        c_s(l,i,j,k) = sqrt(gamma * Pg(l,i,j,k) / max(rho_gas_cell, smallr))
                    end if
                end do; end do; end do; end do

                ! Trapped-IR pressure, same definition as the flux routine so
                ! this CFL bound applies to the drift actually taken.
                do_irtrap = .false.
#if defined(RT) && NENER>0
                do_irtrap = rt_isIR .and. rt_isIRtrap .and. ndust > 0
#endif
                Ptrap = 0.0_dp
#ifndef RT
                ! Without RT nothing fills the share array; it is a save
                ! variable, so zero it rather than read it uninitialised.
                s_IRtrap = 0.0_dp
#endif
#if defined(RT) && NENER>0
                if (do_irtrap) then
                    do k=ku1,ku2; do j=ju1,ju2; do i=iu1,iu2; do l=1,ngrid
                        Ptrap(l,i,j,k) = (gamma_rad(iIRtrapVar-nhydro) - 1.0_dp) &
                                       * max(uloc(l,i,j,k,iIRtrapVar), 0.0_dp)
                    end do; end do; end do; end do
                end if
#endif

                ! ============================================================
                ! STEP 2: Face CFL from true pressure gradient
                ! ============================================================
                do idim = 1, ndim
                    do k = ifind_klo(idim,klo,kf1), ifind_khi(idim,khi,kf2)
                    do j = ifind_jlo(idim,jlo,jf1), ifind_jhi(idim,jhi,jf2)
                    do i = ifind_ilo(idim,ilo,if1), ifind_ihi(idim,ihi,if2)
                        do l = 1, ngrid

                            ! --- Gather left/right cell quantities at this face ---
                            if (idim == 1) then
                                Pg_L      = Pg(l,i-1,j,k);       Pg_R      = Pg(l,i,j,k)
                                Ptrap_L   = Ptrap(l,i-1,j,k);    Ptrap_R   = Ptrap(l,i,j,k)
                                s_IRtrap_face(1:ndust) = half * (s_IRtrap(l,i-1,j,k,1:ndust) + s_IRtrap(l,i,j,k,1:ndust))
                                rho_L     = rho_mix(l,i-1,j,k);  rho_R     = rho_mix(l,i,j,k)
                                eps_tot_L = eps_tot_arr(l,i-1,j,k); eps_tot_R = eps_tot_arr(l,i,j,k)
                                c_s_face  = half * (c_s(l,i-1,j,k) + c_s(l,i,j,k))
#ifdef RT
                                if (dust_radpressure) then
                                    a_rad_g_face = half * (a_rad_g(l,i-1,j,k,idim) + a_rad_g(l,i,j,k,idim))
                                    do jbin = 1, ndust
                                        a_rad_d_face(jbin) = half * (a_rad_d(l,i-1,j,k,jbin,idim) + a_rad_d(l,i,j,k,jbin,idim))
                                    end do
                                else
                                    a_rad_g_face = 0.0_dp
                                    a_rad_d_face = 0.0_dp
                                end if
#else
                                a_rad_g_face = 0.0_dp
                                a_rad_d_face = 0.0_dp
#endif
                            else if (idim == 2) then
                                Pg_L      = Pg(l,i,j-1,k);       Pg_R      = Pg(l,i,j,k)
                                Ptrap_L   = Ptrap(l,i,j-1,k);    Ptrap_R   = Ptrap(l,i,j,k)
                                s_IRtrap_face(1:ndust) = half * (s_IRtrap(l,i,j-1,k,1:ndust) + s_IRtrap(l,i,j,k,1:ndust))
                                rho_L     = rho_mix(l,i,j-1,k);  rho_R     = rho_mix(l,i,j,k)
                                eps_tot_L = eps_tot_arr(l,i,j-1,k); eps_tot_R = eps_tot_arr(l,i,j,k)
                                c_s_face  = half * (c_s(l,i,j-1,k) + c_s(l,i,j,k))
#ifdef RT
                                if (dust_radpressure) then
                                    a_rad_g_face = half * (a_rad_g(l,i,j-1,k,idim) + a_rad_g(l,i,j,k,idim))
                                    do jbin = 1, ndust
                                        a_rad_d_face(jbin) = half * (a_rad_d(l,i,j-1,k,jbin,idim) + a_rad_d(l,i,j,k,jbin,idim))
                                    end do
                                else
                                    a_rad_g_face = 0.0_dp
                                    a_rad_d_face = 0.0_dp
                                end if
#else
                                a_rad_g_face = 0.0_dp
                                a_rad_d_face = 0.0_dp
#endif
                            else
                                Pg_L      = Pg(l,i,j,k-1);       Pg_R      = Pg(l,i,j,k)
                                Ptrap_L   = Ptrap(l,i,j,k-1);    Ptrap_R   = Ptrap(l,i,j,k)
                                s_IRtrap_face(1:ndust) = half * (s_IRtrap(l,i,j,k-1,1:ndust) + s_IRtrap(l,i,j,k,1:ndust))
                                rho_L     = rho_mix(l,i,j,k-1);  rho_R     = rho_mix(l,i,j,k)
                                eps_tot_L = eps_tot_arr(l,i,j,k-1); eps_tot_R = eps_tot_arr(l,i,j,k)
                                c_s_face  = half * (c_s(l,i,j,k-1) + c_s(l,i,j,k))
#ifdef RT
                                if (dust_radpressure) then
                                    a_rad_g_face = half * (a_rad_g(l,i,j,k-1,idim) + a_rad_g(l,i,j,k,idim))
                                    do jbin = 1, ndust
                                        a_rad_d_face(jbin) = half * (a_rad_d(l,i,j,k-1,jbin,idim) + a_rad_d(l,i,j,k,jbin,idim))
                                    end do
                                else
                                    a_rad_g_face = 0.0_dp
                                    a_rad_d_face = 0.0_dp
                                end if
#else
                                a_rad_g_face = 0.0_dp
                                a_rad_d_face = 0.0_dp
#endif
                            end if

                            grad_P       = (Pg_R - Pg_L) / dx
                            grad_Ptrap   = (Ptrap_R - Ptrap_L) / dx
                            Pg_face      = half * (Pg_L + Pg_R)
                            rho_face     = half * (rho_L + rho_R)
                            eps_tot_face = half * (eps_tot_L + eps_tot_R)
                            eps_tot_face = min(max(eps_tot_face, 0.0_dp), 1.0_dp - smallr)
                            rho_g_face   = max(rho_face * (1.0_dp - eps_tot_face), smallr)

                            do jbin = 1, ndust
                                if (idim == 1) then
                                    eps_face_bin = half * (eps_arr(l,i-1,j,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else if (idim == 2) then
                                    eps_face_bin = half * (eps_arr(l,i,j-1,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else
                                    eps_face_bin = half * (eps_arr(l,i,j,k-1,jbin) + eps_arr(l,i,j,k,jbin))
                                end if
                                eps_face_bin = max(eps_face_bin, 0.0_dp)

                                if (tva_test_mode == TVA_TEST_DIFFUSE) then
                                    t_s_face = 0.1_dp
                                else if (tva_test_mode == TVA_TEST_SHOCK) then
                                    t_s_face = eps_face_bin * rho_face / drag_coefficient(jbin)
                                else if (tva_test_mode == TVA_TEST_BLAST1D) then
                                    t_s_face = 6d-3
                                else
                                    t_s_face = epstein_coef * (sgrain_code(jbin) * agrain_code(jbin)) / &
                                        max(rho_g_face * c_s_face, smallr)
                                end if

                                t_s_face_arr(jbin) = t_s_face
                            end do

                            a_rad_mix = (1.0_dp - eps_tot_face) * a_rad_g_face
                            do jbin = 1, ndust
                                if (idim == 1) then
                                    eps_face_bin = half * (eps_arr(l,i-1,j,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else if (idim == 2) then
                                    eps_face_bin = half * (eps_arr(l,i,j-1,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else
                                    eps_face_bin = half * (eps_arr(l,i,j,k-1,jbin) + eps_arr(l,i,j,k,jbin))
                                end if
                                eps_face_bin = max(eps_face_bin, 0.0_dp)
                                a_rad_mix = a_rad_mix + eps_face_bin * a_rad_d_face(jbin)
                            end do

                            sum_eps_ts_D = 0.0_dp
                            do jbin = 1, ndust
                                if (idim == 1) then
                                    eps_face_bin = half * (eps_arr(l,i-1,j,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else if (idim == 2) then
                                    eps_face_bin = half * (eps_arr(l,i,j-1,k,jbin) + eps_arr(l,i,j,k,jbin))
                                else
                                    eps_face_bin = half * (eps_arr(l,i,j,k-1,jbin) + eps_arr(l,i,j,k,jbin))
                                end if
                                eps_face_bin = max(eps_face_bin, 0.0_dp)
                                D_bin(jbin) = grad_P / max(rho_face, smallr) + a_rad_d_face(jbin) - a_rad_mix
                                if (do_irtrap) then
                                    D_bin(jbin) = D_bin(jbin) - grad_Ptrap *          &
                                        ( s_IRtrap_face(jbin)                         &
                                          / max(eps_face_bin * rho_face, smallr)      &
                                          - 1.0_dp / max(rho_face, smallr) )
                                end if
                                ! Draine (2011) drag is nonlinear in the drift, so its
                                ! stopping time needs the driving acceleration D first.
                                if (draine_drag() .and. (tva_test_mode == TVA_TEST_NONE &
                                    .or. tva_test_mode >= TVA_TEST_SPRESS)) then
                                    if (idim == 1) then
                                        drag_state_face = half * (drag_state(l,i-1,j,k,:) + drag_state(l,i,j,k,:))
                                    else if (idim == 2) then
                                        drag_state_face = half * (drag_state(l,i,j-1,k,:) + drag_state(l,i,j,k,:))
                                    else
                                        drag_state_face = half * (drag_state(l,i,j,k-1,:) + drag_state(l,i,j,k,:))
                                    end if
                                    t_s_face_arr(jbin) = draine2011_stopping_time(dustbins_props(jbin)%asize_cm, &
                                        dustbins_props(jbin)%sgrain, D_bin(jbin) * scale_v / scale_t,          &
                                        drag_state_face(1), drag_state_face(2), drag_state_face(2+jbin)) / scale_t
                                end if
                                sum_eps_ts_D = sum_eps_ts_D + eps_face_bin * t_s_face_arr(jbin) * D_bin(jbin)
                            end do

                            ! The dust update sums ndim independent directional flux
                            ! divergences (dust_upwind_correct1/2), so the per-direction
                            ! bound needs a 1/ndim -- cmpdt does the equivalent with
                            ! ndim*c + sum|u| (hydro/courant_fine.f90:288-296). Without it a
                            ! 3D cell can lose more dust in one step than it holds.
                            ! The cap must match the one applied in the flux routines, or
                            ! this bound would not be a bound on the drift actually used.
                            w_cap = tva_wmax_cs * c_s_face
                            w_g = -sum_eps_ts_D
                            if (tva_wmax_cs > 0.0_dp .and. abs(w_g) > w_cap) then
                                tva_nclip = tva_nclip + 1
                                tva_wmax_seen = max(tva_wmax_seen, abs(w_g) / max(c_s_face, smallc))
                                w_g = sign(w_cap, w_g)
                            end if
                            u_drift = abs(w_g)
                            if (u_drift > 0.0_dp) then
                                dtcell = courant_factor * dx / (dble(ndim) * u_drift)
                                dt_loc = min(dt_loc, dtcell)
                            end if

                            do jbin = 1, ndust
                                w_d_val = t_s_face_arr(jbin) * D_bin(jbin) - sum_eps_ts_D
                                if (tva_wmax_cs > 0.0_dp .and. abs(w_d_val) > w_cap) then
                                    tva_nclip = tva_nclip + 1
                                    tva_wmax_seen = max(tva_wmax_seen, abs(w_d_val) / max(c_s_face, smallc))
                                    w_d_val = sign(w_cap, w_d_val)
                                end if
                                u_drift = abs(w_d_val)
                                if (u_drift > 0.0_dp) then
                                    dtcell = courant_factor * dx / (dble(ndim) * u_drift)
                                    dt_loc = min(dt_loc, dtcell)
                                end if
                            end do
                        end do
                    end do; end do; end do
                end do
            end do
        end if
        dt_all = dt_loc
        nclip_all = tva_nclip
        wmax_all = tva_wmax_seen
#ifndef WITHOUTMPI
        call MPI_ALLREDUCE(dt_loc, dt_all, 1, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, info)
        call MPI_ALLREDUCE(tva_nclip, nclip_all, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, info)
        call MPI_ALLREDUCE(tva_wmax_seen, wmax_all, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, info)
#endif
        ! Say so when the drift is what is setting the timestep. t_s ~ 1/rho_gas, so a
        ! single low-density cell can throttle every rank; without this the collapse is
        ! invisible. Once per coarse step, rank 1 only.
        if (myid == 1 .and. dt_all < dtnew(ilevel) .and. nstep_coarse /= tva_last_warn_step) then
            tva_last_warn_step = nstep_coarse
            write(*,'(A,I3,A,ES10.3,A,ES10.3)') &
                ' TVA: dust drift sets dt at level ', ilevel, ': ', dt_all, ' vs hydro ', dtnew(ilevel)
            if (nclip_all > 0) write(*,'(A,I12,A,ES10.3)') &
                '      drift clipped at tva_wmax_cs*c_s in ', nclip_all, &
                ' faces; max |w|/c_s = ', wmax_all
        end if
        tva_nclip = 0
        tva_wmax_seen = 0d0
        dtnew(ilevel) = min(dtnew(ilevel), dt_all)

    end subroutine get_dust_courant_dt

    subroutine dust_upwind_correct1(ind_grid,ncache,ilevel)
        use amr_commons
        use hydro_commons
        implicit none
        integer::ilevel,ncache
        integer,dimension(1:nvector)::ind_grid

        ! Cache blocks matching the sizes found in godfine1
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:nvar_all),save::uloc
        logical ,dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2),save::ok
        real(dp),dimension(1:nvector,if1:if2,jf1:jf2,kf1:kf2,1:ndust,1:ndim),save::dflux
        real(dp),dimension(1:nvector,if1:if2,jf1:jf2,kf1:kf2,1:ndim),save::eflux
        
        integer,dimension(1:nvector),save::ind_cell, ind_father, igrid_nbor, ind_exist, ind_nexist, ind_buffer
        integer,dimension(1:nvector,1:threetondim)::nbors_father_cells
        integer,dimension(1:nvector,0:twondim)::ibuffer_father
        real(dp),dimension(1:nvector,0:twondim,1:nvar_all)::u1
        real(dp),dimension(1:nvector,1:twotondim,1:nvar_all)::u2

        integer::i,j,ivar,idim,iskip,ind_son,nb_noneigh
        integer::i0,j0,k0,i1,j1,k1,i2,j2,k2,i3,j3,k3,nexist,nbuffer,ind_father_idx,i3max_loop,j3max_loop,k3max_loop
        integer::i1min,i1max,j1min,j1max,k1min,k1max
        integer::i2min,i2max,j2min,j2max,k2min,k2max
        integer::i3min,i3max,j3min,j3max,k3min,k3max
        real(dp)::dx,scale,oneontwotondim,dt
        real(dp)::scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2
        real(dp),dimension(1:ndust)::agrain_code,sgrain_code

        ! Get the current code units
        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)

        ! Get grain radii and material density in code units
        do i = 1, ndust
            agrain_code(i) = dustbins_props(i)%asize_cm / scale_l
            sgrain_code(i) = dustbins_props(i)%sgrain / scale_d
        end do

        oneontwotondim = 1d0/dble(twotondim)
        scale=boxlen/dble(icoarse_max-icoarse_min+1)
        dx=0.5d0**ilevel*scale
        dt=dtnew(ilevel)

        ! Integer constants
        i1min=0; i1max=0; i2min=0; i2max=0; i3min=1; i3max=1
        j1min=0; j1max=0; j2min=0; j2max=0; j3min=1; j3max=1
        k1min=0; k1max=0; k2min=0; k2max=0; k3min=1; k3max=1
        if(ndim>0)then
            i1max=2; i2max=1; i3max=2
        end if
        if(ndim>1)then
            j1max=2; j2max=1; j3max=2
        end if
        if(ndim>2)then
            k1max=2; k2max=1; k3max=2
        end if

        ! Gather 3^ndim neighboring father cells
        do i=1,ncache
            ind_cell(i)=father(ind_grid(i))
        end do
        call get3cubefather(ind_cell,nbors_father_cells,ncache,ilevel)

        ! Loop over neighboring grid tree to construct localized uloc stencil block
        do k1=k1min,k1max; do j1=j1min,j1max; do i1=i1min,i1max
            ! Check if neighboring grid exists
            nbuffer=0; nexist=0
            ind_father_idx=1+i1+3*j1+9*k1
            do i=1,ncache
                igrid_nbor(i)=son(nbors_father_cells(i,ind_father_idx))
                if(igrid_nbor(i)>0) then
                    nexist=nexist+1
                    ind_exist(nexist)=i
                else
                    nbuffer=nbuffer+1
                    ind_nexist(nbuffer)=i
                    ind_buffer(nbuffer)=nbors_father_cells(i,ind_father_idx)
                end if
            end do

            ! If not, interpolate hydro variables from parent cells
            if(nbuffer>0) then
                call getnborfather(ind_buffer,ibuffer_father,nbuffer,ilevel)
                do j=0,twondim; do ivar=1,nvar_all; do i=1,nbuffer
                    u1(i,j,ivar)=uold(ibuffer_father(i,j),ivar)
                end do; end do; end do
                call interpol_hydro(u1,u2,nbuffer)
            endif

            do k2=k2min,k2max; do j2=j2min,j2max; do i2=i2min,i2max
                ind_son=1+i2+2*j2+4*k2
                iskip=ncoarse+(ind_son-1)*ngridmax
                do i=1,nexist
                ind_cell(i)=iskip+igrid_nbor(ind_exist(i))
                end do
                i3=1; j3=1; k3=1
                if(ndim>0)i3=1+2*(i1-1)+i2
                if(ndim>1)j3=1+2*(j1-1)+j2
                if(ndim>2)k3=1+2*(k1-1)+k2

                do ivar=1,nvar_all
                    do i=1,nexist;  uloc(ind_exist(i),i3,j3,k3,ivar)=uold(ind_cell(i),ivar); end do
                    do i=1,nbuffer; uloc(ind_nexist(i),i3,j3,k3,ivar)=u2(i,ind_son,ivar); end do
                end do
                do i=1,nexist;  ok(ind_exist(i),i3,j3,k3)=son(ind_cell(i))>0; end do
                do i=1,nbuffer; ok(ind_nexist(i),i3,j3,k3)=.false.; end do
            end do; end do; end do
        end do; end do; end do

        ! Call the actual mathematical worker to get our upwinded mass corrections
        call calculate_pure_drag_fluxes(uloc,dflux,eflux,dx,dt,ncache,agrain_code,sgrain_code)

        ! Synchronize at refinement boundaries: if a finer cell exists next to this face,
        ! zero out the flux; the finer level handles it and restricts it down later
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1
            
            i3max_loop = 1
            j3max_loop = 1
            k3max_loop = 1
            if(ndim>0) i3max_loop = 2+i0
            if(ndim>1) j3max_loop = 2+j0
            if(ndim>2) k3max_loop = 2+k0
            
            do k3=1,k3max_loop; do j3=1,j3max_loop; do i3=1,i3max_loop
                do i=1,ncache
                    if(ok(i,i3-i0,j3-j0,k3-k0) .or. ok(i,i3,j3,k3))then
                        dflux(i,i3,j3,k3,:,idim)=0.0d0
                        eflux(i,i3,j3,k3,idim)=0.0d0
                    end if
                end do
            end do; end do; end do
        end do

        ! Apply the divergence of the fluxes to update unew array
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1
            do k2=0,k2max; do j2=0,j2max; do i2=0,i2max
                ind_son=1+i2+2*j2+4*k2
                iskip=ncoarse+(ind_son-1)*ngridmax
                do i=1,ncache
                    ind_cell(i)=iskip+ind_grid(i)
                end do
                i3=1+i2; j3=1+j2; k3=1+k2
                
                do ivar=1,ndust
                    do i=1,ncache
                        unew(ind_cell(i),idust+ivar-1)=unew(ind_cell(i),idust+ivar-1)+ &
                            (dflux(i,i3,j3,k3,ivar,idim) - dflux(i,i3+i0,j3+j0,k3+k0,ivar,idim))
                    end do
                end do

                do i=1,ncache
                    unew(ind_cell(i),neul)=unew(ind_cell(i),neul)+ &
                        (eflux(i,i3,j3,k3,idim) - eflux(i,i3+i0,j3+j0,k3+k0,idim))
                end do
            end do; end do; end do
        end do

        ! Update neighboring coarser cells (flux correction at coarse-fine boundaries)
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1

            ! Left boundary: check if neighbor coarser cell exists
            nb_noneigh=0
            do i=1,ncache
                if (son(nbor(ind_grid(i),2*idim-1))==0) then
                    nb_noneigh = nb_noneigh + 1
                    ind_buffer(nb_noneigh) = nbor(ind_grid(i),2*idim-1)
                    ind_cell(nb_noneigh) = i
                end if
            end do
            ! Update conservative variables
            do ivar=1,ndust
                do k3=k3min,k3max-k0
                do j3=j3min,j3max-j0
                do i3=i3min,i3max-i0
                    do i=1,nb_noneigh
                        unew(ind_buffer(i),idust+ivar-1)=unew(ind_buffer(i),idust+ivar-1) &
                            - dflux(ind_cell(i),i3,j3,k3,ivar,idim)*oneontwotondim
                    end do
                end do; end do; end do
            end do
            do k3=k3min,k3max-k0
            do j3=j3min,j3max-j0
            do i3=i3min,i3max-i0
                do i=1,nb_noneigh
                    unew(ind_buffer(i),neul)=unew(ind_buffer(i),neul) &
                        - eflux(ind_cell(i),i3,j3,k3,idim)*oneontwotondim
                end do
            end do; end do; end do

            ! Right boundary: check if neighbor coarser cell exists
            nb_noneigh=0
            do i=1,ncache
                if (son(nbor(ind_grid(i),2*idim))==0) then
                    nb_noneigh = nb_noneigh + 1
                    ind_buffer(nb_noneigh) = nbor(ind_grid(i),2*idim)
                    ind_cell(nb_noneigh) = i
                end if
            end do
            ! Update conservative variables
            do ivar=1,ndust
                do k3=k3min+k0,k3max
                do j3=j3min+j0,j3max
                do i3=i3min+i0,i3max
                    do i=1,nb_noneigh
                        unew(ind_buffer(i),idust+ivar-1)=unew(ind_buffer(i),idust+ivar-1) &
                            + dflux(ind_cell(i),i3+i0,j3+j0,k3+k0,ivar,idim)*oneontwotondim
                    end do
                end do; end do; end do
            end do
            do k3=k3min+k0,k3max
            do j3=j3min+j0,j3max
            do i3=i3min+i0,i3max
                do i=1,nb_noneigh
                    unew(ind_buffer(i),neul)=unew(ind_buffer(i),neul) &
                        + eflux(ind_cell(i),i3+i0,j3+j0,k3+k0,idim)*oneontwotondim
                end do
            end do; end do; end do
        end do

    end subroutine dust_upwind_correct1

    subroutine calculate_pure_drag_fluxes(uloc, dflux, eflux, dx, dt, ngrid, &
                                        & agrain_code, sgrain_code)
        use amr_parameters
        use hydro_parameters
        use const
        implicit none

        ! ========================================================================
        ! 1. GLOBAL INPUT
        ! ========================================================================
        integer, intent(in) :: ngrid
        real(dp), intent(in) :: dx, dt

        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:nvar), intent(in)  :: uloc
        real(dp), dimension(1:nvector, if1:if2, jf1:jf2, kf1:kf2, 1:ndust, 1:ndim), intent(out) :: dflux
        real(dp), dimension(1:nvector, if1:if2, jf1:jf2, kf1:kf2, 1:ndim), intent(out) :: eflux
        
        real(dp), dimension(1:ndust), intent(in) :: agrain_code, sgrain_code

        ! ========================================================================
        ! 2. LOCAL WORKSPACE FIELDS
        ! ========================================================================
        integer  :: l, i, j, k, jbin, idim
        integer  :: ilo, ihi, jlo, jhi, klo, khi
        integer  :: ilo_f, ihi_f, jlo_f, jhi_f, klo_f, khi_f
        
        ! Cell-Centered Base Primitives
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: Pg, rho_mix, c_s, eint_cell, eps_tot
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: eps
        
        ! Arrays strictly matching Lebreuilly 2019 formulation
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: rhod_cell, w_d_cell
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: slope_rhod, slope_wd, rhod_pred
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: w_g_cell, slope_wg, wg_pred
        
        real(dp) :: eken, rho_gas_cell, grad_P_cell, erad_cell
        integer  :: irad
        real(dp) :: dlft, drgt, dcen, theta
        real(dp) :: rhod_state_L, rhod_state_R, w_state_L, w_state_R, w_face
        real(dp) :: wg_state_L, wg_state_R, wg_face, Pg_upwind, H_gdnv
        
        real(dp), dimension(1:ndust) :: t_s_intrinsic
        real(dp) :: avg_ts_cell
        real(dp) :: w_cap_cell

        dflux = 0.0_dp
        eflux = 0.0_dp

        ilo = MIN(1, iu1+1); ihi = MAX(1, iu2-1)
        jlo = MIN(1, ju1+1); jhi = MAX(1, ju2-1)
        klo = MIN(1, ku1+1); khi = MAX(1, ku2-1)
        ! Transverse ranges for the FACE loop, clamped to the flux arrays.
        ! dflux/eflux/mflux are dimensioned (if1:if2, jf1:jf2, kf1:kf2) = 1:3,
        ! but ilo/jlo/klo above are MIN(1,iu1+1) = 0 in any active dimension
        ! (iu1 = ju1 = -1), so the transverse index started at 0 and every 2D or
        ! 3D run died with "Index '0' ... below lower bound of 1". RAMSES's own
        ! umuscl uses MIN(1,iu1+2) for exactly this loop. Only indices 1..3 are
        ! ever read back (i3 <= 2 and i3+i0 <= 3 in the update loops), so the
        ! clamp discards nothing. In 1D ju1=ju2=ku1=ku2=1, so jlo=jhi=klo=khi=1
        ! already and this is a no-op.
        ilo_f = MAX(ilo, if1); ihi_f = MIN(ihi, if2)
        jlo_f = MAX(jlo, jf1); jhi_f = MIN(jhi, jf2)
        klo_f = MAX(klo, kf1); khi_f = MIN(khi, kf2)

        ! ========================================================================
        ! STEP 1: BASE PRIMITIVE EXTRACTION
        ! ========================================================================
        do k = ku1, ku2; do j = ju1, ju2; do i = iu1, iu2; do l = 1, ngrid
            if (tva_test_mode == TVA_TEST_DIFFUSE) then
                rho_mix(l,i,j,k) = 1.0_dp
            else
                rho_mix(l,i,j,k) = max(uloc(l,i,j,k,1), smallr) 
            end if 
            
            eps_tot(l,i,j,k) = 0.0_dp
            do jbin = 1, ndust
                eps(l,i,j,k,jbin) = uloc(l,i,j,k,idust+jbin-1) / rho_mix(l,i,j,k) 
                eps_tot(l,i,j,k) = eps_tot(l,i,j,k) + eps(l,i,j,k,jbin)
            end do
            eps_tot(l,i,j,k) = MIN(MAX(eps_tot(l,i,j,k), zero), 1.0_dp - smallr)
            
            rho_gas_cell = rho_mix(l,i,j,k) * (one - eps_tot(l,i,j,k))
            
            eken = 0.0_dp
            do idim = 1, ndim
                eken = eken + uloc(l,i,j,k,idim+1)**2
            end do
            eken = half * eken / rho_mix(l,i,j,k)**2
            
            ! Non-thermal (NENER) energy must come out before the thermal
            ! pressure, exactly as ctoprim does (hydro/umuscl.f90). With
            ! rt_isIRtrap the trapped IR lives in uold(:,inener) and is inside
            ! uold(:,neul), so without this it would be counted as gas thermal
            ! pressure and double-count against the trapped-IR drift term.
            erad_cell = 0.0_dp
#if NENER>0
            do irad = 1, nener
                erad_cell = erad_cell + uloc(l,i,j,k,nhydro+irad) / rho_mix(l,i,j,k)
            end do
#endif
            eint_cell(l,i,j,k) = max((uloc(l,i,j,k,neul) / rho_mix(l,i,j,k)) - eken - erad_cell, smallc**2/gamma/(gamma-one))
            
            if (tva_test_mode == TVA_TEST_DIFFUSE) then
                Pg(l,i,j,k) = 1.0_dp**2 * (one - eps_tot(l,i,j,k)) * rho_mix(l,i,j,k)
                c_s(l,i,j,k) = 1.0_dp
            else
                Pg(l,i,j,k) = max((gamma - 1.0_dp) * rho_gas_cell * eint_cell(l,i,j,k), smallr*smallc**2)
                c_s(l,i,j,k) = sqrt(gamma * Pg(l,i,j,k) / rho_gas_cell)
            end if
        end do; end do; end do; end do


        ! ========================================================================
        ! MAIN DIRECTION SWEEP LOOP
        ! ========================================================================
        do idim = 1, ndim
            ! ====================================================================
            ! STEP 2: CELL-CENTERED KINEMATICS (Lebreuilly 2019)
            ! ====================================================================
            do k = ku1, ku2; do j = ju1, ju2; do i = iu1, iu2; do l = 1, ngrid
                ! 2a. Cell-Centered Pressure Gradients (using one-sided differences at stencil boundaries)
                if (idim == 1) then
                    if (i == iu1) then
                        grad_P_cell = (Pg(l,i+1,j,k) - Pg(l,i,j,k)) / dx
                    else if (i == iu2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i-1,j,k)) / dx
                    else
                        grad_P_cell = (Pg(l,i+1,j,k) - Pg(l,i-1,j,k)) / (2.0_dp * dx)
                    end if
                else if (idim == 2) then
                    if (j == ju1) then
                        grad_P_cell = (Pg(l,i,j+1,k) - Pg(l,i,j,k)) / dx
                    else if (j == ju2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i,j-1,k)) / dx
                    else
                        grad_P_cell = (Pg(l,i,j+1,k) - Pg(l,i,j-1,k)) / (2.0_dp * dx)
                    end if
                else
                    if (k == ku1) then
                        grad_P_cell = (Pg(l,i,j,k+1) - Pg(l,i,j,k)) / dx
                    else if (k == ku2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i,j,k-1)) / dx
                    else
                        grad_P_cell = (Pg(l,i,j,k+1) - Pg(l,i,j,k-1)) / (2.0_dp * dx)
                    end if
                end if

                ! 2b. Cell-Centered Intrinsic Stopping Times
                avg_ts_cell = 0.0_dp
                do jbin = 1, ndust
                    rhod_cell(l,i,j,k,jbin) = rho_mix(l,i,j,k) * eps(l,i,j,k,jbin)

                    if (tva_test_mode == TVA_TEST_DIFFUSE) then
                        t_s_intrinsic(jbin) = 0.1_dp
                    else if (tva_test_mode == TVA_TEST_SHOCK) then
                        t_s_intrinsic(jbin) = (eps(l,i,j,k,jbin) * rho_mix(l,i,j,k)) / drag_coefficient(jbin)
                    else if (tva_test_mode == TVA_TEST_BLAST1D) then
                        t_s_intrinsic(jbin) = 6d-3
                    else
                        t_s_intrinsic(jbin) = epstein_coef * (sgrain_code(jbin) * agrain_code(jbin)) / &
                                            max((one - eps_tot(l,i,j,k)) * rho_mix(l,i,j,k) * c_s(l,i,j,k), smallr) 
                    end if
                    avg_ts_cell = avg_ts_cell + eps(l,i,j,k,jbin) * t_s_intrinsic(jbin)
                end do

                ! 2c. Cell-Centered Drift Velocities
                if (use_w_drift_test) then
                    do jbin = 1, ndust
                        w_d_cell(l,i,j,k,jbin) = w_drift_test(idim)
                    end do
                    w_g_cell(l,i,j,k) = - (eps_tot(l,i,j,k) / max(one - eps_tot(l,i,j,k), smallr)) * w_drift_test(idim)
                else
                    ! Barycentric frame: sum_i rho_i w_i + rho_g w_g = 0 with
                    ! w_d,i = (t_s,i - avg_ts)*D and D = grad_P/rho_mix gives w_g = -avg_ts*D.
                    ! There is no 1/(1-eps_tot) factor here -- this must agree with the
                    ! radiation-pressure solver and with get_dust_courant_dt, which both
                    ! divide by rho_mix alone.
                    w_g_cell(l,i,j,k) = -avg_ts_cell * grad_P_cell / max(rho_mix(l,i,j,k), smallr)
                    do jbin = 1, ndust
                        w_d_cell(l,i,j,k,jbin) = (t_s_intrinsic(jbin) - avg_ts_cell) * grad_P_cell / rho_mix(l,i,j,k)
                    end do
                    ! Same cap as get_dust_courant_dt: TVA is only valid for Stokes << 1.
                    if (tva_wmax_cs > 0.0_dp) then
                        w_cap_cell = tva_wmax_cs * c_s(l,i,j,k)
                        if (abs(w_g_cell(l,i,j,k)) > w_cap_cell) &
                            w_g_cell(l,i,j,k) = sign(w_cap_cell, w_g_cell(l,i,j,k))
                        do jbin = 1, ndust
                            if (abs(w_d_cell(l,i,j,k,jbin)) > w_cap_cell) &
                                w_d_cell(l,i,j,k,jbin) = sign(w_cap_cell, w_d_cell(l,i,j,k,jbin))
                        end do
                    end if
                end if
            end do; end do; end do; end do

            ! ====================================================================
            ! STEP 3: TVD SPATIAL SLOPES 
            ! ====================================================================
            slope_rhod = 0.0_dp; slope_wd = 0.0_dp; slope_wg = 0.0_dp
            
            if (slope_type > 0) then
                if (slope_type == 2) then
                    theta = 2.0_dp
                else
                    theta = 1.0_dp
                end if
                
                do k = klo, khi; do j = jlo, jhi; do i = ilo, ihi; do l = 1, ngrid
                    ! --- Gas Drift Slope ---
                    if (slope_type == 6) then
                        slope_wg(l,i,j,k) = zero
                    else
                        if (idim == 1) then
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i-1,j,k)
                            drgt = w_g_cell(l,i+1,j,k) - w_g_cell(l,i,j,k)
                        else if (idim == 2) then
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i,j-1,k)
                            drgt = w_g_cell(l,i,j+1,k) - w_g_cell(l,i,j,k)
                        else
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i,j,k-1)
                            drgt = w_g_cell(l,i,j,k+1) - w_g_cell(l,i,j,k)
                        end if
                        dcen = half * (dlft + drgt)
                        if (dlft * drgt <= zero) then; slope_wg(l,i,j,k) = zero
                        else; slope_wg(l,i,j,k) = sign(one, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                        end if
                    end if

                    ! --- Dust Density & Drift Slopes ---
                    do jbin = 1, ndust
                        ! Rho_d Slope
                        if (idim == 1) then
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i-1,j,k,jbin) 
                            drgt = rhod_cell(l,i+1,j,k,jbin) - rhod_cell(l,i,j,k,jbin) 
                        else if (idim == 2) then
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i,j-1,k,jbin)
                            drgt = rhod_cell(l,i,j+1,k,jbin) - rhod_cell(l,i,j,k,jbin)
                        else
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i,j,k-1,jbin)
                            drgt = rhod_cell(l,i,j,k+1,jbin) - rhod_cell(l,i,j,k,jbin)
                        end if
                        dcen = half * (dlft + drgt)
                        if (slope_type == 6) then
                            slope_rhod(l,i,j,k,jbin) = dcen
                        else
                            if (dlft * drgt <= 0.0_dp) then; slope_rhod(l,i,j,k,jbin) = 0.0_dp
                            else; slope_rhod(l,i,j,k,jbin) = sign(1.0_dp, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                            end if
                        end if

                        ! Velocity Slope (Delta w_sigma)
                        if (slope_type == 6) then
                            slope_wd(l,i,j,k,jbin) = zero
                        else
                            if (idim == 1) then
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i-1,j,k,jbin) 
                                drgt = w_d_cell(l,i+1,j,k,jbin) - w_d_cell(l,i,j,k,jbin) 
                            else if (idim == 2) then
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i,j-1,k,jbin)
                                drgt = w_d_cell(l,i,j+1,k,jbin) - w_d_cell(l,i,j,k,jbin)
                            else
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i,j,k-1,jbin)
                                drgt = w_d_cell(l,i,j,k+1,jbin) - w_d_cell(l,i,j,k,jbin)
                            end if
                            dcen = half * (dlft + drgt)
                            if (dlft * drgt <= 0.0_dp) then; slope_wd(l,i,j,k,jbin) = 0.0_dp
                            else; slope_wd(l,i,j,k,jbin) = sign(1.0_dp, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                            end if
                        end if
                    end do
                end do; end do; end do; end do
            end if


            ! ====================================================================
            ! STEP 4: POINT 1 - TEMPORAL PREDICTOR (INCLUDING COMPRESSIBILITY)
            ! ====================================================================
            do k = klo, khi; do j = jlo, jhi; do i = ilo, ihi; do l = 1, ngrid
                ! Predictor for Dust Density (w * grad_rho + rho * grad_w)
                do jbin = 1, ndust
                    rhod_pred(l,i,j,k,jbin) = rhod_cell(l,i,j,k,jbin) - 0.5_dp * (dt / dx) * &
                        (w_d_cell(l,i,j,k,jbin) * slope_rhod(l,i,j,k,jbin) + rhod_cell(l,i,j,k,jbin) * slope_wd(l,i,j,k,jbin))
                    rhod_pred(l,i,j,k,jbin) = MAX(rhod_pred(l,i,j,k,jbin), 0.0_dp)
                end do
            end do; end do; end do; end do

            ! ====================================================================
            ! STEP 5: POINTS 2, 3 & 4 - INTERFACE RECONSTRUCTION & UPWIND FLUX
            ! ====================================================================
            do k = ifind_klo(idim, klo_f, kf1), ifind_khi(idim, khi_f, kf2)
            do j = ifind_jlo(idim, jlo_f, jf1), ifind_jhi(idim, jhi_f, jf2)
            do i = ifind_ilo(idim, ilo_f, if1), ifind_ihi(idim, ihi_f, if2)
                do l = 1, ngrid
                    
                    ! --- A. GAS ENTHALPY FLUX ---
                    ! Reconstruct L and R states for the gas drift at the face
                    if (idim == 1) then
                        wg_state_L = w_g_cell(l,i-1,j,k) + half * slope_wg(l,i-1,j,k)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    else if (idim == 2) then
                        wg_state_L = w_g_cell(l,i,j-1,k) + half * slope_wg(l,i,j-1,k)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    else
                        wg_state_L = w_g_cell(l,i,j,k-1) + half * slope_wg(l,i,j,k-1)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    end if
                    
                    ! Point 3: Unique Averaged Face Velocity
                    wg_face = 0.5_dp * (wg_state_L + wg_state_R)
                    
                    ! Upwind Thermal Pressure based on face velocity
                    if (wg_face >= 0.0_dp) then
                        if (idim == 1) then; Pg_upwind = Pg(l,i-1,j,k)
                        else if (idim == 2) then; Pg_upwind = Pg(l,i,j-1,k)
                        else; Pg_upwind = Pg(l,i,j,k-1)
                        end if
                    else
                        Pg_upwind = Pg(l,i,j,k)
                    end if
                    
                    H_gdnv = (gamma / (gamma - 1.0_dp)) * Pg_upwind
                    eflux(l,i,j,k,idim) = wg_face * H_gdnv * (dt / dx)


                    ! --- B. DUST MASS FLUX ---
                    do jbin = 1, ndust
                        ! Point 2: Interpolate rho_d and w to the interfaces
                        if (idim == 1) then
                            rhod_state_L = rhod_pred(l,i-1,j,k,jbin) + half * slope_rhod(l,i-1,j,k,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i-1,j,k,jbin)  + half * slope_wd(l,i-1,j,k,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        else if (idim == 2) then
                            rhod_state_L = rhod_pred(l,i,j-1,k,jbin) + half * slope_rhod(l,i,j-1,k,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i,j-1,k,jbin)  + half * slope_wd(l,i,j-1,k,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        else
                            rhod_state_L = rhod_pred(l,i,j,k-1,jbin) + half * slope_rhod(l,i,j,k-1,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i,j,k-1,jbin)  + half * slope_wd(l,i,j,k-1,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        end if

                        ! Safety clamp
                        rhod_state_L = MAX(rhod_state_L, zero)
                        rhod_state_R = MAX(rhod_state_R, zero)

                        ! Point 3: Unique Averaged Face Velocity
                        w_face = half * (w_state_L + w_state_R)

                        ! Point 4: Upwind Method for Flux Estimation
                        if (w_face >= zero) then 
                            dflux(l,i,j,k,jbin,idim) = w_face * rhod_state_L * (dt / dx)
                        else
                            dflux(l,i,j,k,jbin,idim) = w_face * rhod_state_R * (dt / dx)
                        end if
                    end do

                end do
            end do; end do; end do

        end do
    end subroutine calculate_pure_drag_fluxes

    subroutine dust_diffusion_fine(ilevel)
        use amr_commons
        use hydro_commons
        implicit none
        integer::ilevel

        integer::i,igrid,ncache,ngrid
        integer,dimension(1:nvector),save::ind_grid

        if (numbtot(1,ilevel)==0) return
        if (verbose) write(*,222) ilevel

        ! Loop over active grids by vector sweeps
        ncache = active(ilevel)%ngrid
        do igrid=1,ncache,nvector
            ngrid=min(nvector,ncache-igrid+1)
            do i=1,ngrid
                ind_grid(i)=active(ilevel)%igrid(igrid+i-1)
            end do
            call dust_upwind_correct1(ind_grid,ngrid,ilevel)
        end do
222 format('   Entering dust diffusion for level ',i2)

    end subroutine dust_diffusion_fine
#ifdef RT
    
    subroutine dust_push_fine(ilevel)
        use amr_commons
        use hydro_commons
        implicit none
        integer::ilevel

        integer::i,igrid,ncache,ngrid
        integer,dimension(1:nvector),save::ind_grid

        if (numbtot(1,ilevel)==0) return
        if (verbose) write(*,222) ilevel
        call rp_new_sweep()

        ! Loop over active grids by vector sweeps
        ncache = active(ilevel)%ngrid
        do igrid=1,ncache,nvector
            ngrid=min(nvector,ncache-igrid+1)
            do i=1,ngrid
                ind_grid(i)=active(ilevel)%igrid(igrid+i-1)
            end do
            call dust_upwind_correct2(ind_grid,ngrid,ilevel)
        end do
222 format('   Entering dust push for level ',i2)

    end subroutine dust_push_fine

    subroutine dust_upwind_correct2(ind_grid,ncache,ilevel)
        use amr_commons
        use hydro_commons
        use rt_hydro_commons, only: rtuold, nrtvar
        use dust_radpressure_module, only: compute_gas_dust_radpressure_acc
        implicit none
        integer::ilevel,ncache
        integer,dimension(1:nvector)::ind_grid

        ! Cache blocks matching the sizes found in godfine1
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:nvar_all),save::uloc
        logical ,dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2),save::ok
        real(dp),dimension(1:nvector,if1:if2,jf1:jf2,kf1:kf2,1:ndust,1:ndim),save::dflux
        real(dp),dimension(1:nvector,if1:if2,jf1:jf2,kf1:kf2,1:ndim),save::eflux

        ! Allocate momentum flux and radiation acceleration buffers
        real(dp),dimension(1:nvector,if1:if2,jf1:jf2,kf1:kf2,1:ndim),save::mflux
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:ndim),save::a_rad_g
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:ndust,1:ndim),save::a_rad_d
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,max(1,npah),1:ndim),save::a_rad_pah
        ! Trapped-IR Rosseland opacity share per bin, chi_R,k/chi_R,tot
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:ndust),save::s_IRtrap
        ! n_H, T and per-bin Coulomb coefficient for drag_model='draine2011'
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:ndust+2),save::drag_state
        
        integer,dimension(1:nvector),save::ind_cell, ind_father, igrid_nbor, ind_exist, ind_nexist, ind_buffer
        integer,dimension(1:nvector,1:threetondim)::nbors_father_cells
        integer,dimension(1:nvector,0:twondim)::ibuffer_father
        real(dp),dimension(1:nvector,0:twondim,1:nvar_all)::u1
        real(dp),dimension(1:nvector,1:twotondim,1:nvar_all)::u2
        real(dp),dimension(1:nvector,iu1:iu2,ju1:ju2,ku1:ku2,1:nrtvar),save::rt_uloc
        real(dp),dimension(1:nvector,0:twondim,1:nrtvar)::rt_u1
        real(dp),dimension(1:nvector,1:twotondim,1:nrtvar)::rt_u2

        integer::i,j,ivar,idim,iskip,ind_son,nb_noneigh
        integer::i0,j0,k0,i1,j1,k1,i2,j2,k2,i3,j3,k3,nexist,nbuffer,ind_father_idx,i3max_loop,j3max_loop,k3max_loop
        integer::i1min,i1max,j1min,j1max,k1min,k1max
        integer::i2min,i2max,j2min,j2max,k2min,k2max
        integer::i3min,i3max,j3min,j3max,k3min,k3max
        real(dp)::dx,scale,oneontwotondim,dt
        real(dp)::scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2
        real(dp),dimension(1:ndust)::agrain_code,sgrain_code

        ! Get the current code units
        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)

        ! Get grain radii and material density in code units
        do i = 1, ndust
            agrain_code(i) = dustbins_props(i)%asize_cm / scale_l
            sgrain_code(i) = dustbins_props(i)%sgrain / scale_d
        end do

        oneontwotondim = 1d0/dble(twotondim)
        scale=boxlen/dble(icoarse_max-icoarse_min+1)
        dx=0.5d0**ilevel*scale
        dt=dtnew(ilevel)

        ! Integer constants
        i1min=0; i1max=0; i2min=0; i2max=0; i3min=1; i3max=1
        j1min=0; j1max=0; j2min=0; j2max=0; j3min=1; j3max=1
        k1min=0; k1max=0; k2min=0; k2max=0; k3min=1; k3max=1
        if(ndim>0)then
            i1max=2; i2max=1; i3max=2
        end if
        if(ndim>1)then
            j1max=2; j2max=1; j3max=2
        end if
        if(ndim>2)then
            k1max=2; k2max=1; k3max=2
        end if

        ! Gather 3^ndim neighboring father cells
        do i=1,ncache
            ind_cell(i)=father(ind_grid(i))
        end do
        call get3cubefather(ind_cell,nbors_father_cells,ncache,ilevel)

        ! Loop over neighboring grid tree to construct localized uloc stencil block
        do k1=k1min,k1max; do j1=j1min,j1max; do i1=i1min,i1max
            ! Check if neighboring grid exists
            nbuffer=0; nexist=0
            ind_father_idx=1+i1+3*j1+9*k1
            do i=1,ncache
                igrid_nbor(i)=son(nbors_father_cells(i,ind_father_idx))
                if(igrid_nbor(i)>0) then
                    nexist=nexist+1
                    ind_exist(nexist)=i
                else
                    nbuffer=nbuffer+1
                    ind_nexist(nbuffer)=i
                    ind_buffer(nbuffer)=nbors_father_cells(i,ind_father_idx)
                end if
            end do

            ! If not, interpolate hydro variables from parent cells
            if(nbuffer>0) then
                call getnborfather(ind_buffer,ibuffer_father,nbuffer,ilevel)
                do j=0,twondim; do ivar=1,nvar_all; do i=1,nbuffer
                    u1(i,j,ivar)=uold(ibuffer_father(i,j),ivar)
                end do; end do; end do
                call interpol_hydro(u1,u2,nbuffer)

                do j=0,twondim; do ivar=1,nrtvar; do i=1,nbuffer
                    rt_u1(i,j,ivar)=rtuold(ibuffer_father(i,j),ivar)
                end do; end do; end do
                call rt_interpol_hydro(rt_u1,rt_u2,nbuffer)
            endif

            do k2=k2min,k2max; do j2=j2min,j2max; do i2=i2min,i2max
                ind_son=1+i2+2*j2+4*k2
                iskip=ncoarse+(ind_son-1)*ngridmax
                do i=1,nexist
                ind_cell(i)=iskip+igrid_nbor(ind_exist(i))
                end do
                i3=1; j3=1; k3=1
                if(ndim>0)i3=1+2*(i1-1)+i2
                if(ndim>1)j3=1+2*(j1-1)+j2
                if(ndim>2)k3=1+2*(k1-1)+k2

                do ivar=1,nvar_all
                    do i=1,nexist;  uloc(ind_exist(i),i3,j3,k3,ivar)=uold(ind_cell(i),ivar); end do
                    do i=1,nbuffer; uloc(ind_nexist(i),i3,j3,k3,ivar)=u2(i,ind_son,ivar); end do
                end do

                do ivar=1,nrtvar
                    do i=1,nexist;  rt_uloc(ind_exist(i),i3,j3,k3,ivar)=rtuold(ind_cell(i),ivar); end do
                    do i=1,nbuffer; rt_uloc(ind_nexist(i),i3,j3,k3,ivar)=rt_u2(i,ind_son,ivar); end do
                end do

                ! Compute the radiation pressures for each cell (level cells: once per sweep)
                do i=1,nexist
                    call radpressure_acc_once(ind_cell(i), uloc(ind_exist(i),i3,j3,k3,:),&
                                                            rt_uloc(ind_exist(i),i3,j3,k3,:),& 
                                                            ilevel,dx,& 
                                                            a_rad_g(ind_exist(i),i3,j3,k3,:),&
                                                            a_rad_d(ind_exist(i),i3,j3,k3,:,:),&
                                                            a_rad_pah(ind_exist(i),i3,j3,k3,:,:),&
                                                            s_IRtrap(ind_exist(i),i3,j3,k3,:),&
                                                            drag_state(ind_exist(i),i3,j3,k3,:))
                end do

                do i=1,nbuffer
                    call compute_gas_dust_radpressure_acc(uloc(ind_nexist(i),i3,j3,k3,:),&
                                                            rt_uloc(ind_nexist(i),i3,j3,k3,:),& 
                                                            ilevel,dx,& 
                                                            a_rad_g(ind_nexist(i),i3,j3,k3,:),&
                                                            a_rad_d(ind_nexist(i),i3,j3,k3,:,:),&
                                                            a_rad_pah(ind_nexist(i),i3,j3,k3,:,:),&
                                                            s_IRtrap(ind_nexist(i),i3,j3,k3,:),&
                                                            drag_state(ind_nexist(i),i3,j3,k3,:))
                end do

                do i=1,nexist;  ok(ind_exist(i),i3,j3,k3)=son(ind_cell(i))>0; end do
                do i=1,nbuffer; ok(ind_nexist(i),i3,j3,k3)=.false.; end do
            end do; end do; end do
        end do; end do; end do

        ! Call the actual mathematical worker to get our upwinded mass corrections
        call calculate_drag_rad_fluxes(uloc,dflux,eflux,mflux,dx,dt,ncache,&
                                        a_rad_g,a_rad_d,s_IRtrap,drag_state,agrain_code,sgrain_code)

        ! Synchronize at refinement boundaries: if a finer cell exists next to this face,
        ! zero out the flux; the finer level handles it and restricts it down later
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1
            
            i3max_loop = 1
            j3max_loop = 1
            k3max_loop = 1
            if(ndim>0) i3max_loop = 2+i0
            if(ndim>1) j3max_loop = 2+j0
            if(ndim>2) k3max_loop = 2+k0
            
            do k3=1,k3max_loop; do j3=1,j3max_loop; do i3=1,i3max_loop
                do i=1,ncache
                    if(ok(i,i3-i0,j3-j0,k3-k0) .or. ok(i,i3,j3,k3))then
                        dflux(i,i3,j3,k3,:,idim)=0.0d0
                        eflux(i,i3,j3,k3,idim)=0.0d0
                        mflux(i,i3,j3,k3,idim)=0.0d0
                    end if
                end do
            end do; end do; end do
        end do

        ! Apply the divergence of the fluxes to update unew array
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1
            do k2=0,k2max; do j2=0,j2max; do i2=0,i2max
                ind_son=1+i2+2*j2+4*k2
                iskip=ncoarse+(ind_son-1)*ngridmax
                do i=1,ncache
                    ind_cell(i)=iskip+ind_grid(i)
                end do
                i3=1+i2; j3=1+j2; k3=1+k2
                
                ! 1. Dust Mass Update
                do ivar=1,ndust
                    do i=1,ncache
                        unew(ind_cell(i),idust+ivar-1)=unew(ind_cell(i),idust+ivar-1)+ &
                            (dflux(i,i3,j3,k3,ivar,idim) - dflux(i,i3+i0,j3+j0,k3+k0,ivar,idim))
                    end do
                end do

                do i=1,ncache
                    ! 2. Total Energy Update
                    unew(ind_cell(i),neul)=unew(ind_cell(i),neul)+ &
                        (eflux(i,i3,j3,k3,idim) - eflux(i,i3+i0,j3+j0,k3+k0,idim))
                    ! 3. Mixture Momentum Update
                    unew(ind_cell(i),idim+1)=unew(ind_cell(i),idim+1) + &
                        (mflux(i,i3,j3,k3,idim) - mflux(i,i3+i0,j3+j0,k3+k0,idim))
                end do
            end do; end do; end do
        end do

        ! Update neighboring coarser cells (flux correction at coarse-fine boundaries)
        do idim=1,ndim
            i0=0; j0=0; k0=0
            if(idim==1)i0=1
            if(idim==2)j0=1
            if(idim==3)k0=1

            ! Left boundary: check if neighbor coarser cell exists
            nb_noneigh=0
            do i=1,ncache
                if (son(nbor(ind_grid(i),2*idim-1))==0) then
                    nb_noneigh = nb_noneigh + 1
                    ind_buffer(nb_noneigh) = nbor(ind_grid(i),2*idim-1)
                    ind_cell(nb_noneigh) = i
                end if
            end do
            ! Update conservative variables
            do ivar=1,ndust
                do k3=k3min,k3max-k0
                do j3=j3min,j3max-j0
                do i3=i3min,i3max-i0
                    do i=1,nb_noneigh
                        unew(ind_buffer(i),idust+ivar-1)=unew(ind_buffer(i),idust+ivar-1) &
                            - dflux(ind_cell(i),i3,j3,k3,ivar,idim)*oneontwotondim
                    end do
                end do; end do; end do
            end do
            do k3=k3min,k3max-k0
            do j3=j3min,j3max-j0
            do i3=i3min,i3max-i0
                do i=1,nb_noneigh
                    unew(ind_buffer(i),neul)=unew(ind_buffer(i),neul) &
                        - eflux(ind_cell(i),i3,j3,k3,idim)*oneontwotondim
                    unew(ind_buffer(i),idim+1)=unew(ind_buffer(i),idim+1) &
                        - mflux(ind_cell(i),i3,j3,k3,idim)*oneontwotondim
                end do
            end do; end do; end do



            ! Right boundary: check if neighbor coarser cell exists
            nb_noneigh=0
            do i=1,ncache
                if (son(nbor(ind_grid(i),2*idim))==0) then
                    nb_noneigh = nb_noneigh + 1
                    ind_buffer(nb_noneigh) = nbor(ind_grid(i),2*idim)
                    ind_cell(nb_noneigh) = i
                end if
            end do
            ! Update conservative variables
            do ivar=1,ndust
                do k3=k3min+k0,k3max
                do j3=j3min+j0,j3max
                do i3=i3min+i0,i3max
                    do i=1,nb_noneigh
                        unew(ind_buffer(i),idust+ivar-1)=unew(ind_buffer(i),idust+ivar-1) &
                            + dflux(ind_cell(i),i3+i0,j3+j0,k3+k0,ivar,idim)*oneontwotondim
                    end do
                end do; end do; end do
            end do
            do k3=k3min+k0,k3max
            do j3=j3min+j0,j3max
            do i3=i3min+i0,i3max
                do i=1,nb_noneigh
                    unew(ind_buffer(i),neul)=unew(ind_buffer(i),neul) &
                        + eflux(ind_cell(i),i3+i0,j3+j0,k3+k0,idim)*oneontwotondim
                    unew(ind_buffer(i),idim+1)=unew(ind_buffer(i),idim+1) &
                        + mflux(ind_cell(i),i3+i0,j3+j0,k3+k0,idim)*oneontwotondim
                end do
            end do; end do; end do
        end do

    end subroutine dust_upwind_correct2

    subroutine calculate_drag_rad_fluxes(uloc, dflux, eflux, mflux, dx, dt, ngrid, &
                                        & a_rad_g, a_rad_d, s_IRtrap, drag_state, &
                                        & agrain_code, sgrain_code)
        use amr_parameters
        use hydro_parameters
        use const
        use amr_commons, only: nstep
        use rt_parameters, only: rt_isIR, rt_isIRtrap, iIRtrapVar
        implicit none

        ! ========================================================================
        ! 1. GLOBAL INPUT
        ! ========================================================================
        integer, intent(in) :: ngrid
        real(dp), intent(in) :: dx, dt

        ! Stencil array dimensions bound dynamically by the active refinement level blocks
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:nvar), intent(in)  :: uloc
        real(dp), dimension(1:nvector, if1:if2, jf1:jf2, kf1:kf2, 1:ndust, 1:ndim), intent(out) :: dflux
        real(dp), dimension(1:nvector, if1:if2, jf1:jf2, kf1:kf2, 1:ndim), intent(out) :: eflux
        real(dp), dimension(1:nvector, if1:if2, jf1:jf2, kf1:kf2, 1:ndim), intent(out) :: mflux
        ! Radiation acceleration arrays [code units]
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndim), intent(in) :: a_rad_g
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust, 1:ndim), intent(in) :: a_rad_d
        ! Trapped-IR Rosseland opacity share per bin, chi_R,k/chi_R,tot
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust), intent(in) :: s_IRtrap
        ! n_H, T and per-bin Coulomb coefficient for drag_model='draine2011'
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust+2), intent(in) :: drag_state
        ! Grain properties (sizes and material densities) [code units]
        real(dp), dimension(1:ndust), intent(in)   :: agrain_code, sgrain_code

        ! ========================================================================
        ! 2. LOCAL WORKSPACE FIELDS
        ! ========================================================================
        integer  :: l, i, j, k, jbin, idim
        integer  :: ilo, ihi, jlo, jhi, klo, khi
        integer  :: ilo_f, ihi_f, jlo_f, jhi_f, klo_f, khi_f
        
        ! Cell-centered primitive caches across the localized block
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: Pg, rho_mix, c_s, eint_cell, eps_tot
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: eps
        ! Trapped-IR radiation pressure, exactly as the Riemann solver sees it
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: Ptrap
        real(dp) :: grad_Ptrap_cell
        logical  :: do_irtrap
        
        ! Arrays strictly matching Lebreuilly 2019 formulation
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: rhod_cell, w_d_cell
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2, 1:ndust) :: slope_rhod, slope_wd, rhod_pred
        real(dp), dimension(1:nvector, iu1:iu2, ju1:ju2, ku1:ku2) :: w_g_cell, slope_wg
        
        real(dp) :: eken, rho_gas_cell, grad_P_cell, erad_cell
        integer  :: irad
        real(dp) :: dlft, drgt, dcen, theta
        real(dp) :: rhod_state_L, rhod_state_R, w_state_L, w_state_R, w_face
        real(dp) :: wg_state_L, wg_state_R, wg_face, Pg_upwind, H_gdnv, flux_mass_bin
        real(dp) :: scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2

        real(dp), dimension(1:ndust) :: t_s_intrinsic, D_bin, a_rad_d_cell
        real(dp) :: avg_ts_cell, a_rad_g_cell, a_rad_mix, sum_eps_ts_D
        real(dp) :: w_cap_cell

        ! Initialize output flux arrays
        dflux = 0.0_dp
        eflux = 0.0_dp
        mflux = 0.0_dp
        call units(scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)

        ! Establish safe bounds for cell-centered loops inside the stencil buffer limits
        ilo = MIN(1, iu1+1); ihi = MAX(1, iu2-1)
        jlo = MIN(1, ju1+1); jhi = MAX(1, ju2-1)
        klo = MIN(1, ku1+1); khi = MAX(1, ku2-1)
        ! Transverse ranges for the FACE loop, clamped to the flux arrays.
        ! dflux/eflux/mflux are dimensioned (if1:if2, jf1:jf2, kf1:kf2) = 1:3,
        ! but ilo/jlo/klo above are MIN(1,iu1+1) = 0 in any active dimension
        ! (iu1 = ju1 = -1), so the transverse index started at 0 and every 2D or
        ! 3D run died with "Index '0' ... below lower bound of 1". RAMSES's own
        ! umuscl uses MIN(1,iu1+2) for exactly this loop. Only indices 1..3 are
        ! ever read back (i3 <= 2 and i3+i0 <= 3 in the update loops), so the
        ! clamp discards nothing. In 1D ju1=ju2=ku1=ku2=1, so jlo=jhi=klo=khi=1
        ! already and this is a no-op.
        ilo_f = MAX(ilo, if1); ihi_f = MIN(ihi, if2)
        jlo_f = MAX(jlo, jf1); jhi_f = MIN(jhi, jf2)
        klo_f = MAX(klo, kf1); khi_f = MIN(khi, kf2)

        ! ========================================================================
        ! STEP 1: PRIMITIVE VARIABLE & GAS DENSITY EXTRACTION (CELL-CENTERED)
        ! ========================================================================
        do k = ku1, ku2; do j = ju1, ju2; do i = iu1, iu2; do l = 1, ngrid
            if (tva_test_mode == TVA_TEST_DIFFUSE) then
                rho_mix(l,i,j,k) = 1.0_dp
            else
                rho_mix(l,i,j,k) = max(uloc(l,i,j,k,1), smallr) 
            end if 
            
            eps_tot(l,i,j,k) = 0.0_dp
            do jbin = 1, ndust
                eps(l,i,j,k,jbin) = uloc(l,i,j,k,idust+jbin-1) / rho_mix(l,i,j,k) 
                eps_tot(l,i,j,k) = eps_tot(l,i,j,k) + eps(l,i,j,k,jbin)
            end do
            eps_tot(l,i,j,k) = MIN(MAX(eps_tot(l,i,j,k), zero), 1.0_dp - smallr)
            
            rho_gas_cell = rho_mix(l,i,j,k) * (one - eps_tot(l,i,j,k))
            
            eken = 0.0_dp
            do idim = 1, ndim
                eken = eken + uloc(l,i,j,k,idim+1)**2
            end do
            eken = half * eken / rho_mix(l,i,j,k)**2
            
            ! Non-thermal (NENER) energy must come out before the thermal
            ! pressure, exactly as ctoprim does (hydro/umuscl.f90). With
            ! rt_isIRtrap the trapped IR lives in uold(:,inener) and is inside
            ! uold(:,neul), so without this it would be counted as gas thermal
            ! pressure and double-count against the trapped-IR drift term.
            erad_cell = 0.0_dp
#if NENER>0
            do irad = 1, nener
                erad_cell = erad_cell + uloc(l,i,j,k,nhydro+irad) / rho_mix(l,i,j,k)
            end do
#endif
            eint_cell(l,i,j,k) = max((uloc(l,i,j,k,neul) / rho_mix(l,i,j,k)) - eken - erad_cell, smallc**2/gamma/(gamma-one))
            
            if (tva_test_mode == TVA_TEST_DIFFUSE) then
                Pg(l,i,j,k) = 1.0_dp**2 * (one - eps_tot(l,i,j,k)) * rho_mix(l,i,j,k)
                c_s(l,i,j,k) = 1.0_dp
            else
                Pg(l,i,j,k) = max((gamma - 1.0_dp) * rho_gas_cell * eint_cell(l,i,j,k), smallr*smallc**2)
                c_s(l,i,j,k) = sqrt(gamma * Pg(l,i,j,k) / rho_gas_cell)
            end if
        end do; end do; end do; end do

        ! ========================================================================
        ! STEP 1b: TRAPPED-IR RADIATION PRESSURE (CELL-CENTERED)
        ! ------------------------------------------------------------------------
        ! In optically thick cells the IR is stored in a NENER slot and behaves
        ! as a fluid pressure, so the Godunov solver already applies
        ! -grad(P_trap)/rho_mix to the mixture barycenter. Physically the force
        ! acts on the dust, which carries the IR opacity, so TVA needs the
        ! differential part. Use exactly the same expression the Riemann solver
        ! uses, (gamma_rad-1)*E_trap, so the barycentric term cancels exactly
        ! whatever the rt_c_fraction normalisation turns out to be.
        ! ========================================================================
        do_irtrap = .false.
#if NENER>0
        do_irtrap = rt_isIR .and. rt_isIRtrap .and. ndust > 0
#endif
        Ptrap = 0.0_dp
#if NENER>0
        if (do_irtrap) then
            do k = ku1, ku2; do j = ju1, ju2; do i = iu1, iu2; do l = 1, ngrid
                Ptrap(l,i,j,k) = (gamma_rad(iIRtrapVar-nhydro) - 1.0_dp) &
                               * max(uloc(l,i,j,k,iIRtrapVar), 0.0_dp)
            end do; end do; end do; end do
        end if
#endif

        ! ========================================================================
        ! MAIN DIRECTION SWEEP LOOP (idim = 1: X-sweep, 2: Y-sweep, 3: Z-sweep)
        ! ========================================================================
        do idim = 1, ndim

            ! ====================================================================
            ! STEP 2: CELL-CENTERED KINEMATICS (Lebreuilly 2019 + Radiation Pressure)
            ! ====================================================================
            do k = ku1, ku2; do j = ju1, ju2; do i = iu1, iu2; do l = 1, ngrid
                ! 2a. Cell-Centered Pressure Gradients
                if (idim == 1) then
                    if (i == iu1) then
                        grad_P_cell = (Pg(l,i+1,j,k) - Pg(l,i,j,k)) / dx
                    else if (i == iu2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i-1,j,k)) / dx
                    else
                        grad_P_cell = (Pg(l,i+1,j,k) - Pg(l,i-1,j,k)) / (2.0_dp * dx)
                    end if
                else if (idim == 2) then
                    if (j == ju1) then
                        grad_P_cell = (Pg(l,i,j+1,k) - Pg(l,i,j,k)) / dx
                    else if (j == ju2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i,j-1,k)) / dx
                    else
                        grad_P_cell = (Pg(l,i,j+1,k) - Pg(l,i,j-1,k)) / (2.0_dp * dx)
                    end if
                else
                    if (k == ku1) then
                        grad_P_cell = (Pg(l,i,j,k+1) - Pg(l,i,j,k)) / dx
                    else if (k == ku2) then
                        grad_P_cell = (Pg(l,i,j,k) - Pg(l,i,j,k-1)) / dx
                    else
                        grad_P_cell = (Pg(l,i,j,k+1) - Pg(l,i,j,k-1)) / (2.0_dp * dx)
                    end if
                end if

                ! 2a-bis. Same difference stencil for the trapped-IR pressure:
                ! central in the interior, one-sided on the outermost stencil
                ! planes where no ghost data exists.
                grad_Ptrap_cell = 0.0_dp
                if (do_irtrap) then
                    if (idim == 1) then
                        if (i == iu1) then
                            grad_Ptrap_cell = (Ptrap(l,i+1,j,k) - Ptrap(l,i,j,k)) / dx
                        else if (i == iu2) then
                            grad_Ptrap_cell = (Ptrap(l,i,j,k) - Ptrap(l,i-1,j,k)) / dx
                        else
                            grad_Ptrap_cell = (Ptrap(l,i+1,j,k) - Ptrap(l,i-1,j,k)) / (2.0_dp * dx)
                        end if
                    else if (idim == 2) then
                        if (j == ju1) then
                            grad_Ptrap_cell = (Ptrap(l,i,j+1,k) - Ptrap(l,i,j,k)) / dx
                        else if (j == ju2) then
                            grad_Ptrap_cell = (Ptrap(l,i,j,k) - Ptrap(l,i,j-1,k)) / dx
                        else
                            grad_Ptrap_cell = (Ptrap(l,i,j+1,k) - Ptrap(l,i,j-1,k)) / (2.0_dp * dx)
                        end if
                    else
                        if (k == ku1) then
                            grad_Ptrap_cell = (Ptrap(l,i,j,k+1) - Ptrap(l,i,j,k)) / dx
                        else if (k == ku2) then
                            grad_Ptrap_cell = (Ptrap(l,i,j,k) - Ptrap(l,i,j,k-1)) / dx
                        else
                            grad_Ptrap_cell = (Ptrap(l,i,j,k+1) - Ptrap(l,i,j,k-1)) / (2.0_dp * dx)
                        end if
                    end if
                end if

                ! 2b. Cell-Centered Intrinsic Stopping Times
                avg_ts_cell = 0.0_dp
                do jbin = 1, ndust
                    rhod_cell(l,i,j,k,jbin) = rho_mix(l,i,j,k) * eps(l,i,j,k,jbin)

                    if (tva_test_mode == TVA_TEST_DIFFUSE) then
                        t_s_intrinsic(jbin) = 0.1_dp
                    else if (tva_test_mode == TVA_TEST_SHOCK) then
                        t_s_intrinsic(jbin) = (eps(l,i,j,k,jbin) * rho_mix(l,i,j,k)) / drag_coefficient(jbin)
                    else if (tva_test_mode == TVA_TEST_BLAST1D) then
                        t_s_intrinsic(jbin) = 6d-3
                    else
                        t_s_intrinsic(jbin) = epstein_coef * (sgrain_code(jbin) * agrain_code(jbin)) / &
                                            max((one - eps_tot(l,i,j,k)) * rho_mix(l,i,j,k) * c_s(l,i,j,k), smallr) 
                    end if
                    avg_ts_cell = avg_ts_cell + eps(l,i,j,k,jbin) * t_s_intrinsic(jbin)
                end do

                ! 2c. Cell-Centered Drift Velocities
                if (use_w_drift_test) then
                    do jbin = 1, ndust
                        w_d_cell(l,i,j,k,jbin) = w_drift_test(idim)
                    end do
                    w_g_cell(l,i,j,k) = - (eps_tot(l,i,j,k) / max(one - eps_tot(l,i,j,k), smallr)) * w_drift_test(idim)
                else
                    a_rad_g_cell = a_rad_g(l,i,j,k,idim)
                    a_rad_mix = (1.0_dp - eps_tot(l,i,j,k)) * a_rad_g_cell
                    do jbin = 1, ndust
                        a_rad_d_cell(jbin) = a_rad_d(l,i,j,k,jbin,idim)
                        a_rad_mix = a_rad_mix + eps(l,i,j,k,jbin) * a_rad_d_cell(jbin)
                    end do

                    sum_eps_ts_D = 0.0_dp
                    do jbin = 1, ndust
                        D_bin(jbin) = grad_P_cell / max(rho_mix(l,i,j,k), smallr) + a_rad_d_cell(jbin) - a_rad_mix
                        ! Trapped-IR differential acceleration:
                        !   a_k    = -(s_k/rho_k) grad(P_trap)   (force on bin k)
                        !   a_bary = -(1/rho_mix) grad(P_trap)   (already applied by godunov)
                        !   D_k   += a_k - a_bary
                        if (do_irtrap) then
                            D_bin(jbin) = D_bin(jbin) - grad_Ptrap_cell *            &
                                ( s_IRtrap(l,i,j,k,jbin)                             &
                                  / max(rhod_cell(l,i,j,k,jbin), smallr)             &
                                  - 1.0_dp / max(rho_mix(l,i,j,k), smallr) )
                        end if
                        ! Draine (2011) drag is nonlinear in the drift, so its
                        ! stopping time needs the driving acceleration D first.
                        if (draine_drag() .and. (tva_test_mode == TVA_TEST_NONE &
                            .or. tva_test_mode >= TVA_TEST_SPRESS)) then
                            t_s_intrinsic(jbin) = draine2011_stopping_time(dustbins_props(jbin)%asize_cm, &
                                dustbins_props(jbin)%sgrain, D_bin(jbin) * scale_v / scale_t,          &
                                drag_state(l,i,j,k,1), drag_state(l,i,j,k,2), drag_state(l,i,j,k,2+jbin)) / scale_t
                        end if
                        sum_eps_ts_D = sum_eps_ts_D + eps(l,i,j,k,jbin) * t_s_intrinsic(jbin) * D_bin(jbin)
                        ! dustyirtrap test diagnostic: report the trapped-IR
                        ! drift on the first step, while the state is still the
                        ! analytic one set by condinit, so plot-dustyirtrap.py
                        ! can check it against the closed-form prediction.
                        if (do_irtrap .and. tva_test_mode == TVA_TEST_IRTRAP .and. &
                            (i == 1 .or. i == 2) .and. j == 1 .and. k == 1 .and.   &
                            idim == 1 .and. nstep <= 1) then
                            write(*,'(A,I3,A,I2,6(A,ES14.6))') 'IRTRAP_DIAG step=',nstep, &
                              ' bin=',jbin,                                            &
                              ' grad_Ptrap=',grad_Ptrap_cell,                          &
                              ' share=',s_IRtrap(l,i,j,k,jbin),                        &
                              ' rho_d=',rhod_cell(l,i,j,k,jbin),                       &
                              ' D=',D_bin(jbin),                                       &
                              ' t_s=',t_s_intrinsic(jbin),                             &
                              ' ts_D=',t_s_intrinsic(jbin)*D_bin(jbin)
                        end if
                    end do

                    w_g_cell(l,i,j,k) = -sum_eps_ts_D
                    do jbin = 1, ndust
                        w_d_cell(l,i,j,k,jbin) = t_s_intrinsic(jbin) * D_bin(jbin) - sum_eps_ts_D
                    end do
                    ! Same cap as get_dust_courant_dt: TVA is only valid for Stokes << 1.
                    if (tva_wmax_cs > 0.0_dp) then
                        w_cap_cell = tva_wmax_cs * c_s(l,i,j,k)
                        if (abs(w_g_cell(l,i,j,k)) > w_cap_cell) &
                            w_g_cell(l,i,j,k) = sign(w_cap_cell, w_g_cell(l,i,j,k))
                        do jbin = 1, ndust
                            if (abs(w_d_cell(l,i,j,k,jbin)) > w_cap_cell) &
                                w_d_cell(l,i,j,k,jbin) = sign(w_cap_cell, w_d_cell(l,i,j,k,jbin))
                        end do
                    end if
                end if
            end do; end do; end do; end do

            ! ====================================================================
            ! STEP 3: TVD SPATIAL SLOPES 
            ! ====================================================================
            slope_rhod = 0.0_dp; slope_wd = 0.0_dp; slope_wg = 0.0_dp
            
            if (slope_type > 0) then
                if (slope_type == 2) then
                    theta = 2.0_dp
                else
                    theta = 1.0_dp
                end if
                
                do k = klo, khi; do j = jlo, jhi; do i = ilo, ihi; do l = 1, ngrid
                    ! --- Gas Drift Slope ---
                    if (slope_type == 6) then
                        slope_wg(l,i,j,k) = zero
                    else
                        if (idim == 1) then
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i-1,j,k)
                            drgt = w_g_cell(l,i+1,j,k) - w_g_cell(l,i,j,k)
                        else if (idim == 2) then
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i,j-1,k)
                            drgt = w_g_cell(l,i,j+1,k) - w_g_cell(l,i,j,k)
                        else
                            dlft = w_g_cell(l,i,j,k) - w_g_cell(l,i,j,k-1)
                            drgt = w_g_cell(l,i,j,k+1) - w_g_cell(l,i,j,k)
                        end if
                        dcen = half * (dlft + drgt)
                        if (dlft * drgt <= zero) then; slope_wg(l,i,j,k) = zero
                        else; slope_wg(l,i,j,k) = sign(one, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                        end if
                    end if

                    ! --- Dust Density & Drift Slopes ---
                    do jbin = 1, ndust
                        ! Rho_d Slope
                        if (idim == 1) then
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i-1,j,k,jbin) 
                            drgt = rhod_cell(l,i+1,j,k,jbin) - rhod_cell(l,i,j,k,jbin) 
                        else if (idim == 2) then
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i,j-1,k,jbin)
                            drgt = rhod_cell(l,i,j+1,k,jbin) - rhod_cell(l,i,j,k,jbin)
                        else
                            dlft = rhod_cell(l,i,j,k,jbin) - rhod_cell(l,i,j,k-1,jbin)
                            drgt = rhod_cell(l,i,j+1,k,jbin) - rhod_cell(l,i,j,k,jbin)
                        end if
                        dcen = half * (dlft + drgt)
                        if (slope_type == 6) then
                            slope_rhod(l,i,j,k,jbin) = dcen
                        else
                            if (dlft * drgt <= 0.0_dp) then; slope_rhod(l,i,j,k,jbin) = 0.0_dp
                            else; slope_rhod(l,i,j,k,jbin) = sign(1.0_dp, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                            end if
                        end if

                        ! Velocity Slope
                        if (slope_type == 6) then
                            slope_wd(l,i,j,k,jbin) = zero
                        else
                            if (idim == 1) then
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i-1,j,k,jbin) 
                                drgt = w_d_cell(l,i+1,j,k,jbin) - w_d_cell(l,i,j,k,jbin) 
                            else if (idim == 2) then
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i,j-1,k,jbin)
                                drgt = w_d_cell(l,i,j+1,k,jbin) - w_d_cell(l,i,j,k,jbin)
                            else
                                dlft = w_d_cell(l,i,j,k,jbin) - w_d_cell(l,i,j,k-1,jbin)
                                drgt = w_d_cell(l,i,j+1,k,jbin) - w_d_cell(l,i,j,k,jbin)
                            end if
                            dcen = half * (dlft + drgt)
                            if (dlft * drgt <= 0.0_dp) then; slope_wd(l,i,j,k,jbin) = 0.0_dp
                            else; slope_wd(l,i,j,k,jbin) = sign(1.0_dp, dcen) * min(theta * min(abs(dlft), abs(drgt)), abs(dcen))
                            end if
                        end if
                    end do
                end do; end do; end do; end do
            end if

            ! ====================================================================
            ! STEP 4: TEMPORAL PREDICTOR
            ! ====================================================================
            do k = klo, khi; do j = jlo, jhi; do i = ilo, ihi; do l = 1, ngrid
                do jbin = 1, ndust
                    rhod_pred(l,i,j,k,jbin) = rhod_cell(l,i,j,k,jbin) - 0.5_dp * (dt / dx) * &
                        (w_d_cell(l,i,j,k,jbin) * slope_rhod(l,i,j,k,jbin) + rhod_cell(l,i,j,k,jbin) * slope_wd(l,i,j,k,jbin))
                    rhod_pred(l,i,j,k,jbin) = MAX(rhod_pred(l,i,j,k,jbin), 0.0_dp)
                end do
            end do; end do; end do; end do

            ! ====================================================================
            ! STEP 5: INTERFACE RECONSTRUCTION & UPWIND FLUX
            ! ====================================================================
            do k = ifind_klo(idim, klo_f, kf1), ifind_khi(idim, khi_f, kf2)
            do j = ifind_jlo(idim, jlo_f, jf1), ifind_jhi(idim, jhi_f, jf2)
            do i = ifind_ilo(idim, ilo_f, if1), ifind_ihi(idim, ihi_f, if2)
                do l = 1, ngrid
                    
                    ! --- A. GAS ENTHALPY FLUX ---
                    if (idim == 1) then
                        wg_state_L = w_g_cell(l,i-1,j,k) + half * slope_wg(l,i-1,j,k)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    else if (idim == 2) then
                        wg_state_L = w_g_cell(l,i,j-1,k) + half * slope_wg(l,i,j-1,k)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    else
                        wg_state_L = w_g_cell(l,i,j,k-1) + half * slope_wg(l,i,j,k-1)
                        wg_state_R = w_g_cell(l,i,j,k)   - half * slope_wg(l,i,j,k)
                    end if
                    
                    wg_face = 0.5_dp * (wg_state_L + wg_state_R)
                    
                    if (wg_face >= 0.0_dp) then
                        if (idim == 1) then; Pg_upwind = Pg(l,i-1,j,k)
                        else if (idim == 2) then; Pg_upwind = Pg(l,i,j-1,k)
                        else; Pg_upwind = Pg(l,i,j,k-1)
                        end if
                    else
                        Pg_upwind = Pg(l,i,j,k)
                    end if
                    
                    H_gdnv = (gamma / (gamma - 1.0_dp)) * Pg_upwind
                    eflux(l,i,j,k,idim) = wg_face * H_gdnv * (dt / dx)

                    ! --- B. DUST MASS, MOMENTUM, AND DRIFT ENERGY FLUX ---
                    do jbin = 1, ndust
                        if (idim == 1) then
                            rhod_state_L = rhod_pred(l,i-1,j,k,jbin) + half * slope_rhod(l,i-1,j,k,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i-1,j,k,jbin)  + half * slope_wd(l,i-1,j,k,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        else if (idim == 2) then
                            rhod_state_L = rhod_pred(l,i,j-1,k,jbin) + half * slope_rhod(l,i,j-1,k,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i,j-1,k,jbin)  + half * slope_wd(l,i,j-1,k,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        else
                            rhod_state_L = rhod_pred(l,i,j,k-1,jbin) + half * slope_rhod(l,i,j,k-1,jbin)
                            rhod_state_R = rhod_pred(l,i,j,k,jbin)   - half * slope_rhod(l,i,j,k,jbin)
                            w_state_L    = w_d_cell(l,i,j,k-1,jbin)  + half * slope_wd(l,i,j,k-1,jbin)
                            w_state_R    = w_d_cell(l,i,j,k,jbin)    - half * slope_wd(l,i,j,k,jbin)
                        end if

                        rhod_state_L = MAX(rhod_state_L, zero)
                        rhod_state_R = MAX(rhod_state_R, zero)

                        w_face = half * (w_state_L + w_state_R)

                        if (w_face >= zero) then 
                            flux_mass_bin = w_face * rhod_state_L * (dt / dx)
                        else
                            flux_mass_bin = w_face * rhod_state_R * (dt / dx)
                        end if
                        dflux(l,i,j,k,jbin,idim) = flux_mass_bin

                        ! Momentum flux
                        mflux(l,i,j,k,idim) = mflux(l,i,j,k,idim) + flux_mass_bin * (w_face - wg_face)

                        ! Energy flux (drift kinetic energy only, enthalpy is wg_face * H_gdnv)
                        eflux(l,i,j,k,idim) = eflux(l,i,j,k,idim) + half * flux_mass_bin * (w_face**2 - wg_face**2)
                    end do

                end do
            end do; end do; end do
        end do
        
    end subroutine calculate_drag_rad_fluxes
#endif

end module dust_dynamics