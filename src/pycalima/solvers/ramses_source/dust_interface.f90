module dust_interface
    use amr_commons, only:dp,ndim
    use amr_parameters, only: nvector
    use dust_charging_rtgroups, only: RTGState, RTGResult, RTG_NRI, RTG_ALPHA_GAUSS, RTG_ALPHA_NONE, RTG_RECOMB_IMPORTANT, &
                                      RI_ATOMIC
    use dust_radiation, only: TdustLin
    use constants
    use dust_commons
    implicit none

    private

    public :: compute_dust_rad_rates,compute_dust_precool,&
            compute_dust_coolrates,compute_local_anisotropy_factor,&
            compute_dust_update,rtgroups_reset_warm,report_dust_charging_timer,&
            rtgroups_ir_emitted
    ! RTZ cell sub-steps (rtz_cool_step calls) and their wall-clock time, counted by
    ! rtz_cooling_module when dust_charging_timer: the charging cost per RTZ step
    real(dp), public, save :: trtz = 0d0
    integer(8), public, save :: nrtz = 0

    ! Cache to avoid automatic array allocations in compute_dust_update
    real(dp), allocatable, save, target :: y_gas_cache(:,:), y_gas_out_cache(:,:)
    real(dp), allocatable, save, target :: y_dust_cache(:), y_dust_out_cache(:)
    ! WDB06rt: warm state of every bin in every cell of the RTZ vector (sub-step to
    ! sub-step); the last full solve of each bin, whose d/d ln T serves the cooling
    ! solver's calls at T(1 +- 1e-5) with the other inputs unchanged; the last root
    ! of each bin, the starting guess of a cold solve
    type(RTGState), allocatable, target, save :: rtg_warm(:,:)     ! (nvector + 1, ndust); nvector + 1: no cell index
    type RTGLast
        logical :: valid = .false.
        real(dp) :: T = 0d0, ne = 0d0, n_Hp = 0d0, n_Hep = 0d0, n_Hepp = 0d0, G0 = 0d0, c_red = 0d0
        real(dp), allocatable :: Np(:), egy(:)
        type(RTGResult) :: r
    end type RTGLast
    type(RTGLast), allocatable, save :: rtg_last(:)
    real(dp), allocatable, save :: rtg_Zguess(:)
    real(dp), parameter :: RTG_DLNT = 1d-5, RTG_SERVE = 1.5d-5
    real(dp), parameter :: RTG_SERVE_HI = exp(RTG_SERVE), RTG_SERVE_LO = exp(-RTG_SERVE)   ! rtg_served, without a log
    ! grain charging timer (dust_charging_timer): wall-clock seconds; calls per bin by path
    ! (0: full solve, 1: uniform-G0 table, 2: from d/d ln T, 3: other charging models,
    ! 4: predicted from the cell's last full solve, rtgroups_predict)
    real(dp), save :: tchg = 0d0
    integer(8), save :: nchg(0:4) = 0, nchg_fit = 0, nchg_disc = 0
    ! the dust temperatures of the last update_T_dust and the inputs it had: the cooling solver's
    ! T(1 +- 1e-5) calls with the same other inputs are served from it (serve_T_dust)
    type(TdustLin), allocatable, save :: td_lin(:)
    logical, save :: td_valid = .false., td_hasNp = .false.
    real(dp), save :: td_T = 0d0, td_ne = 0d0, td_G0 = 0d0, td_nH2 = 0d0, td_nCO = 0d0
    real(dp), allocatable, save :: td_nel(:), td_xion(:,:), td_Np(:), td_egy(:)

contains
    subroutine rtgroups_reset_warm()
        ! A new vector of cells (rtz_solve_cooling): their warm states start cold.
        use dust_charging_rtgroups, only: rtgroups_dump_mark
        implicit none
        if (allocated(rtg_warm)) rtg_warm(:,:)%valid = .false.
        if (allocated(rtg_warm)) rtg_warm(:,:)%anchored = .false.
        if (allocated(rtg_last)) rtg_last(:)%valid = .false.
        td_valid = .false.
        if (dust_rtgroups_debug .or. dust_rtgroups_verify) call rtgroups_dump_mark()
    end subroutine rtgroups_reset_warm

    subroutine rtgroups_ir_emitted(ig, Tk, ne, Np_ig)
        ! WDB06rt with rt_isIR: rtz_cool_step adds the sub-step's dust IR emission to group ig
        ! after the precool solve at (Tk, ne) and before the cooling solver's T(1 +- 1e-5)
        ! calls, which only need dC/dT. The bins just solved take the new N_ig as theirs, so
        ! those calls are served from the precool solve's d/d ln T (rtg_served), as without
        ! rt_isIR, instead of two more full solves whose warm-started roots need not agree.
        implicit none
        integer, intent(in) :: ig
        real(dp), intent(in) :: Tk, ne, Np_ig
        integer :: ib
        if (.not. allocated(rtg_last)) return
        do ib = 1, size(rtg_last)
            if (rtg_last(ib)%valid .and. rtg_last(ib)%T == Tk .and. rtg_last(ib)%ne == ne) rtg_last(ib)%Np(ig) = Np_ig
        end do
    end subroutine rtgroups_ir_emitted

    subroutine report_dust_charging_timer()
        ! Wall-clock time of the grain charging and PE heating (dust_charging_timer) and of the
        ! RTZ cell sub-steps, per rank and in total; the charging cost per RTZ step.
        use amr_commons, only: myid, ncpu
        use mpi_mod
        implicit none
        real(dp) :: t, tr, tl(2)
        integer(8) :: n(0:7), nl(0:7)
        real(dp), allocatable :: ta(:,:)
        integer(8), allocatable :: na(:,:)
        integer :: k
#ifndef WITHOUTMPI
        integer :: info
#endif
        if (.not. dust_charging_timer) return
        nl(0:4) = nchg
        nl(5) = nchg_fit
        nl(6) = nchg_disc
        nl(7) = nrtz
        tl = (/ tchg, trtz /)
        allocate(ta(2, ncpu), na(0:7, ncpu))
        ta(:, 1) = tl
        na(:, 1) = nl
#ifndef WITHOUTMPI
        call MPI_GATHER(tl, 2, MPI_DOUBLE_PRECISION, ta, 2, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, info)
        call MPI_GATHER(nl, 8, MPI_INTEGER8, na, 8, MPI_INTEGER8, 0, MPI_COMM_WORLD, info)
#endif
        if (myid /= 1) return
        t = sum(ta(1, :))
        tr = sum(ta(2, :))
        n = sum(na, dim=2)
        write(*,'(A,A)') ' CALIMA grain charging timer, charging_model = ', trim(charging_model)
        write(*,'(A)') '   rank   RTZ steps  t(RTZ steps) [s]  t(charging) [s]  charging/RTZ step [ns]  RTZ step [ns]  charging share'
        do k = 1, ncpu
            write(*,'(I7,ES12.4,2F17.3,F24.1,F15.1,F15.3)') k, dble(na(7, k)), ta(2, k), ta(1, k), &
                1d9*ta(1, k)/max(dble(na(7, k)), 1d0), 1d9*ta(2, k)/max(dble(na(7, k)), 1d0), ta(1, k)/max(ta(2, k), 1d-30)
        end do
        write(*,'(A7,ES12.4,2F17.3,F24.1,F15.1,F15.3)') '  all', dble(n(7)), tr, t, 1d9*t/max(dble(n(7)), 1d0), &
            1d9*tr/max(dble(n(7)), 1d0), t/max(tr, 1d-30)
        write(*,'(A,ES10.3,A,F9.1,A)') '   ', dble(sum(n(0:4))), ' bin calls: ', 1d9*t/max(dble(sum(n(0:4))), 1d0), ' ns per call'
        if (sum(n(0:2)) + n(4) > 0) write(*,'(A,4ES11.3,A,ES11.3,A,ES11.3)') &
            '   WDB06rt calls full / uniform table / from d/dlnT / predicted:', &
            dble(n(0:2)), dble(n(4)), '; discrete', dble(n(6)), '; recombination fits', dble(n(5))
    end subroutine report_dust_charging_timer

    subroutine compute_local_anisotropy_factor(dinfo,Fp,Np)
        ! Computes the local radiation anisotropy factor and solid angle 
        ! subtended by radiation sources based on the local radiation field
        ! dinfo --> the DustChemistryInfo instance to update with the computed values
        ! Fp --> the radiation flux vector (#/s/cm^2) for each radiation group
        ! Np --> the radiation energy density (#/cm^3) for each radiation group
        implicit none

        ! ---- Inputs -----
        class(DustChemistryInfo), intent(inout) :: dinfo
        real(dp), dimension(:,:), intent(in) :: Fp
        real(dp), dimension(:), intent(in) :: Np

        ! ---- Local variables ----
        integer :: i
        real(dp) :: rad_ani

        ! the groups at the floor (skipped below) are isotropic and get no solid angle
        dinfo%local_rad_ani = 0d0
        dinfo%local_solid_angle = 0d0
        if (all(Np.le.dinfo%smallNp)) return

        if (fixed_rad_ani .eq. -1d0) then
            do i = 1, size(Np)
                if (Np(i) .le. dinfo%smallNp) cycle
                rad_ani = sqrt(sum(Fp(:,i)**2d0)) / (Np(i) * dinfo%local_c)
                dinfo%local_rad_ani(i) = rad_ani
                dinfo%local_solid_angle(i) = 2d0 * pi * (1d0 + (1d0 - rad_ani)**2d0) 
            end do
        else
            dinfo%local_rad_ani = fixed_rad_ani
            dinfo%local_solid_angle = 2d0 * pi * (1d0 + (1d0 - fixed_rad_ani)**2d0)
        end if
    end subroutine compute_local_anisotropy_factor

    subroutine compute_dust_rad_rates(dinfo, G0, Tk, ne,&
                                        &dustAbs, dustSc, dustRp,&
                                        &pahAbs, pahSc, pahRp)
        ! Computes the dust and PAH radiative rates (absorption, 
        ! scattering, and radiation pressure) for each radiation group, 
        ! given the local dust properties and radiation field.
        ! dinfo --> the DustChemistryInfo instance containing the dust properties
        ! G0 --> local radiation field strength in units of the Habing field
        ! Tk --> local gas temperature (K)
        ! ne --> local electron density (cm^-3)
        ! dustAbs --> output array for dust absorption rates [1/s]
        ! dustSc --> output array for dust scattering rates [1/s]
        ! dustRp --> output array for dust radiation pressure rates [1/s]
        ! pahAbs --> output array for PAH absorption rates [1/s]
        ! pahSc --> output array for PAH scattering rates [1/s]
        ! pahRp --> output array for PAH radiation pressure rates [1/s]
        use pah_photoelectric_heating, only: interpolate_pah_charge_equilibrium
        use dust_radiation, only: rad_dust_rate, rad_pah_rate

        implicit none

        ! ---- Inputs ----
        class(DustChemistryInfo), intent(inout) :: dinfo
        real(dp), intent(in) :: G0, Tk, ne
        ! ---- Outputs ----
        real(dp), dimension(:), intent(inout) :: dustAbs, dustSc, dustRp
        real(dp), dimension(:), intent(inout) :: pahAbs, pahSc, pahRp
        ! ---- Local variables ----
        integer :: ii

        ! 1. Compute and add the different dust radiative rates
        dustAbs = 0d0; dustSc = 0d0; dustRp = 0d0
        if (dinfo%ndust > 0) then
            do ii = 1, dinfo%nGroups
                dustAbs(ii) = rad_dust_rate(dinfo%csa_dust(:,ii),dinfo%rho_dust(:)) ! [1/s]
                dustSc (ii) = rad_dust_rate(dinfo%css_dust(:,ii),dinfo%rho_dust(:)) ! [1/s]
                dustRp (ii) = rad_dust_rate(dinfo%csr_dust(:,ii),dinfo%rho_dust(:)) ! [1/s]
            end do
            dinfo%dustAbs = dustAbs
        end if

        ! 2. Compute the different PAH radiative rates
        pahAbs = 0d0; pahSc = 0d0; pahRp = 0d0
        if (dinfo%npah > 0) then
            ! Before we move onto computing the PAH contribution
            ! to absorption and scattering, we need to compute the
            ! PAH charge distribution, since the PAH cross-sections
            ! depend on the charge state
            ! TODO: currently we use the interpolation tables since
            ! it's sufficiently accurate and much faster, but we 
            ! could change to use the full calculation if we wanted
            do ii = 1, dinfo%npah
                call interpolate_pah_charge_equilibrium(ii,G0,ne,Tk,&
                   &dinfo%fcharge_pah(:,ii))
            end do
            do ii = 1, dinfo%nGroups
                pahAbs(ii) = pahAbs(ii)+ rad_pah_rate(dinfo%csa_pah(:,ii),dinfo%rho_pah(:),dinfo%fcharge_pah(:,:)) ! [1/s]
                pahSc (ii) = pahSc(ii) + rad_pah_rate(dinfo%css_pah(:,ii),dinfo%rho_pah(:),dinfo%fcharge_pah(:,:)) ! [1/s]
                pahRp (ii) = pahRp(ii) + rad_pah_rate(dinfo%csr_pah(:,ii),dinfo%rho_pah(:),dinfo%fcharge_pah(:,:)) ! [1/s]
            end do
            dinfo%pahAbs = pahAbs
        end if

        ! 3. Add the contribution from dust and PAHs for the 
        !    returned absorption, scattering, and radiation pressure rates
        dustAbs = dustAbs + pahAbs
        dustSc = dustSc + pahSc
        dustRp = dustRp + pahRp
    end subroutine compute_dust_rad_rates

    subroutine compute_dust_precool(dinfo, G0_total, Tk, ne,&
                                    &nElement, xelem_ions, nH2, nCO, &
                                    & Np, rates_only)
        ! Pre-computes the dust and PAH cooling and heating rates for the given local conditions, 
        ! which can then be used in subsequent calls to compute_dust_coolrates to save computational time.
        ! dinfo --> the DustChemistryInfo instance to update with the computed rates
        ! G0_total --> local radiation field strength in units of the Habing field
        ! Tk --> local gas temperature (K)
        ! ne --> local electron density (cm^-3)
        ! nElement --> array of number densities for each element (cm^-3)
        ! xelem_ions --> array of ionization fractions for each element and ionization state
        ! nH2 --> local molecular hydrogen density (cm^-3)
        ! nCO --> local carbon monoxide density (cm^-3)
        ! Np --> the radiation energy density (#/cm^3) for each radiation group
        use dust_charging, only: compute_mean_dust_charge, compute_dust_charge_sigma,&
                                compute_dust_charge_dist, compute_Coulomb_focusing_ions,&
                                isrf_lookup, ISRF_ND, ISRF_DCHARGE
        use dust_photoelectric_heating, only: interpolate_dust_peh_rate,&
                                            compute_dust_peh_rate
        use pah_photoelectric_heating, only: interpolate_pah_charge_equilibrium,&
                                            compute_pah_peh_equilibrium,&
                                            interpolate_pah_peh_equilibrium,&
                                            compute_pah_charge_equilibrium
        use dust_radiation, only: update_T_dust, serve_T_dust
        use dust_surface_chemistry, only: grain_h2_formation_rate
        use dust_charging_rtgroups, only: rtgroups_solve_bin, rtgroups_refine_alpha, rtgroups_uniform_lookup, rtgroups_dump, &
                                          rtgroups_predict, rtgroups_coulomb_ratio
        use amr_commons, only: myid

        implicit none

        ! ---- Inputs ----
        class(DustChemistryInfo), intent(inout) :: dinfo
        real(dp), intent(in) :: G0_total, Tk, ne
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: nH2, nCO
        real(dp), dimension(1:dinfo%nGroups), intent(in), optional :: Np
        logical, intent(in), optional :: rates_only   ! only the rates compute_dust_coolrates returns (not the Coulomb factors)

        ! ---- Local variables ----
        integer :: ii,j,idx_g,idx_T
        integer :: i_neutral, i_charged   ! csa_pah row indices: neutral=2*ii-1, charged=2*ii
        real(dp) :: nHI
        integer :: n_charge
        logical :: use_rtg, no_local, use_isrf, predict, ok, skip_coulomb
        real(dp), dimension(1:ndust) :: isrf_G, isrf_L         ! WDB06isrf: PE heating, recombination cooling
        real(dp), dimension(ISRF_ND, 1:ndust) :: isrf_D        ! WDB06isrf: Coulomb factors over the exact P(Z)
        ! WDB06rt Coulomb factors of the low impactor charges: 1 the Gaussian ones times rtg_D (the
        ! exact-P(Z) ratio of the last full solve), 2 rtg_D (the ISRF tables, background-only cells)
        integer, dimension(1:ndust) :: rtg_Dmode
        real(dp), dimension(ISRF_ND, 1:ndust) :: rtg_D
        real(dp) :: zz, ss, gg, ll
        real(dp) :: n_Hp, n_Hep, n_Hepp, h, t0
        ! sized by the compile-time ndust (= dinfo%ndust): on the stack, not allocated on each call
        real(dp), dimension(1:ndust) :: rtg_Pinj, rtg_Prec
        type(RTGState), pointer :: st
        type(RTGState), allocatable :: st0s(:)      ! the dumps only: an RTGState is large, and initialised on each call
        type(RTGResult) :: rr
        type(RTGResult), dimension(1:ndust) :: rrs
        integer :: path, ic, ndumped
        integer, dimension(1:ndust) :: paths
        logical, dimension(1:ndust) :: refined
        real(dp) :: rate(RTG_NRI, 1:ndust), tot(RTG_NRI), zg_used(1:ndust)
        real(dp) :: alpha_bins(RTG_NRI, 1:ndust), n_grain
        real(kind=8) :: wallclock

        if (dinfo%ndust > 0) then
            ! 1. Compute the equilibrium dust charge
            if (dust_charging_timer) t0 = wallclock()
            use_rtg = (trim(charging_model) == 'WDB06rt') .and. present(Np)
            use_isrf = trim(charging_model) == 'WDB06isrf'
            alpha_bins = 0d0
            if (use_rtg) then
                ! Local-field charge balance on the RT photon groups (WDB06 yields),
                ! which also gives the PE heating, recombination cooling and the
                ! grain-assisted ion recombination (rr%alpha; dust_ion_recombination: dinfo%rec_ion_rate, for the RTZ chemistry)
                n_Hp = nElement(1)*xelem_ions(1,2)
                n_Hep = nElement(2)*xelem_ions(2,2)
                n_Hepp = nElement(2)*xelem_ions(2,3)
                no_local = all(Np .le. dinfo%smallNp)
                if (.not. allocated(rtg_warm)) call rtg_allocate()
                ic = dinfo%icell
                if (ic < 1 .or. ic > nvector) ic = nvector + 1
                predict = rtg_predict_tol > 0d0 .and. ic <= nvector
                do ii = 1, dinfo%ndust
                    paths(ii) = -1
                    rtg_Dmode(ii) = 0
                    zg_used(ii) = rtg_Zguess(ii)
                    if (dinfo%rho_dust(ii) <= 0d0) then
                        ! empty bin: no charge or rates needed
                        dinfo%Z_dust(ii) = 0d0
                        dinfo%Z_sigma(ii) = 0d0
                        rtg_Pinj(ii) = 0d0
                        rtg_Prec(ii) = 0d0
                        cycle
                    end if
                    st => rtg_warm(ic, ii)
                    if (ic == nvector + 1) st%valid = .false.
                    if (dust_rtgroups_debug .or. dust_rtgroups_verify) then
                        if (.not. allocated(st0s)) allocate(st0s(1:dinfo%ndust))
                        st0s(ii) = st
                    end if
                    rr = RTGResult()
                    ok = .false.
                    if (no_local .and. .not. dust_ion_recombination) then
                        ! only the uniform background: start-up table in (T, ne), which has no alpha
                        path = 1
                        call rtgroups_uniform_lookup(ii, dinfo%G0_background, Tk, ne, rr%Zmean, rr%Zsigma, rr%Gamma, rr%Lambda)
                        if (Coulomb_precompute) then
                            call isrf_lookup(ii, dinfo%G0_background, Tk, ne, zz, ss, gg, ll, rtg_D(:, ii))
                            rtg_Dmode(ii) = 2
                        end if
                    else if (rtg_served(ii)) then
                        ! the cooling solver's T(1 +- 1e-5) call: first order in ln T from the last solve
                        path = 2
                        h = log(Tk/rtg_last(ii)%T)
                        rr%Zmean = rtg_last(ii)%r%Zmean + h*rtg_last(ii)%r%dZmean
                        rr%Zsigma = rtg_last(ii)%r%Zsigma + h*rtg_last(ii)%r%dZsigma
                        rr%Gamma = rtg_last(ii)%r%Gamma + h*rtg_last(ii)%r%dGamma
                        rr%Lambda = rtg_last(ii)%r%Lambda + rtg_last(ii)%r%Lambda_auto + h*rtg_last(ii)%r%dLambda
                        rr%Zstar = h
                    else
                        ! first order from this cell's last full solve where the inputs changed little
                        if (predict) call rtgroups_predict(ii, st, Np, dinfo%local_c, Tk, ne, n_Hp, n_Hep, n_Hepp, &
                                                           dinfo%G0_background, rtg_predict_tol, rtg_predict_tol_shape, rr, ok)
                        if (ok) then
                            path = 4
                            if (Coulomb_precompute) then
                                rtg_D(:, ii) = st%Dratio
                                rtg_Dmode(ii) = 1
                            end if
                        else
                            ! full solve; the recombination of a wide P(Z) is the Gaussian estimate here
                            path = 0
                            call rtgroups_solve_bin(ii, Np, dinfo%group_eV, dinfo%local_c, Tk, ne, &
                                                    n_Hp, n_Hep, n_Hepp, dinfo%G0_background, rtg_Zguess(ii), &
                                                    st, RTG_DLNT, merge(RTG_ALPHA_GAUSS, RTG_ALPHA_NONE, dust_ion_recombination), &
                                                    rr, sens=predict)
                            if (Coulomb_precompute) then
                                ! narrow P(Z): its exact Coulomb factors over the Gaussian ones (kept for the
                                ! predictions from this solve); a wide one is close to its Gaussian
                                st%Dratio = 1d0
                                if (rr%discrete) call rtgroups_coulomb_ratio(ii, Np, dinfo%local_c, Tk, ne, n_Hp, n_Hep, &
                                    n_Hepp, dinfo%G0_background, rr%Zstar, rr%Zsigma, rr%Zmean, st%Dratio)
                                rtg_D(:, ii) = st%Dratio
                                rtg_Dmode(ii) = 1
                            end if
                        end if
                        rtg_last(ii)%valid = .true.
                        rtg_last(ii)%T = Tk
                        rtg_last(ii)%ne = ne
                        rtg_last(ii)%n_Hp = n_Hp
                        rtg_last(ii)%n_Hep = n_Hep
                        rtg_last(ii)%n_Hepp = n_Hepp
                        rtg_last(ii)%G0 = dinfo%G0_background
                        rtg_last(ii)%c_red = dinfo%local_c
                        rtg_last(ii)%Np = Np
                        rtg_last(ii)%egy = dinfo%group_eV
                        rtg_last(ii)%r = rr
                        rr%Lambda = rr%Lambda + rr%Lambda_auto
                        rtg_Zguess(ii) = rr%Zstar
                        if (dust_charging_timer .and. rr%discrete .and. path == 0) nchg_disc = nchg_disc + 1
                    end if
                    paths(ii) = path
                    rrs(ii) = rr
                    if (dust_charging_timer) nchg(path) = nchg(path) + 1
                    dinfo%Z_dust(ii) = rr%Zmean
                    dinfo%Z_sigma(ii) = rr%Zsigma
                    rtg_Pinj(ii) = rr%Gamma
                    rtg_Prec(ii) = rr%Lambda
                end do
                ! grain-assisted recombination (pyCALIMA solve_cell): the fit only for the wide bins
                ! solved here that carry more than RTG_RECOMB_IMPORTANT of the cell's rate of some ion
                ! (the predicted bins count in that rate)
                refined = .false.
                if (dust_ion_recombination .and. any(paths == 0)) then
                    rate = 0d0
                    do ii = 1, dinfo%ndust
                        if (paths(ii) == 0 .or. paths(ii) == 4) &
                            rate(:, ii) = (dinfo%rho_dust(ii)/dustbins_props(ii)%mgrain)*rtg_last(ii)%r%alpha
                    end do
                    tot = sum(rate, dim=2)
                    do ii = 1, dinfo%ndust
                        if (paths(ii) /= 0) cycle
                        if (rtg_last(ii)%r%discrete) cycle
                        if (maxval(rate(:, ii)/merge(tot, 1d0, tot > 0d0), mask=tot > 0d0) <= RTG_RECOMB_IMPORTANT) cycle
                        refined(ii) = .true.
                        st => rtg_warm(ic, ii)
                        call rtgroups_refine_alpha(ii, Np, dinfo%group_eV, dinfo%local_c, Tk, ne, n_Hp, n_Hep, n_Hepp, &
                                                   dinfo%G0_background, st, rtg_last(ii)%r)
                        if (st%anchored) st%ar%alpha = rtg_last(ii)%r%alpha      ! predictions scale the fitted alpha
                        if (dust_charging_timer) nchg_fit = nchg_fit + rtg_last(ii)%r%nfit
                    end do
                end if
                if (dust_ion_recombination) then
                    ! alpha of the last full solve of each bin: this cell's, also for the T +- 1e-5 calls it served
                    do ii = 1, dinfo%ndust
                        if (paths(ii) >= 0 .and. rtg_last(ii)%valid) alpha_bins(:, ii) = rtg_last(ii)%r%alpha
                    end do
                end if
                if (dust_rtgroups_debug .or. dust_rtgroups_verify) then
                    do ii = 1, dinfo%ndust
                        if (paths(ii) < 0) cycle
                        rr = rrs(ii)
                        if (paths(ii) == 0) rr = rtg_last(ii)%r
                        path = paths(ii)
                        call rtgroups_dump(ic, ii, path, Np, dinfo%group_eV, dinfo%local_c, Tk, ne, &
                                           n_Hp, n_Hep, n_Hepp, dinfo%G0_background, zg_used(ii), st0s(ii), rr, &
                                           refined(ii), ndumped)
                        call rtg_verify_print(ndumped)
                    end do
                end if
            else if (use_isrf) then
                ! uniform-ISRF tables at the local G0: the charge, the PE heating and recombination
                ! cooling, the Coulomb factors of the low impactor charges over the exact P(Z), and
                ! the grain-assisted recombination (dust_ion_recombination)
                do ii = 1, dinfo%ndust
                    if (dust_ion_recombination) then
                        call isrf_lookup(ii, G0_total, Tk, ne, dinfo%Z_dust(ii), dinfo%Z_sigma(ii), isrf_G(ii), isrf_L(ii), &
                                         isrf_D(:, ii), alpha_bins(:, ii))
                    else
                        call isrf_lookup(ii, G0_total, Tk, ne, dinfo%Z_dust(ii), dinfo%Z_sigma(ii), isrf_G(ii), isrf_L(ii), &
                                         isrf_D(:, ii))
                    end if
                end do
                if (dust_charging_timer) nchg(3) = nchg(3) + dinfo%ndust
            else
                idx_g = -1
                idx_T = -1
                do ii = 1, dinfo%ndust
                    call compute_mean_dust_charge(ii,G0_total,Tk,ne,dinfo%Z_dust(ii),idx_g,idx_T)
                    call compute_dust_charge_sigma(ii,G0_total,Tk,ne,dinfo%Z_sigma(ii),idx_g,idx_T)
                end do
                if (dust_charging_timer) nchg(3) = nchg(3) + dinfo%ndust
            end if
            if (dust_ion_recombination .and. (use_rtg .or. use_isrf)) then
                ! grain-assisted X+ -> X per X+ ion [s^-1]: sum over the bins of n_grain alpha (dust bins only)
                dinfo%rec_ion_rate = 0d0
                do ii = 1, dinfo%ndust
                    if (dinfo%rho_dust(ii) <= 0d0) cycle
                    n_grain = dinfo%rho_dust(ii)/dustbins_props(ii)%mgrain
                    do j = 1, RTG_NRI
                        dinfo%rec_ion_rate(RI_ATOMIC(j)) = dinfo%rec_ion_rate(RI_ATOMIC(j)) + n_grain*alpha_bins(j, ii)
                    end do
                end do
            end if
            if (dust_charging_timer) tchg = tchg + (wallclock() - t0)

            ! 2. If needed, precompute the Coulomb factors, for compute_dust_update: not in a call for
            ! the rates only (the cooling solver's T(1 + 1e-5) call), whose factors differ from those
            ! at T only to first order
            skip_coulomb = .false.
            if (present(rates_only)) skip_coulomb = rates_only
            if (.not. skip_coulomb) then
                dinfo%Coulomb_factor = 1d0
                if (Coulomb_precompute) then
                    do ii = 1, dinfo%ndust
                        call compute_Coulomb_focusing_ions(ii,Tk,dinfo%Z_dust(ii),dinfo%Z_sigma(ii),dinfo%nion_charges,&
                                                           dinfo%Coulomb_factor(ii,-1:dinfo%nion_charges))
                        if (use_isrf) then
                            ! the tabulated factors over the exact P(Z), where the Gaussian of <Z> and
                            ! sigma_Z misses the tail at the opposite sign (small grains in cold gas)
                            do j = 1, ISRF_ND
                                if (ISRF_DCHARGE(j) <= dinfo%nion_charges) &
                                    dinfo%Coulomb_factor(ii, ISRF_DCHARGE(j)) = max(isrf_D(j, ii), 1d-10)
                            end do
                        else if (use_rtg) then
                            do j = 1, ISRF_ND
                                if (ISRF_DCHARGE(j) > dinfo%nion_charges) cycle
                                if (rtg_Dmode(ii) == 1) then
                                    dinfo%Coulomb_factor(ii, ISRF_DCHARGE(j)) = &
                                        max(dinfo%Coulomb_factor(ii, ISRF_DCHARGE(j)) * rtg_D(j, ii), 1d-10)
                                else if (rtg_Dmode(ii) == 2) then
                                    dinfo%Coulomb_factor(ii, ISRF_DCHARGE(j)) = max(rtg_D(j, ii), 1d-10)
                                end if
                            end do
                        end if
                    end do
                end if
            end if

            ! 3. Compute the equilibrium dust photoelectric heating and recombination cooling rates
            if (dust_charging_timer) t0 = wallclock()
            if (dust_pe_heating .and. use_rtg) then
                dinfo%Pinj_dust(1:dinfo%ndust) = rtg_Pinj
                dinfo%Prec_dust(1:dinfo%ndust) = rtg_Prec
            elseif (dust_pe_heating .and. use_isrf) then
                dinfo%Pinj_dust(1:dinfo%ndust) = isrf_G
                dinfo%Prec_dust(1:dinfo%ndust) = isrf_L
            elseif (dust_pe_heating .and. present(Np)) then
                do ii = 1, dinfo%ndust
                    if (dust_pe_heating_isrf .or. all(Np.le.dinfo%smallNp)) then
                        call interpolate_dust_peh_rate(ii,G0_total,ne,Tk,&
                                                        &dinfo%Pinj_dust(ii),&
                                                        &dinfo%Prec_dust(ii))
                    else
                        call compute_dust_charge_sigma(ii,G0_total,Tk,ne,dinfo%Z_sigma(ii))
                        ! Consider the contribution from the UV background
                        call interpolate_dust_peh_rate(ii,dinfo%G0_background,ne,Tk,&
                                                        &dinfo%Pinj_dust(ii),dinfo%Prec_dust(ii))
                        ! Recombination cooling depends on the charge distribution and n_e, not on
                        ! the field: keep only the one compute_dust_peh_rate adds below, at the
                        ! charge of the total field (the table's is at the background's charge)
                        dinfo%Prec_dust(ii) = 0d0
                        ! Now consider the contribution from the local radiation field
                        call compute_dust_peh_rate(ii,dinfo%csa_dust(ii,:),&
                                                    dinfo%l_a(ii,:),dinfo%nGroups,dinfo%local_c,&
                                                    dinfo%local_solid_angle,Np(:),dinfo%group_eV(:),&
                                                    dinfo%Z_dust(ii),dinfo%Z_sigma(ii),Tk,ne,&
                                                    dinfo%Pinj_dust(ii),dinfo%Prec_dust(ii))
                    end if
                end do
            elseif (dust_pe_heating) then
                do ii = 1, dinfo%ndust
                    call interpolate_dust_peh_rate(ii,G0_total,ne,Tk,&
                                                    &dinfo%Pinj_dust(ii),&
                                                    &dinfo%Prec_dust(ii))
                end do
            end if
            if (dust_charging_timer) tchg = tchg + (wallclock() - t0)

            ! 4. Compute the internal energy of the dust grain considering all heating and cooling processes
            if (td_served()) then
                ! the cooling solver's T(1 +- 1e-5) call: first order from the last update_T_dust
                call serve_T_dust(td_lin,Tk,dinfo%Prec_dust(:),dinfo%Pinj_dust(:),dinfo%Z_dust(:),dinfo%Pcoll_dust(:),&
                                  dinfo%Prad_dust(:),dinfo%T_dust(:))
            else
                if (.not. allocated(td_lin)) allocate(td_lin(1:dinfo%ndust), td_nel(size(nElement)), &
                                                      td_xion(size(xelem_ions,1), size(xelem_ions,2)))
                if (present(Np)) then
                    call update_T_dust(dinfo%G0_background,dinfo%Pcoll_dust(:),dinfo%Prec_dust(:),dinfo%Pinj_dust(:),dinfo%Prad_dust(:),&
                                        dinfo%T_dust(:),ne,nElement(:),xelem_ions(:,:),dinfo%Coulomb_factor(:,:),nH2,nCO,Tk,dinfo%Z_dust(:),&
                                        Np(:)*dinfo%group_eV(:),dinfo%csa_dust(:,:),lin=td_lin)
                    if (.not. allocated(td_Np)) allocate(td_Np(dinfo%nGroups), td_egy(dinfo%nGroups))
                    td_Np = Np
                    td_egy = dinfo%group_eV
                    td_G0 = dinfo%G0_background
                else
                    call update_T_dust(G0_total,dinfo%Pcoll_dust(:),dinfo%Prec_dust(:),dinfo%Pinj_dust(:),dinfo%Prad_dust(:),&
                                        dinfo%T_dust(:),ne,nElement(:),xelem_ions(:,:),dinfo%Coulomb_factor(:,:),nH2,nCO,Tk,dinfo%Z_dust(:),&
                                        lin=td_lin)
                    td_G0 = G0_total
                end if
                td_valid = .true.
                td_hasNp = present(Np)
                td_T = Tk
                td_ne = ne
                td_nH2 = nH2
                td_nCO = nCO
                td_nel = nElement
                td_xion = xelem_ions
            end if

            ! 5. Now convert all the rates from erg/s per grain to erg/s/cm3
            do ii = 1, dinfo%ndust
                dinfo%n_dust(ii) = dinfo%rho_dust(ii) / dustbins_props(ii)%mgrain
                dinfo%Pcoll_dust(ii) = dinfo%Pcoll_dust(ii) * dinfo%n_dust(ii)
                ! print*,'ii,dinfo%Pcoll_dust(ii),dinfo%n_dust(ii): ',ii,dinfo%Pcoll_dust(ii),dinfo%n_dust(ii)
                dinfo%Prec_dust(ii) = dinfo%Prec_dust(ii) * dinfo%n_dust(ii)
                dinfo%Pinj_dust(ii) = dinfo%Pinj_dust(ii) * dinfo%n_dust(ii)
                dinfo%Prad_dust(ii) = dinfo%Prad_dust(ii) * dinfo%n_dust(ii)
            end do

            ! 6. Compute the H2 formation rate on dust grains
            if (H2ondust) then
                nHI = nElement(1) * xelem_ions(1,1)
                dinfo%H2_formation_rate = grain_h2_formation_rate(nHI,nElement(1),Tk,dinfo%rho_dust(:),dinfo%T_dust(:))
            end if
        end if

        if (dinfo%npah > 0) then
            ! 7. Compute the PAH PEH model
            if (pah_pe_heating .and. present(Np)) then
                if (pah_pe_heating_isrf) then
                    ! We have rt, but we want to just use the ISRF-averaged model
                        do ii = 1, dinfo%npah
                            call interpolate_pah_peh_equilibrium(ii,G0_total,&
                                                                ne,Tk,dinfo%fcharge_pah(:,ii),&
                                                                dinfo%Pabs_pah(ii,1),dinfo%Pinj_pah(ii),&
                                                                dinfo%Prad_pah(ii),dinfo%Prec_pah(ii))
                        end do
                else
                    ! We want the PAH PEH also have rt, so we use the full model
                    do ii = 1, dinfo%npah
                        i_neutral = 2*ii - 1  ! row for neutral PAH cross-sections in csa_pah
                        i_charged = 2*ii      ! row for charged PAH cross-sections in csa_pah
                        ! Consider the contribution from the UV background first
                        call interpolate_pah_peh_equilibrium(ii,dinfo%G0_background,&
                                                            ne,Tk,dinfo%fcharge_pah(:,ii),&
                                                            dinfo%Pabs_pah(ii,1),dinfo%Pinj_pah(ii),&
                                                            dinfo%Prad_pah(ii),dinfo%Prec_pah(ii))
                        ! As for the grains: the recombination cooling is added once, by the full
                        ! model below, at the charge state of the total field
                        dinfo%Prec_pah(ii) = 0d0
                        ! And now the full model for the local radiation field
                        call compute_pah_peh_equilibrium(ii,dinfo%csa_pah(i_neutral,:),&
                                                            dinfo%csa_pah(i_neutral,:),&
                                                            dinfo%csa_pah(i_charged,:),&
                                                            dinfo%csa_pah(i_charged,:),&
                                                            dinfo%nGroups,dinfo%local_solid_angle(:),&
                                                            Np(:),dinfo%group_eV(:),&
                                                            dinfo%local_c,Tk,ne,dinfo%fcharge_pah(:,ii),&
                                                            dinfo%Pabs_pah(ii,:),dinfo%Pinj_pah(ii),&
                                                            dinfo%Prad_pah(ii),dinfo%Prec_pah(ii))
                    end do
                end if
            else if (pah_pe_heating) then
                ! We want PAH PEH but don't have rt
                do ii = 1, dinfo%npah
                    call interpolate_pah_peh_equilibrium(ii,dinfo%G0_background,&
                                                        ne,Tk,dinfo%fcharge_pah(:,ii),&
                                                        dinfo%Pabs_pah(ii,1),dinfo%Pinj_pah(ii),&
                                                        dinfo%Prad_pah(ii),dinfo%Prec_pah(ii))
                end do
            else if (present(Np)) then
                ! We don't want PAH PEH but have rt, so we still want to compute the PAH charge distribution
                do ii = 1, dinfo%npah
                    i_neutral = 2*ii - 1
                    i_charged = 2*ii
                    call compute_pah_charge_equilibrium(ii,dinfo%csa_pah(i_neutral,:),&
                                                        dinfo%csa_pah(i_neutral,:),&
                                                        dinfo%csa_pah(i_charged,:),&
                                                        dinfo%csa_pah(i_charged,:),&
                                                        dinfo%nGroups,dinfo%local_solid_angle(:),&
                                                        Np(:),dinfo%group_eV(:),dinfo%local_c,&
                                                        Tk,ne,dinfo%fcharge_pah(:,ii))
                end do
            else
                ! We don't want PAH PEH but don't have rt, so we just interpolate
                ! the ISRF-averaged PAH charge tables
                do ii = 1, dinfo%npah
                    call interpolate_pah_charge_equilibrium(ii,G0_total,ne,Tk,&
                       &dinfo%fcharge_pah(:,ii))
                end do
            end if

            ! 8. Now convert from erg/s to erg/s/cm3
            do ii = 1, dinfo%npah
                dinfo%n_pah(ii) = dinfo%rho_pah(ii) / pahbins_props(ii)%mpah
                dinfo%Pabs_pah(ii,:) = dinfo%Pabs_pah(ii,:) * dinfo%n_pah(ii)
                dinfo%Pinj_pah(ii) = dinfo%Pinj_pah(ii) * dinfo%n_pah(ii)
                dinfo%Prad_pah(ii) = dinfo%Prad_pah(ii) * dinfo%n_pah(ii)
                dinfo%Prec_pah(ii) = dinfo%Prec_pah(ii) * dinfo%n_pah(ii)
            end do
        end if
    contains

        logical function rtg_served(ib)
            ! same inputs as the last full solve of bin ib but T, within RTG_SERVE in ln T
            integer, intent(in) :: ib
            real(dp) :: x
            rtg_served = .false.
            if (.not. rtg_last(ib)%valid) return
            x = Tk/rtg_last(ib)%T
            if (x > RTG_SERVE_HI .or. x < RTG_SERVE_LO) return
            if (ne /= rtg_last(ib)%ne .or. n_Hp /= rtg_last(ib)%n_Hp .or. n_Hep /= rtg_last(ib)%n_Hep .or. &
                n_Hepp /= rtg_last(ib)%n_Hepp .or. dinfo%G0_background /= rtg_last(ib)%G0 .or. &
                dinfo%local_c /= rtg_last(ib)%c_red) return
            if (any(Np /= rtg_last(ib)%Np) .or. any(dinfo%group_eV /= rtg_last(ib)%egy)) return
            rtg_served = .true.
        end function rtg_served

        logical function td_served()
            ! same inputs as the last update_T_dust but T, within RTG_SERVE in ln T (and the
            ! recombination and photoelectric terms and the grain charge, which serve_T_dust takes to
            ! first order)
            real(dp) :: x
            td_served = .false.
            if (.not. td_valid .or. (td_hasNp .neqv. present(Np))) return
            x = Tk/td_T
            if (x > RTG_SERVE_HI .or. x < RTG_SERVE_LO) return
            if (ne /= td_ne .or. nH2 /= td_nH2 .or. nCO /= td_nCO) return
            if (present(Np)) then
                if (dinfo%G0_background /= td_G0) return
                if (any(Np /= td_Np) .or. any(dinfo%group_eV /= td_egy)) return
            else
                if (G0_total /= td_G0) return
            end if
            if (any(nElement /= td_nel) .or. any(xelem_ions /= td_xion)) return
            td_served = .true.
        end function td_served

        subroutine rtg_allocate()
            integer :: ib
            allocate(rtg_warm(1:nvector + 1, 1:dinfo%ndust), rtg_last(1:dinfo%ndust), rtg_Zguess(1:dinfo%ndust))
            rtg_Zguess = -huge(1d0)
            do ib = 1, dinfo%ndust
                allocate(rtg_last(ib)%Np(dinfo%nGroups), rtg_last(ib)%egy(dinfo%nGroups))
            end do
        end subroutine rtg_allocate

        subroutine rtg_verify_print(nd)
            ! dust_rtgroups_verify: print the first calls, and stop once dust_rtgroups_debug_max are dumped
            integer, intent(in) :: nd
            if (.not. dust_rtgroups_verify) return
            if (nd <= 12 .and. myid == 1) write(*,'(A,I5,A,I6,A,I2,A,I1,A,ES11.4,A,ES10.3,A,F10.3,A,F8.3,A,L1,A,I1,A,2ES11.3)') &
                ' WDB06rt verify: call', nd, ' cell', ic, ' bin', ii, ' path', path, ' T', Tk, ' ne', ne, &
                ' <Z>', rr%Zmean, ' sigma', rr%Zsigma, ' disc ', rr%discrete, ' fit', rr%nfit, &
                ' alpha(H,C)', rr%alpha(1), rr%alpha(3)
            if (nd >= dust_rtgroups_debug_max) then
                if (myid == 1) then
                    write(*,'(A,I7,A)') ' WDB06rt verify: ', nd, ' calls dumped to dust_rtgroups_debug_*.dat; stopping'
                    write(*,'(A)') '   compare with: python diagnostics/dust_charge/compare_ramses_rtgroups.py RUN_DIR --tables TABLES'
                end if
                call clean_stop
            end if
        end subroutine rtg_verify_print

    end subroutine compute_dust_precool

    subroutine compute_dust_coolrates(dinfo, G0_total, Tk, ne,&
                                    &nElement, xelem_ions, nH2, nCO, &
                                    &total_rec_power, total_inj_power, total_col_power,&
                                    &H2_formation_rate,Np)
        ! Computes the dust and PAH cooling and heating rates for the given local conditions,
        ! using the pre-computed rates from compute_dust_precool if available to save computational time.
        ! dinfo --> the DustChemistryInfo instance to update with the computed rates
        ! G0_total --> local radiation field strength in units of the Habing field
        ! Tk --> local gas temperature (K)
        ! ne --> local electron density (cm^-3)
        ! nElement --> array of number densities for each element (cm^-3)
        ! xelem_ions --> array of ionization fractions for each element and ionization state
        ! nH2 --> local molecular hydrogen density (cm^-3)
        ! nCO --> local carbon monoxide density (cm^-3)
        ! total_rec_power --> output total recombination cooling power from dust and PAHs (erg/s/cm^3)
        ! total_inj_power --> output total photoelectric heating power from dust and PAHs (erg/s/cm^3)
        ! total_col_power --> output total collisional cooling power from dust (erg/s/cm^3)
        ! H2_formation_rate --> output H2 formation rate on dust grains (cm^3/s)
        ! Np --> the radiation energy density (#/cm^3) for each radiation group
        implicit none
        ! ---- Inputs ----
        class(DustChemistryInfo), intent(inout) :: dinfo
        real(dp), intent(in) :: G0_total, Tk, ne
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: nH2, nCO
        real(dp), intent(out) :: total_rec_power, total_inj_power, total_col_power
        real(dp), intent(out) :: H2_formation_rate
        real(dp), dimension(1:dinfo%nGroups), intent(in), optional :: Np


        ! If precomputed rates are not already in dinfo (i.e. compute_dust_precool was not
        ! called before us this step), delegate to it now.  This eliminates ~148 lines
        ! of duplicated physics that previously lived in the non-precomp branch here.
        if (.not. dinfo%use_precomp) then
            call compute_dust_precool(dinfo, G0_total, Tk, ne, nElement, xelem_ions, nH2, nCO, Np=Np, rates_only=.true.)
        end if

        ! Read the precomputed rates from dinfo (populated either just above or by an
        ! external compute_dust_precool call earlier in the RTZ step).
        total_rec_power = 0d0
        total_inj_power = 0d0
        total_col_power = 0d0
        H2_formation_rate = -1d0
        if (dinfo%ndust > 0) then
            total_rec_power = sum(dinfo%Prec_dust)
            total_inj_power = sum(dinfo%Pinj_dust)
            total_col_power = sum(dinfo%Pcoll_dust)
            if (H2ondust) then
                H2_formation_rate = dinfo%H2_formation_rate
            end if
        end if
        if (dinfo%npah > 0) then
            total_rec_power = total_rec_power + sum(dinfo%Prec_pah)
            total_inj_power = total_inj_power + sum(dinfo%Pinj_pah)
        end if
        dinfo%use_precomp = .false.
    end subroutine compute_dust_coolrates

    subroutine compute_dust_update(dinfo,nElement,xelem_ions,dt,Np,step_ok, dUU)

        use ode_driver_mod, only: integrate_dust_ode
        use ode_interface_mod, only: dust_solver_step
        use dust_rhs_mod, only: dust_rhs
        use dust_rates, only: compute_rate_caches
        use dust_radiative_torques, only: total_radiative_torque,IR_damping_factor
#ifdef RTZ
        use rtz_module, only:elements
#endif

        implicit none

        ! ---- Inputs ----
        class(DustChemistryInfo), intent(inout) :: dinfo
        real(dp), intent(in) :: dt
        real(dp), intent(inout) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in), optional :: Np(:)
        logical, intent(out) :: step_ok

        ! --- Local variables ----
        integer :: ii, jj, ndust_total
        real(dp) :: sum_check
        real(dp), pointer :: y_gas(:,:), y_gas_out(:,:)
        real(dp), pointer :: y_dust(:), y_dust_out(:)
        real(dp) :: mass_g(n_elements)   ! element atomic masses — populated once via #ifdef, used throughout
        real(dp) :: h_init_val           ! substep hint for integrate_dust_ode
        ! Cache the last accepted substep's recommended next-step size across calls.
        ! On re-entry with the same (or similar) conditions this skips the firstcall
        ! 1/kmax reduction, cutting substep count at equilibrium for explicit solvers.
        real(dp), save :: last_h_ode = 0.0_dp

        real(dp), intent(out) :: dUU

        dUU = 0.0_dp ! should I keep previous values?

        ! Populate mass_g once so the pack/unpack loops below are #ifdef-free.
#ifdef RTZ
        do ii = 1, n_elements
            mass_g(ii) = elements(ii)%atomic_mass_g
        end do
#else
        mass_g(:) = el_atomic_masses_g(1:n_elements)
#endif

        ! If no dust or PAH process is active, just return
        if (ndust_processes.eq.0 .and. npah_processes.eq.0) then
            step_ok = .true.
            return
        end if

        ! Precompute/cache rate factors for this cell-update step
        call compute_rate_caches(dinfo, nElement)

        ! 1. Compute the local RAT-D quantities if we run with dust_ratd
        if (dust_ratd) then
            if (present(Np)) then
                do ii = 1, dinfo%ndust
                    dinfo%rat_torque(ii) = total_radiative_torque(dinfo%local_rad_ani,Np,&
                                                            dinfo%group_eV(:)*eV2erg,dinfo%nGroups,&
                                                            dinfo%local_c,dinfo%csrat_dust(:,ii))
                end do
            end if
            dinfo%IR_damp_factor = IR_damping_factor(dinfo%local_G0*1.13d0,nElement(1),dinfo%local_Tk,dinfo%T_dust)
            do ii = 1, dinfo%ndust
                dinfo%rat_torque(ii) = dinfo%rat_torque(ii) + dustbins_props(ii)%RAT_torque_0 * dinfo%G0_background
            end do
        end if

        ! 2. Now setup the arrays for gas and dust quantities to send to the ODE solver
        ndust_total = 0
        if (dinfo%ndust > 0) ndust_total = ndust_total + dinfo%ndust
        if (dinfo%npah > 0) ndust_total = ndust_total + dinfo%npah

        if (carry_gas_ions) then
            call ensure_update_cache(n_elements, n_elements + 1, ndust_total)
        else
            call ensure_update_cache(n_elements, 1, ndust_total)
        end if
        y_gas => y_gas_cache; y_gas_out => y_gas_out_cache
        y_dust => y_dust_cache; y_dust_out => y_dust_out_cache

        if (carry_gas_ions) then
            do ii = 1, n_elements
                y_gas(ii,1) = nElement(ii) * mass_g(ii)
                y_gas(ii,2:n_elements+1) = nElement(ii) * xelem_ions(ii,:) * mass_g(ii)
            end do
        else
            do ii = 1, n_elements
                y_gas(ii,1) = nElement(ii) * mass_g(ii)
            end do
        end if
        ! Pack y_dust: PAH bins first [1:npah], then grain bins [npah+1:npah+ndust].
        ! This convention is mirrored in the unpack block below and in integrate_dust_ode.
        if (dinfo%ndust > 0 .and. dinfo%npah > 0) then
            y_dust(1:dinfo%npah) = dinfo%rho_pah(1:dinfo%npah)
            y_dust(dinfo%npah+1:dinfo%npah+dinfo%ndust) = dinfo%rho_dust(1:dinfo%ndust)
        else if (dinfo%ndust > 0) then
            y_dust(1:dinfo%ndust) = dinfo%rho_dust(1:dinfo%ndust)
        else if (dinfo%npah > 0) then
            y_dust(1:dinfo%npah) = dinfo%rho_pah(1:dinfo%npah)
        end if

        ! 3. Now we are ready to call the ODE solver to integrate the dust evolution.
        ! Guard against an uninitialised procedure pointer (set by init_dust_solver based
        ! on dust_solver_type); a null dereference here would be a silent SIGSEGV.
        if (.not. associated(dust_solver_step)) then
            print *, "FATAL ERROR: dust_solver_step is not associated. Check dust_solver_type in namelist."
            call clean_stop
        end if
        step_ok = .true.
        h_init_val = dt
        if (last_h_ode > 0.0_dp) h_init_val = last_h_ode
        call integrate_dust_ode(dinfo,dt,y_gas,y_dust,dust_rhs,dust_solver_step,&
                                y_gas_out,y_dust_out,h_init_val,0d0,dt,&
                                debug_flag=dust_debug,step_ok=step_ok,h_last=last_h_ode)

        ! 4. Update the dinfo with the new values after the ODE step
        if (carry_gas_ions) then
            do ii = 1, n_elements
                nElement(ii) = y_gas_out(ii,1) / mass_g(ii)
                if (nElement(ii) > 1d-30) then
                    xelem_ions(ii,:) = y_gas_out(ii,2:n_elements+1) / nElement(ii) / mass_g(ii)
                    ! Make sure that xelem_ions add up to 1 for each element
                    sum_check = sum(xelem_ions(ii,:))
                    if (sum_check > 1d-30) then
                        xelem_ions(ii,:) = xelem_ions(ii,:) / sum_check
                    end if
                end if
            end do
        else
            do ii = 1, n_elements
                nElement(ii) = y_gas_out(ii,1) / mass_g(ii)
            end do
        end if

        ! Unpack y_dust_out: PAH bins first [1:npah], grain bins [npah+1:npah+ndust].
        if (dinfo%ndust > 0 .and. dinfo%npah > 0) then
            dinfo%rho_pah(1:dinfo%npah) = y_dust_out(1:dinfo%npah)
            dinfo%rho_dust(1:dinfo%ndust) = y_dust_out(dinfo%npah+1:dinfo%npah+dinfo%ndust)
        else if (dinfo%ndust > 0) then
            dinfo%rho_dust(1:dinfo%ndust) = y_dust_out(1:dinfo%ndust)
        else if (dinfo%npah > 0) then
            dinfo%rho_pah(1:dinfo%npah) = y_dust_out(1:dinfo%npah)
        end if

#ifdef RTZ
        ! Compute dUU: the maximum fractional change in any gas or dust species over this step.
        ! dUU is returned *unscaled* (pure fractional change, dimensionless).
        ! The caller (rtz_cool_step in rtz_cooling_module.f90) multiplies by one_over_x_FRAC
        ! before comparing against the RTZ convergence criterion.

        ! check for gas
        do ii = 1, n_elements
            if (elements(ii)%atomic_number > 0) then
                dUU = max(dUU, abs(y_gas_out(ii,1)-y_gas(ii,1))/(y_gas(ii,1)+y_min))

                ! if not, ion part is zero
                if (carry_gas_ions) then
                    do jj=1,elements(ii)%n_ions
                        dUU = max(dUU, abs(y_gas_out(ii,1+jj)-y_gas(ii,1+jj))/(y_gas(ii,1+jj)+y_min))
                    end do
                end if

            end if
        end do

        ! check for dust
        if (dinfo%ndust > 0 .or. dinfo%npah > 0) then
            do ii = 1, dinfo%npah+dinfo%ndust
                dUU = max(dUU, abs(y_dust_out(ii)-y_dust(ii))/(y_dust(ii)+y_min))
            end do
        end if
#endif

    end subroutine compute_dust_update

    subroutine ensure_update_cache(n1, n2, ndust_total)
        integer, intent(in) :: n1, n2, ndust_total
        logical :: need_realloc

        need_realloc = .false.
        if (.not. allocated(y_gas_cache)) then
            need_realloc = .true.
        else if (size(y_gas_cache,1) /= n1 .or. size(y_gas_cache,2) /= n2) then
            need_realloc = .true.
        end if

        if (need_realloc) then
            if (allocated(y_gas_cache)) then
                deallocate(y_gas_cache)
                deallocate(y_gas_out_cache)
            end if
            allocate(y_gas_cache(n1, n2))
            allocate(y_gas_out_cache(n1, n2))
        end if

        need_realloc = .false.
        if (.not. allocated(y_dust_cache)) then
            need_realloc = .true.
        else if (size(y_dust_cache) /= ndust_total) then
            need_realloc = .true.
        end if

        if (need_realloc) then
            if (allocated(y_dust_cache)) then
                deallocate(y_dust_cache)
                deallocate(y_dust_out_cache)
            end if
            allocate(y_dust_cache(ndust_total))
            allocate(y_dust_out_cache(ndust_total))
        end if
    end subroutine ensure_update_cache

end module dust_interface
subroutine report_dust_charging_timer_ext()
    ! for amr/end.f90, compiled before the CALIMA modules
    use dust_interface, only: report_dust_charging_timer
    implicit none
    call report_dust_charging_timer()
end subroutine report_dust_charging_timer_ext
