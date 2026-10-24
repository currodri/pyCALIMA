subroutine run_dust_solver_test()
    use amr_parameters, only: dp, z_ave
    use hydro_parameters, only: n_elements, ndust, npah
    use dust_commons
    use dust_interface, only: compute_dust_update
    use constants, only: amu2g, yr2sec
    use dust_init, only: init_dust_depletion_tests
    use amr_commons, only: myid
#ifdef RTZ
    use rtz_module, only: elements
#endif

    implicit none

    real(dp), dimension(:), allocatable :: nElement
    real(dp), dimension(:,:), allocatable :: xelem_ions
    real(dp), dimension(:), allocatable :: rho_dust
    real(dp), dimension(:), allocatable :: rho_pah
    real(dp) :: dt, t_yr
    integer :: iElement, step, unit_log, i, jj
    logical :: step_ok
    external :: clean_stop
    character(len=64) :: log_filename
    integer :: n_active

    real(dp) :: dummy_dUU

    dummy_dUU = 0.0_dp

    ! Only run on processor 1 to avoid duplicate/interfering outputs and files
    if (myid /= 1) return

    n_active = 0
    if (dust_accretion) n_active = n_active + 1
    if (dust_sputtering) n_active = n_active + 1
    if (dust_coagulation) n_active = n_active + 1
    if (dust_shattering) n_active = n_active + 1

    if (n_active /= 1) then
        write(*,*) 'ERROR: Exactly one dust process (accretion, sputtering, coagulation, shattering)'
        write(*,*) 'must be active for this test.'
        call clean_stop()
    end if

    write(*,*) '========================================='
    write(*,*) 'RUNNING CALIMA DUST SOLVER TEST'
    write(*,*) '  test_nH          = ', test_nH
    write(*,*) '  test_Tk          = ', test_Tk
    write(*,*) '  test_ne          = ', test_ne
    write(*,*) '  test_mu          = ', test_mu
    write(*,*) '  test_nsteps      = ', test_nsteps
    write(*,*) '  test_dt          = ', test_dt, ' yr'
    write(*,*) '  dust_accretion   = ', dust_accretion
    write(*,*) '  dust_sputtering  = ', dust_sputtering
    write(*,*) '  dust_coagulation = ', dust_coagulation
    write(*,*) '  dust_shattering  = ', dust_shattering
    write(*,*) '========================================='

    allocate(nElement(1:n_elements))
    allocate(xelem_ions(1:n_elements, 1:n_elements))
    if (ndust > 0) allocate(rho_dust(1:ndust))
    if (npah > 0) allocate(rho_pah(1:npah))

    nElement = 0.0_dp
    xelem_ions = 0.0_dp
    if (ndust > 0) rho_dust = 0.0_dp
    if (npah > 0) rho_pah = 0.0_dp

    ! Initialise elemental abundances as in rtz_cooling_module.f90
    nElement(1)  = test_nH   
    if (n_elements >= 2)  nElement(2)  = test_nH * 8.51d-02 ! Helium
    if (n_elements >= 6)  nElement(6)  = test_nH * 2.69d-04 * z_ave ! Carbon
    if (n_elements >= 7)  nElement(7)  = test_nH * 6.76d-05 * z_ave ! Nitrogen
    if (n_elements >= 8)  nElement(8)  = test_nH * 4.90d-04 * z_ave ! Oxygen
    if (n_elements >= 10) nElement(10) = test_nH * 8.51d-05 * z_ave ! Neon
    if (n_elements >= 12) nElement(12) = test_nH * 3.98d-05 * z_ave ! Magnesium
    if (n_elements >= 14) nElement(14) = test_nH * 3.24d-05 * z_ave ! Silicon
    if (n_elements >= 16) nElement(16) = test_nH * 1.32d-05 * z_ave ! Sulfur
    if (n_elements >= 26) nElement(26) = test_nH * 3.16d-05 * z_ave ! Iron

    ! Set ionization states to neutral
    xelem_ions = 0.0_dp
    do iElement = 1, n_elements
        xelem_ions(iElement, 1) = 1.0_dp
    end do

    ! Deplete elements and initialize dust and PAH densities
    call init_dust_depletion_tests(nElement, rho_dust, rho_pah)

    ! Set up dust_helper
    call dust_helper%reset()
    dust_helper%local_Tk = test_Tk
    dust_helper%local_nH = test_nH
    dust_helper%local_ne = test_ne
    dust_helper%local_mu = test_mu

    ! Estimate local total mass density
    dust_helper%local_rho = test_nH * (1.4_dp * amu2g)
    dust_helper%local_dx = 1.0d18 ! dummy cell size (e.g. 1 pc)
    dust_helper%local_vol = dust_helper%local_dx**3
    dust_helper%local_Jeans = 4.81973044d19 * sqrt(test_Tk/test_nH)
    dust_helper%local_G0 = 0.0_dp
    dust_helper%local_nCO = 0.0_dp
    dust_helper%G0_background = 0.0_dp
    dust_helper%local_sigma = 1.0d5 ! 1 km/s in cm/s

    ! Copy cross sections
    if (allocated(sigca_dust) .and. ndust > 0) then
        dust_helper%csa_dust = sigca_dust
        dust_helper%css_dust = sigcs_dust
        dust_helper%csr_dust = sigcr_dust
    end if
    if (dust_ratd .and. allocated(sigcrat_dust) .and. ndust > 0) then
        dust_helper%csrat_dust = sigcrat_dust
    end if
    if (allocated(sigca_pah) .and. npah > 0) then
        dust_helper%csa_pah = sigca_pah
        dust_helper%css_pah = sigcs_pah
        dust_helper%csr_pah = sigcr_pah
    end if

    ! Set starting dust densities
    if (ndust > 0) dust_helper%rho_dust = rho_dust
    if (npah > 0) dust_helper%rho_pah = rho_pah

    ! Determine log filename based on active process
    if (dust_accretion) then
        log_filename = 'dust_accretion_test.log'
    else if (dust_sputtering) then
        log_filename = 'dust_sputtering_test.log'
    else if (dust_coagulation) then
        log_filename = 'dust_coagulation_test.log'
    else if (dust_shattering) then
        log_filename = 'dust_shattering_test.log'
    end if

    ! Open log file for output
    unit_log = 888
    open(unit=unit_log, file=trim(log_filename), status='replace')

    ! Header for log file
    write(unit_log, '(A)') '# CALIMA DUST SOLVER EVOLUTION TEST'
    write(unit_log, '(A,ES15.7)') '# nH = ', test_nH
    write(unit_log, '(A,ES15.7)') '# Tk = ', test_Tk
    write(unit_log, '(A,ES15.7)') '# ne = ', test_ne
    write(unit_log, '(A,I0)') '# Steps = ', test_nsteps
    write(unit_log, '(A,ES15.7)') '# dt_yr = ', test_dt
    write(unit_log, '(A,I0)') '# ndust = ', ndust
    write(unit_log, '(A,I0)') '# npah = ', npah
    write(unit_log, '(A,L1)') '# dust_accretion = ', dust_accretion
    write(unit_log, '(A,L1)') '# dust_sputtering = ', dust_sputtering
    write(unit_log, '(A,L1)') '# dust_coagulation = ', dust_coagulation
    write(unit_log, '(A,L1)') '# dust_shattering = ', dust_shattering
    write(unit_log, '(A,ES15.7)') '# local_mu = ', dust_helper%local_mu
    write(unit_log, '(A,A)') '# dust_tables_dir = ', trim(dust_tables_dir)
    do i = 1, n_elements
#ifdef RTZ
        write(unit_log, '(A,I0,A,I0,A,ES15.7E3)') '# GasElement ', i, ' atomic_number = ', elements(i)%atomic_number, ' mass_g = ', elements(i)%atomic_mass_g
#else
        write(unit_log, '(A,I0,A,I0,A,ES15.7E3)') '# GasElement ', i, ' atomic_number = ', i, ' mass_g = ', el_atomic_masses_g(i)
#endif
    end do
    do i = 1, ndust
        write(unit_log, '(A,I0,A,ES15.7,A,ES15.7)') '# Bin ', i, ' k0_acc = ', dustbins_props(i)%k0_acc, ' nhmax_acc = ', dustbins_props(i)%nhmax_acc
        write(unit_log, '(A,I0,A,ES15.7,A,ES15.7,A,ES15.7,A,ES15.7,A,ES15.7,A,ES15.7,A,I0)') &
            '# Bin ', i, ' asize = ', dustbins_props(i)%asize, &
            ' sgrain = ', dustbins_props(i)%sgrain, &
            ' mgrain = ', dustbins_props(i)%mgrain, &
            ' amin = ', dustbins_props(i)%amin, &
            ' amax = ', dustbins_props(i)%amax, &
            ' surf_energy = ', dustbins_props(i)%surf_energy, &
            ' interact_group = ', dustbins_props(i)%interact_group
        write(unit_log, '(A,I0,A,ES15.7,A,ES15.7,A,ES15.7)') &
            '# Bin ', i, ' tensile_strength = ', dustbins_props(i)%tensile_strength, &
            ' Youngs_modulus = ', dustbins_props(i)%Youngs_modulus, &
            ' Poisson_ratio = ', dustbins_props(i)%Poisson_ratio
        write(unit_log, '(A,I0,A,I0)') '# Bin ', i, ' nelements = ', dustbins_props(i)%nelements
        do jj = 1, dustbins_props(i)%nelements
            write(unit_log, '(A,I0,A,I0,A,I0,A,ES15.7,A,ES15.7)') &
                '# Bin ', i, ' element ', jj, ' index = ', dustbins_props(i)%el_index(jj), &
                ' mfraction = ', dustbins_props(i)%el_mfractions(jj), &
                ' mass_g = ', dustbins_props(i)%el_atomic_masses_g(jj)
        end do
    end do
    write(unit_log, '(A)', advance='no') '# Time(yr)  ne(cm-3)'
    do i = 1, n_elements
        write(unit_log, '(A,I0,A)', advance='no') '  Gas_El_', i, '(cm-3)'
    end do
    do i = 1, ndust
        write(unit_log, '(A,I0,A)', advance='no') '  Dust_Bin_', i, '(g/cm3)'
    end do
    do i = 1, npah
        write(unit_log, '(A,I0,A)', advance='no') '  PAH_Bin_', i, '(g/cm3)'
    end do
    write(unit_log, *) ''

    ! Write initial state (t = 0)
    write(unit_log, '(ES15.7E3,ES15.7E3)', advance='no') 0.0_dp, dust_helper%local_ne
    do i = 1, n_elements
        write(unit_log, '(A,ES15.7E3)', advance='no') ' ', nElement(i)
    end do
    do i = 1, ndust
        write(unit_log, '(A,ES15.7E3)', advance='no') ' ', dust_helper%rho_dust(i)
    end do
    do i = 1, npah
        write(unit_log, '(A,ES15.7E3)', advance='no') ' ', dust_helper%rho_pah(i)
    end do
    write(unit_log, *) ''

    ! Step size in seconds
    dt = test_dt * yr2sec
    t_yr = 0.0_dp

    do step = 1, test_nsteps
        dust_helper%local_ne = test_ne

        ! Call the dust solver interface
        call compute_dust_update(dust_helper, nElement, xelem_ions, dt, step_ok=step_ok, dUU=dummy_dUU)
        
        t_yr = t_yr + test_dt

        ! Print to log file
        write(unit_log, '(ES15.7E3,ES15.7E3)', advance='no') t_yr, dust_helper%local_ne
        do i = 1, n_elements
            write(unit_log, '(A,ES15.7E3)', advance='no') ' ', nElement(i)
        end do
        do i = 1, ndust
            write(unit_log, '(A,ES15.7E3)', advance='no') ' ', dust_helper%rho_dust(i)
        end do
        do i = 1, npah
            write(unit_log, '(A,ES15.7E3)', advance='no') ' ', dust_helper%rho_pah(i)
        end do
        write(unit_log, *) ''

        if (.not. step_ok) then
            write(*,*) 'WARNING: step_ok = .false. at step ', step
        end if
    end do

    close(unit_log)
    write(*,*) 'CALIMA DUST SOLVER TEST COMPLETE. Log written to ', trim(log_filename)
    write(*,*) 'Stopping run now.'
    call clean_stop()

end subroutine run_dust_solver_test
