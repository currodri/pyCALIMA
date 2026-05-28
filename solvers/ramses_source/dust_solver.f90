module ode_interface_mod

    use amr_parameters, only: dp
    use hydro_parameters, only: n_elements, ndust, npah
    use dustbin_types, only: DustChemistryInfo

    implicit none
    private
    public :: ZERO, ONE, TWO, HALF, SIXTH, rhs_interface, solver_step_interface

    real(dp), parameter :: ZERO  = 0.0_dp
    real(dp), parameter :: ONE   = 1.0_dp
    real(dp), parameter :: TWO   = 2.0_dp
    real(dp), parameter :: HALF  = 0.5_dp
    real(dp), parameter :: SIXTH = 1.0_dp / 6.0_dp

    abstract interface
        subroutine rhs_interface(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax,debug_flag)
            ! Interface for the ODE right-hand side function. This is the function that will be called 
            ! by the ODE solver to compute the time derivatives of the gas and dust abundances.
            ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
            ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
            ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
            ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
            import dp, DustChemistryInfo
            implicit none
            ! ---- Input/Output variables ----
            type(DustChemistryInfo), intent(in) :: dust_info
            real(dp), intent(in) :: y_gas(:,:), y_dust(:)
            real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:)
            real(dp), intent(inout), optional :: kmax
            logical, intent(in), optional :: debug_flag

        end subroutine rhs_interface
    end interface

    abstract interface
        subroutine solver_step_interface(dust_info,y_gas,y_dust,h,rhs,y_gas_new,y_dust_new,h_new,firstcall,accepted,break,debug_flag)
            ! Interface for the ODE solver step function. This is the function that will be called 
            ! by the ODE solver to compute the right-hand side of the ODE system, and to update the solution.
            ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
            ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
            ! h        --> Time step size [s]
            ! rhs      --> Procedure pointer to the right-hand side function that computes the time derivatives
            ! y_gas_new --> 2D array with the updated gas phase abundances [g cm-3]
            ! y_dust_new --> 1D array with the updated dust phase abundances [g cm-3]
            ! h_new    --> Updated time step size [s]
            ! first_call --> Logical flag indicating whether this is the first call to the subroutine
            ! accepted --> Logical flag indicating whether the step was accepted or not (for adaptive time stepping)
            ! break    --> Logical flag indicating whether the integration should be stopped (e.g., if there are no active dust processes)
            import dp, DustChemistryInfo
            implicit none
            ! ---- Input/Output variables ----
            type(DustChemistryInfo), intent(in) :: dust_info
            real(dp), intent(in) :: y_gas(:,:), y_dust(:)
            real(dp), intent(inout) :: h
            procedure(rhs_interface) :: rhs
            real(dp), intent(out) :: y_gas_new(:,:), y_dust_new(:)
            real(dp), intent(out) :: h_new
            logical, intent(out) :: accepted
            logical, intent(out) :: break
            logical, intent(in) :: firstcall
            logical, intent(in), optional :: debug_flag
        end subroutine solver_step_interface
    end interface

end module ode_interface_mod

module dust_rhs_mod
    use amr_parameters, only: dp
    use dust_commons
    use dust_rates

    implicit none
    private
    public :: dust_rhs, print_last_process_kmax
    public :: reset_timestep_reduction_counters, register_timestep_reduction_cause
    public :: print_timestep_reduction_counters

    real(dp), allocatable, save :: last_kmax_dust(:)
    real(dp), allocatable, save :: last_kmax_pah(:)
    integer, allocatable, save :: reduction_count_dust(:)
    integer, allocatable, save :: reduction_count_pah(:)

    contains

    subroutine ensure_kmax_storage
        implicit none

        if (allocated(last_kmax_dust)) then
            if (size(last_kmax_dust) /= ndust_processes) then
                deallocate(last_kmax_dust)
            end if
        end if
        if (.not. allocated(last_kmax_dust)) then
            allocate(last_kmax_dust(max(0, ndust_processes)))
        end if

        if (allocated(last_kmax_pah)) then
            if (size(last_kmax_pah) /= npah_processes) then
                deallocate(last_kmax_pah)
            end if
        end if
        if (.not. allocated(last_kmax_pah)) then
            allocate(last_kmax_pah(max(0, npah_processes)))
        end if

        if (allocated(reduction_count_dust)) then
            if (size(reduction_count_dust) /= ndust_processes) then
                deallocate(reduction_count_dust)
            end if
        end if
        if (.not. allocated(reduction_count_dust)) then
            allocate(reduction_count_dust(max(0, ndust_processes)))
            reduction_count_dust(:) = 0
        end if

        if (allocated(reduction_count_pah)) then
            if (size(reduction_count_pah) /= npah_processes) then
                deallocate(reduction_count_pah)
            end if
        end if
        if (.not. allocated(reduction_count_pah)) then
            allocate(reduction_count_pah(max(0, npah_processes)))
            reduction_count_pah(:) = 0
        end if
    end subroutine ensure_kmax_storage

    subroutine reset_timestep_reduction_counters
        implicit none

        call ensure_kmax_storage
        if (allocated(reduction_count_dust)) reduction_count_dust(:) = 0
        if (allocated(reduction_count_pah)) reduction_count_pah(:) = 0
    end subroutine reset_timestep_reduction_counters

    subroutine register_timestep_reduction_cause
        implicit none

        integer :: i, max_i
        logical :: is_dust
        real(dp) :: vmax, val

        call ensure_kmax_storage

        vmax = -1d0
        max_i = 0
        is_dust = .true.

        do i = 1, ndust_processes
            val = last_kmax_dust(i)
            if (val > vmax) then
                vmax = val
                max_i = i
                is_dust = .true.
            end if
        end do

        do i = 1, npah_processes
            val = last_kmax_pah(i)
            if (val > vmax) then
                vmax = val
                max_i = i
                is_dust = .false.
            end if
        end do

        if (max_i <= 0 .or. vmax <= 0d0) return

        if (is_dust) then
            reduction_count_dust(max_i) = reduction_count_dust(max_i) + 1
        else
            reduction_count_pah(max_i) = reduction_count_pah(max_i) + 1
        end if
    end subroutine register_timestep_reduction_cause

    subroutine print_last_process_kmax
        implicit none
        integer :: i

        call ensure_kmax_storage

        print *, 'ODE diagnostics: per-process kmax [s^-1] from last RHS evaluation'
        if (ndust_processes > 0) then
            do i = 1, ndust_processes
                print *, '  dust process ', i, ' (', trim(dust_processes_list(i)%name), '): ', last_kmax_dust(i)
            end do
        else
            print *, '  No dust processes active.'
        end if

        if (npah_processes > 0) then
            do i = 1, npah_processes
                print *, '  pah process  ', i, ' (', trim(pah_processes_list(i)%name), '): ', last_kmax_pah(i)
            end do
        else
            print *, '  No PAH processes active.'
        end if
    end subroutine print_last_process_kmax

    subroutine print_timestep_reduction_counters
        implicit none
        integer :: i
        integer :: total_reductions

        call ensure_kmax_storage

        total_reductions = 0
        if (allocated(reduction_count_dust)) total_reductions = total_reductions + sum(reduction_count_dust)
        if (allocated(reduction_count_pah)) total_reductions = total_reductions + sum(reduction_count_pah)

        print *, 'ODE diagnostics: timestep-reduction attributions by dominant process'
        print *, '  Total attributed reductions       = ', total_reductions

        if (ndust_processes > 0) then
            do i = 1, ndust_processes
                print *, '  dust process ', i, ' (', trim(dust_processes_list(i)%name), '): ', reduction_count_dust(i)
            end do
        else
            print *, '  No dust processes active.'
        end if

        if (npah_processes > 0) then
            do i = 1, npah_processes
                print *, '  pah process  ', i, ' (', trim(pah_processes_list(i)%name), '): ', reduction_count_pah(i)
            end do
        else
            print *, '  No PAH processes active.'
        end if
    end subroutine print_timestep_reduction_counters

    subroutine dust_rhs(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,kmax,debug_flag)
        ! Compute the right-hand side of the ODE system for the dust chemistry. This function will be called
        ! by the ODE solver to compute the time derivatives of the gas and dust abundances.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! dydt_gas  <--> 2D array with the time derivative of the gas phase abundances [g cm-3 s-1]
        ! dydt_dust <--> 1D array with the time derivative of the dust phase abundances [g cm-3 s-1]
        ! kmax      <--> Maximum allowed rate for the process (optional output)

        implicit none
        ! ---- Input/Output variables ----
        type(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: dydt_gas(:,:), dydt_dust(:)
        real(dp), intent(inout), optional :: kmax
        logical, intent(in), optional :: debug_flag

        ! ---- Local variables ----
        integer :: i
        real(dp) :: process_kmax

        ! 1. Initialize the time derivatives to zero
        dydt_gas(:,:) = 0.0_dp
        dydt_dust(:) = 0.0_dp

        call ensure_kmax_storage
        if (allocated(last_kmax_dust)) last_kmax_dust(:) = 0.0_dp
        if (allocated(last_kmax_pah)) last_kmax_pah(:) = 0.0_dp

        ! 2. Loop over the dust processes and compute their contribution to the time derivatives
        if (present(kmax)) kmax = 0.0_dp

        do i = 1, ndust_processes
            process_kmax = 0.0_dp
            call dust_processes_list(i)%comp_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,process_kmax)
            last_kmax_dust(i) = process_kmax
            if (present(kmax)) kmax = max(kmax, process_kmax)
        end do

        do i = 1, npah_processes
            process_kmax = 0.0_dp
            call pah_processes_list(i)%comp_rate(dust_info,y_gas,y_dust,dydt_gas,dydt_dust,process_kmax)
            last_kmax_pah(i) = process_kmax
            if (present(kmax)) kmax = max(kmax, process_kmax)
        end do
    end subroutine dust_rhs

end module dust_rhs_mod

module rk4_mod
    use amr_parameters, only: dp
    use dustbin_types, only: DustChemistryInfo
    use dust_commons, only: errmax
    use ode_interface_mod, only: rhs_interface, HALF, TWO, SIXTH

    implicit none
    private
    public :: rk4_step

    contains

    subroutine rk4_raw(dust_info,y_gas,y_dust,h,rhs,y_gas_new,y_dust_new,first_call,break,debug_flag)
        ! Perform a single step of the classical 4th-order Runge-Kutta method (RK4) to solve the ODE system.
        ! This is a "raw" implementation that does not include any adaptive time stepping or error control.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! h        --> Time step size [s]
        ! rhs      --> 2D array with the right-hand side of the ODE system [g cm-3 s-1]
        ! y_gas_new --> 2D array with the updated gas phase abundances [g cm-3]
        ! y_dust_new --> 1D array with the updated dust phase abundances [g cm-3]
        ! first_call --> Logical flag indicating whether this is the first call to the subroutine
        ! break    --> Logical flag indicating whether the integration should be stopped (e.g., if there are no active dust processes)

        implicit none
        ! ---- Input/Output variables ----
        type(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: h
        procedure(rhs_interface) :: rhs
        real(dp), intent(out) :: y_gas_new(:,:), y_dust_new(:)
        logical, intent(in) :: first_call
        logical, intent(out) :: break
        logical, intent(in), optional :: debug_flag

        ! ---- Local variables ----
        real(dp) :: k1_gas(size(y_gas,1),size(y_gas,2))
        real(dp) :: k2_gas(size(y_gas,1),size(y_gas,2))
        real(dp) :: k3_gas(size(y_gas,1),size(y_gas,2))
        real(dp) :: k4_gas(size(y_gas,1),size(y_gas,2))
        real(dp) :: k1_dust(size(y_dust))
        real(dp) :: k2_dust(size(y_dust))
        real(dp) :: k3_dust(size(y_dust))
        real(dp) :: k4_dust(size(y_dust))
        real(dp) :: kmax, h_local
        break = .false.

        ! 1. Perform the four RK4 stages
        if (first_call) then
            ! On the first call, we compute kmax to get an estimate of the necessary time step size for stability.
            if (present(debug_flag)) then
                call rhs(dust_info,y_gas,y_dust,k1_gas,k1_dust,kmax,debug_flag=debug_flag)
            else
                call rhs(dust_info,y_gas,y_dust,k1_gas,k1_dust,kmax)
            end if
            if (kmax.eq.0d0) then
                ! If kmax is zero it means there are no active dust
                ! processes, so there is no point in doing a dust integration.
                break = .true.
                return
            end if
        else
            if (present(debug_flag)) then
                call rhs(dust_info,y_gas,y_dust,k1_gas,k1_dust,debug_flag=debug_flag)
            else
                call rhs(dust_info,y_gas,y_dust,k1_gas,k1_dust)
            end if
        end if

        ! 2. Use the provided kmax to compute a guess of the neccessary
        ! time step size for stability, but never increase the time step 
        ! beyond the provided h
        if (first_call) then
            h_local = min(1d0 / kmax,h)
        else
            h_local = h
        end if
        if (present(debug_flag)) then
            call rhs(dust_info,y_gas+h_local*HALF*k1_gas,y_dust+h_local*HALF*k1_dust,k2_gas,k2_dust,debug_flag=debug_flag)
            call rhs(dust_info,y_gas+h_local*HALF*k2_gas,y_dust+h_local*HALF*k2_dust,k3_gas,k3_dust,debug_flag=debug_flag)
            call rhs(dust_info,y_gas+h_local*k3_gas,y_dust+h_local*k3_dust,k4_gas,k4_dust,debug_flag=debug_flag)
        else
            call rhs(dust_info,y_gas+h_local*HALF*k1_gas,y_dust+h_local*HALF*k1_dust,k2_gas,k2_dust)
            call rhs(dust_info,y_gas+h_local*HALF*k2_gas,y_dust+h_local*HALF*k2_dust,k3_gas,k3_dust)
            call rhs(dust_info,y_gas+h_local*k3_gas,y_dust+h_local*k3_dust,k4_gas,k4_dust)
        end if

        ! 3. Combine the stages to compute the new solution
        y_gas_new(:,:) = y_gas(:,:) + (h_local * SIXTH) * (k1_gas(:,:) + TWO*k2_gas(:,:) + TWO*k3_gas(:,:) + k4_gas(:,:))
        y_dust_new(:) = y_dust(:) + (h_local * SIXTH) * (k1_dust(:) + TWO*k2_dust(:) + TWO*k3_dust(:) + k4_dust(:))

        ! 4. Update the time step size for the next step (this will be used by the adaptive RK4 wrapper)
        h = h_local
    end subroutine rk4_raw

    subroutine rk4_step(dust_info,y_gas,y_dust,h,rhs,y_gas_new,y_dust_new,h_new,first_call,accepted,break,debug_flag)
        ! Perform a single step of the RK4 method with adaptive time stepping and error control.
        ! This is a wrapper around the rk4_raw subroutine that includes logic for adjusting the time step size
        ! based on the estimated error of the solution.
        ! y_gas     --> 2D array with the gas phase abundances [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances [g cm-3]
        ! h        --> Time step size [s]
        ! rhs      --> 2D array with the right-hand side of the ODE system [g cm-3 s-1]
        ! y_gas_new --> 2D array with the updated gas phase abundances [g cm-3]
        ! y_dust_new --> 1D array with the updated dust phase abundances [g cm-3 s-1]
        ! h_new    --> Updated time step size [s]
        ! first_call --> Logical flag indicating whether this is the first call to the subroutine
        ! accepted --> Logical flag indicating whether the step was accepted or not (for adaptive time stepping)
        ! break    --> Logical flag indicating whether the integration should be stopped (e.g., if there are no active dust processes)

        implicit none
        ! ---- Input/Output variables ----
        type(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        real(dp), intent(inout) :: h
        procedure(rhs_interface) :: rhs
        real(dp), intent(out) :: y_gas_new(:,:), y_dust_new(:)
        real(dp), intent(out) :: h_new
        logical, intent(out) :: accepted,break
        logical, intent(in) :: first_call
        logical, intent(in), optional :: debug_flag

        ! ---- Local variables ----
        real(dp) :: y_gas_temp(size(y_gas,1),size(y_gas,2)), y_dust_temp(size(y_dust))
        real(dp) :: error_gas(size(y_gas,1),size(y_gas,2)), error_dust(size(y_dust))
        real(dp) :: max_error, scale

        ! 1. Perform a raw RK4 step to get the new solution
        if (present(debug_flag)) then
            call rk4_raw(dust_info,y_gas,y_dust,h,rhs,y_gas_temp,y_dust_temp,first_call,break,debug_flag=debug_flag)
        else
            call rk4_raw(dust_info,y_gas,y_dust,h,rhs,y_gas_temp,y_dust_temp,first_call,break)
        end if
        if (break) then
            ! If break is true, it means there are no active dust processes, so we can skip the rest of the logic.
            accepted = .true.
            h_new = h
            return
        end if

        ! 2. Max relative change w.r.t. old state
        error_gas(:,:) = abs(y_gas_temp(:,:) - y_gas(:,:)) / max(abs(y_gas(:,:)), 1.0d-40)
        error_dust(:) = abs(y_dust_temp(:) - y_dust(:)) / max(abs(y_dust(:)), 1.0d-40)
        max_error = maxval(error_gas(:,:))
        max_error = max(max_error, maxval(error_dust(:)))

        accepted = (max_error <= errmax)

        ! 3. Update the step size based on the error
        scale = 0.9d0 * (errmax / max(max_error,1.0d-10))
        h_new = h * min(2.0_dp, max(0.1_dp, scale))  ! Limit the change in step size

        ! 4. If the step is accepted, update the solution; otherwise, keep the old solution
        if (accepted) then
            y_gas_new(:,:) = y_gas_temp(:,:)
            y_dust_new(:) = y_dust_temp(:)
        end if
    end subroutine rk4_step
end module rk4_mod

module ode_driver_mod
    use amr_parameters, only: dp
    use dustbin_types, only: DustChemistryInfo
    use dust_commons
    use ode_interface_mod
    use dust_rhs_mod, only: print_last_process_kmax, reset_timestep_reduction_counters
    use dust_rhs_mod, only: register_timestep_reduction_cause, print_timestep_reduction_counters

    implicit none
    private
    public :: integrate_dust_ode

    contains

    subroutine integrate_dust_ode(dust_info,dt,y_gas,y_dust,rhs,step_fn,&
                                    &y_gas_final,y_dust_final,h_init,h_min,h_max,debug_flag)
        ! Integrate the dust ODE system over a time step dt using the provided step function.
        ! dust_info --> DustChemistryInfo type with all the necessary information to compute the accretion rate.
        ! dt        --> Time step size [s]
        ! y_gas     --> 2D array with the gas phase abundances at the beginning of the step [g cm-3]
        ! y_dust    --> 1D array with the dust phase abundances at the beginning of the step [g cm-3]
        ! rhs       --> 2D array with the right-hand side of the ODE system [g cm-3 s-1]
        ! step_fn   --> ODE solver step function that will be called to perform the integration
        ! y_gas_final --> 2D array with the gas phase abundances at the end of the step [g cm-3]
        ! y_dust_final --> 1D array with the dust phase abundances at the end of the step [g cm-3]
        ! h_init    --> Initial guess for the time step size to be used by the ODE solver [s]
        ! h_min     --> Minimum allowed time step size for the ODE solver [s]
        ! h_max     --> Maximum allowed time step size for the ODE solver [s]

        implicit none
        ! ---- Input/Output variables ----
        type(DustChemistryInfo), intent(in) :: dust_info
        real(dp), intent(in) :: dt
        real(dp), intent(in) :: y_gas(:,:), y_dust(:)
        procedure(rhs_interface) :: rhs
        procedure(solver_step_interface) :: step_fn
        real(dp), intent(out) :: y_gas_final(:,:), y_dust_final(:)
        real(dp), intent(in) :: h_init, h_min, h_max
        logical, intent(in), optional :: debug_flag

        ! ---- Local variables ----
        integer :: icount
        integer :: naccepted, nrejected
        integer :: nreduced
        real(dp) :: h, h_new, tau, h_candidate
        real(dp) :: y_gas_temp(size(y_gas,1),size(y_gas,2)), y_dust_temp(size(y_dust))
        real(dp) :: y_gas_new(size(y_gas,1),size(y_gas,2)), y_dust_new(size(y_dust))
        logical :: accepted,break,firstcall
        logical :: debug_enabled

        debug_enabled = dust_log
        if (present(debug_flag)) debug_enabled = debug_flag

        ! 1. Initialize the time step size for the ODE solver
        h = min(min(max(h_init, h_min), h_max),dt)
        tau = 0.0_dp
        icount = 0
        naccepted = 0
        nrejected = 0
        nreduced = 0
        firstcall = .true.

        call reset_timestep_reduction_counters

        y_gas_temp(:,:) = y_gas(:,:)
        y_dust_temp(:) = y_dust(:)

        ! 2. Integrate the ODE system until we have covered the full time step dt
        do while (tau < dt)
            h = min(h, dt - tau)  ! Adjust the time step size to not overshoot the time step

            ! 3. Call the ODE solver step function to perform a single integration step
            if (present(debug_flag)) then
                call step_fn(dust_info,y_gas_temp,y_dust_temp,h,rhs,y_gas_new,y_dust_new,h_new,firstcall,accepted,break,debug_flag=debug_flag)
            else
                call step_fn(dust_info,y_gas_temp,y_dust_temp,h,rhs,y_gas_new,y_dust_new,h_new,firstcall,accepted,break)
            end if
            firstcall = .false.
            if (break) then
                ! If break is true, it means there are no active dust processes, so we can skip
                ! the rest of the integration and just return the initial state.
                y_gas_final(:,:) = y_gas(:,:)
                y_dust_final(:) = y_dust(:)
                return
            end if

            ! 4. If the step was accepted, update tau and the solution
            if (accepted) then
                tau  = tau + h
                y_gas_temp(:,:) = y_gas_new(:,:)
                y_dust_temp(:) = y_dust_new(:)
                naccepted = naccepted + 1
            else
                nrejected = nrejected + 1
            end if

            ! 5. Update the time step size for the next iteration
            h_candidate = min(max(h_new, h_min), h_max)
            if (h_candidate < h) then
                nreduced = nreduced + 1
                call register_timestep_reduction_cause
            end if
            h = h_candidate
            icount = icount + 1
            if (icount > countmax) then
                print *, "Warning: Maximum number of ODE solver iterations reached. Integration may not have converged."
                print *, '  Requested total dt [s]        = ', dt
                print *, '  Integrated tau [s]            = ', tau
                print *, '  Last attempted timestep h [s] = ', h
                print *, '  Last proposed h_new [s]       = ', h_new
                print *, '  Accepted steps                = ', naccepted
                print *, '  Rejected steps                = ', nrejected
                print *, '  Number of timestep reductions = ', nreduced
                call print_last_process_kmax
                call print_timestep_reduction_counters
                call clean_stop
            end if
        end do

        ! 6. Set the final solution
        y_gas_final(:,:) = y_gas_temp(:,:)
        y_dust_final(:) = y_dust_temp(:)

        if (debug_enabled) then
            if (any(y_gas_final < 0.0_dp) .or. any(y_dust_final < 0.0_dp)) then
                print *, 'DEBUG integrate_dust_ode: negative final density detected.'
                call clean_stop
            end if
        end if
    end subroutine integrate_dust_ode

end module ode_driver_mod