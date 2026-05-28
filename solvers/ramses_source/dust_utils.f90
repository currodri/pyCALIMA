module dust_utils
    use amr_parameters, only:dp
    use constants
    use safe_math, only: safe_exp, safe_erf
    contains

    subroutine cmp_sigma_turb(icell,sigma2,ilevel)
        use amr_commons
        use hydro_commons
        implicit none
        integer::ilevel,icell,ncell
        real(dp)::d,d1,d2,d3,d4,d5,d6,ul,ur
        real(dp)::sigma2,sigma2_comp,sigma2_sole
        integer ,dimension(1:nvector)::ind_cell2
        integer ,dimension(1:nvector,0:twondim)::ind_nbor
      
        ! We need to estimate the norm of the gradient of the velocity field in the cell (tensor of 2nd rank)
        ! i.e. || A ||^2 = trace( A A^T) where A = grad vec(v) is the tensor.
        ! So construct values of velocity field on the 6 faces of the cell using simple linear interpolation
        ! from neighbouring cell values and differentiate.
        ! Get neighbor cells if they exist, otherwise use straight injection from local cell
        ncell = 1 ! we just want the neighbors of that cell
        ind_cell2(1)=icell
        d=uold(icell,1)
        call getnbor(ind_cell2,ind_nbor,ncell,ilevel)
        d1           = uold(ind_nbor(1,1),1) ; d2 = uold(ind_nbor(1,2),1) ; d3 = uold(ind_nbor(1,3),1)
        d4           = uold(ind_nbor(1,4),1) ; d5 = uold(ind_nbor(1,5),1) ; d6 = uold(ind_nbor(1,6),1)
        sigma2       = 0d0 ; sigma2_comp = 0d0 ; sigma2_sole = 0d0
        !!!!!!!!!!!!!!!!!!
        ! Divergence terms
        !!!!!!!!!!!!!!!!!!
        ul        = (uold(ind_nbor(1,2),2) + uold(icell,2))/(d2+d)
        ur        = (uold(ind_nbor(1,1),2) + uold(icell,2))/(d1+d)
        sigma2_comp = sigma2_comp + (ur-ul)**2
        ul        = (uold(ind_nbor(1,4),3) + uold(icell,3))/(d4+d)
        ur        = (uold(ind_nbor(1,3),3) + uold(icell,3))/(d3+d)
        sigma2_comp = sigma2_comp + (ur-ul)**2
        ul        = (uold(ind_nbor(1,6),4) + uold(icell,4))/(d6+d)
        ur        = (uold(ind_nbor(1,5),4) + uold(icell,4))/(d5+d)
        sigma2_comp = sigma2_comp + (ur-ul)**2
        !!!!!!!!!!!!
        ! Curl terms
        !!!!!!!!!!!!
        ul        = (uold(ind_nbor(1,6),3) + uold(icell,3))/(d6+d)
        ur        = (uold(ind_nbor(1,5),3) + uold(icell,3))/(d5+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        ul        = (uold(ind_nbor(1,4),4) + uold(icell,4))/(d4+d)
        ur        = (uold(ind_nbor(1,3),4) + uold(icell,4))/(d3+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        ul        = (uold(ind_nbor(1,6),2) + uold(icell,2))/(d6+d)
        ur        = (uold(ind_nbor(1,5),2) + uold(icell,2))/(d5+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        ul        = (uold(ind_nbor(1,2),4) + uold(icell,4))/(d2+d)
        ur        = (uold(ind_nbor(1,1),4) + uold(icell,4))/(d1+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        ul        = (uold(ind_nbor(1,4),2) + uold(icell,2))/(d4+d)
        ur        = (uold(ind_nbor(1,3),2) + uold(icell,2))/(d3+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        ul        = (uold(ind_nbor(1,2),3) + uold(icell,3))/(d2+d)
        ur        = (uold(ind_nbor(1,1),3) + uold(icell,3))/(d1+d)
        sigma2_sole = sigma2_sole + (ur-ul)**2
        sigma2    = sigma2_comp+sigma2_sole
        
    end subroutine cmp_sigma_turb

    subroutine size2char(a,char)
        implicit none
        real(dp), intent(in) :: a
        character(7),intent(inout) :: char

        write(char,'(F6.4)') a
    end subroutine size2char
    
    function locate(xx,n,x)
        ! Locates position j of a value x in an ordered array xx of n elements
        ! After: xx(j) <= x <= xx(j+1) (assuming increasing order)
        ! j is lower bound, so it can be zero but not larger than n
        !-------------------------------------------------------------------------
        use amr_commons,only:dp
        integer ::  n,j,jl,ju,jm
        integer :: locate
        real(dp)::  xx(n),x
        !-------------------------------------------------------------------------
        jl = 0
        ju = n+1
        do while (ju-jl > 1) 
            jm = (ju+jl)/2
            if ((xx(n) > xx(1)) .eqv. (x > xx(jm))) then
                jl = jm
            else
                ju = jm
            endif
        enddo
        j = jl
        if (j.eq.0)   j = 1
        if (j.ge.n)   j = n - 1
        locate = j
    end function locate

    function locate_eqw(xx,n,x)
        ! Fast locator for equally spaced ordered arrays.
        ! Returns lower index j such that x is bracketed by xx(j), xx(j+1)
        ! and clamps to edges: j in [1, n-1].
        implicit none
        integer, intent(in) :: n
        real(dp), intent(in) :: xx(n), x
        integer :: locate_eqw
        integer :: j
        real(dp) :: dx, invdx

        if (n <= 2) then
            locate_eqw = 1
            return
        end if

        dx = (xx(n) - xx(1)) / dble(n - 1)
        if (dx == 0d0) then
            locate_eqw = 1
            return
        end if
        invdx = 1d0 / dx

        if (dx > 0d0) then
            if (x <= xx(1)) then
                locate_eqw = 1
                return
            else if (x >= xx(n)) then
                locate_eqw = n - 1
                return
            end if
            j = int((x - xx(1)) * invdx) + 1
        else
            if (x >= xx(1)) then
                locate_eqw = 1
                return
            else if (x <= xx(n)) then
                locate_eqw = n - 1
                return
            end if
            j = int((xx(1) - x) / abs(dx)) + 1
        end if

        if (j < 1) j = 1
        if (j >= n) j = n - 1
        locate_eqw = j
    end function locate_eqw

    subroutine interpolate1D(x, results, ni, xi, interp_val, non_eqw)
        implicit none
        integer, intent(in) :: ni
        real(dp), intent(in) :: x(ni)
        real(dp), intent(in) :: results(ni)
        real(dp), intent(in) :: xi
        real(dp), intent(out) :: interp_val
        logical, intent(in), optional :: non_eqw
        integer :: i
        real(dp) :: x1, x2
        logical :: use_eqw
        ! If xi is outside bounds, return continuation at the limits (no extrapolation)
        if (xi <= x(1)) then
            interp_val = results(1)
            return
        else if (xi >= x(ni)) then
            interp_val = results(ni)
            return
        end if

        ! Find index for interpolation (xi now within bounds)
        use_eqw = .true.
        if (present(non_eqw)) use_eqw = .not. non_eqw
        if (use_eqw) then
            i = locate_eqw(x, ni, xi)
        else
            i = locate(x, ni, xi)
        end if

        ! Get bounding values
        x1 = x(i)
        x2 = x(i+1)

        ! Perform linear interpolation
        interp_val = (x2 - xi) / (x2 - x1) * results(i) + (xi - x1) / (x2 - x1) * results(i+1)
    end subroutine interpolate1D


    subroutine interpolate2D(x, y, results, ni, nj, xi, yi, interp_val, non_eqw)
        implicit none
        integer, intent(in) :: ni, nj
        real(dp), intent(in) :: x(ni), y(nj)
        real(dp), intent(in) :: results(ni, nj)
        real(dp), intent(in) :: xi, yi
        real(dp), intent(out) :: interp_val
        logical, intent(in), optional :: non_eqw
        integer :: i, j
        real(dp) :: x1, x2, y1, y2
        real(dp) :: c0, c1
        real(dp) :: xi_copy, yi_copy
        logical :: use_eqw

        ! Copy and clamp coordinates to domain limits so we continue at edges (no extrapolation)
        xi_copy = xi
        yi_copy = yi
        if (xi_copy < x(1)) xi_copy = x(1)
        if (xi_copy > x(ni)) xi_copy = x(ni)
        if (yi_copy < y(1)) yi_copy = y(1)
        if (yi_copy > y(nj)) yi_copy = y(nj)

        ! Find indices for interpolation using clamped coordinates
        use_eqw = .true.
        if (present(non_eqw)) use_eqw = .not. non_eqw
        if (use_eqw) then
            i = locate_eqw(x, ni, xi_copy)
            j = locate_eqw(y, nj, yi_copy)
        else
            i = locate(x, ni, xi_copy)
            j = locate(y, nj, yi_copy)
        end if

        ! Ensure indices are within bounds
        if (i < 1) i = 1
        if (i >= ni) i = ni - 1
        if (j < 1) j = 1
        if (j >= nj) j = nj - 1

        ! Get bounding values
        x1 = x(i)
        x2 = x(i+1)
        y1 = y(j)
        y2 = y(j+1)

        ! Perform bilinear interpolation (with clamped coordinates)
        c0 = (x2 - xi_copy) / (x2 - x1) * results(i, j) + (xi_copy - x1) / (x2 - x1) * results(i+1, j)
        c1 = (x2 - xi_copy) / (x2 - x1) * results(i, j+1) + (xi_copy - x1) / (x2 - x1) * results(i+1, j+1)

        interp_val = (y2 - yi_copy) / (y2 - y1) * c0 + (yi_copy - y1) / (y2 - y1) * c1
        ! print*,'DEBUG: xi=',xi,' yi=',yi,' i=',i,' j=',j,' x1=',x1,' x2=',x2,' y1=',y1,' y2=',y2,' c0=',c0,' c1=',c1,' interp_val=',interp_val
    end subroutine interpolate2D

    subroutine interpolate3D(x,y,z,results,ni,nj,nk,xi,yi,zi,interp_val,non_eqw)
        implicit none
        integer, intent(in) :: ni, nj, nk
        real(dp), intent(in) :: x(ni), y(nj), z(nk)
        real(dp), intent(in) :: results(ni, nj, nk)
        real(dp), intent(in) :: xi, yi, zi
        real(dp), intent(out) :: interp_val
        logical, intent(in), optional :: non_eqw
        integer :: i, j, k
        real(dp) :: x1, x2, y1, y2, z1, z2
        real(dp) :: c00, c01, c10, c11, c0, c1
        real(dp) :: xi_copy, yi_copy, zi_copy
        logical :: use_eqw

        ! Copy and clamp values so we continue at the domain limits (no extrapolation)
        xi_copy = xi
        yi_copy = yi
        zi_copy = zi
        if (xi_copy < x(1)) xi_copy = x(1)
        if (xi_copy > x(ni)) xi_copy = x(ni)
        if (yi_copy < y(1)) yi_copy = y(1)
        if (yi_copy > y(nj)) yi_copy = y(nj)
        if (zi_copy < z(1)) zi_copy = z(1)
        if (zi_copy > z(nk)) zi_copy = z(nk)

        ! Find indices for interpolation using clamped coordinates
        use_eqw = .true.
        if (present(non_eqw)) use_eqw = .not. non_eqw
        if (use_eqw) then
            i = locate_eqw(x, ni, xi_copy)
            j = locate_eqw(y, nj, yi_copy)
            k = locate_eqw(z, nk, zi_copy)
        else
            i = locate(x, ni, xi_copy)
            j = locate(y, nj, yi_copy)
            k = locate(z, nk, zi_copy)
        end if

        ! Ensure indices are within bounds
        if (i < 1) i = 1
        if (i >= ni) i = ni - 1
        if (j < 1) j = 1
        if (j >= nj) j = nj - 1
        if (k < 1) k = 1
        if (k >= nk) k = nk - 1


        ! Get bounding values
        x1 = x(i)
        x2 = x(i+1)
        y1 = y(j)
        y2 = y(j+1)
        z1 = z(k)
        z2 = z(k+1)


        ! Perform trilinear interpolation
        c00 = (x2 - xi_copy) / (x2 - x1) * results(i, j, k) + (xi_copy - x1) / (x2 - x1) * results(i+1, j, k)
        c01 = (x2 - xi_copy) / (x2 - x1) * results(i, j, k+1) + (xi_copy - x1) / (x2 - x1) * results(i+1, j, k+1)
        c10 = (x2 - xi_copy) / (x2 - x1) * results(i, j+1, k) + (xi_copy - x1) / (x2 - x1) * results(i+1, j+1, k)
        c11 = (x2 - xi_copy) / (x2 - x1) * results(i, j+1, k+1) + (xi_copy - x1) / (x2 - x1) * results(i+1, j+1, k+1)

        c0 = (y2 - yi_copy) / (y2 - y1) * c00 + (yi_copy - y1) / (y2 - y1) * c10
        c1 = (y2 - yi_copy) / (y2 - y1) * c01 + (yi_copy - y1) / (y2 - y1) * c11

        interp_val = (z2 - zi_copy) / (z2 - z1) * c0 + (zi_copy - z1) / (z2 - z1) * c1
    end subroutine interpolate3D

    subroutine read_next_data_line(iunit, out_line, io_status)
        implicit none
        integer, intent(in) :: iunit
        character(len=*), intent(out) :: out_line
        integer, intent(out) :: io_status

        do
            read(iunit,'(A)',iostat=io_status) out_line
            if (io_status /= 0) return
            out_line = adjustl(out_line)
            if (len_trim(out_line) == 0) cycle
            if (out_line(1:1) == '#') cycle
            return
        end do
    end subroutine read_next_data_line

    subroutine replace_pipe_with_space(str)
        implicit none
        character(len=*), intent(inout) :: str
        integer :: ipos

        ipos = index(str, '|')
        do while (ipos > 0)
            str(ipos:ipos) = ' '
            ipos = index(str, '|')
        end do
    end subroutine replace_pipe_with_space

    function sigmoid_function(k,x0,x)
        ! Sigmoid function, useful to smooth dust functions
        ! k     => steepness of transition
        ! x0    => transition central value
        ! x     => x value for function
        implicit none
        real(dp), intent(in) :: k,x0,x

        real(dp)             :: sigmoid_function
        real(dp)             :: normx

        normx = x / x0
        sigmoid_function = 1d0 / (1d0 + exp(-k*(normx-1d0)))
    end function sigmoid_function

    pure elemental function sticking_probability_from_velocity(v_rel, v_thresh, width_frac) result(p_stick)
        ! Smooth sticking probability using a Maxwellian relative-speed distribution.
        ! The sticking probability is P(v < v_thresh), i.e. the Maxwell CDF.
        ! v_rel      => deterministic relative speed [cm s-1]
        ! v_thresh   => coagulation threshold speed [cm s-1]
        ! width_frac => fractional transition width around v_thresh (optional)
        implicit none
        real(dp), intent(in) :: v_rel, v_thresh
        real(dp), intent(in), optional :: width_frac
        real(dp) :: p_stick
        real(dp) :: sigma_v, frac_loc
        real(dp) :: ratio, ratio2, arg
        real(dp), parameter :: frac_default = 1d-1
        real(dp), parameter :: frac_min = 1d-4
        real(dp), parameter :: frac_max = 5d-1
        real(dp), parameter :: arg_hi = 12d0
        real(dp), parameter :: tiny_v = 1d-40
        real(dp), parameter :: inv_sqrt2 = 7.071067811865475d-1
        real(dp), parameter :: sqrt_2_over_pi = 7.978845608028654d-1

        frac_loc = frac_default
        if (present(width_frac)) frac_loc = width_frac
        frac_loc = min(max(frac_loc, frac_min), frac_max)

        sigma_v = max(v_rel, v_thresh, tiny_v) * frac_loc
        ratio = v_thresh / sigma_v
        arg = ratio * inv_sqrt2
        ratio2 = ratio * ratio

        if (arg <= 0d0) then
            p_stick = 0d0
        else if (arg >= arg_hi) then
            p_stick = 1d0
        else
            p_stick = safe_erf(arg) - sqrt_2_over_pi * ratio * safe_exp(-5d-1 * ratio2)
            p_stick = min(max(p_stick, 0d0), 1d0)
        end if
    end function sticking_probability_from_velocity

    function planck_function(wavelength, T) result(emittance)
        use constants, only: hplanck, c_cgs, kB
        use safe_math, only: safe_exp
        ! This function computes the Planck function for a given wavelength and temperature
        ! The Planck function is given by:
        ! B_lambda = (2 * h * c^2 / lambda^5) / (exp(h * c / (lambda * k * T)) - 1)
        ! where h is the Planck constant, c is the speed of light, lambda is the wavelength,
        ! k is the Boltzmann constant, and T is the temperature.
        ! wavelength => Wavelength in cm
        ! T          => Temperature in K
        ! emittance  => Emittance in erg/s/cm^2/cm/steradian
        implicit none
        ! Input arguments
        real(dp), intent(in) :: wavelength   ! Wavelength in cm
        real(dp), intent(in) :: T            ! Temperature in K
        ! Output
        real(dp) :: emittance                ! Emittance in erg/s/cm^2/cm/steradian
        ! Local variables
        real(dp) :: exponent

        ! Compute the Planck function
        exponent = hplanck * c_cgs / (wavelength * kB * T)
        emittance = (2.0d0 * hplanck * c_cgs**2 / wavelength**5) / (safe_exp(exponent) - 1.0d0)
    end function planck_function

    function planck_function_derivative(wavelength, T) result(derivative)
        use constants, only: hplanck, c_cgs, kB
        use safe_math, only: safe_exp
        ! This function computes the derivative of the Planck function for a given wavelength and temperature
        ! The derivative of the Planck function is given by:
        ! dB_lambda/dT = B_lambda^2 * (h * c / lambda) / (k * T^2)
        ! where h is the Planck constant, c is the speed of light, lambda is the wavelength,
        ! k is the Boltzmann constant, and T is the temperature.
        ! wavelength => Wavelength in cm
        ! T          => Temperature in K
        ! derivative => Derivative in erg/s/cm^2/cm/steradian/K
        implicit none
        ! Input arguments
        real(dp), intent(in) :: wavelength   ! Wavelength in cm
        real(dp), intent(in) :: T            ! Temperature in K
        ! Output
        real(dp) :: derivative               ! Derivative in erg/s/cm^2/cm/steradian/K
        ! Local variables
        real(dp) :: exponent, expo, prefactor

        ! Compute the Planck function derivative
        exponent = hplanck * c_cgs / (wavelength * kB * T)
        expo = safe_exp(exponent)
        prefactor = 2.0d0 * hplanck * c_cgs**2 / wavelength**5
        derivative = prefactor * expo * exponent / (T * (expo - 1.0d0)**2)
        if (isnan(derivative)) derivative = 0d0
    end function planck_function_derivative

    function a_to_Nc(a) result(Nc)
        ! Converts PAH size in cm to number of carbon atoms using the relation:
        ! Nc = 468 * (a / 1e-7 cm)^3
        implicit none
        real(dp), intent(in) :: a   ! PAH size in cm
        integer :: Nc             ! Number of carbon atoms

        Nc = int(468d0 * (a / 1d-7)**3)
    end function a_to_Nc

    function Nc_to_a(Nc) result(a)
        ! Converts number of carbon atoms in a PAH to its size in cm using the relation:
        ! a = 1e-7 cm * (Nc / 468)^(1/3)
        implicit none
        integer, intent(in) :: Nc  ! Number of carbon atoms
        real(dp) :: a               ! PAH size in cm

        a = 1d-7 * (real(Nc, dp) / 468d0)**(1d0/3d0)
    end function Nc_to_a

    function Nc_to_Nh(Nc) result(Nh)
        ! Converts number of carbon atoms in a PAH
        ! to the hydrogenated number of hydrogen atoms
        implicit none
        integer, intent(in) :: Nc  ! Number of carbon atoms
        integer :: Nh              ! Number of hydrogen atoms

        if (Nc <= 25) then
            Nh = int(0.5d0 * Nc + 0.5d0)
        else if (Nc <= 100) then
            Nh = int(2.5d0 * sqrt(real(Nc, dp)) + 0.5d0)
        else
            Nh = int(0.25d0 * real(Nc, dp) + 0.5d0)
        end if
    end function Nc_to_Nh

    function Nc_to_mass(Nc) result(mass)
        ! Converts number of carbon atoms in a PAH to its mass in grams
        implicit none
        integer, intent(in) :: Nc  ! Number of carbon atoms

        integer :: Nh              ! Number of hydrogen atoms
        real(dp) :: mass           ! Mass in grams

        Nh = Nc_to_Nh(Nc)
        mass = (real(Nc, dp) * mC_amu + real(Nh, dp) * mH_amu) * amu2g
    end function Nc_to_mass
end module dust_utils