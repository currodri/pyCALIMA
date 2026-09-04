module dust_charging

    use dust_commons
    use dust_utils

    private
    public :: compute_mean_dust_charge, compute_dust_charge_sigma,&
              compute_dust_charge_dist, compute_Coulomb_focusing,&
              three_point_charge_mix, two_point_charge_mix

    contains

    subroutine compute_charge_moments_Ibanez2019(G0,Tgas,ne,ispecie,agrain,Z_avg,dist_sigma)
        ! Compute mean and sigma of the charge distribution from the fitting
        ! functions of Ibanez-Mejia et al. (2019).
        implicit none

        integer, intent(in) :: ispecie
        real(dp), intent(in) :: G0,Tgas,ne,agrain
        real(dp), intent(inout) :: Z_avg,dist_sigma

        integer :: isize
        real(dp),dimension(1:7) :: Ibanez2019_sizes = (/3.5d-4,5d-4,1d-3,5d-3,1d-2,5d-2,1d-1/) ! in microns
        real(dp),dimension(1:2,1:7) :: alpha,k,b,hz,cplus,etaplus,d,cminus,etaminus
        real(dp) :: charPar

        ! Parameters from Table 1 in Ibanez-Mejia et al. (2019)
        alpha = transpose(reshape((/0.4699d0,0.4386d0,0.4994d0,0.6009d0,0.2900d0,0.3400d0,0.3500d0,&
                                    &0.3263d0,0.3141d0,0.3535d0,0.5115d0,0.3525d0,0.3643d0,0.3927d0/),[7,2]))
        k = transpose(reshape((/0.0085d0,0.0195d0,0.0199d0,0.0523d0,2.2310d0,5.8944d0,9.6536d0,&
                                &0.0149d0,0.0372d0,0.0494d0,0.0717d0,0.6591d0,2.6283d0,3.6493d0/),[7,2]))
        b = transpose(reshape((/-0.1162d0,-0.3084d0,-0.4959d0,-0.4092d0,-0.2061d0,0.1727d0,0.4183d0,&
                                &-0.1212d0,-0.3043d0,-0.4865d0,-0.4106d0,-0.1649d0,0.5217d0,0.8389d0/),[7,2]))
        hz = transpose(reshape((/48d0,95d0,78d0,218d0,1063d0,1034d0,1273d0,&
                                &57d0,86d0,73d0,107d0,384d0,345d0,372d0/),[7,2]))
        cplus = transpose(reshape((/0.3103d0,0.3699d0,0.6511d0,1.6536d0,2.5445d0,5.9455d0,8.7003d0,&
                                &0.4123d0,0.2734d0,0.4353d0,1.0758d0,1.6245d0,4.0732d0,5.9813d0/),[7,2]))
        etaplus = transpose(reshape((/0.2744d0,0.5654d0,0.9839d0,2.6688d0,4.3352d0,18.3186d0,36.1014d0,&
                                &0.2513d0,0.2925d0,0.7459d0,1.7832d0,2.8390d0,11.0200d0,20.6410d0/),[7,2]))
        d = transpose(reshape((/0.2551d0,0.4158d0,0.5275d0,0.6671d0,0.7010d0,0.8377d0,0.9094d0,&
                                &0.1891d0,0.3233d0,0.4451d0,0.5860d0,0.6346d0,0.6797d0,0.6961d0/),[7,2]))
        cminus = transpose(reshape((/0.3766d0,0.2890d0,-0.0213d0,-9.5138d0,-2.5341d3,-2.4189d3,-2.6009d3,&
                                &0.4845d0,0.3615d0,0.1053d0,-1.0379d3,-4.2075d2,-0.2418d0,-0.1885d0/),[7,2]))
        etaminus = transpose(reshape((/0.5241d0,1.6241d0,0.0977d0,35.3519d0,8.1962d3,4.9424d3,4.7029d3,&
                                &0.3532d0,0.6532d0,0.5803d0,7.7069d3,1.9840d3,0.5910d0,0.4237d0/),[7,2]))

        charPar = G0 * sqrt(Tgas) / ne

        ! TODO: This should be determine from the input sizes, because we may
        ! have grain sizes not in the results of Ibanez-Mejia et al. (2019)
        isize = minloc(abs(Ibanez2019_sizes-agrain),1)

        ! Eq. 17-19 in Ibanez-Mejia et al. (2019)
        Z_avg = k(ispecie,isize) * (1d0 - exp(-charPar / hz(ispecie,isize))) * (charPar**alpha(ispecie,isize)) + b(ispecie,isize)
        if (Z_avg>0d0) then
            dist_sigma = cplus(ispecie,isize) * (1d0 - exp(-Z_avg / etaplus(ispecie,isize))) + d(ispecie,isize)
        else
            dist_sigma = cminus(ispecie,isize) * (1d0 - exp(-abs(Z_avg) / etaminus(ispecie,isize))) + d(ispecie,isize)
        end if
    end subroutine compute_charge_moments_Ibanez2019

    subroutine compute_dust_charge_dist_Ibanez2019(G0,Tgas,ne,ispecie,agrain,Zdust,fcharge)
        ! ====== CHARGE DISTRIBUTION ======
        ! This is computed from the parametric fitting results of
        ! Ibanez-Mejia et al. (2019) - 
        ! (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
        ! What this assumes, and is shown in this work to be a pretty good
        ! assumption, is that charging is a very quick process, much faster
        ! than typical ISM/hydrodynamical scales
        use constants, only: sq2pi
        implicit none
        
        integer, intent(in) :: ispecie
        real(dp), intent(in) :: G0,Tgas,ne, agrain
        real(dp), dimension(:), allocatable, intent(inout) :: Zdust
        real(dp), dimension(:), allocatable, intent(inout) :: fcharge

        integer :: j
        integer :: Zmin,Zmax
        real(dp) :: Z_avg,dist_sigma

        call compute_charge_moments_Ibanez2019(G0,Tgas,ne,ispecie,agrain,Z_avg,dist_sigma)

        ! Compute approx. min and max of distribution by considering the points
        ! 3 sigma away from the mean (also, charge should be the nearest integer value)
        Zmin = nint(Z_avg - 3 * dist_sigma)
        Zmax = nint(Z_avg + 3 * dist_sigma)
        if (allocated(Zdust)) deallocate(Zdust)
        if (allocated(fcharge)) deallocate(fcharge)
        allocate(Zdust(1:(Zmax-Zmin+1)))
        allocate(fcharge(1:(Zmax-Zmin+1)))
        ! And now compute charge values and the Gaussian distribution
        do j=1,Zmax-Zmin+1
            Zdust(j) = dble(Zmin + j - 1)
            fcharge(j) = (1d0 / (dist_sigma * sq2pi)) * exp(-0.5d0*((Zdust(j) - Z_avg) / dist_sigma)**2)
        end do
        ! Renormalise distribution to make sure it adds to 1
        fcharge(:) = fcharge(:) / sum(fcharge(:))

    end subroutine compute_dust_charge_dist_Ibanez2019

    subroutine compute_mean_dust_charge_Ibanez2019(G0,Tgas,ne,ispecie,agrain,Zdust)
        ! ====== Mean Dust Charge ======
        ! This is computed from the parametric fitting results of
        ! Ibanez-Mejia et al. (2019) - 
        ! (https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1220I/abstract)
        ! What this assumes, and is shown in this work to be a pretty good
        ! assumption, is that charging is a very quick process, much faster
        ! than typical ISM/hydrodynamical scales
        implicit none
        
        integer, intent(in) :: ispecie
        real(dp), intent(in) :: G0,Tgas,ne, agrain
        real(dp), intent(inout) :: Zdust

        real(dp) :: dist_sigma

        call compute_charge_moments_Ibanez2019(G0,Tgas,ne,ispecie,agrain,Zdust,dist_sigma)
        Zdust = idnint(Zdust)  ! Should be the nearest integer value
    end subroutine compute_mean_dust_charge_Ibanez2019

    subroutine compute_mean_dust_charge(i_dust,G0,Tgas,ne,Zdust)
        ! ====== Mean Dust Charge ======
        ! This subroutine computes the mean dust charge following inteporlation
        ! of the equilibrium distribution properties presented in Rodríguez Montero
        ! et al. (2026), which in turn is based on the model from Weingartner &
        ! Draine (2001)
        implicit none
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: G0,Tgas,ne
        real(dp), intent(inout) :: Zdust

        real(dp) :: lgamma,lT
        real(dp) :: Zsigma
        integer :: ispecie

        if (trim(charging_model).eq.'Ibanez2019') then
            if (dustbins_props(i_dust)%separate_refractive_index) then
                ispecie = 1 ! Carbonaceous grains
            else
                ispecie = 2
            end if
            call compute_charge_moments_Ibanez2019(G0,Tgas,ne,ispecie,dustbins_props(i_dust)%asize,Zdust,Zsigma)
            return
        end if

        ! 1. Compute charging parameter
        lgamma = log10(G0 * sqrt(Tgas) / ne)
        lT = log10(Tgas)

        ! 2. Interpolate the pre-computed per-grain table
        if (.not. dustbins_props(i_dust)%mean_charg_tab%initialised) then
            Zdust = 0d0
            return
        end if
        call interpolate2D(dustbins_props(i_dust)%mean_charg_tab%tab1d(1:dustbins_props(i_dust)%mean_charg_tab%npts(1),1), &
                           dustbins_props(i_dust)%mean_charg_tab%tab1d(1:dustbins_props(i_dust)%mean_charg_tab%npts(2),2), &
                           dustbins_props(i_dust)%mean_charg_tab%tab2d(1:dustbins_props(i_dust)%mean_charg_tab%npts(1),1:dustbins_props(i_dust)%mean_charg_tab%npts(2),1), &
                           dustbins_props(i_dust)%mean_charg_tab%npts(1), dustbins_props(i_dust)%mean_charg_tab%npts(2), lgamma, lT, Zdust)
    end subroutine compute_mean_dust_charge

    subroutine compute_dust_charge_sigma(i_dust,G0,Tgas,ne,Zsigma)
        ! ====== Dust Charge Sigma ======
        ! This subroutine computes the dust charge sigma following inteporlation
        ! of the equilibrium distribution properties presented in Rodríguez Montero
        ! et al. (2026), which in turn is based on the model from Weingartner &
        ! Draine (2001)
        implicit none
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: G0,Tgas,ne
        real(dp), intent(inout) :: Zsigma

        real(dp) :: lgamma,lT
        real(dp) :: Z_avg
        integer :: ispecie

        if (trim(charging_model).eq.'Ibanez2019') then
            if (dustbins_props(i_dust)%separate_refractive_index) then
                ispecie = 1 ! Carbonaceous grains
            else
                ispecie = 2
            end if
            call compute_charge_moments_Ibanez2019(G0,Tgas,ne,ispecie,dustbins_props(i_dust)%asize,Z_avg,Zsigma)
            return
        end if

        ! 1. Compute charging parameter
        lgamma = log10(G0 * sqrt(Tgas) / ne)
        lT = log10(Tgas)

        ! 2. Interpolate the pre-computed per-grain table
        if (.not. dustbins_props(i_dust)%sigma_charg_tab%initialised) then
            Zsigma = 1d0
            return
        end if

        call interpolate2D(dustbins_props(i_dust)%sigma_charg_tab%tab1d(1:dustbins_props(i_dust)%sigma_charg_tab%npts(1),1), &
                           dustbins_props(i_dust)%sigma_charg_tab%tab1d(1:dustbins_props(i_dust)%sigma_charg_tab%npts(2),2), &
                           dustbins_props(i_dust)%sigma_charg_tab%tab2d(1:dustbins_props(i_dust)%sigma_charg_tab%npts(1),1:dustbins_props(i_dust)%sigma_charg_tab%npts(2),1), &
                           dustbins_props(i_dust)%sigma_charg_tab%npts(1), dustbins_props(i_dust)%sigma_charg_tab%npts(2), lgamma, lT, Zsigma)

    end subroutine compute_dust_charge_sigma

    subroutine compute_dust_charge_dist(i_dust,G0,Tgas,ne,Zdust,fcharge)
        ! ====== CHARGE DISTRIBUTION ======
        ! This subroutine computes the dust charge distribution following inteporlation
        ! of the equilibrium distribution properties presented in Rodríguez Montero
        ! et al. (2026), which in turn is based on the model from Weingartner &
        ! Draine (2001)
        use constants, only: sq2pi
        implicit none
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: G0,Tgas,ne
        real(dp), dimension(:), allocatable, intent(inout) :: Zdust
        real(dp), dimension(:), allocatable, intent(inout) :: fcharge

        integer :: j,kk,isize
        integer :: Zmin,Zmax
        real(dp) :: gamma,Z_avg,Zsigma

        ! 1. Compute charging parameter
        gamma = G0 * sqrt(Tgas) / ne

        ! 2. Get the interpolated mean and sigma of the distribution
        call compute_mean_dust_charge(i_dust,G0,Tgas,ne,Z_avg)
        call compute_dust_charge_sigma(i_dust,G0,Tgas,ne,Zsigma)

        ! 3. Compute approx. min and max of distribution by considering the points
        ! 3 sigma away from the mean (also, charge should be the nearest integer value)
        Zmin = nint(Z_avg - 3 * Zsigma)
        Zmax = nint(Z_avg + 3 * Zsigma)
        if (allocated(Zdust)) deallocate(Zdust)
        if (allocated(fcharge)) deallocate(fcharge)
        allocate(Zdust(1:(Zmax-Zmin+1)))
        allocate(fcharge(1:(Zmax-Zmin+1)))
        ! And now compute charge values and the Gaussian distribution
        do j=1,Zmax-Zmin+1
            Zdust(j) = dble(Zmin + j - 1)
            fcharge(j) = (1d0 / (Zsigma * sq2pi)) * exp(-0.5d0*((Zdust(j) - Z_avg) / Zsigma)**2)
        end do
        ! Renormalise distribution to make sure it adds to 1
        fcharge(:) = fcharge(:) / sum(fcharge(:))
    end subroutine compute_dust_charge_dist

    subroutine compute_Coulomb_focusing(Tgas,agrain,fcharge,Zdust,Zion,D_Coulomb)
        ! ====== Coulomb enhancement factor =====
        ! This is based on Eq. 6-7 in Weingartner & Draine (1999) which allows
        ! the computation of the Coulomb enhancement factor from the charge
        ! distribution (https://iopscience.iop.org/article/10.1086/307197)
        ! Remember that this equation is in CGS, so the grain size should be
        ! instead in cm, not in microns
        use cooling_module, only: kB
        use constants, only: pi,e2instatC
        implicit none
        
        real(dp), dimension(:), intent(in) :: fcharge
        real(dp), dimension(:), intent(in) :: Zdust
        real(dp), intent(in) :: Zion,agrain,Tgas
        real(dp), intent(inout) :: D_Coulomb

        integer :: j
        real(dp) :: Zg,Bfact

        D_Coulomb = 0d0
        if (Zion.ne.0d0) then
            ! Loop over the charge distribution, adding each contribution
            do j=1,size(Zdust,1)
                Zg = Zdust(j)
                if (Zg*Zion.gt.0) then
                    Bfact = exp(-Zg*Zion*e2instatC / (kB*Tgas*agrain))
                elseif (Zg*Zion.lt.0) then
                    Bfact = 1d0 - Zg*Zion*e2instatC / (kB*Tgas*agrain)
                elseif (Zg.eq.0) then
                    Bfact = 1d0 + sqrt(pi*Zion**2*e2instatC / (2d0*kB*Tgas*agrain))
                end if
                D_Coulomb = D_Coulomb + fcharge(j) * Bfact
            end do
            D_Coulomb = max(D_Coulomb,1d-10)
        else
            ! In the case of neutral atom, there is no Coulomb focusing
            D_Coulomb = 1d0
        end if

    end subroutine compute_Coulomb_focusing

    subroutine two_point_charge_mix(mu, zmin, zlo, zhi, wlo, whi)
        implicit none
        real(dp), intent(in)  :: mu
        integer,  intent(in)  :: zmin
        integer,  intent(out) :: zlo, zhi
        real(dp), intent(out) :: wlo, whi

        zlo = ifloor(mu)
        zhi = zlo + 1

        whi = mu - real(zlo, dp)
        wlo = 1.0_dp - whi

        ! Enforce physical lower bound exactly as in the Python logic.
        if (zlo < zmin) then
        zlo = zmin
        zhi = zmin
        wlo = 1.0_dp
        whi = 0.0_dp
        else if (zhi < zmin) then
        zlo = zmin
        zhi = zmin
        wlo = 1.0_dp
        whi = 0.0_dp
        end if
    end subroutine two_point_charge_mix


    subroutine three_point_charge_mix(mu, sigma, zmin, z1, z2, z3, w1, w2, w3, used_two_point)
        implicit none
        real(dp), intent(in)  :: mu, sigma
        integer,  intent(in)  :: zmin
        integer,  intent(out) :: z1, z2, z3
        real(dp), intent(out) :: w1, w2, w3
        logical,  intent(out) :: used_two_point

        real(dp), parameter :: tol = 1.0d-10
        real(dp) :: sig, m2_target
        integer  :: half_span, zl, zh
        integer  :: i, j, k
        real(dp) :: a, b, c, a2, b2, c2
        real(dp) :: da, db, da2, db2, rhs1, rhs2, det
        real(dp) :: ww1, ww2, ww3
        real(dp) :: score_span, score_mid
        real(dp) :: best_span, best_mid
        logical  :: found_nonneg

        ! Safe sigma handling.
        sig = sigma
        if (sig /= sig) sig = 0.0_dp
        if (sig < 0.0_dp) sig = 0.0_dp

        m2_target = mu*mu + sig*sig

        ! Search window around mu.
        half_span = max(3, iceil(4.0_dp*sig + 2.0_dp))

        zl = max(zmin, ifloor(mu) - half_span)
        zh = iceil(mu) + half_span
        if (zh - zl < 2) zh = zl + 2

        found_nonneg = .false.
        best_span = huge(1.0_dp)
        best_mid  = huge(1.0_dp)

        ! Initialize outputs.
        z1 = zl
        z2 = zl + 1
        z3 = zl + 2
        w1 = 0.0_dp
        w2 = 0.0_dp
        w3 = 0.0_dp

        do i = zl, zh - 2
        do j = i + 1, zh - 1
            do k = j + 1, zh
            a = real(i, dp)
            b = real(j, dp)
            c = real(k, dp)
            a2 = a*a
            b2 = b*b
            c2 = c*c

            ! Solve:
            ! w1+w2+w3=1
            ! w1*a + w2*b + w3*c = mu
            ! w1*a2+w2*b2+w3*c2 = m2_target
            !
            ! Reduced 2x2 system for w1,w2 with w3=1-w1-w2.
            da   = a - c
            db   = b - c
            da2  = a2 - c2
            db2  = b2 - c2
            rhs1 = mu - c
            rhs2 = m2_target - c2

            det = da*db2 - db*da2
            if (abs(det) <= 1.0d-18) cycle

            ww1 = ( rhs1*db2 - rhs2*db ) / det
            ww2 = ( da*rhs2  - da2*rhs1 ) / det
            ww3 = 1.0_dp - ww1 - ww2

            if ((ww1 >= -tol) .and. (ww2 >= -tol) .and. (ww3 >= -tol)) then
                ! Prefer compact support and center near mu.
                score_span = real(k - i, dp)
                score_mid  = abs(real(j, dp) - mu)

                if ((.not. found_nonneg) .or. &
                    (score_span < best_span) .or. &
                    (abs(score_span - best_span) <= 1.0d-12 .and. score_mid < best_mid)) then
                found_nonneg = .true.
                best_span = score_span
                best_mid  = score_mid

                z1 = i
                z2 = j
                z3 = k
                w1 = ww1
                w2 = ww2
                w3 = ww3
                end if
            end if

            end do
        end do
        end do

        if (found_nonneg) then
        ! Clip tiny negatives and renormalize.
        if (w1 < 0.0_dp) w1 = 0.0_dp
        if (w2 < 0.0_dp) w2 = 0.0_dp
        if (w3 < 0.0_dp) w3 = 0.0_dp
        call renormalize3(w1, w2, w3)
        used_two_point = .false.
        else
        ! Fallback: two-point method.
        call two_point_charge_mix(mu, zmin, z1, z2, w1, w2)
        z3 = z2
        w3 = 0.0_dp
        used_two_point = .true.
        end if
    end subroutine three_point_charge_mix


    subroutine renormalize3(w1, w2, w3)
        implicit none
        real(dp), intent(inout) :: w1, w2, w3
        real(dp) :: s

        s = w1 + w2 + w3
        if (s > 0.0_dp) then
        w1 = w1 / s
        w2 = w2 / s
        w3 = w3 / s
        else
        ! Defensive fallback (should not happen in normal flow).
        w1 = 1.0_dp
        w2 = 0.0_dp
        w3 = 0.0_dp
        end if
    end subroutine renormalize3


    integer function ifloor(x)
        implicit none
        real(dp), intent(in) :: x
        integer :: t

        t = int(x)
        if (real(t, dp) > x) t = t - 1
        ifloor = t
    end function ifloor


    integer function iceil(x)
        implicit none
        real(dp), intent(in) :: x
        integer :: t

        t = int(x)
        if (real(t, dp) < x) t = t + 1
        iceil = t
    end function iceil

end module dust_charging