module dust_rates
    use amr_parameters, only: dp
    use constants, only: yr2sec,Myr2sec,kB,mH,amu2g,e2instatC
    use dust_commons
    use dust_utils

#ifdef RTZ
    use rtz_module, only: elements
#endif

    implicit none

contains

    subroutine compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                       T_dust,Zmean_dust,gamma_RAT,FIR, &
                                       nElement,xelem_ions,fcharge_pahs, &
                                       lead_elem,nions_lead,int_dust_ratd, &
                                       boost_acc,boost_coa,rhoZ_lim, &
                                       t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                       do_sputtering,do_accretion,do_coagulation,do_shattering,do_ratd, &
                                       do_pah_sputtering,do_pah_sublimation,do_pah_coalescence,do_pah_freezing,do_pah_evaporation)
        implicit none

        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)
        logical, intent(in), optional :: do_sputtering,do_accretion,do_coagulation,do_shattering,do_ratd
        logical, intent(in), optional :: do_pah_sputtering,do_pah_sublimation,do_pah_coalescence,do_pah_freezing,do_pah_evaporation

        integer :: ii, jj, kk, jj1, jj2, kk1, kk2, nd_ctype
        logical :: run_sputtering,run_accretion,run_coagulation,run_shattering,run_ratd
        logical :: run_pah_sputtering,run_pah_sublimation,run_pah_coalescence,run_pah_freezing,run_pah_evaporation

        run_sputtering = dust_sputtering
        if (present(do_sputtering)) run_sputtering = do_sputtering
        run_accretion = dust_accretion
        if (present(do_accretion)) run_accretion = do_accretion
        run_coagulation = dust_coagulation
        if (present(do_coagulation)) run_coagulation = do_coagulation
        run_shattering = dust_shattering
        if (present(do_shattering)) run_shattering = do_shattering
        run_ratd = dust_ratd
        if (present(do_ratd)) run_ratd = do_ratd

        run_pah_sputtering = pah_sputtering
        if (present(do_pah_sputtering)) run_pah_sputtering = do_pah_sputtering
        run_pah_sublimation = pah_photolysis
        if (present(do_pah_sublimation)) run_pah_sublimation = do_pah_sublimation
        run_pah_coalescence = pah_coalescence
        if (present(do_pah_coalescence)) run_pah_coalescence = do_pah_coalescence
        run_pah_freezing = pah_freezing
        if (present(do_pah_freezing)) run_pah_freezing = do_pah_freezing
        run_pah_evaporation = pah_cluster_evaporation
        if (present(do_pah_evaporation)) run_pah_evaporation = do_pah_evaporation

        if (run_sputtering) call compute_t_sputtering
        if (run_accretion) call compute_t_accretion
        if (run_coagulation) call compute_t_coagulation
        if (run_shattering) call compute_t_shattering
        if (run_ratd) call compute_t_ratd

        if (dust_pahs) then
            if (run_pah_sputtering) call compute_t_pah_sputtering
            if (run_pah_sublimation) call compute_t_pah_sublimation
            if (run_pah_coalescence) call compute_t_pah_coalescence
            if (run_pah_freezing) call compute_t_pah_freezing
            if (run_pah_evaporation) call compute_t_pah_cluster_evaporation
        end if

    contains
        subroutine compute_t_accretion
            ! Using local gas properties, this routine
            ! allows for the computation of the accretion timescale
            ! for the full range of dust species and sizes used
            implicit none
            integer::k
            real(dp)::mx,fx,tacc_max,t

            t_acc = 1d15 * yr2sec

            tacc_max = 5
            select case (accretion_model)
                case ('subgrid')
                    if(Tk.gt.1d4.or.nH.lt.0.1d0 &
                        & .or.lambda_jeans>4d0*dx_loc) then 
                        ! Assume Le Bourlot+2012 (https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..76L/abstract)
                        do jj=1,ndchemtype
                            jj1 = istart_chemtype(jj)
                            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                            do ii = 1, nions_lead(jj)
                                do k=jj1,jj2
                                    t = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * (mH/rhoZ_lim(jj,ii)) * &
                                        & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                        & (1d0+1d-4*Tk**1.5d0) * dustbins_props(k)%el_mfractions(lead_elem(jj))
                                    t = log10(t/yr2sec/1d6)
                                    t = (1-sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH)) * t + &
                                        & sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH) * tacc_max
                                    t = 10d0**t
                                    t_acc(k,ii) = t*1d6*yr2sec
                                end do
                            end do
                        end do
                    else
                        do jj=1,ndchemtype
                            jj1 = istart_chemtype(jj)
                            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                            do ii = 1, nions_lead(jj)
                                do k=jj1,jj2
                                    t = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * (mH/rhoZ_lim(jj,ii)) * &
                                        & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                        & dustbins_props(k)%el_mfractions(lead_elem(jj)) / Sconstant &
                                        & / boost_acc(k)
                                    t = log10(t/yr2sec/1d6)
                                    t = (1-sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH)) * t + &
                                        & sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH) * tacc_max
                                    t = 10d0**t
                                    t_acc(k,ii) = t*1d6*yr2sec
                                end do
                            end do
                        enddo
                    endif
                case ('LeBourlot2012')
                    ! Taken from Le Bourlot+2012 (https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..76L/abstract)
                    do jj=1,ndchemtype
                        jj1 = istart_chemtype(jj)
                        jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                        do ii = 1, nions_lead(jj)
                            do k=jj1,jj2
                                t = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * (mH/rhoZ_lim(jj,ii)) * &
                                    & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                    & (1d0+1d-4*Tk**1.5d0) * dustbins_props(k)%el_mfractions(lead_elem(jj))
                                t = log10(t/yr2sec/1d6)
                                t = (1-sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH)) * t + &
                                    & sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH) * tacc_max
                                t = 10d0**t
                                t_acc(k,ii) = t*1d6*yr2sec
                            end do
                        end do
                    end do
                case ('Chaabouni2012')
                    ! Taken from Chaabouni+2012 (https://ui.adsabs.harvard.edu/#abs/2012A%26A...538A.128C)
                    do jj=1,ndchemtype
                        jj1 = istart_chemtype(jj)
                        jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                        do ii = 1, nions_lead(jj)
                            do k=jj1,jj2
                                t = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * (mH/rhoZ_lim(jj,ii)) * &
                                    & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                    & dustbins_props(k)%el_mfractions(lead_elem(jj)) / (0.95d0*(1d0+2.22d0*Tk/56d0)/(1d0+Tk/56d0)**2.22d0)
                                t = log10(t/yr2sec/1d6)
                                t = (1-sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH)) * t + &
                                    & sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH) * tacc_max
                                t = 10d0**t
                                t_acc(k,ii) = t*1d6*yr2sec
                            end do
                        end do
                    end do
                case ('LDW1985')
                    ! Taken from the fittings to the experimetal data of Leitch-Devlin & Williams (1985) done by
                    ! Grassi+2014 (https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.1259G/abstract) for the KROME code
                    do jj=1,ndchemtype
                        jj1 = istart_chemtype(jj)
                        jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                        do ii = 1, nions_lead(jj)
                            do k=jj1,jj2
                                t = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * (mH/rhoZ_lim(jj,ii)) * &
                                    & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                    & dustbins_props(k)%el_mfractions(lead_elem(jj)) / &
                                    & (1.9d-2*Tk*(1.7d-3*T_dust(k-npah)+4d-1)*exp(-7d-3*Tk))
                                t = log10(t/yr2sec/1d6)
                                t = (1-sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH)) * t + &
                                    & sigmoid_function(tacc_max,dustbins_props(k)%nhmax_acc,nH) * tacc_max
                                t = 10d0**t
                                t_acc(k,ii) = t*1d6*yr2sec
                            end do
                        end do
                    end do
                case ('Dubois2024')
                    ! Same as our prescription, but in the original Dubois+2024 paper there is no
                    ! smoothing function, but rather a sharp threshold at nH=0.1 cm**-3.
                    ! NOTE: The temperature at which accretion is taking place is fixed at 100 K
                    ! when the gas is underresolved.
                    ! In the computation of the Jeans length, Yohan assumes a mean molecular
                    ! weight of 1
                    if(Tk.gt.1d4.or.nH.lt.0.1d0 &
                        & .or.lambda_jeans>4d0*dx_loc) then 
                        ! Assume Le Bourlot+2012 (https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..76L/abstract)
                        do jj=1,ndchemtype
                            jj1 = istart_chemtype(jj)
                            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                            do ii = 1, nions_lead(jj)
                                do k=jj1,jj2
                                    t_acc(k-npah,ii) = dustbins_props(k)%t0_acc * sqrt(50d0/Tk) * &
                                                       & (mH/rhoZ_lim(jj,ii)) * sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) * &
                                                       & (1d0+1d-4*Tk**1.5d0) * dustbins_props(k)%el_mfractions(lead_elem(jj))
                                end do
                            end do
                        end do
                    else
                        do jj=1,ndchemtype
                            jj1 = istart_chemtype(jj)
                            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                            do ii = 1, nions_lead(jj)
                                do k=jj1,jj2
                                    t_acc(k-npah,ii) = dustbins_props(k)%t0_acc * sqrt(50d0/100d0) * (mH/rhoZ_lim(jj,ii)) * &
                                                       & sqrt(dustbins_props(k)%el_atomic_masses_amu(lead_elem(jj))) / Sconstant &
                                                       & / boost_acc(k)
                                end do
                            end do
                        enddo
                    endif
            end select

            if (dust_acc_coulomb) then
                do jj=1,ndchemtype
                    jj1 = istart_chemtype(jj)
                    jj2 = jj1 + dustbins_per_chemtype(jj) - 1
                    do ii = 1, nions_lead(jj)
                        do k=jj1,jj2
                            t_acc(k,ii) = t_acc(k,ii) / dust_helper%Coulomb_factor(k,ii-1)
                        end do
                    end do
                end do
            else if ((Tk.lt.2d4) .or. (nH.gt.10.0d0)) then
                do k=1,ndust
                    t_acc(k,1) = t_acc(k,1) / dustbins_props(k)%Coulomb_enhance
                end do
            end if

        end subroutine compute_t_accretion

        subroutine compute_t_coagulation
            ! This routine allows for the computation of 
            ! coagulation timescales from the small to the large grains
            ! for different species.
            implicit none
            integer::k
            !TODO: Add all the cases from Yohan

            t_coa = 1d15 * yr2sec

            select case (coagulation_model)
            case ('subgrid')
                if (dust_turbulent_model) then
                    ! This considers the subgrid model described in Rodriguez Montero et al.
                    ! (2024) in Section 2.4.2. Instead of using a fixed relative velocity
                    ! between small grains, this is computed using the subgrid turbulent model
                    ! but only allowing coagulation below a threshold velocity v_coag 
                    ! (Chokshi et al. 1993)
                    call compute_turbulent_coagulation
                else
                    ! This model assumes that above nh_coa half the gas is at a higher density
                    ! of 1e+3 Hcc, but boosted using the subgrid turbulence model
                    do k=1,ndust
                        if ((Tk.gt.1d4).or.(nH<dustbins_props(k)%nh_coa) &
                            &.or.(lambda_jeans>4d0*dx_loc)) then
                            t_coa(k,:) = 1d15 * yr2sec
                        else
                            t_coa(k,:) = dustbins_props(k)%t0_coa * 1d3 / (nH * boost_coa(k) * local_mu)
                        end if
                    end do
                end if
            case ('aoyama17')
                ! This is the default model used in Dubois+2024 for coagulation,
                ! which is based on Aoyama+2017 which assumes that 0.5 of the 
                ! gas mass, which Jeans length is unresolved and with gas density
                ! above nh_coa and temperature below 1e4 K, is at a density 
                ! of 1e3 cm**-3 with a small grain turbulent velocity of 0.1 km/s.
                ! NOTE: This assumes a fixed mean molecular weight of 1.4 that
                ! is already introduced in t0_coa as well as F=0.5
                ! In the computation of the Jeans length, Yohan assumes a mean molecular
                ! weight of 1
                do k=1,ndust
                    if ((Tk.gt.1d4).or.(nH.lt.dustbins_props(k)%nh_coa) &
                        &.or.(lambda_jeans>4d0*dx_loc)) then
                        t_coa(k,:) = 1d15 * yr2sec
                    else
                        t_coa(k,:) = dustbins_props(k)%t0_coa
                    end if
                end do
            end select
        end subroutine compute_t_coagulation

        subroutine compute_turbulent_coagulation
            use constants, only: pi,pc2cm
            use dust_dynamics, only: grain_relative_velocity

            implicit none

            real(dp)                         :: v_rel,v_coag,R,coll_factor,t
            real(dp)                         :: tcoag_max, enhan_factor
            real(dp)                         :: temp_sigma,temp_L

            tcoag_max = 5d0
            t_coa = 1d15 * yr2sec
            
            if (dust_eq_test) then
                temp_sigma = 5.67d5 * (nH/1d2)**(-0.25)
                temp_L = 10d0 * pc2cm * (nH/1d2)**(-1d0/3d0)
            else
                temp_sigma = sigma
                temp_L = dx_loc
            end if
            do ii = 1, ndchemtype
                jj1 = istart_chemtype(ii)
                jj2 = jj1 + dustbins_per_chemtype(ii) - 1
                ! Loop over all dust bins besides the largest grain size
                do jj = jj1, jj2 - 1
                    kk1 = jj1
                    kk2 = jj2
                    do kk = kk1, kk2
                        ! 1. Compute the relative velocity of the two grains
                        v_rel = grain_relative_velocity(dust_velocity_model,Tk,rho,nH,temp_sigma,&
                                                        local_mu,temp_L,&
                                                        dustbins_props(jj)%asize_cm,&
                                                        dustbins_props(kk)%asize_cm,&
                                                        dustbins_props(jj)%sgrain,&
                                                        dustbins_props(kk)%sgrain,&
                                                        dustbins_props(jj)%mgrain,&
                                                        dustbins_props(kk)%mgrain)
                        ! 2. Compute the enhancement due to ice mantles
                        v_coag = dustbins_props(jj)%vthresh_coag(kk)
                        if (poppe_ice_enhancement) then
                            enhan_factor = (1d0-sigmoid_function(4d0,log10(dustbins_props(jj)%nhmax_acc),log10(nH))) + &
                                           sigmoid_function(4d0,log10(dustbins_props(jj)%nhmax_acc),log10(nH)) * 4d0
                            v_coag = enhan_factor * v_coag
                        end if
                        ! 3. Collision rate calculation
                        ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                        ! Marchand et al. (2021) and considers that grain velocities along the x-,
                        ! y-, and z-axes are Gaussian distributed
                        coll_factor = sqrt(8d0/(3d0*pi)) * pi * (dustbins_props(jj)%asize_cm + dustbins_props(kk)%asize_cm)**2d0 * v_rel
                        if (jj.eq.kk) then
                            coll_factor = coll_factor * 0.5d0 ! Avoid double counting
                        end if
                        t = log10(dustbins_props(jj)%mgrain/(coll_factor * rho)/yr2sec/1d6)
                        t = (1d0-sigmoid_function(tcoag_max,v_coag,v_rel)) * t + sigmoid_function(tcoag_max,v_coag,v_rel) * tcoag_max
                        t = 10d0**t
                        t_coa(jj,kk) = t*1d6*yr2sec
                    end do
                end do
            end do

        end subroutine compute_turbulent_coagulation

        subroutine compute_t_shattering
            ! This routine allows for the computation of 
            ! shattering timescales from the large to the small grains
            ! for different species.
            implicit none
            integer::k
            real(dp)::mx,fx,sfunc
            !TODO: Add all the cases from Yohan

            t_sha = 1d15 * yr2sec

            select case (shattering_model)
            case ('subgrid')
                if (dust_turbulent_model) then
                    ! This computes the shattering timescale of large-large grain collisions
                    ! turning into a distribution of fragments, which may mean for the large
                    ! mass to either be destroyed, turned into PAHs (for carbonaceous grains),
                    ! turn into small grains or just kept into the large grain bins
                    ! For further information see Section 2.3.1 in Rodriguez Montero et al. (2023)
                    call compute_turbulent_shattering
                else
                    ! This model instead uses the velocity dispersion obtained
                    ! locally for the cell
                    if (nH<1d-1) then
                        t_sha = 1d15 * yr2sec
                    else
                        if (sigma/1d5 .lt. 2.0d0) then
                            t_sha = 1d15 * yr2sec
                        else
                            do k = 1, ndchemtype
                                nd_ctype = dustbins_per_chemtype(k)
                                jj1 = istart_chemtype(k)
                                jj2 = jj1 + nd_ctype - 1
                                do ii = jj1, jj2 - 1
                                    ! Compute the shattering timescale for each dust bin
                                    t_sha(ii,ii,ii+1) = dustbins_props(ii)%t0_sha / (nH * local_mu) * (1d6/sigma)
                                end do
                            end do
                        end if
                    end if
                end if
            case ('Granato2021')
                ! This models follows the prescription by Granato+2021
                ! Eq. 8 and 9 in https://ui.adsabs.harvard.edu/abs/2021MNRAS.503..511G/abstract
                if (nH < 1.0d0) then
                    do k = 1, ndchemtype
                        nd_ctype = dustbins_per_chemtype(k)
                        jj1 = istart_chemtype(k)
                        jj2 = jj1 + nd_ctype - 1
                        do ii = jj1, jj2 - 1
                            t_sha(ii,ii,ii+1) = dustbins_props(ii)%t0_sha / (nH * local_mu)
                        end do
                    end do
                else
                    ! We impose an exponential suppresion above the nhmax_sha, based on the
                    ! assumption that shattering is not efficient in the cold, dense ISM due
                    ! grain relative velocities being too low for fragmentation (see Yan+2004)
                    do k = 1, ndchemtype
                        nd_ctype = dustbins_per_chemtype(k)
                        jj1 = istart_chemtype(k)
                        jj2 = jj1 + nd_ctype - 1
                        do ii = jj1, jj2 - 1
                            sfunc = min(nH**(2.d0/3.d0)*exp(nH/dustbins_props(ii)%nhmax_sha),1d15 * yr2sec)
                            t_sha(ii,ii,ii+1) = dustbins_props(ii)%t0_sha * sfunc / (nH * local_mu)
                        end do
                    end do
                end if
            case ('Dubois2024')
                ! This model follows the prescription by Dubois+2024
                ! which is basically the same as for Granato+2021, with the 
                ! exception that the suppression is not exponential but sharp
                ! NOTE: The mean molecular weight is kept fixed in this model to mu=1.4
                if (nH < 1.0d0) then
                    do k = 1, ndchemtype
                        nd_ctype = dustbins_per_chemtype(k)
                        jj1 = istart_chemtype(k)
                        jj2 = jj1 + nd_ctype - 1
                        do ii = jj1, jj2 - 1
                            t_sha(ii,ii,ii+1) = dustbins_props(ii)%t0_sha / nH
                        end do
                    end do
                elseif (nH < 1d3) then
                    do k = 1, ndchemtype
                        nd_ctype = dustbins_per_chemtype(k)
                        jj1 = istart_chemtype(k)
                        jj2 = jj1 + nd_ctype - 1
                        do ii = jj1, jj2 - 1
                            t_sha(ii,ii,ii+1) = dustbins_props(ii)%t0_sha / (nH**(1.0d0/3.0d0))
                        end do
                    end do
                else
                    t_sha = 1d15 * yr2sec
                end if
            end select
        end subroutine compute_t_shattering

        subroutine compute_turbulent_shattering
            use constants, only: pi,pc2cm
            use dust_dynamics, only: grain_relative_velocity
            ! Following Hirashita & Li (2013), the shattering model uses the
            ! prescription by Kobayashi & Tanaka (2010). The relative velocity
            ! is given by the subrgrid model as explained in Section 2.3.1 of
            ! Rodriguez Montero et al. (2023). As detailed in the paper,
            ! collisions between large grains are only considered, as the
            ! collision of small and large grains are inneficient in breaking
            ! large grains.
            
            implicit none

            integer  :: pp
            real(dp) :: temp_sigma,temp_L
            real(dp) :: coll_factor, v_rel
            real(dp) :: chi_frag_dest
            real(dp),dimension(:),allocatable :: chi_frag
            real(dp),dimension(1:npah) :: chi_frag_pah


            if (dust_eq_test) then
                temp_sigma = 5.67d5 * (nH/1d2)**(-0.25)
                temp_L = 10d0 * pc2cm * (nH/1d2)**(-1d0/3d0)
            else
                temp_sigma = sigma
                temp_L = dx_loc
            end if

            do ii=1,ndchemtype
                jj1 = istart_chemtype(ii)
                jj2 = jj1 + dustbins_per_chemtype(ii) - 1
                allocate(chi_frag(jj1:jj2))
                do jj = jj2, jj1, -1
                    ! We only consider fragment production from collisions of grain i = j.
                    ! This excludes large-small collision shattering channels.
                    call compute_shattered_fragments(jj,jj,temp_sigma,temp_L,v_rel,chi_frag_dest,chi_frag,chi_frag_pah)

                    ! The factor of sqrt(8/(3*pi)) is taken from Guillet et al. (2020) and
                    ! Marchand et al. (2021) and considers that grain velocities along the x-,
                    ! y-, and z-axes are Gaussian distributed.
                    coll_factor = sqrt(8d0/(3d0*pi)) * pi * (2d0*dustbins_props(jj)%asize_cm)**2d0 * v_rel / dustbins_props(jj)%mgrain**2d0
                    t_sha(jj,jj,1) = min(1d0 / (coll_factor * rho * dustbins_props(jj)%mgrain * max(chi_frag_dest,1d-10)), 1d15 * yr2sec)
                    do pp = jj1, jj2
                        t_sha(jj,jj,pp) = min(1d0 / (coll_factor * rho * dustbins_props(jj)%mgrain * max(chi_frag(pp),1d-10)), 1d15 * yr2sec)
                    end do

                    ! In the case of following PAHs, also compute their fragment-production timescales.
                    do pp = 1, npah
                        t_sha_pah(jj,jj,pp) = min(1d0 / (coll_factor * rho * dustbins_props(jj)%mgrain * max(chi_frag_pah(pp),1d-10)), 1d15 * yr2sec)
                    end do
                end do
            end do
        end subroutine compute_turbulent_shattering

        subroutine compute_shattered_fragments(id1,id2,local_sigma,local_L,v_rel,chi_frag_dest,chi_frag,chi_frag_pah)
            use dust_dynamics, only: grain_relative_velocity
            implicit none

            integer,intent(in) :: id1, id2
            real(dp),intent(in) :: local_sigma, local_L
            real(dp),intent(inout) :: v_rel, chi_frag_dest
            real(dp),dimension(:),intent(inout) :: chi_frag
            real(dp),dimension(:),intent(inout) :: chi_frag_pah

            integer  :: pp,ll
            real(dp) :: E_imp, phi, m_ej, m_remnant, m_max, m_min
            real(dp) :: prefactor,m_tot

            ! 1. Compute the relative velocity of two grains
            v_rel = grain_relative_velocity(dust_velocity_model,Tk,rho,nH,local_sigma,&
                                            local_mu,local_L,&
                                            dustbins_props(id1)%asize_cm,&
                                            dustbins_props(id2)%asize_cm,&
                                            dustbins_props(id1)%sgrain,&
                                            dustbins_props(id2)%sgrain,&
                                            dustbins_props(id1)%mgrain,&
                                            dustbins_props(id2)%mgrain)

            ! 2. Disrupted mass computation (Eqs. 20-22 of Hirashita & Aoyama 2019)
            E_imp = 5d-1 * (dustbins_props(id1)%mgrain*dustbins_props(id2)%mgrain)&
                    /(dustbins_props(id1)%mgrain+dustbins_props(id2)%mgrain) &
                    * v_rel**2d0
            phi = E_imp / (dustbins_props(id1)%mgrain*dustbins_props(id1)%catastrophic_spec_energy)
            m_ej = phi / (1d0 + phi) * dustbins_props(id1)%mgrain

            ! 3. Compute the maximum and minimum masses of the fragment distribution
            m_remnant = dustbins_props(id1)%mgrain - m_ej
            m_max = 2d-2*m_ej
            m_min = 1d-6*m_max

            ! 4. Compute the distribution prefactor
            prefactor = m_ej / (m_max**slope_frag_func - m_min**slope_frag_func)

            ! 5. Compute the mass fraction that is put into fragments
            !    so small that are simply returned to the gas phase
            chi_frag_dest = 0d0
            chi_frag(:) = 0d0
            chi_frag_pah(:) = 0d0
            if (dustbins_props(id1)%interact_pah .and. npah>0) then
                if (m_min .ge. pahbins_props(1)%mpah_min) then
                    chi_frag_dest = 0d0
                    if (m_remnant<pahbins_props(1)%mpah_min) chi_frag_dest = chi_frag_dest + m_remnant
                else
                    chi_frag_dest = prefactor * (min(pahbins_props(1)%mpah_min,m_max)**slope_frag_func - m_min**slope_frag_func)
                    if (m_remnant<pahbins_props(1)%mpah_min) chi_frag_dest = chi_frag_dest + m_remnant
                end if
            else
                if (m_min .ge. dustbins_props(jj1)%mgrain_min) then
                    chi_frag_dest = 0d0
                    if (m_remnant<dustbins_props(jj1)%mgrain_min) chi_frag_dest = chi_frag_dest + m_remnant
                else
                    chi_frag_dest = prefactor * (min(dustbins_props(jj1)%mgrain_min,m_max)**slope_frag_func - m_min**slope_frag_func)
                    if (m_remnant<dustbins_props(jj1)%mgrain_min) chi_frag_dest = chi_frag_dest + m_remnant
                end if
            end if

            ! 6. In the case of following PAHs, compute the first the
            !    mass fraction that is put into fragments of PAH size
            if (dustbins_props(id1)%interact_pah .and. npah>0) then
                do pp = 1, npah
                    if ((m_min.ge.pahbins_props(pp)%mpah_max).or.(m_max<pahbins_props(pp)%mpah_min)) then
                        chi_frag_pah(pp) = 0d0
                        if ((pahbins_props(pp)%mpah_min.le.m_remnant).and.(m_remnant<pahbins_props(pp)%mpah_max)) &
                            & chi_frag_pah(pp) = chi_frag_pah(pp) + m_remnant
                    else
                        chi_frag_pah(pp) = prefactor * (min(pahbins_props(pp)%mpah_max,m_max)**slope_frag_func - max(pahbins_props(pp)%mpah_min,m_min)**slope_frag_func)
                        if ((pahbins_props(pp)%mpah_min.le.m_remnant).and.(m_remnant<pahbins_props(pp)%mpah_max)) &
                            & chi_frag_pah(pp) = chi_frag_pah(pp) + m_remnant
                    end if
                end do
            end if

            ! 7. Loop over the grain sizes for the given chemical
            !    using the jj1 and jj2 that all defined in 
            !    compute_turbulent_shattering
            do ll = jj1, jj2
                if ((m_min.ge.dustbins_props(ll)%mgrain_max).or.(m_max<dustbins_props(ll)%mgrain_min)) then
                    chi_frag(ll) = 0d0
                    if ((dustbins_props(ll)%mgrain_min.le.m_remnant).and.(m_remnant<dustbins_props(ll)%mgrain_max)) &
                        & chi_frag(ll) = chi_frag(ll) + m_remnant
                else
                    chi_frag(ll) = prefactor * (min(dustbins_props(ll)%mgrain_max,m_max)**slope_frag_func - max(dustbins_props(ll)%mgrain_min,m_min)**slope_frag_func)
                    if ((dustbins_props(ll)%mgrain_min.le.m_remnant).and.(m_remnant<dustbins_props(ll)%mgrain_max)) &
                        & chi_frag(ll) = chi_frag(ll) + m_remnant
                end if
            end do

            ! 8. Normalise the fragmentation mass fractions
            m_tot = chi_frag_dest + sum(chi_frag(:)) + sum(chi_frag_pah(:))
            if (m_tot > 0d0) then
                chi_frag_dest = chi_frag_dest / m_tot
                chi_frag(:) = chi_frag(:) / m_tot
                chi_frag_pah(:) = chi_frag_pah(:) / m_tot
            end if

        end subroutine compute_shattered_fragments

        subroutine compute_t_sputtering
            ! This routine allows for the computation of 
            ! sputtering timescales from all grains
            ! for different species.
            use cooling_module, only:mH
            use constants, only:eV2erg
            implicit none
            integer::k, iel, iion, iphi0, nT_loc, nphi_loc
            real(dp)::ngas
            real(dp)::lT,ySi,yC,y,rate_total,T6
            real(dp)::xion,irate,phi_charge
            integer :: izion,nions_loc
            
            t_spu = 1d15 * yr2sec
            T6 = Tk / 1d6

            ngas = rho / mH
            select case (sputtering_model)
            case ('Novak2012')
                ! Draine & Salpeter (1979) (see also Novak et al. 2012)  
                do k=1,ndust
                    t_spu(k) = dustbins_props(k)%t0_spu/ngas * (1d0+1d0/T6**3d0)
                end do
            case ('Tsai1998')
                ! Tsai & Matthews (1998)
                do k=1,ndust
                    t_spu(k) = 1.65d0 * dustbins_props(k)%t0_spu/ngas * (1d0+(2d0/T6)**2.5d0)
                end do
            case ('Nozawa2006')
                ! Nozawa et al. (2006) - https://ui.adsabs.harvard.edu/abs/2006ApJ...648..435N/abstract
                lT = log10(TK*0.60d0)
                ySi=aSith(1)+aSith(2)*lT+aSith(3)*lT**2d0+aSith(4)*lT**3d0+aSith(5)*lT**4d0+aSith(6)*lT**5d0
                yC =aCth (1)+aCth (2)*lT+aCth (3)*lT**2d0+aCth (4)*lT**3d0+aCth (5)*lT**4d0+aCth (6)*lT**5d0
                ySi=10d0**ySi
                yC =10d0**yC
                do k=1,ndust
                    if (dustbins_props(k)%interact_group.eq.1) then
                        t_spu(k) = dustbins_props(k)%asize/nH/yC/3d0 * yr2sec
                    elseif (dustbins_props(k)%interact_group.eq.2) then
                        t_spu(k) = dustbins_props(k)%asize/nH/ySi/3d0 * yr2sec
                    else
                        t_spu(k) = 1d15 * yr2sec
                    end if
                end do
            case ('RM2026')
                ! This uses the tables computed for the Rodriguez Montero et al. (2024)
                ! thermal sputtering, now summed over elements and ion fractions.
                ! If dust_sputtering_charge=.false., rates are interpolated only in T
                ! using the phi=0 slice indicated by ipos_zero(2).
                ! If dust_sputtering_charge=.true., rates are interpolated in (T,phi),
                ! where phi = -Z_ion * Z_grain * e^2 / a_grain [eV].
                lT = log10(Tk)
                do k = 1, ndust
                    rate_total = 0d0
                    do iel = 1, n_elements
                        if (.not. dustbins_props(k)%sputtering_tab(iel)%initialised) cycle
                        if (nElement(iel) <= 1d-20) cycle ! Ignore elements with negligible abundance
                        nT_loc = dustbins_props(k)%sputtering_tab(iel)%npts(1)
                        nphi_loc = dustbins_props(k)%sputtering_tab(iel)%npts(2)
                        if (dust_sputtering_charge) then
                            nions_loc = n_elements
#ifdef RTZ
                            nions_loc = max(1,elements(iel)%n_ions)
#endif
                            do iion = 1, nions_loc
                                if (xelem_ions(iel,iion) <= 1d-10) cycle ! Ignore ionisation states with negligible abundance
                                izion = iion-1
                                phi_charge = Zmean_dust(k) * dustbins_props(k)%phi_prefact(izion)
                                call interpolate2D(dustbins_props(k)%sputtering_tab(iel)%tab1d(1:nT_loc,1), &
                                    dustbins_props(k)%sputtering_tab(iel)%tab1d(1:nphi_loc,2), &
                                    dustbins_props(k)%sputtering_tab(iel)%tab2d(1:nT_loc,1:nphi_loc,1), &
                                    dustbins_props(k)%sputtering_tab(iel)%npts(1), dustbins_props(k)%sputtering_tab(iel)%npts(2), lT, phi_charge, irate)
                                rate_total = rate_total + nElement(iel) * xelem_ions(iel,iion) * dust_helper%Coulomb_factor(k,izion) * 10d0**irate
                            end do
                        else
                            iphi0 = dustbins_props(k)%sputtering_tab(iel)%ipos_zero(2)
                            call interpolate1D(dustbins_props(k)%sputtering_tab(iel)%tab1d(1:nT_loc,1), &
                                dustbins_props(k)%sputtering_tab(iel)%tab2d(1:nT_loc,iphi0,1), &
                                dustbins_props(k)%sputtering_tab(iel)%npts(1), lT, irate)
                            rate_total = rate_total + nElement(iel) * 10d0**irate
                        end if
                    end do
                    if (rate_total > 0d0) then
                        t_spu(k) = dustbins_props(k)%asize / (3d0 * rate_total) * yr2sec
                    end if
                end do
            case default
                ! Tsai & Matthews (1998)
                do k=1,ndust
                    t_spu(k) = 1.65d0 * dustbins_props(k)%t0_spu/ngas * (1d0+(2d0/T6)**2.5d0)
                end do
            end select
        end subroutine compute_t_sputtering
#if NPAH>0
        subroutine compute_t_pah_sputtering
            ! This routine computes the thermal ion and electron
            ! sputtering timescale of PAHs
            use dust_charging, only: compute_Coulomb_focusing
            implicit none

            integer :: k, nT_loc, iel, iion, nions_loc, izion
            real(dp) :: lT, R_total, Dtemp, xion, Zel
            real(dp), dimension(0:n_elements) :: J_rate
            real(dp), dimension(:), allocatable :: coulomb_focus_tab

            lT = log10(Tk)
            do k = 1, npah
                R_total = 0d0

                if (allocated(coulomb_focus_tab)) deallocate(coulomb_focus_tab)
                allocate(coulomb_focus_tab(-1:n_elements))
                Zel = -1d0
                call compute_Coulomb_focusing(Tk,pahbins_props(k)%apah_cm, &
                    fcharge_pahs(1:pahbins_props(k)%ncharge_states,k), &
                    pahbins_props(k)%charge_states(1:pahbins_props(k)%ncharge_states), &
                    Zel,coulomb_focus_tab(-1))

                Zel = 0d0
                call compute_Coulomb_focusing(Tk,pahbins_props(k)%apah_cm, &
                    fcharge_pahs(1:pahbins_props(k)%ncharge_states,k), &
                    pahbins_props(k)%charge_states(1:pahbins_props(k)%ncharge_states), &
                    Zel,coulomb_focus_tab(0))

                do izion = 1, n_elements
                    Zel = dble(izion)
                    call compute_Coulomb_focusing(Tk,pahbins_props(k)%apah_cm, &
                        fcharge_pahs(1:pahbins_props(k)%ncharge_states,k), &
                        pahbins_props(k)%charge_states(1:pahbins_props(k)%ncharge_states), &
                        Zel,coulomb_focus_tab(izion))
                end do

                J_rate = -100d0
                do iel = 0, n_elements
                    if (.not. pahbins_props(k)%sputtering_tab(iel)%initialised) cycle
                    nT_loc = pahbins_props(k)%sputtering_tab(iel)%npts(1)
                    call interpolate1D(pahbins_props(k)%sputtering_tab(iel)%tab1d(1:nT_loc,1), &
                                       pahbins_props(k)%sputtering_tab(iel)%tab2d(1:nT_loc,1,1), &
                                       nT_loc, lT, J_rate(iel))
                end do
                J_rate = 10d0**J_rate

                ! Electrons (table slot 0)
                R_total = R_total + ne * coulomb_focus_tab(-1) * J_rate(0)

                ! Ions: loop all tracked elements and skip uninitialized tables.
                do iel = 1, n_elements
                    if (.not. pahbins_props(k)%sputtering_tab(iel)%initialised) cycle
                    if (nElement(iel) <= 1d-20) cycle

#ifdef RTZ
                    nions_loc = max(1, elements(iel)%n_ions)
#else
                    nions_loc = 1
#endif

                    if (nions_loc > 1) then
                        do iion = 1, nions_loc
                            xion = xelem_ions(iel,iion)
                            if (xion <= 1d-20) cycle
                            izion = min(n_elements, iion-1)
                            Dtemp = coulomb_focus_tab(izion)
                            R_total = R_total + nElement(iel) * xion * Dtemp * J_rate(iel)
                        end do
                    else
                        R_total = R_total + nElement(iel) * J_rate(iel)
                    end if
                end do

                if (R_total > 0d0) then
                    t_spu_pah(k) = pah_nc(k) / R_total
                else
                    t_spu_pah(k) = 1d15 * yr2sec
                end if

                if (allocated(coulomb_focus_tab)) deallocate(coulomb_focus_tab)
            end do

        end subroutine compute_t_pah_sputtering

        subroutine compute_t_pah_sublimation
            ! This routine computes the UV sublimation of
            ! PAHs for the local interstellar radiation
            
            implicit none

            integer :: ipahbin, nG0_loc, nnH_loc
            real(dp) :: U,kdiss

            t_subl = 1d15 *yr2sec

            if (G0_total.gt.0d0) then
                select case (TRIM(sublimation_model))
                case ('Galliano')
                    ! This is based on the plots done by Galliano (private comm.) in this website
                    ! https://irfu.cea.fr/Pisp/frederic.galliano/HDR/hdrch6.html#x7-3420004.2.2.1
                    ! See Section 4.2.2.1, Figure 4.11. The sublimation timescales are taken for
                    ! very small graphite grains and depends on the grain size (from 3 Angstrom
                    ! to 8 Angstrom) and the interstellar radiation intensity U from Mathis+1983
                    ! which is integrated from 0.09 microns to 8 microns, giving a total value of
                    ! 2.17e-2 erg/s/cm2. We have then obtained a fitting function for the averaging 
                    ! over the basic PAH lognormal distribution with:
                    ! - peak size = 5 angstrom
                    ! - min size = 1 angstrom
                    ! - max size = 20 angstrom
                    ! - sigma = 0.4
                    ! - PAH density = 2 g/cm3
                    ! TODO: These fit would ideally be computed during the initialisation steps of RAMSES

                    U = G0_total * 8.772d-4 ! Scaling factor between Habing band and Mathis+1983 and W/m2
                    U = U / 2.2d-5
                    t_subl(1) = 10d0**(18.0664d0 - 2.2674d0*log10(U) + 4.4058d-7*U) * Myr2sec
                case ('Allain1996')
                    ! This is based on Table 6 of Allain et al. (1996) using the Ghondalekar et al. (1980)
                    ! ISRF which is very similar to Mathis+1983 
                    ! (https://ui.adsabs.harvard.edu/abs/1996A%26A...305..602A/abstract)
                    ! The sublimation timescales are given for molecules beginning in Benzene 
                    ! to PAHs with 50 C. We have then obtained a fitting function for the averaging 
                    ! over the basic PAH lognormal distribution with:
                    ! - peak size = 5 angstrom
                    ! - min size = 1 angstrom
                    ! - max size = 20 angstrom
                    ! - sigma = 0.4
                    ! - PAH density = 2 g/cm3
                    ! TODO: These fit would ideally be computed during the initialisation steps of RAMSES

                    U = G0_total * 8.772d-4 ! Scaling factor between Habing band and Mathis+1983 and W/m2
                    U = U / 2.2d-5
                    t_subl(1) = 10**(2.4418d0 - log10(U)) * Myr2sec
                case ('Murga2019')
                    ! UV photodestruction in the SHIVA model (Murga et al. 2019)
                    ! This is based on a smooth broken power-law fitting for the UV destruction of
                    ! 5 Angstrom PAHs in their Fig. 4 (https://ui.adsabs.harvard.edu/abs/2019MNRAS.488..965M/abstract)
                    U = G0_total * 8.772d-4 ! Scaling factor between Habing band and Mathis+1983 and W/m2
                    U = U / 2.2d-5
                    t_subl(1) = 10d0**(4.905d0 - 2.006d0*log10(U) + log10(1d0 + 1.235d-3*(U**9.408d-1))) * Myr2sec
                case ('RM24')
                    ! This is the model in Rodriguez Montero et al. (2024)
                    do ipahbin = 1, npah
                        if (.not. pahbins_props(ipahbin)%dissociation_tab%initialised) cycle
                        nG0_loc = pahbins_props(ipahbin)%dissociation_tab%npts(1)
                        nnH_loc = pahbins_props(ipahbin)%dissociation_tab%npts(2)
                        call interpolate2D(pahbins_props(ipahbin)%dissociation_tab%tab1d(1:nG0_loc,1), &
                                           pahbins_props(ipahbin)%dissociation_tab%tab1d(1:nnH_loc,2), &
                                           pahbins_props(ipahbin)%dissociation_tab%tab2d(1:nG0_loc,1:nnH_loc,1), &
                                           nG0_loc,nnH_loc,log10(G0_total),log10(nH),kdiss)
                        kdiss = 10d0**kdiss ! [1/s]
                        if (kdiss > 0d0) t_subl(ipahbin) = 1d0 / kdiss
                    end do
                end select
            end if
        end subroutine compute_t_pah_sublimation

        subroutine compute_t_pah_cluster_evaporation
            ! This subroutine computes the evaporation timescale of PAH
            ! clusters as per Rapacioli et al. (2006), using a fitting
            ! to the results of Montillaud and Joblin (2014) numerical
            ! experiments.

            implicit none

            real(dp) :: t_single, t_multi

            t_evap = 1d15 * yr2sec

            t_single = 0.19306d0 / G0_total
            t_multi = 10d0**(-3.1692061d0 * log10(G0_total) + 13.5642486d0)
            t_evap(2) = max(t_single,t_multi) * yr2sec

        end subroutine compute_t_pah_cluster_evaporation

        subroutine compute_t_pah_coalescence
            ! This routine determines the timescale in which small PAHs
            ! coalesce to form large PAH clusters, similar to the turbulent
            ! coagulation of small grains but only considering that PAHs are
            ! usually dominated by thermal (Brownian) motions
            use constants, only: pi,pc2cm
            use dust_dynamics, only: grain_relative_velocity
            implicit none

            real(dp) :: reduced_mass,dV_thermal,C_eff,pah_ion_fraction
            real(dp) :: R1,R2,coll_section
            integer :: cation_start, nstates

            t_coal = 1d15 * yr2sec

            select case(trim(coalescence_model))
            case ('Totton2012')
                ! 1. Compute the reduced mass in g
                reduced_mass = 5d-1 * pahbins_props(1)%mpah
                
                ! 2. Compute the sticking probability based on the fitting to the
                ! results of Totton et al. (2012) given in Rodriguez Montero et al. (2024)
                ! NOTE: This has been computed for the small PAH with Nc=54 (circumcoronene)
                C_eff = 1d0 / (1d0 + 9.92807181d-7 * (log10(Tk))**1.37933821d1)

                ! 3. Compute the thermal velocity difference
                dV_thermal = sqrt(8.d0 * kB * Tk / reduced_mass)

                ! 4. And finally the timescale
                coll_section = 4d0*pi*(pahbins_props(1)%apah_cm)**2d0
                t_coal(1) = pahbins_props(1)%mpah / (coll_section * dV_thermal * C_eff * rho)
            case ('Tielens2021')
                nstates = pahbins_props(1)%ncharge_states
                cation_start = pahbins_props(1)%cation_start_idx
                if (cation_start <= nstates) then
                    pah_ion_fraction = sum(fcharge_pahs(cation_start:nstates,1))
                else
                    pah_ion_fraction = 0d0
                end if
                ! 1. Compute the rate for neutral PAHs following Tielens (2021)
                R1 = 4d-11 * sqrt(Tk/10d0) * sqrt(pahbins_props(1)%nc/50d0)

                ! 2. Compute the rate for ionised PAHs following Tielens (2021)
                ! (using the Langevin rate with standard polarisability for small PAHs)
                reduced_mass = 5d-1 * pahbins_props(1)%mpah
                R2 = 6d-9 * sqrt(pahbins_props(1)%nc/50d0) * sqrt((12d0*amu2g)/reduced_mass)

                ! 3. Compute the total timescale
                t_coal(1) = 1d0 / ( (R1 * (1d0-pah_ion_fraction) * rho) / pahbins_props(1)%mpah + &
                                    & (R2 * pah_ion_fraction * rho) / pahbins_props(1)%mpah)
            end select

        end subroutine compute_t_pah_coalescence

        subroutine compute_t_pah_freezing
            use constants, only: pi,eV2erg,pc2cm
            use dust_dynamics, only: grain_relative_velocity
            use dust_charging, only: compute_dust_charge_dist, compute_Coulomb_focusing
            ! This routine compute the freezing (or sequestration) of PAHs
            ! on the surface of small and large carbonaceous grains

            implicit none
            
            integer :: k
            real(dp) :: v_rel, D_temp, D_av, Z_single, E_col, coll_factor, t, t_fre_max
            real(dp) :: temp_sigma, temp_L
            real(dp),dimension(:),allocatable :: Z_grain
            real(dp),dimension(:),allocatable :: fcharge_grain

            t_fre = 0d0
            t_fre_max = 5d0

            if (dust_eq_test) then
                temp_sigma = 5.67d5 * (nH/1d2)**(-0.25)
                temp_L = 10d0 * pc2cm * (nH/1d2)**(-1d0/3d0)
            else
                temp_sigma = sigma
                temp_L = dx_loc
            end if

            do k = 1, npah
                jj1 = pahbins_props(k)%dust_index_interact
                jj2 = jj1 + dustbins_per_chemtype(dustbins_props(jj1)%interact_group) - 1
                do kk=jj1,jj2
                    ! 1. Compute the relative velocity of two grains
                    v_rel = grain_relative_velocity(dust_velocity_model,Tk,rho,nH,temp_sigma,&
                                                    local_mu,temp_L,&
                                                    dustbins_props(kk)%asize_cm,pahbins_props(k)%apah_cm,&
                                                    dustbins_props(kk)%sgrain,pahbins_props(k)%spah,&
                                                    dustbins_props(kk)%mgrain,pahbins_props(k)%mpah)

                    ! 2. Compute the dust grain charge distribution and the consequent Coulomb
                    !   focusing caused by it (similar to metal accretion, PAH freezing is affected
                    !   by the relative charge of the PAH and the dust grain)
                    call compute_dust_charge_dist(kk,G0_total,Tk,ne,Z_grain,fcharge_grain)
                    D_av = 0.0d0
                    do ii = 1, pahbins_props(k)%ncharge_states
                        Z_single = pahbins_props(k)%charge_states(ii)
                        call compute_Coulomb_focusing(Tk,dustbins_props(kk)%asize_cm,fcharge_grain,Z_grain,Z_single,D_temp)
                        D_av = D_av + fcharge_pahs(ii,k) * D_temp
                    end do
                    D_av = max(D_av,1d-10)

                    ! 3. Collision energy in eV
                    E_col = 5d-1 * (pahbins_props(k)%mpah*dustbins_props(kk)%mgrain)/(pahbins_props(k)%mpah+dustbins_props(kk)%mgrain) &
                            & * v_rel**2d0 / eV2erg

                    ! 4. Collision rate calculation (we limit to collision energies that are no larger
                    !   than the typical van der Waals bonding energies of ~1 eV, as PAH monomers and
                    !   clusters are attached to grain surfaces via these forces)
                    coll_factor = pi * (pahbins_props(k)%apah_cm + dustbins_props(kk)%asize_cm)**2d0 * v_rel * D_av
                    t = log10(dustbins_props(kk)%mgrain/(coll_factor * rho)/yr2sec/1d6)
                    t = (1d0-sigmoid_function(t_fre_max,1d0,E_col)) * t + sigmoid_function(t_fre_max,1d0,E_col) * t_fre_max
                    t = 10d0**t
                    t_fre(k,kk) = min(t * 1d6 * yr2sec, 1d15 * yr2sec)
                end do
            end do

        end subroutine compute_t_pah_freezing
#endif


        subroutine compute_t_ratd
            ! Radiative torques induced by anisotropic radiation fields can accelerate
            ! grain rotation to high angular speeds. When the grain rotation induced is
            ! sufficiently high that the tensile stress is larger than the tensile strength
            ! of the grain, it will be disrupted into smaller fragments
            ! This is all based in the fantastic work presented in the Hoang et al. (2019,2020)
            ! about radiative torque disruption (https://www.nature.com/articles/s41550-019-0763-6)
            use dust_radiative_torques, only: RAT_frequency
            implicit none
            integer :: k, pp, dust_parent, id_start, id_end, id_child, ich
            real(dp) :: t_temp,t_comb, chi_val
            real(dp),dimension(1:ndust) :: w_RAT
            t_ratd = 1d15 * yr2sec
            t_ratd_dest = 1d15 * yr2sec

            ! IR photons emitted by grain carry away part of the grain's angular momentum,
            ! and collisions with gas atoms (with consequent desorptions) can also cause loss
            ! of angular momentum. These timescales set a minimum size for the distruption of
            ! grains by radiative torques Draine & Lazarian (1998)
            ! (https://ui.adsabs.harvard.edu/abs/1998ApJ...508..157D/abstract)
            if (.not.(all(gamma_RAT.eq.0d0))) then
                w_RAT = RAT_frequency(nH,Tk,local_mu,gamma_RAT,FIR)
                do k=1,ndchemtype
                    id_start = istart_chemtype(k)
                    id_end = id_start + dustbins_per_chemtype(k) - 1
                    dust_parent = id_end
                    pp = dust_parent + npah
                    t_comb = 0d0
                    t_temp = 0d0
                    if (w_RAT(dust_parent) .ge. dustbins_props(dust_parent)%w_disr) then
                        int_dust_ratd(k) = .true.
                        t_temp = dustbins_props(dust_parent)%grain_inertia * dustbins_props(dust_parent)%w_disr / gamma_RAT(dust_parent)

                        ! Contribution to gas-phase destruction
                        chi_val = max(dustbins_props(dust_parent)%chi_frag_ratd(0), tiny(1d0))
                        t_ratd_dest(pp) = min(t_temp/chi_val,1d15*yr2sec)
                        t_comb = t_comb + 1d0 / t_ratd_dest(pp)

                        ! Contribution to PAH bins (if this chemistry has PAH coupling)
                        if (dustbins_props(dust_parent)%interact_pah .and. dust_pahs .and. npah>0) then
                            do ii=1,npah
                                chi_val = max(dustbins_props(dust_parent)%chi_frag_ratd(ii), tiny(1d0))
                                t_ratd(ii) = min(t_temp/chi_val,1d15*yr2sec)
                                t_comb = t_comb + 1d0 / t_ratd(ii)
                            end do
                        end if

                        ! Contribution to lower dust bins of the same chemistry
                        do id_child=id_start,id_end-1
                            if (dustbins_props(dust_parent)%interact_pah .and. npah>0) then
                                ich = npah + (id_child - id_start + 1)
                            else
                                ich = id_child - id_start + 1
                            end if
                            chi_val = max(dustbins_props(dust_parent)%chi_frag_ratd(ich), tiny(1d0))
                            t_ratd(id_child+npah) = min(t_temp/chi_val,1d15*yr2sec)
                            t_comb = t_comb + 1d0 / t_ratd(id_child+npah)
                        end do

                        if (t_comb > 0d0) then
                            t_ratd(pp) = min(1d0/t_comb,1d15*yr2sec)
                        end if
                    end if
                end do
            end if
        end subroutine compute_t_ratd

    end subroutine compute_dust_timescales

    subroutine compute_t_sputtering_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                          T_dust,Zmean_dust,gamma_RAT,FIR, &
                                          nElement,xelem_ions,fcharge_pahs, &
                                          lead_elem,nions_lead,int_dust_ratd, &
                                          boost_acc,boost_coa,rhoZ_lim, &
                                          t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_sputtering=.true.)
    end subroutine compute_t_sputtering_rates

    subroutine compute_t_accretion_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                         T_dust,Zmean_dust,gamma_RAT,FIR, &
                                         nElement,xelem_ions,fcharge_pahs, &
                                         lead_elem,nions_lead,int_dust_ratd, &
                                         boost_acc,boost_coa,rhoZ_lim, &
                                         t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_accretion=.true.)
    end subroutine compute_t_accretion_rates

    subroutine compute_t_coagulation_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                           T_dust,Zmean_dust,gamma_RAT,FIR, &
                                           nElement,xelem_ions,fcharge_pahs, &
                                           lead_elem,nions_lead,int_dust_ratd, &
                                           boost_acc,boost_coa,rhoZ_lim, &
                                           t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_coagulation=.true.)
    end subroutine compute_t_coagulation_rates

    subroutine compute_t_shattering_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                          T_dust,Zmean_dust,gamma_RAT,FIR, &
                                          nElement,xelem_ions,fcharge_pahs, &
                                          lead_elem,nions_lead,int_dust_ratd, &
                                          boost_acc,boost_coa,rhoZ_lim, &
                                          t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_shattering=.true.)
    end subroutine compute_t_shattering_rates

    subroutine compute_t_ratd_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                    T_dust,Zmean_dust,gamma_RAT,FIR, &
                                    nElement,xelem_ions,fcharge_pahs, &
                                    lead_elem,nions_lead,int_dust_ratd, &
                                    boost_acc,boost_coa,rhoZ_lim, &
                                    t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_ratd=.true.)
    end subroutine compute_t_ratd_rates

    subroutine compute_t_pah_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                   T_dust,Zmean_dust,gamma_RAT,FIR, &
                                   nElement,xelem_ions,fcharge_pahs, &
                                   lead_elem,nions_lead,int_dust_ratd, &
                                   boost_acc,boost_coa,rhoZ_lim, &
                                   t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        implicit none
        real(dp), intent(in) :: Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total
        real(dp), intent(inout) :: ne
        real(dp), intent(in) :: T_dust(:), Zmean_dust(:), gamma_RAT(:), FIR(:)
        real(dp), intent(in) :: nElement(:), xelem_ions(:,:)
        real(dp), intent(in) :: fcharge_pahs(:,:)
        integer, intent(in) :: lead_elem(:), nions_lead(:)
        logical, intent(inout) :: int_dust_ratd(:)
        real(dp), intent(in) :: boost_acc(:), boost_coa(:), rhoZ_lim(:,:)
        real(dp), intent(inout) :: t_acc(:,:), t_coa(:,:), t_sha(:,:,:), t_sha_pah(:,:,:)
        real(dp), intent(inout) :: t_spu(:), t_spu_pah(:), t_subl(:), t_coal(:), t_fre(:,:), t_evap(:), t_ratd(:), t_ratd_dest(:)

        call compute_dust_timescales(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                     T_dust,Zmean_dust,gamma_RAT,FIR, &
                                     nElement,xelem_ions,fcharge_pahs, &
                                     lead_elem,nions_lead,int_dust_ratd, &
                                     boost_acc,boost_coa,rhoZ_lim, &
                                     t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest, &
                                     do_pah_sputtering=.true.,do_pah_sublimation=.true.,do_pah_coalescence=.true., &
                                     do_pah_freezing=.true.,do_pah_evaporation=.true.)
    end subroutine compute_t_pah_rates

end module dust_rates
