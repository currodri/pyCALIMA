! Dust chemistry module for radiation-hydrodynamics.
! For details, see Dubois et al. 2024 and Rodriguez Montero et al. 2024
! By: Yohan Dubois (Original: 1 Feb 2022)
! Changes:
!       - Curro Rodriguez (Cleaning into external 
!                           module: 21 Feb 2022)
!       - Curro Rodriguez (Rewritting routines for
!                           for consistency
!                           module: 13 Jun 2022)
!       - Curro Rodriguez (Adding PAH modelling:
!                           6 Dec 2022)



module dust_chemistry_solver
    use amr_parameters
    use hydro_parameters, only:ndust,ndchemtype,n_elements
    use hydro_commons, only:nmetals
    use constants, only:yr2sec,Myr2sec,amu2g,mH,kB
    use dust_commons
    use dust_utils
    use dust_rates, only: compute_t_sputtering_rates, compute_t_accretion_rates,&
                            compute_t_coagulation_rates, compute_t_shattering_rates,&
                            compute_t_ratd_rates, compute_t_pah_rates
#ifdef RTZ
    use rtz_module, only: elements
#endif

    implicit none

    ! Global quantities
    real(dp),dimension(1:ndust+npah)               :: drhoD_acc,drhoD_spu,drhoD_coa
    real(dp),dimension(1:ndust+npah)               :: drhoD_ratd,drhoD_sha,drhoD_subl,drhoD_sha_dest
    real(dp),dimension(1:ndust+npah)               :: drhoD_ratd_dest
    real(dp),dimension(1:ndust+npah)               :: drhoD_coal,drhoD_fre,drhoD_deso,drhoD_evap

    contains

    function check_update_dust(t0,ddt,rho,rho_dust,rho_pah,nElement,tracked_elements)
        implicit none
        logical                               :: check_update_dust
        real(dp),dimension(1:ndust+npah,1:14) :: t0
        real(dp)                              :: ddt
        real(dp)                              :: rho
        real(dp),dimension(1:ndust)           :: rho_dust
        real(dp),dimension(1:npah)            :: rho_pah
        real(dp),dimension(1:n_elements)      :: nElement
        logical,dimension(1:n_elements)       :: tracked_elements

        integer  :: i
        logical  :: tiny_tracked_metals,has_tracked_elements
        real(dp) :: rho_el

        check_update_dust = .true.
        ! 1. If the dust density is zero, do not update the dust
        !    (this is the case for the first timestep)
        if (all(rho_dust.eq.0d0).and.all(rho_pah.eq.0d0)) check_update_dust = .false.

        ! 2. If the dust and metal densities are ridiculously small
        !   (usually caused by numerical diffusion over a 0 metallicity
        !   region) do not update the dust
        tiny_tracked_metals = .true.
        has_tracked_elements = .false.
        do i=1,n_elements
            if (.not. tracked_elements(i)) cycle
            has_tracked_elements = .true.
#ifdef RTZ
            rho_el = nElement(i) * elements(i)%atomic_mass * amu2g
#else
            rho_el = nElement(i) * el_atomic_masses_amu(i) * amu2g
#endif
            if (rho_el/rho.ge.1d-40) then
                tiny_tracked_metals = .false.
                exit
            end if
        end do
        if (.not.has_tracked_elements) tiny_tracked_metals = .true.

        if (all(rho_dust/rho.lt.smallr_dust).and.all(rho_pah/rho.lt.smallr_dust)&
            .and.tiny_tracked_metals) check_update_dust = .false.

        ! 2. If all timescales are much longer than the age of the Universe
        !   just do not update the dust.
        if (all(t0.ge.1d15*yr2sec)) check_update_dust = .false.
        if (.not.check_update_dust) return

        ! 3. If the introduced cooling timestep is so much shorter compared
        !   to the dust timescales there is no point in computing the change
        !   in dust if it will be below the 1e-10 precission. This speeds up
        !   the code.
        if (all(t0.ge.1e9*ddt)) check_update_dust = .false.
        if (.not.check_update_dust) return
    end function check_update_dust

    subroutine update_dust_counter
        implicit none
        integer :: ii

        ! Add changes to counting global variables
        do ii=npah+1,ndust+npah
            ! Sum on every cell
            if(dust_accretion) dM_acc(ii) = dM_acc(ii) + drhoD_acc(ii)
            if(dust_sputtering) dM_spu(ii) = dM_spu(ii) + drhoD_spu(ii)
            if(dust_coagulation) dM_coa(ii) = dM_coa(ii) + drhoD_coa(ii)
            if(dust_shattering) dM_sha(ii) = dM_sha(ii) + drhoD_sha(ii)
            if(dust_ratd) then
                                dM_ratd(ii) = dM_ratd(ii) + drhoD_ratd(ii)
                                dM_ratd_dest(ii) = dM_ratd_dest(ii) + drhoD_ratd_dest(ii)
            end if
            if(dust_turbulent_model.and.dust_shattering) dM_sha_dest(ii) = dM_sha_dest(ii) + drhoD_sha_dest(ii)
            if(pah_freezing) dM_fre(ii) = dM_fre(ii) + drhoD_fre(ii)
        enddo
        
        if (npah>0) then
            do ii=1,npah
                if(pah_accretion) dM_acc(ii) = dM_acc(ii) + drhoD_acc(ii)
                if(pah_sputtering) dM_spu(ii) = dM_spu(ii) + drhoD_spu(ii)
                if(pah_photolysis) dM_subl(ii) = dM_subl(ii) + drhoD_subl(ii)
                if(pah_cluster_evaporation) dM_evap(ii) = dM_evap(ii) + drhoD_evap(ii)
                if(pah_coalescence) dM_coal(ii) = dM_coal(ii) + drhoD_coal(ii)
                if(pah_freezing) dM_fre(ii) = dM_fre(ii) + drhoD_fre(ii)
                ! if(pah_desorption) dM_deso = dM_deso + drhoD_deso
                dM_sha(ii) = dM_sha(ii) + drhoD_sha(ii)
                if (dust_ratd) dM_ratd(ii) = dM_ratd(ii) + drhoD_ratd(ii)
            end do
        end if
    end subroutine update_dust_counter

    subroutine cmp_lim_elem(dust_index,n_el,el_density,lim_index)
        ! This subroutine compares the element densities tracked in
        ! the composition of a dust bin and finds the limiting element
        ! for the accretion process. It is used to determine which
        ! element is limiting the accretion of dust in a given dust bin.
        ! dust_index ==> index of the dust bin in dustbins_props
        ! n_el       ==> number of elements in the dust bin
        ! el_density ==> array of element densities in the dust bin
        ! lim_index  <== index of the limiting element in the dust bin

        implicit none
        integer,intent(in) :: dust_index
        integer,intent(in) :: n_el
        real(dp),dimension(1:n_el),intent(in) :: el_density
        integer,intent(inout) :: lim_index

        real(dp),dimension(1:n_el) :: el_density_lim
        integer :: i

        do i = 1, n_el
            el_density_lim(i) = el_density(i) / dustbins_props(dust_index)%el_mfractions(i)
        end do
        ! Find the limiting element
        lim_index = minloc(el_density_lim,1)
    end subroutine cmp_lim_elem

    subroutine dust_fine(ddt, dx_loc, Tk, rho_pah &
                            , rho_dust, T_dust &
                            , Zdust_mean, fcharge_pahs &
                            , rho, G0_total &
                            , sigma,local_mu &
                            , gamma_RAT, FIR &
                            , ne, nElement &
                            , xelem_ions &
                            , global_check)
        ! Main routine for the update of dust during cooling
        !
        ! (everything in physical units unless stated)
        ! ddt => timestep asked by the cooling
        ! dx_loc   => size of cell
        ! Tk       => gas temperature (NOTE: without mu factor)
        ! rho_pah  => density in PAHs
        ! rho_dust => dust density
        ! rho      => full gas density
        ! nH       => hydrogen number density
        ! sigma    => local velocity dispersion
        ! nX       => number density of element X in the gas phase

        use hydro_parameters
        implicit none
        real(dp), intent(in)                      :: ddt
        real(dp), intent(in)                      :: dx_loc
        real(dp), intent(in)                      :: Tk, rho, G0_total, sigma, local_mu
        real(dp), intent(inout)                   :: ne
        real(dp),dimension(1:ndust),intent(in)    :: gamma_RAT,FIR
        real(dp),dimension(1:npah),intent(inout)  :: rho_pah
        real(dp),dimension(1:ndust),intent(inout) :: rho_dust,T_dust,Zdust_mean
        real(dp),intent(in)                       :: fcharge_pahs(:,:)
        real(dp),intent(inout)                    :: nElement(1:n_elements)
        real(dp),intent(inout)                    :: xelem_ions(1:n_elements,1:n_elements)

        logical                                   :: okdust,global_check
        logical,dimension(1:ndchemtype)           :: int_dust_ratd=.false.,int_acc_switch=.false.
        logical,dimension(1:ndust+npah)           :: update_switch
        logical                                   :: has_pah_interaction
        logical,dimension(1:n_elements)           :: tracked_elements
        integer,dimension(1:ndust)                :: lead_elem
        integer                                   :: ii, jj, kk, jj1, jj2, kk1, kk2, ikey
        integer                                   :: iel_local, iel_global, iatomic, nions_el
        integer                                   :: ratd_start_dust_global, ratd_source_local, ratd_dest_local
        integer                                   :: nd_ctype
        integer,dimension(1:2)                    :: minloc_t0
        integer,dimension(1:ndchemtype)           :: nions_lead
        real(dp),dimension(:,:),allocatable       :: ion_frac
        real(dp)                                  :: scale_nH, scale_T2, &
                                                     scale_l, scale_d, &
                                                     scale_t, scale_v, &
                                                     lambda_jeans
        real(dp)                                  :: mach,nH
        real(dp)                                  :: sigs, sigs2
        real(dp)                                  :: T6, dtremain, vol_loc
        real(dp)                                  :: dlead,rhoGZ0_sum
        real(dp),dimension(:),allocatable         :: rho_metals_1,rho_metals_2,rho_metals_3
        real(dp),dimension(1:ndust)               :: smax,boost_acc,boost_coa,boost_coulomb
        real(dp),dimension(:,:),allocatable       :: t_acc,oneovertacc
        real(dp),dimension(1:ndust,1:ndust)       :: t_coa
        real(dp),dimension(1:ndust,1:ndust,0:ndust):: t_sha,t_sha_dest
        real(dp),dimension(:,:,:),allocatable     :: t_sha_pah,t_sha_dest_pah
        real(dp),dimension(1:ndust)               :: t_spu
        real(dp),dimension(1:npah)                :: t_subl_pah,t_coal,t_spu_pah
        real(dp),dimension(:,:),allocatable       :: t_fre,oneovertfre
        real(dp),dimension(1:ndust+npah)          :: t_ratd=0d0
        real(dp),dimension(1:ndust+npah)          :: t_ratd_dest=0d0,t_evap=0d0
        real(dp),dimension(1:ndust,1:ndust)       :: oneovertcoa
        real(dp),dimension(1:ndust,1:ndust,0:ndust):: oneovertsha,oneovertsha_dest
        real(dp),dimension(:,:,:),allocatable     :: oneovertsha_pah,oneovertsha_dest_pah
        real(dp),dimension(1:ndust)               :: oneovertspu,oneovertratd,oneovertratd_dest
        real(dp),dimension(1:npah)                :: oneovertsubl_pah,oneovertcoal,oneovertevap,oneovertspu_pah
        real(dp),dimension(:,:),allocatable       :: rhoZ0,rhoGZ0,rhoGZ00,diff_rhoGZ,rhoZ0_lim
        real(dp),dimension(1:ndust+npah,1:14)     :: t0
        real(dp) :: renorm_chi_frag_ratd

        ! Initialise the counters
        drhoD_acc(:)=0.0d0;drhoD_spu(:)=0.0d0;drhoD_coa(:)=0.0d0
        drhoD_sha(:)=0.0d0;drhoD_ratd(:)=0d0;drhoD_ratd_dest(:)=0.0d0;drhoD_subl(:)=0.0d0
        drhoD_sha_dest(:)=0.0d0
        drhoD_coal(:)=0.0d0;drhoD_fre(:)=0.0d0;drhoD_deso(:)=0.0d0;drhoD_evap(:)=0.0d0
        if (npah>0) then
            jj1 = istart_chemtype(pahbins_props(1)%dust_index_interact)
            jj2 = jj1 + dustbins_per_chemtype(pahbins_props(1)%dust_index_interact) - 1
            allocate(t_sha_pah(jj1:jj2,jj1:jj2,1:npah))
            allocate(t_sha_dest_pah(jj1:jj2,jj1:jj2,1:npah))
            allocate(oneovertsha_pah(jj1:jj2,jj1:jj2,1:npah))
            allocate(oneovertsha_dest_pah(jj1:jj2,jj1:jj2,1:npah))
            allocate(t_fre(1:npah,jj1:jj2))
            allocate(oneovertfre(1:npah,jj1:jj2))
        end if

        ! Compute the local cell volume
        vol_loc = dx_loc**ndim

        ! Compute the local Jeans length
        nH = nElement(1)
        lambda_jeans = 4.81973044d19 * sqrt(Tk/nH) ! Prefactor is sqrt(kB*pi/(G*mH**2))

        ! Add up all the dust mass
        if (dust_log) then
            if (npah>0) then
                do ii=1,npah
                    total_mdust(ii) = total_mdust(ii) + rho_pah(ii) * vol_loc
                end do
                do ii=1,ndust
                    total_mdust(ii+npah) = total_mdust(ii+npah) + rho_dust(ii) * vol_loc
                end do
            else
                do ii=1,ndust
                    total_mdust(ii) = total_mdust(ii) + rho_dust(ii) * vol_loc
                end do
            end if
        end if

        call units(scale_l,scale_t,scale_d,scale_v,scale_nH,scale_T2)

        if (dust_acc_coulomb) then
            allocate(rhoZ0(1:ndchemtype,1:n_elements)); rhoZ0 = 0d0
            allocate(rhoGZ0(1:ndchemtype,1:n_elements)); rhoGZ0 = 0d0
            allocate(rhoGZ00(1:ndchemtype,1:n_elements)); rhoGZ00 = 0d0
            allocate(diff_rhoGZ(1:ndchemtype,1:n_elements)); diff_rhoGZ = 0d0
            allocate(ion_frac(1:ndchemtype,1:n_elements)); ion_frac = 0d0
            allocate(rhoZ0_lim(1:ndchemtype,1:n_elements)); rhoZ0_lim = 0d0
            allocate(t_acc(1:ndust,1:n_elements)); t_acc = 0d0
            allocate(oneovertacc(1:ndust+npah,1:n_elements))
        else
            allocate(rhoZ0(1:ndchemtype,1:1)); rhoZ0 = 0d0
            allocate(rhoGZ0(1:ndchemtype,1:1)); rhoGZ0 = 0d0
            allocate(rhoGZ00(1:ndchemtype,1:1)); rhoGZ00 = 0d0
            allocate(diff_rhoGZ(1:ndchemtype,1:1)); diff_rhoGZ = 0d0
            allocate(rhoZ0_lim(1:ndchemtype,1:1)); rhoZ0_lim = 0d0
            allocate(t_acc(1:ndust,1:1)); t_acc = 0d0
            allocate(ion_frac(1:ndchemtype,1:1)); ion_frac = 0d0
            allocate(oneovertacc(1:ndust+npah,1:1))
        end if
        tracked_elements = .false.
        T6 = Tk/1d6
        speciesloop: do jj=1,ndchemtype ! Loop over chemical dust species
            jj1 = istart_chemtype(jj)
            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
            do ii = 1, dustbins_props(jj1)%nelements
                tracked_elements(dustbins_props(jj1)%el_atomic_number(ii)) = .true.
            end do
            if (dustbins_props(jj1)%nelements == 1) then
                if (dust_acc_coulomb) then
                    nions_lead(jj) = dustbins_props(jj1)%el_nions(1)
                    ion_frac(jj,1:nions_lead(jj)) = xelem_ions(dustbins_props(jj1)%el_index(1),1:nions_lead(jj))
                    rhoZ0(jj,1:nions_lead(jj)) = nElement(dustbins_props(jj1)%el_atomic_number(1)) * &
                        dustbins_props(jj1)%el_atomic_masses_g(1) * ion_frac(jj,1:nions_lead(jj))
                else
                    nions_lead(jj) = 1
                    ion_frac(jj,1) = 1d0
                    rhoZ0(jj,1) = nElement(dustbins_props(jj1)%el_atomic_number(1)) * &
                        dustbins_props(jj1)%el_atomic_masses_g(1)
                end if
                lead_elem(jj) = 1
            else
                ! TODO: Improve!
                ! NOTE: The handling of metal densities and limiting elements is a bit
                ! convoluted, but this is necessary for accurate dust evolution modeling.
                allocate(rho_metals_1(1:dustbins_props(jj1)%nelements))
                allocate(rho_metals_2(1:dustbins_props(jj1)%nelements))
                allocate(rho_metals_3(1:dustbins_props(jj1)%nelements))
                do ii = 1, dustbins_props(jj1)%nelements
                    ! rho_metals_1: this is the density in the gas phase of the metals converted
                    ! to the equivalent dust mass.
                    rho_metals_1(ii) = nElement(dustbins_props(jj1)%el_atomic_number(ii)) * &
                        dustbins_props(jj1)%el_atomic_masses_g(ii) / dustbins_props(jj1)%el_mfractions(ii)
                    ! rho_metals_2: this is the density of the elements in the gas phase plus the
                    ! mass depleted in the dust.
                    rho_metals_2(ii) = rho_metals_1(ii) + sum(rho_dust(jj1:jj2)) * dustbins_props(jj1)%el_mfractions(ii)
                    ! rho_metals_3: this is not really a density but rather the correct way for obtaining the limiting element
                    ! which includes not just abundances but also the effect of ion mass on the interaction rate with the dust.
                    ! see Eq. 8 in Rodriguez Montero et al. 2025
                    rho_metals_3(ii) = rho_metals_2(ii) / (dustbins_props(jj1)%el_mfractions(ii) * &
                        dustbins_props(jj1)%el_atomic_masses_amu(ii)**0.5)
                end do
                ! Find the limiting element
                ikey = minloc(rho_metals_3, dim=1)
                lead_elem(jj) = ikey
                if (dust_acc_coulomb) then
                    nions_lead(jj) = dustbins_props(jj1)%el_nions(ikey)
                    ion_frac(jj,1:nions_lead(jj)) = xelem_ions(dustbins_props(jj1)%el_index(ikey),1:nions_lead(jj))
                    rhoZ0(jj,1:nions_lead(jj)) = minval(rho_metals_1) * ion_frac(jj,1:nions_lead(jj))
                else
                    nions_lead(jj) = 1
                    ion_frac(jj,1) = 1d0
                    rhoZ0(jj,1) = minval(rho_metals_1)
                end if
            end if
            ! Turn off accretion if the metal density is zero
            if (sum(rhoZ0(jj,:)).le.0d0) int_acc_switch(jj) = .false.

            ! Save the gas phase metal densities
            rhoGZ0(jj,1:nions_lead(jj)) = rhoZ0(jj,1:nions_lead(jj))
            rhoGZ00(jj,1:nions_lead(jj)) = rhoZ0(jj,1:nions_lead(jj))

            ! Now add the dust density to the total gas + dust metal densities
            if (npah>0 .and. dustbins_props(jj1)%interact_pah) then
                rhoZ0(jj,:) = rhoZ0(jj,:) + SUM(rho_pah(:)) + SUM(rho_dust(jj1:jj2))
            else
                rhoZ0(jj,:) = rhoZ0(jj,:) + SUM(rho_dust(jj1:jj2))
            end if

            ! Save the true limiting density
            if (dustbins_props(jj1)%nelements == 1) then
                rhoZ0_lim(jj,1:nions_lead(jj)) = rhoZ0(jj,1:nions_lead(jj))
            else
                if (dust_acc_coulomb) then
                    rhoZ0_lim(jj,1:nions_lead(jj)) = rho_metals_2(ikey) * ion_frac(jj,1:nions_lead(jj))
                else
                    rhoZ0_lim(jj,1) = rho_metals_2(ikey)
                end if
            end if
        end do speciesloop

        ! Compute local dust timescales
        mach      = MAX(1d-5,sigma/sqrt(1.666667d0*kB*Tk/mH))
        sigs      = log(1d0+(0.4d0*mach)**2)
        sigs2     = sigs*sigs
        do ii = 1, ndust
            smax(ii) = log(dustbins_props(ii)%nhmax_acc/nH)
            boost_acc(ii) = 0.5d0*exp(sigs2)*erfc((1.5d0*sigs2-smax(ii))/(dsqrt(2d0)*sigs))
            smax(ii) = log(dustbins_props(ii)%nhmax_coa/nH)
            boost_coa(ii) = 0.5d0*exp(sigs2)*erfc((1.5d0*sigs2-smax(ii))/(dsqrt(2d0)*sigs))
        end do

        call compute_dust_local_rates

        ! Initialize destruction timescales from primary timescales
        if(dust_shattering) t_sha_dest = t_sha
        if(dust_pahs .and. dust_shattering) t_sha_dest_pah = t_sha_pah

        if(dust_sputtering) oneovertspu = 1d0/t_spu
        if(dust_accretion) oneovertacc(1+npah:ndust+npah,:) = 1d0/t_acc
        if(dust_shattering)oneovertsha = 1d0/t_sha

        if(dust_shattering.and.dust_turbulent_model) oneovertsha_dest = 1d0/t_sha_dest
        if(dust_coagulation) oneovertcoa = 1d0/t_coa
        if(dust_ratd) then
            oneovertratd(1:ndust) = 1d0/t_ratd(1+npah:ndust+npah)
            oneovertratd_dest(1:ndust) = 1d0/t_ratd_dest(1+npah:ndust+npah)
        end if
        if (npah>0) then
            if(pah_sputtering) oneovertspu_pah = 1d0/t_spu_pah
            if(pah_accretion) oneovertacc(1:npah,:) = 1d0/t_acc
            if(pah_photolysis) oneovertsubl_pah = 1d0/t_subl_pah
            if(pah_coalescence) oneovertcoal = 1d0/t_coal
            if(pah_freezing) oneovertfre(1:npah,:) = 1d0/t_fre(1:npah,:)
            if(pah_cluster_evaporation) oneovertevap = 1d0/t_evap(1:npah)
            if(dust_ratd) oneovertratd(1:npah) = 1d0/t_ratd(1:npah)
            if(dust_shattering.and.dust_turbulent_model) oneovertsha_dest_pah = 1d0/t_sha_dest_pah
        end if
        
        t0 = 1d15 * yr2sec
        do ii=npah+1,ndust+npah
            if(dust_accretion)    t0(ii,1) = 0.1d0 * minval(t_acc(ii-npah,:))
            if(dust_sputtering)   t0(ii,2) = 0.1d0 * t_spu(ii-npah)
            if(dust_shattering)   t0(ii,3) = 0.1d0 * minval(t_sha(ii-npah,ii-npah,:))
            if(dust_coagulation)  t0(ii,4) = 0.1d0 * minval(t_coa(ii-npah,:))
            if(dust_ratd)then
                                  t0(ii,5) = 0.1d0 * t_ratd(ii)
                                  t0(ii,6) = 0.1d0 * t_ratd_dest(ii)
            end if
        enddo
        if (npah>0) then
            do ii=1,npah
                if(pah_accretion)    t0(ii,1) = 0.1d0 * minval(t_acc(1,:))
                if(pah_sputtering) t0(ii,2) = 0.1d0 * t_spu_pah(ii)
                if(pah_photolysis) t0(ii,10) = 0.1d0 * t_subl_pah(ii)
                if(pah_coalescence) t0(ii,11) = 0.1d0 * t_coal(ii)
                ! if(pah_desorption) t0(ii,12) = 0.1d0 * t_deso(ii)
                if(pah_freezing) t0(ii,13) = 0.1d0 * minval(t_fre(ii,:))
                if(pah_cluster_evaporation) t0(ii,14) = 0.1d0 * t_evap(ii)
                if(dust_ratd) t0(ii,5) = 0.1d0 * t_ratd(ii)
            end do
        end if

        ! Save to the counter the dust process that limits the timestep
        ! of the dust chemistry solver
        if (dust_log) then
            ! Make sure that at least one of the values is the array is different
            ! from the initial value (1d15*yr2sec)
            if (.not. all(t0.eq.1d15*yr2sec)) then
                minloc_t0 = minloc(t0)
                nt0_limiter(minloc_t0(1),minloc_t0(2)) = nt0_limiter(minloc_t0(1),minloc_t0(2)) + 1
            end if
        end if

        global_check = check_update_dust(t0,ddt,rho,rho_dust,rho_pah,nElement,tracked_elements)
        print*,'check_update_dust ',global_check,' t0 ',t0,' ddt ',ddt,' rho ',rho,' rho_dust ',rho_dust,' rho_pah ',rho_pah, ' tracked_elements ',tracked_elements,' nElement ',nElement
        call clean_stop
        if (.not.global_check) return
        print*,'t_acc ',t_acc
        call clean_stop

        ! Now, loop over dust species and update their densities
        rkloop: do jj=1,ndchemtype
            jj1 = npah + istart_chemtype(jj)
            jj2 = jj1 + dustbins_per_chemtype(jj) - 1
            has_pah_interaction = dust_pahs .and. dustbins_props(istart_chemtype(jj))%interact_pah
            if (has_pah_interaction) then
                jj1 = 1
                jj2 = npah + dustbins_per_chemtype(jj)
            end if
            
            dtremain = ddt
            ! THE UPDATE SWITCH
            update_switch = .true.

            ! TODO: This is a temporary fix
            ! Sometimes when the temperature is too high, the thermal sputtering
            ! of PAHs becomes ridiculously quick compared to the rest of timescales
            ! and the cooling timestep
            if (dust_pahs .and. pah_sputtering) then
                if (has_pah_interaction .and. (any(t_spu_pah(1:npah)/ddt<5d-3))) then
                    ! write(*,*)'PAH sputtering too quick'
                    ! write(*,*)'t_spu_pah/ddt ',t_spu_pah(1:npah)/ddt
                    ! write(*,*)'nH,rho,Tk ',nH,rho,Tk
                    ! write(*,*)'rho_pah ',rho_pah
                    do kk = 1,npah
                        if (all(t0(kk,:)>=0.1d0*t_spu_pah(kk)).and.(t_spu_pah(kk)<minval(t_sha(1,:,:)))) then
                            update_switch(kk) = .false.
                            if (dust_log) then
                                ! Save the number of cells and the amount of dust mass that is\
                                ! subject to the skipping of the RK4 step
                                ncells_skipped(kk,2) = ncells_skipped(kk,2) + 1
                                mdust_skipped(kk,2) = mdust_skipped(kk,2) + rho_pah(kk) * vol_loc
                            end if
                            rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_pah(kk)
                            drhoD_spu(kk) = -rho_pah(kk)
                            rho_pah(kk) = 0d0
                            t0(kk,:) = 1d15 * yr2sec
                            oneovertspu(kk) = 1d0/t0(kk,2)
                        end if
                    end do
                end if
            end if
            
            ! TODO: This is a temporary fix
            ! When too close to a source the radiation field is very intense,
            ! large PAHs quickly dissociate into small ones, much faster than
            ! any of the other timescales.
            if (dust_pahs .and. pah_cluster_evaporation) then
                if (has_pah_interaction .and. (any(t_evap(1:npah)/ddt<1d-2))) then
                    do kk = 1,npah
                        if (all(t0(kk,:)>=0.1d0*t_evap(kk)).and.(t_evap(kk)<minval(t_sha(1,:,:)))) then
                            update_switch(kk) = .false.
                            if (dust_log) then
                                ! Save the number of cells and the amount of dust mass that is\
                                ! subject to the skipping of the RK4 step
                                ncells_skipped(kk,14) = ncells_skipped(kk,14) + 1
                                mdust_skipped(kk,14) = mdust_skipped(kk,14) + rho_pah(kk) * vol_loc
                            end if
                            if (update_switch(kk-1)) then
                                rho_pah(kk-1) = rho_pah(kk-1) + rho_pah(kk)
                                drhoD_evap(kk-1) = rho_pah(kk)
                                drhoD_evap(kk) = -rho_pah(kk)
                                rho_pah(kk) = 0d0
                            else
                                rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_pah(kk)
                                drhoD_evap(kk) = -rho_pah(kk)
                                rho_pah(kk) = 0d0
                            end if
                            t0(kk,:) = 1d15 * yr2sec
                            oneovertevap(kk) = 1d0/t0(kk,14)
                        end if
                    end do
                end if
            end if

            ! TODO: This is a temporary fix
            ! If RATD timescales become incredibly tinny with high G0, we just
            ! destroy all dust and stop the update of that grain
            if (dust_ratd) then
                do ii=jj2,jj1,-1
                    if (t_ratd(ii)/ddt < 1d-2) then
                        update_switch(ii) = .false.
                        if (dust_log) then
                            ! Save the number of cells and the amount of dust mass that is
                            ! subject to the skipping of the RK4 step
                            ncells_skipped(ii,5) = ncells_skipped(ii,5) + 1
                            mdust_skipped(ii,5) = mdust_skipped(ii,5) + rho_dust(ii) * vol_loc
                        end if

                        ratd_start_dust_global = npah + istart_chemtype(jj)
                        if (has_pah_interaction) ratd_start_dust_global = npah + pahbins_props(1)%dust_index_interact
                        ratd_source_local = ii - ratd_start_dust_global + 1
                        
                        if (has_pah_interaction) then
                            renorm_chi_frag_ratd = sum(dustbins_props(ii-npah)%chi_frag_ratd(0:npah+ratd_source_local-1))
                            rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(0)/renorm_chi_frag_ratd
                            drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii) -rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(0)/renorm_chi_frag_ratd
                            if (npah>0) then
                                do kk=1,npah
                                    if (update_switch(kk)) then
                                        rho_pah(kk) = rho_pah(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(kk)/renorm_chi_frag_ratd
                                        drhoD_ratd(kk) = drhoD_ratd(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(kk)/renorm_chi_frag_ratd
                                    else
                                        rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(kk)/renorm_chi_frag_ratd
                                        drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii) - rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(kk)/renorm_chi_frag_ratd
                                    end if
                                end do
                            end if
                            do kk=1+npah,ii-1
                                ratd_dest_local = npah + (kk - ratd_start_dust_global + 1)
                                if (update_switch(kk)) then
                                    rho_dust(kk) = rho_dust(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                    drhoD_ratd(kk) = drhoD_ratd(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                else
                                    rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                    drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii) - rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                end if
                            end do
                        else
                            renorm_chi_frag_ratd = sum(dustbins_props(ii-npah)%chi_frag_ratd(0:ratd_source_local-1))
                            rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(0)/renorm_chi_frag_ratd
                            drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii) - rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(0)/renorm_chi_frag_ratd
                            do kk=jj1,ii-1
                                ratd_dest_local = kk - ratd_start_dust_global + 1
                                if (update_switch(kk)) then
                                    rho_dust(kk) = rho_dust(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                    drhoD_ratd(kk) = drhoD_ratd(kk) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                else
                                    rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                    drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii) - rho_dust(ii) * dustbins_props(ii-npah)%chi_frag_ratd(ratd_dest_local)/renorm_chi_frag_ratd
                                end if
                            end do
                        end if
                        drhoD_ratd(ii) = -rho_dust(ii)
                        rho_dust(ii) = 0d0
                        t0(ii,:) = 1d15 * yr2sec
                        oneovertratd(ii) = 1d0/t0(ii,5)
                        oneovertratd_dest(ii) = 1d0/t0(ii,6)
                        ! print*,'RATD too quick'
                        ! print*,'t_ratd/ddt ',t_ratd(ii)/ddt
                        ! print*,'nH,rho,Tk ',nH,rho,Tk
                        ! print*,'rho_dust(jj1:jj2) ',rho_dust(jj1:jj2)
                        ! print*,'rho_dust_init(jj1:jj2) ',rho_dust_init(jj1:jj2)
                        ! print*,'rho_pah ',rho_pah
                        ! print*,'rho_pah_init ',rho_pah_init
                        ! print*,'sum(rho_dust(jj1:jj2)) - sum(rho_dust_init(jj1:jj2))',sum(rho_dust(jj1:jj2)) - sum(rho_dust_init(jj1:jj2))
                        ! print*,'sum(rho_pah) - sum(rho_pah_init)',sum(rho_pah) - sum(rho_pah_init)
                        ! print*,'sum(rho_dust(jj1:jj2))+sum(rho_pah) - sum(rho_dust_init(jj1:jj2))+sum(rho_pah_init)',sum(rho_dust(jj1:jj2))+sum(rho_pah) - sum(rho_dust_init(jj1:jj2))+sum(rho_pah_init)
                        ! print*,'sum(rho_dust(jj1:jj2))+sum(rho_pah) - sum(rho_dust_init(jj1:jj2))+sum(rho_pah_init) + sum(rhoGZ0(jj,1:nions_lead(jj))) - sum(rhoGZ00(jj,1:nions_lead(jj)))',sum(rho_dust(jj1:jj2))+sum(rho_pah) - sum(rho_dust_init(jj1:jj2))+sum(rho_pah_init) + sum(rhoGZ0(jj,1:nions_lead(jj))) - sum(rhoGZ00(jj,1:nions_lead(jj)))
                        ! print*,'sum(rhoGZ0(jj,1:nions_lead(jj))) - sum(rhoGZ00(jj,1:nions_lead(jj)))',sum(rhoGZ0(jj,1:nions_lead(jj))) - sum(rhoGZ00(jj,1:nions_lead(jj)))
                        ! print*,'drhoD_ratd(ii) = ',drhoD_ratd(ii)
                        ! print*,'drhoD_ratd_dest(ii) = ',drhoD_ratd_dest(ii)
                        ! print*,'drhoD_ratd(jj1:jj2) = ',drhoD_ratd(jj1:jj2)
                        ! print*,'drhoD_ratd_dest(jj1:jj2) = ',drhoD_ratd_dest(jj1:jj2)
                        ! print*,'chi_frag_ratd(ii-npah,:) = ',dustbins_props(ii-npah)%chi_frag_ratd(:)
                        ! print*,'sum(chi_frag_ratd(ii-npah,:)) = ',sum(dustbins_props(ii-npah)%chi_frag_ratd(:))
                    end if
                end do
            end if

            ! TODO: This is a temporary fix
            ! If the thermal sputtering becomes incredibly tiny with high density
            ! and high temperature, we just destroy all dust and stop the update
            ! of that grain
            if (dust_sputtering) then
                do ii=jj1,jj2
                    if ((t_spu(ii)/ddt < 5d-3)) then
                        update_switch(ii) = .false.
                        if (dust_log) then
                            ! Save the number of cells and the amount of dust mass that is\
                            ! subject to the skipping of the RK4 step
                            ncells_skipped(ii,2) = ncells_skipped(ii,2) + 1
                            mdust_skipped(ii,2) = mdust_skipped(ii,2) + rho_dust(ii) * vol_loc
                        end if
                        rhoGZ0(jj,1) = rhoGZ0(jj,1) + rho_dust(ii)
                        drhoD_spu(ii) = -rho_dust(ii)
                        rho_dust(ii) = 0d0
                        t0(ii,:) = 1d15 * yr2sec
                        oneovertspu(ii) = 1d0/t0(ii,2)
                    end if
                end do
            end if

            okdust=.false.
            ! Run Runge-Kutta 4th order integration of dust
            ! evolution equation
            ! =============================================
            if (has_pah_interaction &
                &.and. (any(update_switch(1:npah)))) then
                call dust_with_pah_update_RK4
            else
                if (has_pah_interaction) jj1 = jj1 + npah
                call dust_update_RK4
            end if
            ! =============================================
        end do rkloop

        ! Check that the pahs and dust densities are not negative
        if (npah>0) then
            do ii=1,npah
                if (rho_pah(ii)<0d0.or.rho_pah(ii).ne.rho_pah(ii)) then
                    write(*,*)'Negative PAH density: ',rho_pah(ii)
                    rho_pah(ii) = 0d0
                end if
            end do
        end if
        do ii=1,ndust
            if (rho_dust(ii)<0d0.or.rho_dust(ii).ne.rho_dust(ii)) then
                write(*,*)'Negative dust density: ',rho_dust(ii)
                rho_dust(ii) = 0d0
            end if
        end do

        ! Update metal and ion densities from the gas-phase limiting element updates
        diff_rhoGZ = rhoGZ0 - rhoGZ00
        updateloop: do jj=1,ndchemtype
            jj1 = istart_chemtype(jj)
            dlead = SUM(diff_rhoGZ(jj,1:nions_lead(jj)))

            do iel_local=1,dustbins_props(jj1)%nelements
                iatomic = dustbins_props(jj1)%el_atomic_number(iel_local)
                nElement(iatomic) = nElement(iatomic) + dlead * dustbins_props(jj1)%el_conv_factors(iel_local)
            end do

            if (dust_acc_coulomb) then
                do iel_local=1,dustbins_props(jj1)%nelements
                    iel_global = dustbins_props(jj1)%el_index(iel_local)
                    if (iel_local == lead_elem(jj)) then
                        rhoGZ0_sum = sum(rhoGZ0(jj,1:nions_lead(jj)))
                        if (rhoGZ0_sum.gt.0d0) then
                            xelem_ions(iel_global,:) = 0d0
                            xelem_ions(iel_global,1:nions_lead(jj)) = rhoGZ0(jj,1:nions_lead(jj)) / rhoGZ0_sum
                        end if
                        do kk = 2, nions_lead(jj)
                            ! Keep the electron budget consistent with the ion mass moved in each charge state.
                            ne = ne - dble(kk) * diff_rhoGZ(jj,kk) * dustbins_props(jj1)%el_conv_factors(iel_local)
                        end do
                    else
                        nions_el = dustbins_props(jj1)%el_nions(iel_local)
                        do kk = 2, nions_el
                            ne = ne - dble(kk) * dlead * dustbins_props(jj1)%el_conv_factors(iel_local) * xelem_ions(iel_global,kk)
                        end do
                    end if
                end do
            end if
        end do updateloop

        ! Check that the metal densities are not negative
        if (any((nElement(1:n_elements)<0d0).and.tracked_elements)) then
            write(*,*)'Negative metal density detected at end of dust_fine'
            call print_tracked_elements_state('Tracked element number densities:')
            write(*,*)'rhoGZ0 = ',rhoGZ0
            write(*,*)'rhoGZ00 = ',rhoGZ00
            write(*,*)'rhoZ0 = ',rhoZ0
            write(*,*)'lead_elem = ',lead_elem
            write(*,*)'rho_dust = ',rho_dust
            if (npah>0) write(*,*)'rho_pah = ',rho_pah
            stop
        end if
        ! Add changes to counting global variables
        if (dust_log) then
            vol_loc = dx_loc**ndim
            do ii=npah+1,ndust+npah
                ! Sum on every cell
                if(dust_accretion) drhoD_acc(ii) = drhoD_acc(ii)*vol_loc 
                if(dust_sputtering) drhoD_spu(ii) = drhoD_spu(ii)*vol_loc 
                if(dust_coagulation) drhoD_coa(ii)= drhoD_coa(ii)*vol_loc 
                if(dust_shattering) drhoD_sha(ii) = drhoD_sha(ii)*vol_loc 
                if(dust_ratd) then
                                    drhoD_ratd(ii) = drhoD_ratd(ii)*vol_loc
                                    drhoD_ratd_dest(ii) = drhoD_ratd_dest(ii)*vol_loc
                end if
                if(dust_shattering.and.dust_turbulent_model) drhoD_sha_dest(ii) = drhoD_sha_dest(ii)*vol_loc

                if(pah_freezing) drhoD_fre(ii) = drhoD_fre(ii)*vol_loc
            enddo
            if (npah>0) then
                do ii=1,npah
                    if(pah_sputtering) drhoD_spu(ii) = drhoD_spu(ii)*vol_loc 
                    if(pah_photolysis) drhoD_subl(ii) = drhoD_subl(ii)*vol_loc
                    if(pah_cluster_evaporation) drhoD_evap(ii) = drhoD_evap(ii)*vol_loc
                    drhoD_sha(ii) = drhoD_sha(ii)*vol_loc
                    if(pah_coalescence) drhoD_coal(ii) = drhoD_coal(ii)*vol_loc 
                    if(pah_freezing) drhoD_fre(ii) = drhoD_fre(ii)*vol_loc
                    if(dust_ratd) drhoD_ratd(ii) = drhoD_ratd(ii)*vol_loc
                end do
            end if
        end if

        if (dust_acc_coulomb) then
            deallocate(rhoZ0)
            deallocate(rhoGZ0)
            deallocate(diff_rhoGZ)
            deallocate(ion_frac)
            deallocate(t_acc)
            deallocate(oneovertacc)
        else
            deallocate(rhoZ0)
            deallocate(rhoGZ0)
            deallocate(diff_rhoGZ)
            deallocate(t_acc)
            deallocate(oneovertacc)
        end if

        contains

        subroutine compute_dust_local_rates
            implicit none

            if(dust_sputtering) call compute_t_sputtering_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                         T_dust,Zdust_mean,gamma_RAT,FIR, &
                                         nElement,xelem_ions,fcharge_pahs, &
                                         lead_elem,nions_lead,int_dust_ratd, &
                                         boost_acc,boost_coa,rhoZ0_lim, &
                                         t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
            if(dust_accretion) call compute_t_accretion_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                       T_dust,Zdust_mean,gamma_RAT,FIR, &
                                       nElement,xelem_ions,fcharge_pahs, &
                                       lead_elem,nions_lead,int_dust_ratd, &
                                       boost_acc,boost_coa,rhoZ0_lim, &
                                       t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
            if(dust_coagulation) call compute_t_coagulation_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                           T_dust,Zdust_mean,gamma_RAT,FIR, &
                                           nElement,xelem_ions,fcharge_pahs, &
                                           lead_elem,nions_lead,int_dust_ratd, &
                                           boost_acc,boost_coa,rhoZ0_lim, &
                                           t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
            if(dust_shattering) call compute_t_shattering_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                         T_dust,Zdust_mean,gamma_RAT,FIR, &
                                         nElement,xelem_ions,fcharge_pahs, &
                                         lead_elem,nions_lead,int_dust_ratd, &
                                         boost_acc,boost_coa,rhoZ0_lim, &
                                         t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
            if(dust_ratd) call compute_t_ratd_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                   T_dust,Zdust_mean,gamma_RAT,FIR, &
                                   nElement,xelem_ions,fcharge_pahs, &
                                   lead_elem,nions_lead,int_dust_ratd, &
                                   boost_acc,boost_coa,rhoZ0_lim, &
                                   t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
            if(npah>0) call compute_t_pah_rates(Tk,nH,rho,dx_loc,sigma,local_mu,lambda_jeans,G0_total,ne, &
                                   T_dust,Zdust_mean,gamma_RAT,FIR, &
                                   nElement,xelem_ions,fcharge_pahs, &
                                   lead_elem,nions_lead,int_dust_ratd, &
                                   boost_acc,boost_coa,rhoZ0_lim, &
                                   t_acc,t_coa,t_sha,t_sha_pah,t_spu,t_spu_pah,t_subl_pah,t_coal,t_fre,t_evap,t_ratd,t_ratd_dest)
        end subroutine compute_dust_local_rates

        subroutine dust_update_RK4
            ! This routine solves the dust evolution equations
            ! using the integration method Runge-Kutta of 
            ! 4th order

            implicit none

            logical, dimension(jj1:jj2)::okdt_bin
            integer, dimension(jj1:jj2):: icount,idec_step
            real(dp)::rhoDT0,dtloc,halfdtloc,dd,dtloc_init
            real(dp)::error_rel,error_rel1,error_rel2,den0,den
            real(dp), dimension(jj1:jj2):: rhoD0, drhoD,fD0
            real(dp),dimension(1:nions_lead(jj)):: Gvar,drhoGZ0,rhoGZ
            real(dp),dimension(jj1:jj2)::rhoD
            real(dp), dimension(jj1:jj2)::d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest
            real(dp), dimension(jj1:jj2)::d_sha_dest
            real(dp), dimension(jj1:jj2)::dtloc_bin
            real(dp),dimension(jj1:jj2)::k1,k2,k3,k4
            real(dp),dimension(jj1:jj2)::rhoD0k1,rhoD0k2,rhoD0k3
            real(dp),dimension(1:nions_lead(jj))::rhoGZ0k1,rhoGZ0k2,rhoGZ0k3
            real(dp),dimension(1:nions_lead(jj))::k1_gas,k2_gas,k3_gas,k4_gas
            real(dp),dimension(jj1:jj2)::fD0k1,fD0k2,fD0k3

            icount = 0
            idec_step = 0
            drhoGZ0(:) = 0d0
            
            solveloop: do while (okdust .eqv. .false.)
                d_acc(:)=0.0d0;d_spu(:)=0.0d0;d_coa(:)=0.0d0;d_sha(:)=0.0d0;d_ratd(:)=0.0d0;d_sha_dest(:)=0.0d0
                d_ratd_dest(:)=0.0d0

                ! Get density of dust
                rhoDT0 = 0.d0
                do ii=jj1,jj2
                    rhoD0(ii) = rho_dust(ii-npah)
                    fD0(ii) = rho_dust(ii-npah) / rho
                    rhoDT0 = rhoDT0 + rhoD0(ii)
                    if (icount(ii)==0) dtloc_bin(ii) = MINVAL(t0(ii,:))
                end do

                ! Initialise local timestep
                dtloc = MINVAL(dtloc_bin(jj1:jj2))
                dtloc = MIN(dtloc,dtremain)
                if (all(icount(:)==0)) dtloc_init = dtloc
                halfdtloc = 0.5d0 * dtloc

                ! Begin RK4 for all processes
                k1=0.0d0;k1_gas=0d0;Gvar=0.0d0
                call apply_dust_stage(rhoD0,fD0,rhoGZ0(jj,1:nions_lead(jj)),1d0,k1,Gvar,&
                                      d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                      d_sha_dest)
                rhoD0k1 (:)=rhoD0 (:)+halfdtloc*k1(:)
                fD0k1(:) = rhoD0k1(:)/rho
                rhoGZ0k1(:)=rhoGZ0(jj,:)+halfdtloc*Gvar(:)
                k1_gas(:) = Gvar(:)

                k2=0.0d0;k2_gas=0.d0;Gvar=0.0d0
                call apply_dust_stage(rhoD0k1,fD0k1,rhoGZ0k1,2d0,k2,Gvar,&
                                      d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                      d_sha_dest)
                rhoD0k2 (:)=rhoD0 (:)+halfdtloc*k2(:)
                fD0k2(:) = rhoD0k2(:)/rho
                rhoGZ0k2(:)=rhoGZ0(jj,:)+halfdtloc*Gvar(:)
                k2_gas(:) = Gvar(:)

                k3=0.0d0;k3_gas=0.d0;Gvar=0.0d0
                call apply_dust_stage(rhoD0k2,fD0k2,rhoGZ0k2,2d0,k3,Gvar,&
                                      d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                      d_sha_dest)
                rhoD0k3 (:)=rhoD0 (:)+dtloc*k3(:)
                fD0k3(:) = rhoD0k3(:)/rho
                rhoGZ0k3(:)=rhoGZ0(jj,:)+dtloc*Gvar(:)
                k3_gas(:) = Gvar(:)

                k4=0.0d0;k4_gas=0.d0
                call apply_dust_stage(rhoD0k3,fD0k3,rhoGZ0k3,1d0,k4,k4_gas,&
                                      d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                      d_sha_dest)
                call finalize_dust_stage(dtloc,rhoD0,k1,k2,k3,k4,drhoD,rhoD,&
                                         d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                         d_sha_dest)

                ! Update gas/ion metal densities
                do kk=1,nions_lead(jj)
                    drhoGZ0(kk) = (dtloc/6d0) * (k1_gas(kk)+2d0*k2_gas(kk)+2d0*k3_gas(kk)+k4_gas(kk))
                    rhoGZ(kk) = rhoGZ0(jj,kk) + drhoGZ0(kk)
                end do

                if (abs(abs(sum(drhoGZ0))-abs(sum(drhoD)))/sum(drhoD).gt.1d-15&
                    & .and. sum(drhoD).gt.0d0 .and. sum(drhoGZ0).gt.0d0 &
                    &.and. sum(drhoD)/rho.gt.smallr_dust) then
                    print*,'Error in dust processing: sum(drhoGZ0)/sum(drhoD) /= 1.0'
                    print*,'sum(drhoGZ0)/sum(drhoD) = ',abs(sum(drhoGZ0))/abs(sum(drhoD))
                    print*,'sum(drhoGZ0)-sum(drhoD) = ',abs(sum(drhoGZ0))-abs(sum(drhoD))
                    print*,'k1,k2,k3,k4',k1(jj1:jj2),k2(jj1:jj2),k3(jj1:jj2),k4(jj1:jj2)
                    print*,'k1_gas,k2_gas,k3_gas,k4_gas',k1_gas(:),k2_gas(:),k3_gas(:),k4_gas(:)
                    print*,'sum(k1),sum(k2),sum(k3),sum(k4)',sum(k1(jj1:jj2)),sum(k2(jj1:jj2)),sum(k3(jj1:jj2)),sum(k4(jj1:jj2))
                    print*,'sum(k1_gas),sum(k2_gas),sum(k3_gas),sum(k4_gas)',sum(k1_gas(:)),sum(k2_gas(:)),sum(k3_gas(:)),sum(k4_gas(:))
                    stop
                end if

                ! Check now for integration errors
                okdt_bin = .false.
                do ii=jj1,jj2
                    if(rhoD0(ii)>0.d0) then
                        error_rel1 = ABS(drhoD(ii)) / MIN(rhoD0(ii),rhoD(ii))
                        den0 = (1d0-rhoD0(ii)/sum(rhoZ0(jj,:)))*rhoD0(ii)
                        den  = (1d0-rhoD (ii)/sum(rhoZ0(jj,:)))*rhoD (ii)
                        if(MIN(den0,den)<0d0) then
                            error_rel = error_rel1
                        else
                            error_rel2 = ABS(drhoD(ii)) / MIN(den0,den)
                            error_rel  = MAX(error_rel1,error_rel2)
                        end if
                    else
                        error_rel = 0.d0
                    end if

                    ! If still in the do while
                    if(.not.okdust) then
                        if(error_rel.le.errmax.and.error_rel.ge.0.0d0) then
                            okdt_bin(ii) = .true.
                            ! Check whether the timestep can be increased
                            if(error_rel.le.0.5d0*errmax) dtloc_bin(ii) = dtloc*2.0d0
                        else if (error_rel.gt.errmax.or.error_rel.lt.0.0d0) then
                            ! Error too large -> deacrease timestep
                            dtloc_bin(ii)=0.5d0*dtloc
                            idec_step(ii) = idec_step(ii) + 1
                        end if
                        icount(ii) = icount(ii) + 1
                    end if

                    if(icount(ii)>countmax)then
                        write(*,*)'stopping in dust processing ii,icount,idec_step>',ii,icount(jj1:jj2),idec_step(jj1:jj2)
                        write(*,*)'dtloc_init/ddt,dtloc/ddt,dtloc_bin/ddt,dtremain/ddt',dtloc_init/ddt,dtloc/ddt,dtloc_bin(jj1:jj2)/ddt,dtremain/ddt
                        write(*,*)'update_switch ',update_switch(jj1:jj2)
                        write(*,*)'int_dust_ratd ',int_dust_ratd(jj)
                        call print_tracked_elements_state('Tracked element number densities:')
                        write(*,*)'rhoD0',rhoD0
                        write(*,*)'rhoD',rhoD
                        write(*,*)'rho,T,mach',rho,TK,mach
                        write(*,*)'rhoZ0',rhoZ0(jj,:)
                        write(*,*)'t_spu/ddt',t_spu(jj1-npah:jj2-npah)/ddt
                        write(*,*)'t_acc/ddt',t_acc(jj1-npah:jj2-npah,1:nions_lead(jj))/ddt
                        if (dust_acc_coulomb) write(*,*)'Coulomb_enhance_ion exists'
                        write(*,*)'t_ratd/ddt',t_ratd(jj1:jj2)/ddt
                        write(*,*)'t_ratd_dest/ddt',t_ratd_dest(jj1:jj2)/ddt
                        write(*,*)'t0/ddt ',t0(jj1:jj2,:)/ddt
                        write(*,*)'minloc t0/ddt',minloc(t0(jj1:jj2,:)),minval(t0(jj1:jj2,:))/1d-1/ddt
                        write(*,*)'d_acc(jj1:jj2) ',d_acc(jj1:jj2)
                        write(*,*)'d_spu(jj1:jj2) ',d_spu(jj1:jj2)
                        write(*,*)'d_sha(jj1:jj2) ',d_sha(jj1:jj2)
                        write(*,*)'d_sha_dest(jj1:jj2) ',d_sha_dest(jj1:jj2)
                        write(*,*)'d_coa(jj1:jj2) ',d_coa(jj1:jj2)
                        write(*,*)'d_ratd(jj1:jj2) ',d_ratd(jj1:jj2)
                        write(*,*)'d_ratd_dest(jj1:jj2) ',d_ratd_dest(jj1:jj2)
                        write(*,*)'oneovertacc(jj1:jj2,1:nions_lead(jj))',oneovertacc(jj1:jj2,1:nions_lead(jj))
                        write(*,*)'oneovertcoa(local)',oneovertcoa(jj1-npah:jj2-npah,jj1-npah:jj2-npah)
                        stop
                    endif
                end do

                if(.not.okdust) then
                    if(ALL(okdt_bin(jj1:jj2).eqv..true.)) then
                        dtremain = dtremain - dtloc
                        do ii=jj1,jj2
                            ! Update dust density
                            rho_dust(ii-npah)  = MAX(rhoD(ii),0d0)
                            ! Update gas phase metal density
                            do kk=1,nions_lead(jj)
                                rhoGZ0(jj,kk) = rhoGZ(kk)
                                if (rhoGZ0(jj,kk)<0.0d0 .and. (abs(rhoGZ0(jj,kk))>1d-30)) then
                                    write(*,*)'Failed in dust_update_RK4 with negative gas metallicity!'
                                    write(*,*)ii,rhoZ0(jj,kk), rhoZ0(jj,kk)-SUM(rhoD0(jj1:jj2)),SUM(rhoD0(jj1:jj2)),rhoGZ0(jj,kk),icount,lead_elem(jj1:jj2)
                                    call print_tracked_elements_state('Tracked element number densities:')
                                    STOP
                                else if (rhoGZ0(jj,kk)<0.0d0 ) then
                                    rhoGZ0(jj,kk) = 0.d0
                                end if
                            end do
                            drhoD_acc(ii) = drhoD_acc(ii) + d_acc(ii)
                            drhoD_spu(ii)=drhoD_spu(ii)+d_spu(ii)
                            drhoD_coa(ii)=drhoD_coa(ii)+d_coa(ii)
                            drhoD_sha(ii)=drhoD_sha(ii)+d_sha(ii)
                            drhoD_ratd(ii)=drhoD_ratd(ii)+d_ratd(ii)
                            drhoD_ratd_dest(ii)=drhoD_ratd_dest(ii)+d_ratd_dest(ii)
                            drhoD_sha_dest(ii)=drhoD_sha_dest(ii)+d_sha_dest(ii)
                        end do
                    end if
                    if(dtremain.le.0.0d0) then
                        okdust=.true.
                        if (dust_log) then
                            ntot_dust_loopcnt(jj) = ntot_dust_loopcnt(jj) + maxval(icount(jj1:jj2))
                            nmax_dust_loopcnt(jj) = max(maxval(icount(jj1:jj2)),nmax_dust_loopcnt(jj))
                            nmin_dust_loopcnt(jj) = min(maxval(icount(jj1:jj2)),nmin_dust_loopcnt(jj))
                        end if
                    end if
                end if

            end do solveloop

        end subroutine dust_update_RK4

        subroutine apply_dust_stage(rhoD_stage,fD_stage,rhoGZ_stage,stage_weight,k_stage,k_gas_stage,&
                                    acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage,&
                                    sha_dest_stage)
            implicit none
            real(dp), intent(in) :: stage_weight
            real(dp), dimension(jj1:jj2), intent(in) :: rhoD_stage,fD_stage
            real(dp), dimension(1:nions_lead(jj)), intent(in) :: rhoGZ_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: k_stage
            real(dp), dimension(1:nions_lead(jj)), intent(inout) :: k_gas_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: sha_dest_stage
            real(dp) :: dd

            do ii=jj1,jj2
                if(dust_accretion.and.int_acc_switch(jj))then
                    do kk=1,nions_lead(jj)
                        if(rhoGZ0(jj,kk)/rhoZ0(jj,kk).gt.1d-10)then
                            dd=(rhoGZ_stage(kk)/rhoZ0(jj,kk))*rhoD_stage(ii)*oneovertacc(ii,kk)
                            k_stage(ii)=k_stage(ii) + dd
                            acc_stage(ii)=acc_stage(ii) + stage_weight*dd
                            k_gas_stage(kk)=k_gas_stage(kk)-dd
                        end if
                    end do
                end if
                if(dust_sputtering)then
                    dd = rhoD_stage(ii)*oneovertspu(ii)
                    k_stage(ii) = k_stage(ii) - dd
                    k_gas_stage(1) = k_gas_stage(1) + dd
                    spu_stage(ii)=spu_stage(ii) - stage_weight*dd
                end if
                if(dust_shattering.and.update_switch(jj1).and.(ii==jj2)) then
                    if (dust_turbulent_model) then
                        dd = rhoD_stage(ii)*oneovertsha_dest(ii-npah,ii-npah,1)*fD_stage(ii)
                        k_gas_stage(1) = k_gas_stage(1) + dd
                        k_stage(ii)=k_stage(ii) - dd
                        sha_dest_stage(ii)=sha_dest_stage(ii) - stage_weight*dd
                        sha_stage(ii)=sha_stage(ii) - stage_weight*dd
                        do kk=jj1,jj2-1
                            dd = rhoD_stage(ii)*oneovertsha(ii-npah,ii-npah,kk-npah)*fD_stage(ii)
                            k_stage(kk)=k_stage(kk) + dd
                            k_stage(ii)=k_stage(ii) - dd
                            sha_stage(kk)=sha_stage(kk) + stage_weight*dd
                            sha_stage(ii)=sha_stage(ii) - stage_weight*dd
                        end do
                    else
                        dd = rhoD_stage(ii)*minval(oneovertsha(ii-npah,ii-npah,:))*fD_stage(ii)
                        k_stage(ii)=k_stage(ii) - dd
                        sha_stage(ii)=sha_stage(ii) - stage_weight*dd
                        k_stage(ii-1)=k_stage(ii-1) + dd
                        sha_stage(ii-1)=sha_stage(ii-1) + stage_weight*dd
                    end if
                end if
                if(dust_coagulation.and.update_switch(jj2))then
                    dd = rhoD_stage(jj1)*oneovertcoa(jj1-npah,jj2-npah)*fD_stage(jj1)
                    if(ii==jj1)then
                        k_stage(ii)= k_stage(ii) - dd
                        coa_stage(ii)=coa_stage(ii) - stage_weight*dd
                    endif
                    if(ii==jj2)then
                        k_stage(ii)= k_stage(ii) + dd
                        coa_stage(ii)=coa_stage(ii) + stage_weight*dd
                    endif
                end if
                if(int_dust_ratd(jj).and.update_switch(jj1).and.(ii==jj2)) then
                    dd = rhoD_stage(ii)*oneovertratd_dest(ii)
                    k_gas_stage(1) = k_gas_stage(1) + dd
                    k_stage(ii)= k_stage(ii) - dd
                    ratd_dest_stage(ii)=ratd_dest_stage(ii) - stage_weight*dd
                    ratd_stage(ii)=ratd_stage(ii) - stage_weight*dd
                    do kk=jj1,jj2-1
                        dd = rhoD_stage(ii)*oneovertratd(kk)
                        k_stage(kk)=k_stage(kk) + dd
                        k_stage(ii)=k_stage(ii) - dd
                        ratd_stage(kk)=ratd_stage(kk) + stage_weight*dd
                        ratd_stage(ii)=ratd_stage(ii) - stage_weight*dd
                    end do
                end if
            end do
        end subroutine apply_dust_stage

        subroutine finalize_dust_stage(dt_stage,rhoD_base,k1_stage,k2_stage,k3_stage,k4_stage,drho_stage,rho_stage,&
                                       acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage,&
                                       sha_dest_stage)
            implicit none
            real(dp), intent(in) :: dt_stage
            real(dp), dimension(jj1:jj2), intent(in) :: rhoD_base,k1_stage,k2_stage,k3_stage,k4_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: drho_stage,rho_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: sha_dest_stage

            do ii=jj1,jj2
                acc_stage(ii)=dt_stage/6d0*acc_stage(ii)
                spu_stage(ii)=dt_stage/6d0*spu_stage(ii)
                sha_stage(ii)=dt_stage/6d0*sha_stage(ii)
                sha_dest_stage(ii)=dt_stage/6d0*sha_dest_stage(ii)
                coa_stage(ii)=dt_stage/6d0*coa_stage(ii)
                ratd_stage(ii)=dt_stage/6d0*ratd_stage(ii)
                ratd_dest_stage(ii)=dt_stage/6d0*ratd_dest_stage(ii)

                drho_stage(ii)=(dt_stage/6d0)*(k1_stage(ii)+2d0*k2_stage(ii)+2d0*k3_stage(ii)+k4_stage(ii))
                rho_stage(ii)=rhoD_base(ii)+drho_stage(ii)
            end do
        end subroutine finalize_dust_stage

        subroutine apply_dust_pah_stage(rhoD_stage,fD_stage,rhoGZ_stage,stage_weight,k_stage,k_gas_stage,&
                                        acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage,&
                                        subl_stage,fre_stage,sha_dest_stage,&
                                        coal_stage,evap_stage)
            implicit none
            real(dp), intent(in) :: stage_weight
            real(dp), dimension(jj1:jj2), intent(in) :: rhoD_stage,fD_stage
            real(dp), dimension(1:nions_lead(jj)), intent(in) :: rhoGZ_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: k_stage
            real(dp), dimension(1:nions_lead(jj)), intent(inout) :: k_gas_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: acc_stage,spu_stage,coa_stage,sha_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: ratd_stage,ratd_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: subl_stage,fre_stage,sha_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: coal_stage,evap_stage
            real(dp) :: dd_stage

            do ii=jj1,jj2
                if ((ii <= jj1+npah-1) .and. update_switch(ii)) then
                    if (pah_sputtering) then
                        dd_stage = rhoD_stage(ii)*oneovertspu_pah(ii)
                        k_stage(ii) = k_stage(ii) - dd_stage
                        k_gas_stage(1) = k_gas_stage(1) + dd_stage
                        spu_stage(ii) = spu_stage(ii) - stage_weight*dd_stage
                    end if
                    if (pah_photolysis.and.ii==jj1) then
                        dd_stage = rhoD_stage(ii)*oneovertsubl_pah(ii)
                        k_stage(ii) = k_stage(ii) - dd_stage
                        k_gas_stage(1) = k_gas_stage(1) + dd_stage
                        subl_stage(ii) = subl_stage(ii) - stage_weight*dd_stage
                    end if
                    if (pah_cluster_evaporation.and.ii==jj1+1) then
                        dd_stage = rhoD_stage(ii)*oneovertevap(ii)
                        k_stage(ii) = k_stage(ii) - dd_stage
                        evap_stage(ii) = evap_stage(ii) - stage_weight*dd_stage
                        k_stage(ii-1) = k_stage(ii-1) + dd_stage
                        evap_stage(ii-1) = evap_stage(ii-1) + stage_weight*dd_stage
                    end if
                    if (pah_coalescence.and.ii==jj1) then
                        dd_stage = rhoD_stage(ii)*oneovertcoal(ii)*fD_stage(ii)
                        k_stage(ii) = k_stage(ii) - dd_stage
                        coal_stage(ii) = coal_stage(ii) - stage_weight*dd_stage
                        k_stage(ii+1) = k_stage(ii+1) + dd_stage
                        coal_stage(ii+1) = coal_stage(ii+1) + stage_weight*dd_stage
                    end if
                    if (pah_freezing) then
                        do kk=jj1+npah,jj2
                            dd_stage = rhoD_stage(ii)*oneovertfre(ii,kk)*fD_stage(kk)
                            k_stage(kk) = k_stage(kk) + dd_stage
                            fre_stage(kk) = fre_stage(kk) + stage_weight*dd_stage
                            k_stage(ii) = k_stage(ii) - dd_stage
                            fre_stage(ii) = fre_stage(ii) - stage_weight*dd_stage
                        end do
                    end if
                elseif (update_switch(ii)) then
                    if(dust_accretion.and.int_acc_switch(jj))then
                        do kk=1,nions_lead(jj)
                            if(rhoGZ0(jj,kk)/rhoZ0(jj,kk).gt.1d-10)then
                                dd_stage=(rhoGZ_stage(kk)/rhoZ0(jj,kk))*rhoD_stage(ii)*oneovertacc(ii,kk)
                                k_stage(ii)=k_stage(ii) + dd_stage
                                acc_stage(ii)=acc_stage(ii) + stage_weight*dd_stage
                                k_gas_stage(kk)=k_gas_stage(kk)-dd_stage
                            end if
                        end do
                    end if
                    if(dust_sputtering)then
                        dd_stage = rhoD_stage(ii)*oneovertspu(ii)
                        k_stage(ii) = k_stage(ii) - dd_stage
                        k_gas_stage(1) = k_gas_stage(1) + dd_stage
                        spu_stage(ii)=spu_stage(ii) - stage_weight*dd_stage
                    end if
                    if(dust_shattering .and. (ii==jj2)) then
                        if (dust_turbulent_model) then
                            dd_stage = rhoD_stage(ii)*oneovertsha_dest(ii-npah,ii-npah,1)*fD_stage(ii)
                            k_gas_stage(1) = k_gas_stage(1) + dd_stage
                            k_stage(ii) = k_stage(ii) - dd_stage
                            sha_stage(ii)=sha_stage(ii) - stage_weight*dd_stage
                            sha_dest_stage(ii)=sha_dest_stage(ii) - stage_weight*dd_stage
                            do kk=jj1,jj2-1
                                dd_stage = rhoD_stage(ii)*oneovertsha(ii-npah,ii-npah,kk-npah)*fD_stage(ii)
                                k_stage(kk) = k_stage(kk) + dd_stage
                                k_stage(ii) = k_stage(ii) - dd_stage
                                sha_stage(ii)=sha_stage(ii) - stage_weight*dd_stage
                                sha_stage(kk)=sha_stage(kk) + stage_weight*dd_stage
                            end do
                        else
                            dd_stage = rhoD_stage(ii)*minval(oneovertsha(ii-npah,ii-npah,:))*fD_stage(ii)
                            k_stage(ii)=k_stage(ii) - dd_stage
                            sha_stage(ii)=sha_stage(ii) - stage_weight*dd_stage
                            k_stage(ii-1)=k_stage(ii-1) + dd_stage
                            sha_stage(ii-1)=sha_stage(ii-1) + stage_weight*dd_stage
                        end if
                    end if
                    if(dust_coagulation.and.(ii==jj2-1).and.update_switch(jj2))then
                        dd_stage = rhoD_stage(ii)*oneovertcoa(ii-npah,jj2-npah)*fD_stage(ii)
                        k_stage(ii)= k_stage(ii) - dd_stage
                        coa_stage(ii)= coa_stage(ii) - stage_weight*dd_stage
                        k_stage(jj2)= k_stage(jj2) + dd_stage
                        coa_stage(jj2)= coa_stage(jj2) + stage_weight*dd_stage
                    end if
                    if(int_dust_ratd(jj).and.(ii==jj2))then
                        dd_stage = rhoD_stage(ii)*oneovertratd_dest(ii)
                        k_gas_stage(1) = k_gas_stage(1) + dd_stage
                        k_stage(ii) = k_stage(ii) - dd_stage
                        ratd_stage(ii)=ratd_stage(ii) - stage_weight*dd_stage
                        ratd_dest_stage(ii)=ratd_dest_stage(ii) - stage_weight*dd_stage
                        do kk=jj1,jj2-1
                            dd_stage = rhoD_stage(ii)*oneovertratd(kk)
                            k_stage(kk) = k_stage(kk) + dd_stage
                            k_stage(ii) = k_stage(ii) - dd_stage
                            ratd_stage(ii)=ratd_stage(ii) - stage_weight*dd_stage
                            ratd_stage(kk)=ratd_stage(kk) + stage_weight*dd_stage
                        end do
                    end if
                end if
            end do
        end subroutine apply_dust_pah_stage

        subroutine finalize_dust_pah_stage(dt_stage,rhoD_base,k1_stage,k2_stage,k3_stage,k4_stage,drho_stage,rho_stage,&
                                           acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage,&
                                           subl_stage,fre_stage,sha_dest_stage,coal_stage,evap_stage)
            implicit none
            real(dp), intent(in) :: dt_stage
            real(dp), dimension(jj1:jj2), intent(in) :: rhoD_base,k1_stage,k2_stage,k3_stage,k4_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: drho_stage,rho_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: acc_stage,spu_stage,coa_stage,sha_stage,ratd_stage,ratd_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: subl_stage,fre_stage,sha_dest_stage
            real(dp), dimension(jj1:jj2), intent(inout) :: coal_stage,evap_stage

            do ii=jj1,jj2
                acc_stage(ii)=dt_stage/6d0*acc_stage(ii)
                spu_stage(ii)=dt_stage/6d0*spu_stage(ii)
                sha_stage(ii)=dt_stage/6d0*sha_stage(ii)
                sha_dest_stage(ii)=dt_stage/6d0*sha_dest_stage(ii)
                coa_stage(ii)=dt_stage/6d0*coa_stage(ii)
                subl_stage(ii)=dt_stage/6d0*subl_stage(ii)
                ratd_stage(ii)=dt_stage/6d0*ratd_stage(ii)
                ratd_dest_stage(ii)=dt_stage/6d0*ratd_dest_stage(ii)
                fre_stage(ii)=dt_stage/6d0*fre_stage(ii)
                coal_stage(ii)=dt_stage/6d0*coal_stage(ii)
                evap_stage(ii)=dt_stage/6d0*evap_stage(ii)

                drho_stage(ii)=(dt_stage/6d0)*(k1_stage(ii)+2d0*k2_stage(ii)+2d0*k3_stage(ii)+k4_stage(ii))
                rho_stage(ii)=rhoD_base(ii)+drho_stage(ii)
            end do
        end subroutine finalize_dust_pah_stage

        subroutine dust_with_pah_update_RK4
            ! This routine solves the dust evolution equations
            ! using the integration method Runge-Kutta of 
            ! 4th order

            implicit none

            logical, dimension(jj1:jj2)::okdt_bin
            integer, dimension(jj1:jj2):: icount,idec_step
            real(dp)::rhoDT0,dtloc,halfdtloc,dd,dtloc_init
            real(dp)::error_rel,error_rel1,error_rel2,den0,den
            real(dp), dimension(jj1:jj2):: rhoD0, drhoD,fD0
            real(dp),dimension(1:nions_lead(jj)):: Gvar,drhoGZ0,rhoGZ
            real(dp),dimension(jj1:jj2)::rhoD
            real(dp), dimension(jj1:jj2)::d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest
            real(dp), dimension(jj1:jj2)::d_subl,d_fre,d_sha_dest,d_coal,d_evap
            real(dp), dimension(jj1:jj2)::dtloc_bin
            real(dp),dimension(jj1:jj2)::k1,k2,k3,k4
            real(dp),dimension(jj1:jj2)::rhoD0k1,rhoD0k2,rhoD0k3
            real(dp),dimension(1:nions_lead(jj))::rhoGZ0k1,rhoGZ0k2,rhoGZ0k3
            real(dp),dimension(1:nions_lead(jj))::k1_gas,k2_gas,k3_gas,k4_gas
            real(dp),dimension(jj1:jj2)::fD0k1,fD0k2,fD0k3

            icount = 0
            idec_step = 0
            drhoGZ0(:) = 0d0

            solveloop: do while (okdust .eqv. .false.)
                d_acc=0.0d0;d_spu=0.0d0;d_coa=0.0d0;d_sha=0.0d0;d_subl=0.0d0;d_ratd=0.0d0;d_sha_dest(:)=0.0d0
                d_fre=0.0d0;d_ratd_dest=0.0d0;d_coal=0.0d0;d_evap=0.0d0

                ! Get density in the gas phase
                rhoDT0 = 0.d0
                do ii=jj1,jj1+npah-1
                    rhoD0(ii) = rho_pah(ii)
                    fD0(ii) = rho_pah(ii) / rho
                    rhoDT0 = rhoDT0 + rhoD0(ii)
                    if (icount(ii)==0) dtloc_bin(ii) = MINVAL(t0(ii,:))
                end do
                do ii=jj1+npah,jj2
                    rhoD0(ii) = rho_dust(ii-npah)
                    fD0(ii) = rho_dust(ii-npah) / rho
                    rhoDT0 = rhoDT0 + rhoD0(ii)
                    if (icount(ii)==0) dtloc_bin(ii) = MINVAL(t0(ii,:))
                end do

                ! Initialise local timestep
                dtloc = MINVAL(dtloc_bin(jj1:jj2))
                dtloc = MIN(dtloc,dtremain)
                if (all(icount(:)==0)) dtloc_init = dtloc
                halfdtloc = 0.5d0 * dtloc

                ! Begin RK4 for all processes
                k1=0.0d0;k1_gas=0.d0;Gvar=0.0d0
                call apply_dust_pah_stage(rhoD0,fD0,rhoGZ0(jj,1:nions_lead(jj)),1d0,k1,Gvar,d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                          d_subl,d_fre,d_sha_dest,d_coal,d_evap)
                rhoD0k1 (:)=rhoD0 (:)+halfdtloc*k1(:)
                fD0k1(:) = rhoD0k1(:)/rho
                rhoGZ0k1(:)=rhoGZ0(jj,:)+halfdtloc*Gvar(:)
                k1_gas(:) = Gvar(:)
                ! print*,'k1',sum(k1(jj1:jj2)),'k1_gas',sum(k1_gas(:)),jj1,jj2

                k2=0.0d0;k2_gas=0.d0;Gvar=0.0d0
                call apply_dust_pah_stage(rhoD0k1,fD0k1,rhoGZ0k1,2d0,k2,Gvar,d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                          d_subl,d_fre,d_sha_dest,d_coal,d_evap)
                rhoD0k2 (:)=rhoD0 (:)+halfdtloc*k2(:)
                fD0k2(:) = rhoD0k2(:)/rho            
                rhoGZ0k2(:)=rhoGZ0(jj,:)+halfdtloc*Gvar(:)
                k2_gas(:) = Gvar(:)

                k3=0.0d0;k3_gas=0.d0;Gvar=0.0d0
                call apply_dust_pah_stage(rhoD0k2,fD0k2,rhoGZ0k2,2d0,k3,Gvar,d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                          d_subl,d_fre,d_sha_dest,d_coal,d_evap)
                rhoD0k3 (:)=rhoD0 (:)+dtloc*k2(:)
                fD0k3(:) = rhoD0k3(:)/rho
                rhoGZ0k3(:)=rhoGZ0(jj,:)+dtloc*Gvar(:)
                k3_gas(:) = Gvar(:)

                k4=0.0d0;k4_gas=0.d0
                call apply_dust_pah_stage(rhoD0k3,fD0k3,rhoGZ0k3,1d0,k4,k4_gas,d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                          d_subl,d_fre,d_sha_dest,d_coal,d_evap)
                call finalize_dust_pah_stage(dtloc,rhoD0,k1,k2,k3,k4,drhoD,rhoD,&
                                             d_acc,d_spu,d_coa,d_sha,d_ratd,d_ratd_dest,&
                                             d_subl,d_fre,d_sha_dest,d_coal,d_evap)

                ! Update gas/ion metal densities
                do kk=1,nions_lead(jj)
                    drhoGZ0(kk) = (dtloc/6d0) * (k1_gas(kk)+2d0*k2_gas(kk)+2d0*k3_gas(kk)+k4_gas(kk))
                    rhoGZ(kk) = rhoGZ0(jj,kk) + drhoGZ0(kk)
                end do
                if (sum(drhoD).gt.1d-15 .and. sum(drhoGZ0).eq.0d0) then
                    print*,'Error in dust+PAHs processing: sum(drhoD)/sum(rhoD0) /= 1.0 but sum(drhoGZ0) == 0.0'
                    print*,'sum(drhoD)/sum(rhoD0) = ',abs(sum(drhoD))/abs(sum(rhoD0))
                    print*,'sum(drhoGZ0) = ',abs(sum(drhoGZ0))
                    print*,'sum(drhoD) = ',abs(sum(drhoD))
                    print*,'k1,k2,k3,k4',k1(jj1:jj2),k2(jj1:jj2),k3(jj1:jj2),k4(jj1:jj2)
                    print*,'k1_gas,k2_gas,k3_gas,k4_gas',k1_gas(:),k2_gas(:),k3_gas(:),k4_gas(:)
                    print*,'sum(k1),sum(k2),sum(k3),sum(k4)',sum(k1(jj1:jj2)),sum(k2(jj1:jj2)),sum(k3(jj1:jj2)),sum(k4(jj1:jj2))
                    print*,'sum(k1_gas),sum(k2_gas),sum(k3_gas),sum(k4_gas)',sum(k1_gas(:)),sum(k2_gas(:)),sum(k3_gas(:)),sum(k4_gas(:))
                    print*,'sum(rhoD0) = ',abs(sum(rhoD0))
                    print*,'sum(rhoD) = ',abs(sum(rhoD))
                    print*,'sum(rhoGZ0(jj,:)) = ',abs(sum(rhoGZ0(jj,:)))
                    print*,'sum(rhoGZ) = ',abs(sum(rhoGZ))
                    print*,'sum(rhoZ0(jj,:)) = ',abs(sum(rhoZ0(jj,:)))
                    stop
                end if
                    
                if (abs(abs(sum(drhoGZ0))-abs(sum(drhoD)))/sum(drhoD).gt.1d-15&
                    & .and. sum(drhoD).gt.0d0 .and. sum(drhoGZ0).gt.0d0 &
                    & .and. sum(drhoD)/rho.gt.smallr_dust) then
                    print*,'Error in dust+PAHs processing: sum(drhoGZ0)/sum(drhoD) /= 1.0'
                    print*,'sum(drhoGZ0)/sum(drhoD) = ',abs(sum(drhoGZ0))/abs(sum(drhoD))
                    print*,'sum(drhoGZ0)-sum(drhoD) = ',abs(sum(drhoGZ0))-abs(sum(drhoD))
                    print*,'sum(drhoGZ0) = ',abs(sum(drhoGZ0))
                    print*,'sum(drhoD) = ',abs(sum(drhoD))
                    print*,'k1,k2,k3,k4',k1(jj1:jj2),k2(jj1:jj2),k3(jj1:jj2),k4(jj1:jj2)
                    print*,'k1_gas,k2_gas,k3_gas,k4_gas',k1_gas(:),k2_gas(:),k3_gas(:),k4_gas(:)
                    print*,'sum(k1),sum(k2),sum(k3),sum(k4)',sum(k1(jj1:jj2)),sum(k2(jj1:jj2)),sum(k3(jj1:jj2)),sum(k4(jj1:jj2))
                    print*,'sum(k1_gas),sum(k2_gas),sum(k3_gas),sum(k4_gas)',sum(k1_gas(:)),sum(k2_gas(:)),sum(k3_gas(:)),sum(k4_gas(:))
                    stop
                end if

                ! Check now for integration errors
                okdt_bin = .false.
                do ii=jj1,jj2
                    if(rhoD0(ii)>0.d0) then
                        error_rel1 = ABS(drhoD(ii)) / MIN(rhoD0(ii),rhoD(ii))
                        den0 = (1d0-rhoD0(ii)/sum(rhoZ0(jj,:)))*rhoD0(ii)
                        den  = (1d0-rhoD (ii)/sum(rhoZ0(jj,:)))*rhoD (ii)
                        if(MIN(den0,den)<0d0) then
                            error_rel = error_rel1
                        else
                            error_rel2 = ABS(drhoD(ii)) / MIN(den0,den)
                            error_rel  = MAX(error_rel1,error_rel2)
                        end if
                    else
                        error_rel = 0.d0
                    end if

                    ! If still in the do while
                    if(.not.okdust) then
                        if(error_rel.le.errmax.and.error_rel.ge.0.0d0) then
                            okdt_bin(ii) = .true.
                            ! Check whether the timestep can be increased
                            if(error_rel.le.0.5d0*errmax) dtloc_bin(ii) = dtloc*2.0d0
                        else if (error_rel.gt.errmax.or.error_rel.lt.0.0d0) then
                            ! Error too large -> deacrease timestep
                            dtloc_bin(ii)=0.5d0*dtloc
                            idec_step(ii) = idec_step(ii) + 1
                        end if
                        icount(ii) = icount(ii) + 1
                    end if

                    if(icount(ii)>countmax)then
                        write(*,*)'stopping in dust+PAHs processing ii,icount,idec_step>',ii,icount(jj1:jj2),idec_step(jj1:jj2)
                        write(*,*)'dtloc_init/ddt,dtloc/ddt,dtloc_bin/ddt,dtremain/ddt',dtloc_init/ddt,dtloc/ddt,dtloc_bin(jj1:jj2)/ddt,dtremain/ddt
                        write(*,*)'update_switch ',update_switch(jj1:jj2)
                        write(*,*)'int_dust_ratd ',int_dust_ratd(jj)
                        call print_tracked_elements_state('Tracked element number densities:')
                        write(*,*)'rhoD0',rhoD0(jj1:jj2)
                        write(*,*)'rhoD',rhoD(jj1:jj2)
                        write(*,*)'rho,T,mach',rho,TK,mach
                        write(*,*)'rhoZ0',rhoZ0(jj,:)
                        write(*,*)'t_spu/ddt - index details limited in PAH context'
                        write(*,*)'t_acc/ddt',t_acc(jj1:jj2-npah,1:nions_lead(jj))/ddt
                        ! Array dimensions differ between dust and PAH bins
                        write(*,*)'t_ratd/ddt',t_ratd(jj1:jj2)/ddt
                        write(*,*)'t_ratd_dest/ddt',t_ratd_dest(jj1:jj2)/ddt
                        write(*,*)'t_evap/ddt',t_evap(jj1:jj2)/ddt
                        write(*,*)'t0/ddt ',t0(jj1:jj2,:)/ddt
                        write(*,*)'d_acc(jj1+npah) ',d_acc(jj1+npah)
                        write(*,*)'d_spu(jj1+npah) ',d_spu(jj1+npah)
                        write(*,*)'d_sha(jj1+npah) ',d_sha(jj1+npah)
                        write(*,*)'d_sha_dest(jj1+npah) ',d_sha_dest(jj1+npah)
                        write(*,*)'d_coa(jj1+npah) ',d_coa(jj1+npah)
                        write(*,*)'d_subl(jj1+npah) ',d_subl(jj1+npah)
                        write(*,*)'d_ratd(jj1+npah) ',d_ratd(jj1+npah)
                        write(*,*)'d_ratd_dest(jj1+npah) ',d_ratd_dest(jj1+npah)
                        write(*,*)'d_fre(jj1+npah) ',d_fre(jj1+npah)
                        write(*,*)'d_coal(jj1+npah) ',d_coal(jj1+npah)
                        write(*,*)'d_evap(jj1+npah) ',d_evap(jj1+npah)
                        stop
                    endif
                end do
                if(.not.okdust) then
                    if(ALL(okdt_bin(jj1:jj2).eqv..true.)) then
                        dtremain = dtremain - dtloc
                        do ii=jj1,jj2
                            ! Update dust density
                            if (ii.le.jj1+npah-1) then
                                rho_pah(ii) = max(rhoD(ii),0d0)
                            else
                                rho_dust(ii-npah)  = max(rhoD(ii),0d0)
                            end if
                            ! Update gas phase metal densities
                            do kk=1,nions_lead(jj)
                                rhoGZ0(jj,kk) = rhoGZ(kk)
                                if (rhoGZ0(jj,kk)<0.0d0 .and. abs(rhoGZ0(jj,kk))>1d-30) then
                                    write(*,*)'Failed in dust_with_pah_update_RK4 with negative gas metallicity!'
                                    write(*,*)ii,rhoZ0(jj,kk), SUM(rhoD0(jj1:jj2)),rho,rhoD0(jj1:jj2)
                                    STOP
                                else if (rhoGZ0(jj,kk)<0.0d0) then
                                    rhoGZ0(jj,kk) = 0.d0
                                end if
                            end do
                            drhoD_acc(ii)=drhoD_acc(ii)+d_acc(ii)
                            drhoD_spu(ii)=drhoD_spu(ii)+d_spu(ii)
                            drhoD_coa(ii)=drhoD_coa(ii)+d_coa(ii)
                            drhoD_sha(ii)=drhoD_sha(ii)+d_sha(ii)
                            drhoD_sha_dest(ii)=drhoD_sha_dest(ii)+d_sha_dest(ii)
                            drhoD_subl(ii)=drhoD_subl(ii)+d_subl(ii)
                            drhoD_ratd(ii)=drhoD_ratd(ii)+d_ratd(ii)
                            drhoD_ratd_dest(ii)=drhoD_ratd_dest(ii)+d_ratd_dest(ii)
                            drhoD_fre(ii)=drhoD_fre(ii)+d_fre(ii)
                            drhoD_coal(ii)=drhoD_coal(ii)+d_coal(ii)
                            drhoD_evap(ii)=drhoD_evap(ii)+d_evap(ii)
                        end do
                    end if
                    if(dtremain.le.0.0d0) then
                        okdust=.true.
                        if (dust_log) then
                            ntot_dust_loopcnt(jj) = ntot_dust_loopcnt(jj) + maxval(icount(jj1:jj2))
                            nmax_dust_loopcnt(jj) = max(maxval(icount(jj1:jj2)),nmax_dust_loopcnt(jj))
                            nmin_dust_loopcnt(jj) = min(maxval(icount(jj1:jj2)),nmin_dust_loopcnt(jj))
                        end if
                    end if
                end if

            end do solveloop

        end subroutine dust_with_pah_update_RK4

        subroutine print_tracked_elements_state(header)
            implicit none
            character(len=*),intent(in) :: header
            integer :: iel

            write(*,*)trim(header)
            do iel=1,n_elements
                if (.not. tracked_elements(iel)) cycle
#ifdef RTZ
                write(*,*)trim(elements(iel)%symbol),' = ',nElement(iel)
#else
                write(*,*)trim(el_names(iel)),' = ',nElement(iel)
#endif
            end do
        end subroutine print_tracked_elements_state

    end subroutine dust_fine

end module dust_chemistry_solver
