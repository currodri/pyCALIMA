module dust_init
    use amr_parameters, only: dp, metal
    use hydro_parameters, only:ndust,ndchemtype,npah,nmetals
    use constants
    use dust_utils
    use dustbin_types
    use dust_commons
    use dust_charging, only: init_dust_charging_isrf
    use dust_cooling, only: init_dust_coll_heating_BH80_cache
#ifdef RTZ
    use rtz_module, only:elements
#endif
    implicit none

    contains

    subroutine print_dust_parameters
        ! subroutine that prints to screen the dust parameters used
        use hydro_parameters, only:idust,ipah
        implicit none
        integer :: ii, jj
        character(len=256) :: composition_string

        write(*,*) '>>> DUST PARAMETERS ======================================'
        write(*,*) '   idust      = ',idust,',   ndust    = ',ndust,',   ndchemtype = ',ndchemtype
        if (dust_solver_type == 1) then
            write(*,*) 'dust_solver_type      = ',dust_solver_type, ' (RK4)'
        else if (dust_solver_type == 2) then
            write(*,*) 'dust_solver_type      = ',dust_solver_type, ' (Anninos)'
        else if (dust_solver_type == 3) then
            write(*,*) 'dust_solver_type      = ',dust_solver_type, ' (RK54)'
        else
            write(*,*) 'dust_solver_type      = ',dust_solver_type, ' (Unknown)'
        end if
        write(*,*) 'dust_acc              = ',dust_accretion  ,',        dust_sput     = ',dust_sputtering
        write(*,*) 'dust_coa              = ',dust_coagulation,',        dust_sha      = ',dust_shattering
        write(*,*) 'dust_acc_coulomb      = ',dust_acc_coulomb,',        dust_ratd     = ',dust_ratd
        write(*,*) 'max_accretion_rate    = ',max_accretion_rate
        write(*,*) 'dust_turbulent_model  = ',dust_turbulent_model,',        H2ondust      = ',H2ondust
        write(*,*) 'dust_shattering_SN      = ',dust_shattering_SN
        write(*,*) 'dust_sublimation      = ',dust_sublimation
        do ii = 1, ndchemtype
            write(*,*)'dust chemical group ',ii,' has ',dustbins_per_chemtype(ii),' dust bins',' starting with dust bin ',istart_chemtype(ii)
            write(*,*)'composition: '
            do jj = 1, dustbins_props(istart_chemtype(ii))%nelements
                write(composition_string,'(A,A,A,I2,A,F6.3)') '   element ',dustbins_props(istart_chemtype(ii))%el_names(jj),' (',dustbins_props(istart_chemtype(ii))%el_index(jj),') : ',dustbins_props(istart_chemtype(ii))%stoichiometry(jj)
                write(*,'(A)') trim(composition_string)
            end do
            write(*,*)'sizes (µm) and densities (g/cm^3): '
            do jj = istart_chemtype(ii), istart_chemtype(ii) + dustbins_per_chemtype(ii) - 1
                write(*,'(A,I2,A,E10.2,A,E10.2)') '   dust bin ',jj,' : a =',dustbins_props(jj)%asize,',  s =',dustbins_props(jj)%sgrain
            end do
        end do        
#if NPAH>0
        if(dust_pahs) then
            write(*,*) '>>> PAHs PARAMETERS ======================================'
            write(*,*) '   ipah      = ',ipah,',   npah    = ',npah
            write(*,*) 'pah_coalescence      = ',pah_coalescence,   ',pah_sputtering      = ',pah_sputtering
            write(*,*) 'pah_freezing         = ',pah_freezing,      ',pah_evaporation     = ',pah_cluster_evaporation
            write(*,*) 'pah_photolysis   = ',pah_photolysis,',pah_sn_destruction  = ',pah_sn_destruction
            write(*,*) 'H2onpah              = ',H2onpah,           ',pah_pe_heating      = ',pah_pe_heating
            write(*,*) 'pah_accretion        = ',pah_accretion
            do ii = 1, npah
                write(*,*)'pah bin ',ii,' has size ',pahbins_props(ii)%apah,' µm and density ',pahbins_props(ii)%spah,' g/cm^3'
                write(*,*)'   pah_minmass = ',pahbins_props(ii)%mpah_min,',   pah_maxmass = ',pahbins_props(ii)%mpah_max
                write(*,*)'   pah_nc = ',pahbins_props(ii)%nc
                write(*,*)'   pah_inwind = ',fpah_inwind(ii)
            end do
        end if
#endif
        write(*,*) '>>> DUST DYNAMICS ========================================'
        write(*,*) 'dust_tva             = ',dust_tva,',    dust_radpressure    = ',dust_radpressure
        if (dust_tva) then
            write(*,*) 'use_w_drift_test     = ',use_w_drift_test
            if (use_w_drift_test) write(*,*) '   w_drift_test      = ',w_drift_test
            write(*,*) 'tva_wmax_cs          = ',tva_wmax_cs,' (cap on |w_drift| / c_s)'
            write(*,*) 'epstein_coef         = ',epstein_coef,' = sqrt(pi*gamma/8)'
            write(*,*) 'drag_model           = ',trim(drag_model)
        end if
        write(*,*) '=============================================================='
    end subroutine print_dust_parameters

    
    function check_params_dust(myid)
        ! This function goes through the hydro and compilation parameters
        ! to make sure that it is compliant with the required dust settings
        use hydro_parameters
#ifdef RT
        use rt_parameters, only: rt_isIRtrap, rt_isIR, rt_use_hll
#endif
        implicit none
        logical :: check_params_dust
        integer,intent(in) :: myid
        
        logical :: metals_for_dust
        real(dp):: fpah_total

        check_params_dust = .true.
        !-------------------------------------------------
        ! Check we have metals ON for dust
        !-------------------------------------------------
        if(.not. metal)then
            if(myid==1)then
                write(*,*)'WARNING: metal=.false., gas-phase elements are not defined.'
                write(*,*)'Disabling CALIMA gas-exchange processes (accretion, sputtering, SN/wind seeding & destruction)'
            end if
            dust_accretion = .false.
            pah_accretion = .false.
            dust_sputtering = .false.
            pah_sputtering = .false.
            dust_inSN = .false.
            dust_inSNIa = .false.
            dust_inSW = .false.
            pah_AGBwinds = .false.
            dust_SNdest = .false.
            pah_sn_destruction = .false.
            pah_acc_spu = .false.
        end if
#ifdef RTZ 
        metals_for_dust=(N_OXYGEN_IONS>0).and.(N_MAGNESIUM_IONS>0).and.(N_CARBON_IONS>0).and.(N_IRON_IONS>0).and.(N_SILICON_IONS>0)
        if (.not. metals_for_dust) then
            if (myid==1) then
                write(*,*) "WARNING: missing metals required to track dust composition."
                write(*,*) "Disabling CALIMA gas-exchange processes."
            end if
            dust_accretion = .false.
            pah_accretion = .false.
            dust_sputtering = .false.
            pah_sputtering = .false.
            dust_inSN = .false.
            dust_inSNIa = .false.
            dust_inSW = .false.
            pah_AGBwinds = .false.
            dust_SNdest = .false.
            pah_sn_destruction = .false.
            pah_acc_spu = .false.
        end if
#endif
        !-------------------------------------------------
        ! Check variables for PAHs are ok
        !-------------------------------------------------
        fpah_total = sum(fpah_inwind)
        if (fpah_total.ne.1d0.and.pah_AGBwinds) then
            if(myid==1)write(*,*)'Error: fpah_inwind needs to add up to 1'
            check_params_dust=.false.
        end if

        if (dust_turbulent_model.and.dust_shattering) then
            if (trim(shattering_model).ne.'subgrid') then
                if(myid==1)write(*,*)'Error: shattering_model needs to be set to subgrid when using dust_turbulent_model'
                check_params_dust=.false.
            end if
        end if
        if (dust_turbulent_model.and.dust_coagulation) then
            if (trim(coagulation_model).ne.'subgrid') then
                if(myid==1)write(*,*)'Error: coagulation_model needs to be set to subgrid when using dust_turbulent_model'
                check_params_dust=.false.
            end if
        end if

        !-------------------------------------------------
        ! Check everything is correct to do dust dynamics
        !-------------------------------------------------
#ifndef RT
        if (dust_radpressure) then
            if(myid==1)write(*,*)'Error: dust_radpressure can only be used with RT'
            check_params_dust=.false.
        end if
#endif
        if (dust_tva .and. ndust <= 0) then
            if(myid==1)write(*,*)'Error: dust_tva requires at least one dust bin (NDUST>0)'
            check_params_dust=.false.
        end if
        if (dust_radpressure .and. .not.dust_tva) then
            if(myid==1)write(*,*)'Error: dust_radpressure has no effect without dust_tva=.true.'
            check_params_dust=.false.
        end if
        if (trim(drag_model).ne.'epstein' .and. .not.draine_drag()) then
            if(myid==1)write(*,*)'Error: drag_model must be epstein, draine2011 or draine2011_pz, not ',trim(drag_model)
            check_params_dust=.false.
        end if
        ! The Draine drag needs n_H, T and the grain charge, which the TVA gets
        ! from the per-cell radiation-pressure call; the pure-drag path has none.
        if (draine_drag() .and. .not.dust_radpressure) then
            if(myid==1)write(*,*)'Error: drag_model=',trim(drag_model),' requires dust_radpressure=.true.'
            check_params_dust=.false.
        end if
        if (dust_tva .and. myid==1) then
            if (use_w_drift_test) then
                write(*,*)'WARNING: use_w_drift_test=.true. replaces the physical drift ', &
                          'with the constant w_drift_test. This is a test mode, not physics.'
            end if
            if (npah > 0) then
                write(*,*)'WARNING: TVA advects the ',ndust,' dust bins only. The ',npah, &
                          ' PAH bins stay perfectly coupled to the gas (see calima/README.md).'
            end if
            if (tva_wmax_cs <= 0d0) then
                write(*,*)'WARNING: tva_wmax_cs<=0 disables the drift cap. TVA assumes ', &
                          'Stokes<<1; without a cap dt can collapse in hot/diffuse cells.'
            end if
        end if
#ifdef RT
        ! Trapped-IR radiation pressure on the dust
        if (dust_tva .and. rt_isIRtrap) then
            if (nener <= 0) then
                if(myid==1)write(*,*)'Error: rt_isIRtrap needs a non-thermal energy ', &
                                     'variable; recompile with NENER>=1 (and NVAR+1)'
                check_params_dust=.false.
            end if
            if (.not. rt_isIR) then
                if(myid==1)write(*,*)'Error: rt_isIRtrap requires rt_isIR=.true. ', &
                                     '(the trapped variable is the IR group)'
                check_params_dust=.false.
            end if
            if (ndust <= 0) then
                if(myid==1)write(*,*)'Error: trapped-IR pressure on dust requires NDUST>0'
                check_params_dust=.false.
            end if
            if (rt_use_hll) then
                ! Rosdahl & Teyssier 2015, footnote 3: the trapped/streaming
                ! partition is matched to the numerical diffusion of the GLF
                ! flux, so it is only consistent with rt_flux_scheme='glf'.
                if(myid==1)write(*,*)'Error: rt_isIRtrap is only consistent with ', &
                                     "rt_flux_scheme='glf', not 'hll'"
                check_params_dust=.false.
            end if
        end if
#endif
        select case (trim(charging_model))
        case ('Ibanez2019', 'WDB06isrf')
        case ('WDB06rt')
#ifndef RT
            if (myid == 1) write(*,*) 'Error: charging_model='//trim(charging_model)//' needs RT photon groups; ', &
                'use WDB06isrf (uniform ISRF) or Ibanez2019'
            check_params_dust = .false.
#endif
        case default
            if (myid == 1) write(*,*) 'Error: unknown charging_model='//trim(charging_model)//' (WDB06rt, ', &
                'WDB06isrf or Ibanez2019; the old (gamma, T) charge tables and WDB06tab were removed)'
            check_params_dust = .false.
        end select
        if (dust_ion_recombination .and. trim(charging_model) /= 'WDB06rt' .and. trim(charging_model) /= 'WDB06isrf') then
            if (myid == 1) write(*,*) 'WARNING: dust_ion_recombination needs charging_model WDB06rt or WDB06isrf; ', &
                'the RTZ chemistry keeps its grain recombination fit'
        end if
#if defined(RT) && !defined(RTZ)
        ! RT + CALIMA without RTZ. The radiative dust coupling is ported into
        ! rt_cooling_module, but the dust<->gas THERMAL coupling is not: it
        ! lives in rtz_coolrates_module's all_cooling, which does not exist in
        ! this build. Warn rather than fail -- the configuration is useful and
        ! deliberate -- but do not let the limitation pass silently.
        if (myid == 1) then
            write(*,*)'------------------------------------------------------------'
            write(*,*)'WARNING: CALIMA is compiled without RTZ.'
            write(*,*)'  ACTIVE : per-bin dust temperature, IR Rosseland/Planck'
            write(*,*)'           opacities, dust radiation pressure, dust drift.'
            write(*,*)'  MISSING: dust->gas thermal coupling. Dust recombination'
            write(*,*)'           cooling, photoelectric heating and dust-gas'
            write(*,*)'           collisional cooling are computed but NOT applied'
            write(*,*)'           to the gas temperature.'
            write(*,*)'  MISSING: metal abundances. The gas state handed to the'
            write(*,*)'           grain physics carries H and He only, so T_dust is'
            write(*,*)'           biased LOW (no heavy-species collisional heating)'
            write(*,*)'           and grain charging sees no metal-donated'
            write(*,*)'           electrons. See rt_calima_gas_state.'
            write(*,*)'  With rt_kIR_RT15=.true. the one-temperature RT15 closure'
            write(*,*)'  is used instead and the gas DOES receive IR heating.'
            write(*,*)'------------------------------------------------------------'
        end if
#endif
    end function check_params_dust

    subroutine init_dust_depletion(myq,Hfrac,force_zero)
        ! This routine is used for initialising isolated galaxy simulations
        ! following the fractional contributions of the BARE-GR-S model
        ! from Zubko et al. (2004) - see Table 6
        ! (https://ui.adsabs.harvard.edu/abs/2004ApJS..152..211Z/abstract)
        ! and Dopita et al. (2000) - see Table 1
        ! (https://ui.adsabs.harvard.edu/abs/2000ApJ...539..742D/abstract)
        use amr_commons, only: myid
        use hydro_commons, only: smallr,nvar,imetal
        implicit none
        real(dp),dimension(1:nvar),intent(inout) :: myq
        real(dp),intent(in) :: Hfrac
        logical,intent(in),optional :: force_zero
        integer :: ii,jj,jj1,jj2,kk,ilim
        logical :: myforce
        real(dp) :: GD,GDfactor,DTMfactor
        real(dp) :: dustC,dustPAH,dustMass,Z_interest
        real(dp) :: ftot,metalM
        real(dp),dimension(:),allocatable :: M_el

        myforce = .false.
        if (present(force_zero)) myforce = force_zero

        if (DTMinit.eq.-1d0) then
            ! OPTION A: Initialise based on G/D for the MW
            if (GDinit.eq.-1d0) then
                GD = GD_RR14(myq(imetal+8-1)/mO_amu, Hfrac/mH_amu)
            else
                GD = GDinit
            end if
            ! Add up the total metallicity due only to the metals depleted on dust
            Z_interest = 0d0
            do ii = 1, ndchemtype
                jj1 = istart_chemtype(ii)
                do jj = 1, dustbins_props(jj1)%nelements
                    Z_interest = Z_interest + myq(imetal + dustbins_props(jj1)%el_index(jj) - 1)
                end do
            end do
            if (Z_interest .eq. 0d0) then
                myforce = .true.
                GD = 0d0
            end if
            if ((GD.ne.0d0).and. (.not.myforce)) then
                if (GD.lt.1d0/Z_interest) then
                    GD = 1d0/Z_interest
                end if
                GDfactor = GD_solar / GD ! Scaled to solar G/D=162 (Zubko et al. 2004)
                do jj=1,ndchemtype
                    jj1 = istart_chemtype(jj)
                    if (dustbins_props(jj1)%interact_pah .and. npah>0) then
                        jj2 = dustbins_props(jj1)%el_index(1) - 1
                        ! In the case we follow PAHs and carbonaceous grains
                        dustC = myq(imetal + jj2) * min(fDust_depletions(6) * GDfactor,1d0 - smallr_dust / myq(imetal + jj2))
                        if (dust_pahs .and. (fpah_ini.eq.-1d0)) then
                            dustPAH = fCDust_inPAH * dustC
                        elseif (dust_pahs) then
                            dustPAH = min(max(fpah_ini,0d0),1.d0) * dustC
                        endif
                        do kk = jj1, jj1 + dustbins_per_chemtype(jj) - 1
                            myq(idust + kk - 1) = fdustmass_ini(kk) * dustC
                        end do
                        if (dust_pahs) then
                            do kk = 1, npah
                                myq(ipah + kk - 1) = fpahmass_ini(kk) * dustPAH
                            end do
                        end if
                        ! Deplete carbon
                        myq(imetal+jj2) = max(myq(imetal+jj2) - (dustC + dustPAH),0d0)
                    else
                        ! Now for a general dust chemistry in which we look for the limiting element
                        allocate(M_el(1:dustbins_props(jj1)%nelements))
                        do ii = 1, dustbins_props(jj1)%nelements
                            M_el(ii) = fDust_depletions(dustbins_props(jj1)%el_atomic_number(ii)) * &
                                        myq(imetal + dustbins_props(jj1)%el_index(ii) - 1)
                        end do
                        call cmp_lim_elem(jj1,dustbins_props(jj1)%nelements,M_el,ilim)
                        dustMass = GDfactor * M_el(ilim) / dustbins_props(jj1)%el_mfractions(ilim)
                        do kk = jj1, jj1 + dustbins_per_chemtype(jj) - 1
                            myq(idust + kk - 1) = fdustmass_ini(kk) * dustMass
                        end do
                        ! Deplete the elements
                        do ii = 1, dustbins_props(jj1)%nelements
                            myq(imetal + dustbins_props(jj1)%el_index(ii) - 1) = max(myq(imetal + dustbins_props(jj1)%el_index(ii) - 1) - &
                                (dustMass * dustbins_props(jj1)%el_mfractions(ii)),0d0)
                        end do
                    end if
                end do
                if (any(myq(idust:idust+ndust-1) .lt. 0d0)) then
                    print*,'ZERO OR NEGATIVE DUST IN INIT DEPLETION!!'
                    print*,'dust:',myq(idust:idust+ndust-1)
                    stop
                end if
                if (dust_pahs .and. any(myq(ipah:ipah+npah-1) .lt. 0d0)) then
                    print*,'ZERO OR NEGATIVE PAHs IN INIT DEPLETION!!'
                    print*,'pahs:',myq(ipah:ipah+npah-1)
                    stop
                end if
                if (any(myq(imetal:imetal+nmetals-1) .le. 0d0)) then
                    print*,'ZERO OR NEGATIVE METAL IN INIT DEPLETION!!'
                    print*,'metals:',myq(imetal:imetal+nmetals-1)
                    print*,'G/D:',GD
                    stop
                end if
            else
                ! This is just for the case of 0 dust
                do jj=1,ndust
                    myq(idust+jj-1) = 0d0
                end do
                if (dust_pahs) then
                    do jj=1,npah
                        myq(ipah+jj-1) = 0d0
                    end do
                end if
            end if
        else
            ! OPTION B: Initialise based on DTMinit
            if ((DTMinit.ne.0d0).and.(.not.myforce)) then
                DTMfactor = DTMinit / DTM_solar
                do jj=1,ndchemtype
                    jj1 = istart_chemtype(jj)
                    if (dustbins_props(jj1)%interact_pah .and. npah>0) then
                        jj2 = dustbins_props(jj1)%el_index(1) - 1
                        ! In the case we follow PAHs and carbonaceous grains
                        dustC = myq(imetal + jj2) * min(fDust_depletions(6) * DTMfactor,1d0 - smallr_dust / myq(imetal + jj2))
                        if (dust_pahs .and. (fpah_ini.eq.-1d0)) then
                            dustPAH = fCDust_inPAH * dustC
                        elseif (dust_pahs) then
                            dustPAH = min(max(fpah_ini,0d0),1.d0) * dustC
                        endif
                        do kk = jj1, jj1 + dustbins_per_chemtype(jj) - 1
                            myq(idust + kk - 1) = fdustmass_ini(kk) * dustC
                        end do
                        if (dust_pahs) then
                            do kk = 1, npah
                                myq(ipah + kk - 1) = fpahmass_ini(kk) * dustPAH
                            end do
                        end if
                        ! Deplete carbon
                        myq(imetal+jj2) = max(myq(imetal+jj2) - (dustC + dustPAH),0d0)
                    else
                        ! Now for a general dust chemistry in which we look for the limiting element
                        allocate(M_el(1:dustbins_props(jj1)%nelements))
                        do ii = 1, dustbins_props(jj1)%nelements
                            M_el(ii) = fDust_depletions(dustbins_props(jj1)%el_atomic_number(ii)) * &
                                        myq(imetal + dustbins_props(jj1)%el_index(ii) - 1)
                        end do
                        call cmp_lim_elem(jj1,dustbins_props(jj1)%nelements,M_el,ilim)
                        dustMass = DTMfactor * M_el(ilim) / dustbins_props(jj1)%el_mfractions(ilim)
                        do kk = jj1, jj1 + dustbins_per_chemtype(jj) - 1
                            myq(idust + kk - 1) = fdustmass_ini(kk) * dustMass
                        end do
                        ! Deplete the elements
                        do ii = 1, dustbins_props(jj1)%nelements
                            myq(imetal + dustbins_props(jj1)%el_index(ii) - 1) = max(myq(imetal + dustbins_props(jj1)%el_index(ii) - 1) - &
                                (dustMass * dustbins_props(jj1)%el_mfractions(ii)),0d0)
                        end do
                    end if
                end do
            else
                ! This is just for the case of 0 dust
                do jj=1,ndust
                    myq(idust+jj-1) = 0d0
                end do
                if (dust_pahs) then
                    do jj=1,npah
                        myq(ipah+jj-1) = 0d0
                    end do
                end if
            end if
        end if
    end subroutine init_dust_depletion

    subroutine init_dust_depletion_tests(nElement,rho_dust,rho_pah)
        ! This routine is for setting the dust and PAH abundances
        ! for the equilibrium test runs in rtz_cooling_module.f90.
        ! nElement <--> element number densities [cm^-3]
        ! rho_dust <--> dust mass densities [g/cm^3]
        ! rho_pah  <--> PAH mass densities [g/cm^3]
        implicit none

        ! ---- Input/Output ----
        real(dp),dimension(1:n_elements), intent(inout) :: nElement
        real(dp),dimension(1:ndust), intent(inout) :: rho_dust
        real(dp),dimension(1:npah), intent(inout) :: rho_pah

        ! ---- Local variables ----
        integer :: ii,ii1,ii2,jj,kk,ilim
        real(dp) :: DTM_factor, GD
        real(dp) :: dustC,dustPAH,dustMass
        real(dp) :: total_dust_pah, total_metals, dust_fraction
        real(dp),dimension(:),allocatable :: M_el

        if (DTMinit .eq. -1d0) then
            ! G/D-based initialisation (mirrors OPTION A of init_dust_depletion)
            if (GDinit .eq. -1d0) then
                if (nElement(1) .gt. 0d0) then
                    GD = GD_RR14(nElement(8), nElement(1))
                else
                    GD = GD_solar
                end if
            else
                GD = GDinit
            end if
            DTM_factor = GD_solar / GD
        else
            DTM_factor = DTMinit / DTM_solar
        end if

        do ii = 1, ndchemtype
            ii1 = istart_chemtype(ii)
            if (dustbins_props(ii1)%interact_pah .and. npah>0) then
                ! In the case we follow PAHs and carbonaceous grains
                kk = dustbins_props(ii1)%el_index(1)
                dustC = dustbins_props(ii1)%el_atomic_masses_g(1) * &
                        nElement(kk) * fDust_depletions(6) * DTM_factor
                if (dust_pahs .and. (fpah_ini.eq.-1d0)) then
                    dustPAH = fCDust_inPAH * dustC
                elseif (dust_pahs) then
                    dustPAH = min(max(fpah_ini,0d0),1.d0) * dustC
                end if
                ii2 = ii1 + dustbins_per_chemtype(ii) - 1
                do jj = ii1, ii2
                    rho_dust(jj) = fdustmass_ini(jj) * dustC
                end do
                if (dust_pahs) then
                    do jj = 1, npah
                        rho_pah(jj) = fpahmass_ini(jj) * dustPAH
                    end do
                end if
                ! Deplete carbon
                nElement(kk) = max(nElement(kk) - (dustC + dustPAH) / &
                                dustbins_props(ii1)%el_atomic_masses_g(1),0d0)
            else
                ! Now for a general dust chemistry in which we look for the limiting element
                allocate(M_el(1:dustbins_props(ii1)%nelements))
                do jj = 1, dustbins_props(ii1)%nelements
                    kk = dustbins_props(ii1)%el_index(jj)
                    M_el(jj) = fDust_depletions(dustbins_props(ii1)%el_atomic_number(jj)) &
                                * nElement(kk) * dustbins_props(ii1)%el_atomic_masses_g(jj)
                end do
                call cmp_lim_elem(ii1,dustbins_props(ii1)%nelements,M_el,ilim)
                dustMass = DTM_factor * M_el(ilim) / dustbins_props(ii1)%el_mfractions(ilim)
                ii2 = ii1 + dustbins_per_chemtype(ii) - 1
                do jj = ii1, ii2
                    rho_dust(jj) = fdustmass_ini(jj) * dustMass
                end do
                ! Deplete the elements
                do jj = 1, dustbins_props(ii1)%nelements
                    kk = dustbins_props(ii1)%el_index(jj)
                    nElement(kk) = max(nElement(kk) - (dustMass * dustbins_props(ii1)%el_mfractions(jj))&
                                    / dustbins_props(ii1)%el_atomic_masses_g(jj),0d0)
                end do
                deallocate(M_el)
            end if
        end do

        ! Safety check: depletion setup must not produce negative densities.
        if (any(nElement(1:n_elements) < 0d0)) then
            write(*,*) 'NEGATIVE METAL DENSITY IN INIT DUST DEPLETION TESTS!!'
            write(*,*) 'nElement:', nElement(1:n_elements)
            stop
        end if
        if (any(rho_dust(1:ndust) < 0d0)) then
            write(*,*) 'NEGATIVE DUST DENSITY IN INIT DUST DEPLETION TESTS!!'
            write(*,*) 'rho_dust:', rho_dust(1:ndust)
            stop
        end if
        if (npah > 0) then
            if (any(rho_pah(1:npah) < 0d0)) then
                write(*,*) 'NEGATIVE PAH DENSITY IN INIT DUST DEPLETION TESTS!!'
                write(*,*) 'rho_pah:', rho_pah(1:npah)
                stop
            end if
        end if

        ! Verify that dust+PAH fractions are consistent with DTMinit
        if (DTMinit > 0d0) then
            total_dust_pah = sum(rho_dust(1:ndust))
            if (npah > 0) total_dust_pah = total_dust_pah + sum(rho_pah(1:npah))
            ! Convert element number densities [cm^-3] to mass density [g/cm^3]
            total_metals = 0d0
            do jj = 1, n_elements
#ifdef RTZ
                total_metals = total_metals + nElement(jj) * elements(jj)%atomic_mass * amu2g
#else
                total_metals = total_metals + nElement(jj) * el_atomic_masses_amu(jj) * amu2g
#endif
            end do
            if (total_metals > 0d0) then
                ! The dust/metal ratio should be DTM_factor (which is DTMinit/DTM_solar scaled by depletion factors)
                dust_fraction = total_dust_pah / total_metals
                if (dust_fraction > DTMinit) then
                    write(*,*) 'DUST DEPLETION TEST CHECK:'
                    write(*,*) '  DTMinit=', DTMinit, ' DTM_solar=', DTM_solar, ' DTM_factor=', DTM_factor
                    write(*,*) '  total_dust_pah=', total_dust_pah, ' g/cm^3'
                    write(*,*) '  total_metals=', total_metals, 'g/cm^3'
                    write(*,*) '  dust_fraction=', dust_fraction, ' (dust mass / metal number density)'
                end if
            end if
        end if
    end subroutine init_dust_depletion_tests

    function GD_RR14(nO,nH)
        ! This function returns an estimate of the gas to dust ratio
        ! (G/D) based on a broken power-law fit by Remy-Ruyer et al. (2014)
        ! (see their parameters in Table 1:
        ! https://ui.adsabs.harvard.edu/abs/2014A%26A...563A..31R/abstract)
        ! This assumes that the solar abundance is (O/H)sun = 4.9e-4
        ! nO and nH are any quantities proportional to the O and H number densities
        implicit none
        real(dp),intent(in) :: nO, nH
        real(dp) :: a, alpha_H, b, alpha_L, x_t, xsun
        real(dp) :: GD_RR14,x,y

        ! This uses the XCO,Z case (right column of Table 1)
        a = 2.21d0; alpha_H = 1d0; b = 0.96d0
        alpha_L = 3.10d0; x_t = 8.10d0; xsun = 8.69d0

        x = 12 + log10(max(nO/nH,1.d-40))
        if (x .gt. x_t) then
            y = a + alpha_H * (xsun - x)
        else
            y = b + alpha_L * (xsun - x)
        end if
        GD_RR14 = 10**y
    end function GD_RR14

    subroutine init_dust_processes
        use dust_rates
        implicit none
        integer :: ii

        ! 1. We begin by counting how many dust processes will be included based on
        ! on the namelist parameters, and allocate the array of dust_processes_list
        ndust_processes = 0
        if (ndust > 0) then
            if (dust_accretion) then
                ndust_processes = ndust_processes + 1
            end if
            if (dust_sputtering) then
                ndust_processes = ndust_processes + 1
            end if
            if (dust_sublimation) then
                ndust_processes = ndust_processes + 1
            end if
            if (dust_coagulation) then
                ndust_processes = ndust_processes + 1
            end if
            if (dust_shattering) then
                ndust_processes = ndust_processes + 1
            end if
            if (dust_ratd) then
                ndust_processes = ndust_processes + 1
            end if
            if (ndust_processes > 0) then
                allocate(dust_processes_list(1:ndust_processes))
                ndust_processes = 0
                if (dust_accretion) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'accretion'
                    dust_processes_list(ndust_processes)%source = .true.
                    dust_processes_list(ndust_processes)%sink = .false.
                    if (dust_acc_coulomb) then
                        carry_gas_ions = .true.
                        dust_processes_list(ndust_processes)%comp_rate => coulomb_accretion_rate
                    else if (accretion_model.eq.'LeBourlot2012') then
                        dust_processes_list(ndust_processes)%comp_rate => LeBourlot2012_accretion_rate
                    else
                        dust_processes_list(ndust_processes)%comp_rate => LeBourlot2012_accretion_rate
                    end if
                end if
                if (dust_sputtering) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'sputtering'
                    dust_processes_list(ndust_processes)%source = .false.
                    dust_processes_list(ndust_processes)%sink = .true.
                    if (sputtering_model.eq.'RM2026') then
                        if (dust_sputtering_charge) then
                            carry_gas_ions = .true.
                            dust_processes_list(ndust_processes)%comp_rate => charged_sputtering_rate
                        else
                            dust_processes_list(ndust_processes)%comp_rate => sputtering_rate
                        end if
                    else
                        dust_processes_list(ndust_processes)%comp_rate => sputtering_rate
                    end if
                end if
                if (dust_sublimation) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'sublimation'
                    dust_processes_list(ndust_processes)%source = .false.
                    dust_processes_list(ndust_processes)%sink = .true.
                    dust_processes_list(ndust_processes)%comp_rate => sublimation_rate
                end if
                if (dust_coagulation) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'coagulation'
                    dust_processes_list(ndust_processes)%source = .false.
                    dust_processes_list(ndust_processes)%sink = .false.
                    if (coagulation_model.eq.'Aoyama2017') then
                        dust_processes_list(ndust_processes)%comp_rate => Aoyama2017_coagulation_rate
                    else if (coagulation_model.eq.'Hirashita2015') then
                        dust_processes_list(ndust_processes)%comp_rate => turbulent_coagulation_rate
                    else if (coagulation_model.eq.'Smoluchowski1916') then
                        dust_processes_list(ndust_processes)%comp_rate => turbulent_all_coagulation_rate
                    else
                        dust_processes_list(ndust_processes)%comp_rate => Aoyama2017_coagulation_rate
                    end if
                end if
                if (dust_shattering) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'shattering'
                    dust_processes_list(ndust_processes)%source = .false.
                    dust_processes_list(ndust_processes)%sink = .true.
                    if (shattering_model.eq.'Hirashita2015') then
                        dust_processes_list(ndust_processes)%comp_rate => turbulent_shattering_rate
                    else if (shattering_model.eq.'Smoluchowski1916') then
                        dust_processes_list(ndust_processes)%comp_rate => turbulent_all_shattering_rate
                    else
                        dust_processes_list(ndust_processes)%comp_rate => turbulent_shattering_rate
                    end if
                end if
                if (dust_ratd) then
                    ndust_processes = ndust_processes + 1
                    dust_processes_list(ndust_processes)%name = 'ratd'
                    dust_processes_list(ndust_processes)%source = .false.
                    dust_processes_list(ndust_processes)%sink = .true.
                    ! comp_rate not yet implemented for RATD; guard below will abort.
                end if
            end if
        end if

        ! 2. Do the same for the PAH processes
        if (npah > 0) then
            if (pah_accretion) then
                npah_processes = npah_processes + 1
            end if
            if (pah_sputtering) then
                npah_processes = npah_processes + 1
            end if
            if (pah_coalescence) then
                npah_processes = npah_processes + 1
            end if
            if (pah_freezing) then
                npah_processes = npah_processes + 1
            end if
            if (pah_desorption) then
                npah_processes = npah_processes + 1
            end if
            if (pah_cluster_evaporation) then
                npah_processes = npah_processes + 1
            end if
            if (pah_photolysis) then
                npah_processes = npah_processes + 1
            end if
            if (npah_processes > 0) then
                allocate(pah_processes_list(1:npah_processes))
                npah_processes = 0
                if (pah_accretion) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'accretion'
                    ! comp_rate not yet implemented for PAH accretion; guard below will abort.
                end if
                if (pah_sputtering) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'sputtering'
                    pah_processes_list(npah_processes)%source = .false.
                    pah_processes_list(npah_processes)%sink = .true.
                    if (pah_sputtering_model.eq.'RM2026') then
                        pah_processes_list(npah_processes)%comp_rate => pah_sputtering_rate
                    else
                        pah_processes_list(npah_processes)%comp_rate => pah_sputtering_rate
                    end if
                end if
                if (pah_coalescence) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'coalescence'
                    pah_processes_list(npah_processes)%source = .false.
                    pah_processes_list(npah_processes)%sink = .false.
                    if (coalescence_model.eq.'Totton2012') then
                        pah_processes_list(npah_processes)%comp_rate => Totton2012_pah_coalescence_rate
                    else if (coalescence_model.eq.'Tielens2021') then
                        pah_processes_list(npah_processes)%comp_rate => Tielens2021_pah_coalescence_rate
                    else
                        pah_processes_list(npah_processes)%comp_rate => Totton2012_pah_coalescence_rate
                    end if
                end if
                if (pah_freezing) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'freezing'
                    pah_processes_list(npah_processes)%source = .false.
                    pah_processes_list(npah_processes)%sink = .false.
                    pah_processes_list(npah_processes)%comp_rate => pah_freezing_rate
                end if
                if (pah_desorption) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'desorption'
                    ! comp_rate not yet implemented for PAH desorption; guard below will abort.
                end if
                if (pah_cluster_evaporation) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'cluster_evaporation'
                    pah_processes_list(npah_processes)%source = .false.
                    pah_processes_list(npah_processes)%sink = .false.
                    if (cluster_evaporation_model.eq.'RM2026') then
                        pah_processes_list(npah_processes)%comp_rate => pah_cluster_evaporation_rate
                    else
                        pah_processes_list(npah_processes)%comp_rate => pah_cluster_evaporation_rate
                    end if
                end if
                if (pah_photolysis) then
                    npah_processes = npah_processes + 1
                    pah_processes_list(npah_processes)%name = 'photolysis'
                    pah_processes_list(npah_processes)%source = .false.
                    pah_processes_list(npah_processes)%sink = .true.
                    if (photolysis_model.eq.'RM2026') then
                        pah_processes_list(npah_processes)%comp_rate => pah_photolysis_rate
                    else
                        pah_processes_list(npah_processes)%comp_rate => pah_photolysis_rate
                    end if
                end if
            end if
        end if

        ! 3. Guard: abort at startup if any registered process has no comp_rate.
        !    This catches stub entries ('ratd', PAH 'accretion', PAH 'desorption')
        !    before they trigger a null procedure-pointer call at runtime.
        if (allocated(dust_processes_list)) then
            do ii = 1, ndust_processes
                if (.not. associated(dust_processes_list(ii)%comp_rate)) then
                    write(*,'(a,a,a)') 'ERROR: dust process "', &
                        trim(dust_processes_list(ii)%name), &
                        '" has no comp_rate implementation (not yet supported).'
                    call clean_stop()
                end if
            end do
        end if
        if (allocated(pah_processes_list)) then
            do ii = 1, npah_processes
                if (.not. associated(pah_processes_list(ii)%comp_rate)) then
                    write(*,'(a,a,a)') 'ERROR: PAH process "', &
                        trim(pah_processes_list(ii)%name), &
                        '" has no comp_rate implementation (not yet supported).'
                    call clean_stop()
                end if
            end do
        end if
    end subroutine init_dust_processes

    subroutine init_CALIMA_dust(nGroups)
        ! This function initialises dust constants that will be used
        ! for the dust routines, such that they only need to be computed at startup
        ! This is called during init_time.f90
        use hydro_parameters
        use amr_commons, only:myid
        use amr_parameters, only:condinit_kind
        use dust_photoelectric_heating, only: most_negative_allowed_charge
        use dust_optics, only:getRATCrosssection
        use rk4_mod, only: rk4_step
        use anninos_mod, only: anninos_step
        use rk54_mod, only: rk54_step
        use ode_interface_mod, only: dust_solver_step
        implicit none
        integer, intent(in) :: nGroups
        logical :: check_for_pahs
        integer :: ii,jj,kk,kk_loc,n_el,nd_ctype,id_start,id_end,ichemtype,i
        integer :: jbin,kbin,idest
        integer :: idust_pah_interact
        real(dp) :: mf_max,mf_min,prefactor,chi_total,frac_tot,mcoag
        real(dp) :: R
        integer :: iend_chemtype
        ! external :: run_dust_solver_test

        ! Epstein drag coefficient. t_s = rho_s*a/(rho_g*v_th) with v_th the MEAN thermal
        ! speed sqrt(8kT/(pi mu mH)); the solver's c_s is adiabatic, and
        ! v_th = sqrt(8/(pi*gamma))*c_s, so the coefficient is sqrt(pi*gamma/8) = 0.809 for
        ! gamma=5/3. Cf. patch/mrn-scratch/synchro_fine.f90, which carries the isothermal
        ! form of the same constant, sqrt(pi/8) = 0.62665706865775.
        epstein_coef = sqrt(pi * gamma / 8d0)

        ! Resolve the TVA test IC once, here, instead of string-comparing condinit_kind
        ! inside the per-cell/per-face/per-bin solver loops.
        select case (trim(condinit_kind))
        case ('dustydiffuse'); tva_test_mode = TVA_TEST_DIFFUSE
        case ('dustyshock');   tva_test_mode = TVA_TEST_SHOCK
        case ('dustyblast1d'); tva_test_mode = TVA_TEST_BLAST1D
        case ('dustyspress');  tva_test_mode = TVA_TEST_SPRESS
        case ('dustygauss');   tva_test_mode = TVA_TEST_GAUSS
        case ('dustyirtrap');  tva_test_mode = TVA_TEST_IRTRAP
        case default;          tva_test_mode = TVA_TEST_NONE
        end select
        if (myid == 1 .and. tva_test_mode /= TVA_TEST_NONE) then
            if (tva_test_mode == TVA_TEST_IRTRAP) then
                ! This one keeps the real Epstein drag and real thermodynamics;
                ! it only freezes the analytic trapped-IR profile from condinit
                ! by skipping the re-partition in cooling_fine.
                write(*,*) 'WARNING: condinit_kind="',trim(condinit_kind),'" selects a TVA TEST ', &
                           'branch: the trapped-IR energy is held at its analytic profile ', &
                           '(no trapped/streaming re-partition). Drag and thermodynamics are physical.'
            else
                write(*,*) 'WARNING: condinit_kind="',trim(condinit_kind),'" selects a TVA TEST ', &
                           'branch: the stopping time and/or thermodynamics are hardcoded, not physical.'
            end if
        end if

        ! 0. Build the bin-to-chemtype mapping from per-chemtype bin counts
        if (sum(dustbins_per_chemtype) /= ndust) then
            if (myid == 1) then
                write(*,*) 'ERROR: sum(dustbins_per_chemtype)=', sum(dustbins_per_chemtype), &
                    & ' but ndust=', ndust
            end if
            call clean_stop()
        end if
        if (any(dustbins_per_chemtype <= 0)) then
            if (myid == 1) then
                write(*,*) 'ERROR: all dustbins_per_chemtype entries must be > 0'
            end if
            call clean_stop()
        end if
        istart_chemtype(1) = 1
        do ii = 2, ndchemtype
            istart_chemtype(ii) = istart_chemtype(ii-1) + dustbins_per_chemtype(ii-1)
        end do

        call init_dust_processes

        ! 1. Loop over dust bins and set up their properties based on the namelist parameters
        idust_pah_interact = 0
        ichemtype = 1
        iend_chemtype = dustbins_per_chemtype(1)
        do ii = 1, ndust
            do while (ii > iend_chemtype .and. ichemtype < ndchemtype)
                ichemtype = ichemtype + 1
                iend_chemtype = iend_chemtype + dustbins_per_chemtype(ichemtype)
            end do

            ! 1.1 Set the dust variable ordering
            dustbins_props(ii)%dust_index = ii
            dustbins_props(ii)%u_hydro_idx = idust + ii - 1

            ! 1.2 Get the number of elements followed for the dust bin
            n_el = 0
            check_for_pahs = .false.
            do jj = 1, n_elements
                if (dust_composition(ichemtype,jj) > 0d0) then
                    n_el = n_el + 1
                    if (jj == 6) check_for_pahs = .true. ! Check if the grain follows carbon
                end if
            end do
            dustbins_props(ii)%nelements = n_el

            ! 1.3 Allocate and set the composition arrays
            if (allocated(dustbins_props(ii)%el_index)) deallocate(dustbins_props(ii)%el_index)
            if (allocated(dustbins_props(ii)%stoichiometry)) deallocate(dustbins_props(ii)%stoichiometry)
            if (allocated(dustbins_props(ii)%el_mfractions)) deallocate(dustbins_props(ii)%el_mfractions)
            if (allocated(dustbins_props(ii)%el_atomic_masses_amu)) deallocate(dustbins_props(ii)%el_atomic_masses_amu)
            if (allocated(dustbins_props(ii)%el_atomic_masses_g)) deallocate(dustbins_props(ii)%el_atomic_masses_g)
            if (allocated(dustbins_props(ii)%el_conv_factors)) deallocate(dustbins_props(ii)%el_conv_factors)
#ifdef RTZ
            if (allocated(dustbins_props(ii)%el_nions)) deallocate(dustbins_props(ii)%el_nions)
#endif
            if (allocated(dustbins_props(ii)%el_names)) deallocate(dustbins_props(ii)%el_names)
            if (allocated(dustbins_props(ii)%el_atomic_number)) deallocate(dustbins_props(ii)%el_atomic_number)
            allocate(dustbins_props(ii)%el_index(1:n_el), &
                     dustbins_props(ii)%stoichiometry(1:n_el), &
                     dustbins_props(ii)%el_mfractions(1:n_el), &
                     dustbins_props(ii)%el_atomic_masses_amu(1:n_el), &
                     dustbins_props(ii)%el_atomic_masses_g(1:n_el), &
                     dustbins_props(ii)%el_conv_factors(1:n_el), &
                     dustbins_props(ii)%el_names(1:n_el), &
                     dustbins_props(ii)%el_atomic_number(1:n_el))
#ifdef RTZ
            allocate(dustbins_props(ii)%el_nions(1:n_el))
#endif
            kk = 0
            do jj = 1, n_elements
                if (dust_composition(ichemtype,jj) > 0d0) then
                    kk = kk + 1
                    dustbins_props(ii)%stoichiometry(kk) = dust_composition(ichemtype,jj)
#ifdef RTZ
                    dustbins_props(ii)%el_atomic_number(kk) = elements(jj)%atomic_number
                    dustbins_props(ii)%el_index(kk) = jj
                    if (elements(jj)%atomic_mass .le. 0d0) then
                        if (myid == 1) then
                            write(*,*)'ERROR: the element ', elements(jj)%symbol, &
                                & ' has an atomic mass of ', elements(jj)%atomic_mass, &
                                & ' but is needed for dust bin ', ii, &
                                & ' with composition ', dust_composition(ichemtype,jj)
                        end if
                        stop
                    end if
                    dustbins_props(ii)%el_mfractions(kk) = elements(jj)%atomic_mass * dust_composition(ichemtype,jj)
                    dustbins_props(ii)%el_atomic_masses_amu(kk) = elements(jj)%atomic_mass
                    dustbins_props(ii)%el_atomic_masses_g(kk) = elements(jj)%atomic_mass_g
                    dustbins_props(ii)%el_nions(kk) = elements(jj)%n_ions
                    dustbins_props(ii)%el_names(kk) = elements(jj)%symbol
#else
                    dustbins_props(ii)%el_atomic_number(kk) = real(jj,kind=dp)
                    dustbins_props(ii)%el_index(kk) = jj
                    dustbins_props(ii)%el_mfractions(kk) = el_atomic_masses_amu(jj) * dust_composition(ichemtype,jj)
                    dustbins_props(ii)%el_atomic_masses_amu(kk) = el_atomic_masses_amu(jj)
                    dustbins_props(ii)%el_atomic_masses_g(kk) = el_atomic_masses_amu(jj) * amu2g
                    dustbins_props(ii)%el_names(kk) = el_names(jj)
#endif
                end if
            end do

            dustbins_props(ii)%el_mfractions(:) = dustbins_props(ii)%el_mfractions(:) / sum(dustbins_props(ii)%el_mfractions(:)) ! Normalise the mass fractions
            dustbins_props(ii)%el_conv_factors(:) = dustbins_props(ii)%el_mfractions(:) / sum(dustbins_props(ii)%el_mfractions(:)) &
                                                    & / dustbins_props(ii)%el_atomic_masses_g(:)  ! Normalise the conversion factors

            ! 1.4 Set the dust interaction properties
            dustbins_props(ii)%interact_group = ichemtype
            dustbins_props(ii)%interact_pah = check_for_pahs .and. (n_el.eq.1) ! If it has only one element and is carbonaceous
            if (dustbins_props(ii)%interact_pah .and. idust_pah_interact == 0) idust_pah_interact = ii
            ! 1.5 Set the dust grain properties and distribution limits
            dustbins_props(ii)%asize = asize(ii)
            dustbins_props(ii)%asize_cm = asize(ii)*1d-4
            dustbins_props(ii)%sgrain = sgrain(ii)
            dustbins_props(ii)%mgrain = (4D0/3D0)*pi*(asize(ii)*1D-4)**3D0*sgrain(ii)
            dustbins_props(ii)%amin = amin(ii)
            dustbins_props(ii)%amax = amax(ii)
            dustbins_props(ii)%mgrain_min = (4D0/3D0)*pi*(amin(ii)*1D-4)**3D0*sgrain(ii)
            dustbins_props(ii)%mgrain_max = (4D0/3D0)*pi*(amax(ii)*1D-4)**3D0*sgrain(ii)
            dustbins_props(ii)%Youngs_modulus = Youngs_modulus(ichemtype)
            dustbins_props(ii)%Poisson_ratio = Poisson_ratio(ichemtype)
            dustbins_props(ii)%shear_modulus = Youngs_modulus(ichemtype) / (2d0 * (1d0 + Poisson_ratio(ichemtype)))
            dustbins_props(ii)%catastrophic_spec_energy = dustbins_props(ii)%shear_modulus / (2D0 * sgrain(ii))
            dustbins_props(ii)%tensile_strength = tensile_strength(ichemtype)
            dustbins_props(ii)%surf_energy = surf_energy(ichemtype)
            dustbins_props(ii)%work_function = work_function(ichemtype)
            dustbins_props(ii)%band_gap = band_gap(ichemtype)
            dustbins_props(ii)%e_escape_length = e_escape_length(ichemtype)
            dustbins_props(ii)%separate_refractive_index = separate_refractive_index(ichemtype)
            dustbins_props(ii)%Zmin = most_negative_allowed_charge(dustbins_props(ii)%asize_cm,&
                                                            &separate_refractive_index(ichemtype))
            if (allocated(dustbins_props(ii)%phi_prefact)) deallocate(dustbins_props(ii)%phi_prefact)
            allocate(dustbins_props(ii)%phi_prefact(-1:n_elements))
            dustbins_props(ii)%phi_prefact(:) = 1d0
            do jj = -1, n_elements
                dustbins_props(ii)%phi_prefact(jj) = - dble(jj) * e2instatC / (dustbins_props(ii)%asize_cm * eV2erg)
            end do

            ! 1.6 Initialise the dust rates and parameters
            ! This is the common factor for accretion, which is in units of [cm3/s * sqrt(1/(g * K))]
            dustbins_props(ii)%k0_acc = dustbins_props(ii)%asize_cm**2 / dustbins_props(ii)%mgrain * sqrt(8d0 * kB * pi)

            dustbins_props(ii)%nh_coa = nh_coa(ichemtype)
            dustbins_props(ii)%nhmax_coa = nhmax_coa(ichemtype)
            dustbins_props(ii)%nhmax_acc = nhmax_acc(ichemtype)
            dustbins_props(ii)%nhmax_sha = nhmax_sha(ichemtype)

            ! 1.7 Set the injection and destruction parameters
            dustbins_props(ii)%SNII_cond_eff = dust_SNII_cond_eff(ichemtype)
            dustbins_props(ii)%SNIa_cond_eff = dust_SNIa_cond_eff(ichemtype)
            dustbins_props(ii)%AGB_cond_eff = dust_AGB_cond_eff(ichemtype)
            dustbins_props(ii)%SNdest_eff = dust_SNdest_eff(ichemtype)
            dustbins_props(ii)%SNsha_eff = dust_SNsha_eff(ichemtype)
        end do

        ! 2. Initialise the PAH bin properties
        if (dust_pahs) then
            ! Loop over PAH bins
            do ii = 1, npah
                ! 2.1 Set the PAH variable ordering
                pahbins_props(ii)%pah_index = ii
                pahbins_props(ii)%u_hydro_idx = ipah + ii - 1
                pahbins_props(ii)%C_index = 6
                pahbins_props(ii)%dust_index_interact = idust_pah_interact
                pahbins_props(ii)%nd_bins = dustbins_per_chemtype(dustbins_props(idust_pah_interact)%interact_group)

                ! 2.2 Set the PAH properties and distribution limits
                pahbins_props(ii)%nc = pah_nc(ii)
                pahbins_props(ii)%nc_min = pah_nc_min(ii)
                pahbins_props(ii)%nc_max = pah_nc_max(ii)
                pahbins_props(ii)%apah_cm = Nc_to_a(pahbins_props(ii)%nc)
                pahbins_props(ii)%apah = pahbins_props(ii)%apah_cm * 1d4
                pahbins_props(ii)%spah = spah(ii)
                pahbins_props(ii)%mpah = Nc_to_mass(pahbins_props(ii)%nc)
                pahbins_props(ii)%mpah_min = Nc_to_mass(pahbins_props(ii)%nc_min)
                pahbins_props(ii)%mpah_max = Nc_to_mass(pahbins_props(ii)%nc_max)
                pahbins_props(ii)%is_cluster = pah_is_cluster(ii)

                ! 2.3 Set the PAH injection and destruction parameters
                pahbins_props(ii)%AGB_cond_eff = fpah_inwind(ii)
                pahbins_props(ii)%SNdest_eff = pah_SNdest_eff(ii)

                ! 2.4 Set the PAH charges
                ncharge_pah_max = max(ncharge_pah_max, pah_ncharge_states(ii))
                pahbins_props(ii)%ncharge_states = pah_ncharge_states(ii)
                if (allocated(pahbins_props(ii)%charge_states)) deallocate(pahbins_props(ii)%charge_states)
                allocate(pahbins_props(ii)%charge_states(1:pah_ncharge_states(ii)))
                ! Charge states start from -1, then 0, then +1, etc.
                pahbins_props(ii)%charge_states = [(jj-2, jj=1,int(pahbins_props(ii)%ncharge_states))]
                ! Save where cation charge states (>0) start in fcharge_pahs for this PAH bin.
                pahbins_props(ii)%cation_start_idx = 1
                do jj = 1, pahbins_props(ii)%ncharge_states
                    if (pahbins_props(ii)%charge_states(jj) > 0d0) then
                        pahbins_props(ii)%cation_start_idx = jj
                        exit
                    end if
                end do
                if (all(pahbins_props(ii)%charge_states(1:pahbins_props(ii)%ncharge_states) <= 0d0)) then
                    pahbins_props(ii)%cation_start_idx = pahbins_props(ii)%ncharge_states + 1
                end if
            end do
        end if

        ! Allocate the reusable dust chemistry workspace once per rank.
        call dust_helper%init(ndust, npah, nGroups, ncharge_pah_max, n_elements)
        do jj = 1, n_elements
#ifdef RTZ
            dust_helper%el_atomic_mass_g(jj) = elements(jj)%atomic_mass * amu2g
#else
            dust_helper%el_atomic_mass_g(jj) = el_atomic_masses_amu(jj) * amu2g
#endif
            dust_helper%inv_atomic_mass(jj) = 1.0_dp / max(dust_helper%el_atomic_mass_g(jj), tiny(1.0_dp))
        end do

        ! 3. Add the RAT-D parameters
        if (dust_ratd) then
            do ii = 1, ndust
                ichemtype = dustbins_props(ii)%interact_group
                ! Maximum rotational rate for centrifugal disruption [rad/s]
                dustbins_props(ii)%w_disr = 2d0 / (asize(ii) * 1d-4) * sqrt(tensile_strength(ichemtype)/sgrain(ii))

                ! Grain moment of inertia (assuming sphere) [g cm^2]
                dustbins_props(ii)%grain_inertia = 8d0 * pi * sgrain(ii) * (asize(ii) * 1d-4)**5d0 / 15d0

                ! G0-averaged RAT torque [erg]
                dustbins_props(ii)%RAT_torque_0 = getRATCrosssection(fixed_lambda_mean,ii) * u_Mathis1983 * fixed_rad_ani * &
                    & (fixed_lambda_mean * 1d-4/twopi)

                ! Reference gas rotation damping scale [1/g/cm^4]
#ifdef RTZ
                dustbins_props(ii)%tau_gas_0 = 3d0 / (4d0 * sqrt(pi) * elements(1)%atomic_mass * amu2g * (asize(ii) * 1d-4)**4d0)
#else
                dustbins_props(ii)%tau_gas_0 = 3d0 / (4d0 * sqrt(pi) * el_atomic_masses_amu(1) * amu2g * (asize(ii) * 1d-4)**4d0)
#endif
                mf_max = 2d-2 * dustbins_props(ii)%mgrain
                mf_min = 1d-6 * mf_max
                prefactor = 1d0 / (mf_max**slope_frag_func - mf_min**slope_frag_func)
                nd_ctype = dustbins_per_chemtype(dustbins_props(ii)%interact_group)
                if (dustbins_props(ii)%interact_pah .and. npah>0) then
                    if (allocated(dustbins_props(ii)%chi_frag_ratd)) deallocate(dustbins_props(ii)%chi_frag_ratd)
                    allocate(dustbins_props(ii)%chi_frag_ratd(0:npah+nd_ctype))
                    if (pahbins_props(ii)%mpah_min<mf_min) then
                        dustbins_props(ii)%chi_frag_ratd(0) = 0d0
                    else
                        dustbins_props(ii)%chi_frag_ratd(0) = prefactor * (pahbins_props(ii)%mpah_min**slope_frag_func - mf_min**slope_frag_func)
                    end if
                    ! Loop over PAH bins
                    do jj = 1, npah
                        if ((mf_min.ge.pahbins_props(jj)%mpah_max).or.(mf_max<pahbins_props(jj)%mpah_min)) then
                            dustbins_props(ii)%chi_frag_ratd(jj) = 0d0
                        else
                            dustbins_props(ii)%chi_frag_ratd(jj) = prefactor * (min(pahbins_props(jj)%mpah_max,mf_max)**slope_frag_func-max(pahbins_props(jj)%mpah_min,mf_min)**slope_frag_func)
                        end if
                    end do
                else
                    if (allocated(dustbins_props(ii)%chi_frag_ratd)) deallocate(dustbins_props(ii)%chi_frag_ratd)
                    allocate(dustbins_props(ii)%chi_frag_ratd(0:nd_ctype))
                    if (dustbins_props(ii)%mgrain_min<mf_min) then
                        dustbins_props(ii)%chi_frag_ratd(0) = 0d0
                    else
                        dustbins_props(ii)%chi_frag_ratd(0) = prefactor * (dustbins_props(ii)%mgrain_min**slope_frag_func - mf_min**slope_frag_func)
                    end if
                end if
                ! Loop over grain bins below the current dust bin only for the bins with the same composition
                id_start = istart_chemtype(dustbins_props(ii)%interact_group)
                kk = 1
                if (dustbins_props(ii)%interact_pah) kk = npah + kk
                do jj = id_start, dustbins_props(ii)%dust_index
                    if ((mf_min.ge.dustbins_props(jj)%mgrain_max).or.(mf_max<dustbins_props(jj)%mgrain_min)) then
                        dustbins_props(ii)%chi_frag_ratd(kk) = 0d0
                    else
                        dustbins_props(ii)%chi_frag_ratd(kk) = prefactor * (min(dustbins_props(jj)%mgrain_max,mf_max)**slope_frag_func-max(dustbins_props(jj)%mgrain_min,mf_min)**slope_frag_func)
                    end if
                    kk = kk + 1
                end do
                ! Renormalise the fractions
                chi_total = sum(dustbins_props(ii)%chi_frag_ratd(:))
                dustbins_props(ii)%chi_frag_ratd(:) = dustbins_props(ii)%chi_frag_ratd(:) / chi_total
            end do
        end if

        ! 4. Make sure that the fractions of dust masses per bin is okay
        do ii = 1, ndchemtype
            ! 4.1 Check the initial fractions
            frac_tot = sum(fdustmass_ini(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1))
            fdustmass_ini(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1) = fdustmass_ini(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1) / frac_tot

            ! 4.2 Check the fractions of mass in stellar ejecta
            frac_tot = sum(fmass_ej(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1))
            fmass_ej(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1) = fmass_ej(istart_chemtype(ii):istart_chemtype(ii)+dustbins_per_chemtype(ii)-1) / frac_tot
        end do

        if (dust_pahs) then
            ! 4.3 Check the initial fractions for PAHs
            frac_tot = sum(fpahmass_ini(1:npah))
            fpahmass_ini(1:npah) = fpahmass_ini(1:npah) / frac_tot

            ! 4.4 Check the fractions of mass in stellar winds for PAHs
            frac_tot = sum(fpah_inwind(1:npah))
            fpah_inwind(1:npah) = fpah_inwind(1:npah) / frac_tot
        end if

        ! 5. Determine to what dust bins the individual grain coagulation events should transfer mass to
        do ii = 1, ndust
            ichemtype = dustbins_props(ii)%interact_group
            nd_ctype = dustbins_per_chemtype(ichemtype)
            id_start = ii - istart_chemtype(ichemtype) + 1
            id_end = istart_chemtype(ichemtype) + nd_ctype - 1
            if (ii .lt. id_end) then
                if (trim(coagulation_model).eq.'Aoyama2017') then
                    if (allocated(dustbins_props(ii)%k0_coa)) deallocate(dustbins_props(ii)%k0_coa)
                    allocate(dustbins_props(ii)%k0_coa(1))
                    ! For the Aoyama2017 coagulation kernel, the coagulation rate coefficient is just a constant (see their Eq. 14), so we can precompute it here.
                    dustbins_props(ii)%k0_coa(1) = 0.5d0 * 4d0 * pi * dustbins_props(ii)%asize_cm**2d0 * 1d4 * 1d3&
                                                & / dustbins_props(ii)%mgrain ! [cm3/s/g]
                else if (trim(coagulation_model).eq.'Hirashita2015') then
                    ! 5.1 This is the classic approximation of Hirashita et al. (2015)
                    ! which assumes that small grains always moves mass to the next larger grain bin
                    if (allocated(dustbins_props(ii)%idend_coag)) deallocate(dustbins_props(ii)%idend_coag)
                    if (allocated(dustbins_props(ii)%vthresh_coag)) deallocate(dustbins_props(ii)%vthresh_coag)
                    if (allocated(dustbins_props(ii)%k0_coa)) deallocate(dustbins_props(ii)%k0_coa)
                    allocate(dustbins_props(ii)%idend_coag(1))
                    allocate(dustbins_props(ii)%vthresh_coag(1))
                    allocate(dustbins_props(ii)%k0_coa(1))
                    dustbins_props(ii)%idend_coag(1) = ii + 1
                    ! Compute the threshold velocity for coagulation (Choski et al. 1993)
                    R = 0.5d0 * dustbins_props(ii)%asize_cm
                    dustbins_props(ii)%vthresh_coag(1) = &
                        & 10.7d0 * dustbins_props(ii)%surf_energy**(5d0/3d0) &
                        & / (dustbins_props(ii)%Youngs_modulus**(1d0/3d0) &
                        & * R**(5d0/6d0) * sqrt(dustbins_props(ii)%sgrain))
                    dustbins_props(ii)%k0_coa(1) = sqrt(8d0/(3d0*pi)) * 4d0 * pi * dustbins_props(ii)%asize_cm**2d0&
                                                & / dustbins_props(ii)%mgrain ! [cm3/s/g]
                else if (trim(coagulation_model).eq.'Smoluchowski1916') then
                    ! 5.2 This is the correct treatment of individual grain coagulation
                    ! from Smoluchowski (1916) which requires the computation of the final
                    ! coagulated grain mass and to determine in which dust bin it ends up
                    if (allocated(dustbins_props(ii)%idend_coag)) deallocate(dustbins_props(ii)%idend_coag)
                    if (allocated(dustbins_props(ii)%vthresh_coag)) deallocate(dustbins_props(ii)%vthresh_coag)
                    if (allocated(dustbins_props(ii)%k0_coa)) deallocate(dustbins_props(ii)%k0_coa)
                    allocate(dustbins_props(ii)%idend_coag(1:nd_ctype))
                    allocate(dustbins_props(ii)%vthresh_coag(1:nd_ctype))
                    allocate(dustbins_props(ii)%k0_coa(1:nd_ctype))
                    ! Default to no bin change; overwritten below when a larger-bin destination exists.
                    dustbins_props(ii)%idend_coag(1:nd_ctype) = ii
                    dustbins_props(ii)%vthresh_coag(1:nd_ctype) = 0d0
                    do kk_loc = 1, nd_ctype
                        jbin = istart_chemtype(ichemtype) + kk_loc - 1
                        mcoag = dustbins_props(ii)%mgrain + dustbins_props(jbin)%mgrain
                        ! Identify the destination bin for the newly coagulated grain mass.
                        idest = id_end
                        do kk = 1, nd_ctype
                            kbin = istart_chemtype(ichemtype) + kk - 1
                            if (mcoag .ge. dustbins_props(kbin)%mgrain_min .and. &
                                & mcoag .lt. dustbins_props(kbin)%mgrain_max) then
                                idest = kbin
                                exit
                            end if
                        end do
                        dustbins_props(ii)%idend_coag(kk_loc) = idest
                        ! Compute the threshold velocity for coagulation (Choski et al. 1993)
                        R = (dustbins_props(ii)%asize_cm * dustbins_props(jbin)%asize_cm) &
                            & / (dustbins_props(ii)%asize_cm + dustbins_props(jbin)%asize_cm)
                        dustbins_props(ii)%vthresh_coag(kk_loc) = &
                            & 21.4d0 * sqrt((dustbins_props(ii)%asize_cm**3d0+dustbins_props(jbin)%asize_cm**3d0)/&
                            & (dustbins_props(ii)%asize_cm + dustbins_props(jbin)%asize_cm)**3d0) &
                            & * dustbins_props(ii)%surf_energy**(5d0/3d0) / (dustbins_props(ii)%Youngs_modulus**(1d0/3d0) * R**(5d0/6d0) * &
                            & sqrt(dustbins_props(ii)%sgrain))
                        dustbins_props(ii)%k0_coa(kk_loc) = sqrt(8d0/(3d0*pi)) * pi * (dustbins_props(ii)%asize_cm + dustbins_props(jbin)%asize_cm)**2d0&
                                                    & / (dustbins_props(ii)%mgrain * dustbins_props(jbin)%mgrain / (dustbins_props(ii)%mgrain + dustbins_props(jbin)%mgrain)) ! [cm3/s/g]
                    end do
                end if
            end if
        end do

        ! 6. Allocate per-process tracking arrays (sizes known after init_dust_processes).
        !    dM_ode_* are only needed when dust_debug=.true. (mass budget logging).
        !    ode_reduction_count_* are needed for dust_log=.true. (solver stats).
        if (ndust_processes > 0) then
            if (dust_debug) then
                if (.not. allocated(dM_ode_dust)) &
                    allocate(dM_ode_dust(ndust+npah, ndust_processes))
                dM_ode_dust(:,:) = 0.0_dp
            end if
            if (.not. allocated(ode_reduction_count_dust)) &
                allocate(ode_reduction_count_dust(ndust_processes))
            ode_reduction_count_dust(:) = 0_8
        end if
        if (npah_processes > 0) then
            if (dust_debug) then
                if (.not. allocated(dM_ode_pah)) &
                    allocate(dM_ode_pah(ndust+npah, npah_processes))
                dM_ode_pah(:,:) = 0.0_dp
            end if
            if (.not. allocated(ode_reduction_count_pah)) &
                allocate(ode_reduction_count_pah(npah_processes))
            ode_reduction_count_pah(:) = 0_8
        end if
        smallr_dust = smallr

        ! 7. Other constants and parameters
#if NPAH>0
        if (dust_pahs) then
            if (pah_sputtering) call init_pah_sputtering_tables
            if (pah_photolysis) call init_pah_dissociation_tables
            ! Always: the PAH charge tables (fcharge_tab) built here are also used by
            ! the RT absorption (compute_dust_rad_rates), radiation pressure and the
            ! cooling pre-computation, whether or not pah_pe_heating is on
            call init_pah_peh_tables
        end if
#endif

        if (dust_accretion.or.dust_shattering.or.dust_coagulation&
            &.or.pah_freezing) then
            comp_sigma_turb = .true.
        end if

        if (dust_acc_coulomb.or.pah_freezing) then
            Coulomb_precompute = .true.
        end if

        ! 8. Read the dust thermal sputtering tables
        if (dust_sputtering .and. sputtering_model.eq.'RM2026')  call init_thermal_sputtering_tables

        ! 8b. Read the dust thermal sublimation tables
        if (dust_sublimation) call init_dust_sublimation_tables

        ! 9. Read the dust collisional tables
        if (dust_coll_cooling) call init_dust_collisional_tables

        ! 10. Read the uniform-ISRF charging tables: charging_model='WDB06isrf', the PE heating of the
        ! models that do not compute their own (WDB06rt does), and the Coulomb factors of the WDB06rt
        ! cells without local photons
        if (trim(charging_model) == 'WDB06isrf' .or. (dust_pe_heating .and. trim(charging_model) /= 'WDB06rt') &
            .or. (Coulomb_precompute .and. trim(charging_model) == 'WDB06rt')) call init_dust_charging_isrf

        ! 11b. Read the dust IR emission tables
        call init_dust_IR_emission_tables

        ! 12. Cache the BH80 collisional heating factors that only depend on the dust bins
        if (dust_coll_cooling) call init_dust_coll_heating_BH80_cache


        ! Select the ODE solver procedure pointer
        if (dust_solver_type == 1) then
            dust_solver_step => rk4_step
            solver_substepped = .true.
        else if (dust_solver_type == 2) then
            dust_solver_step => anninos_step
            solver_substepped = .false.
        else if (dust_solver_type == 3) then
            dust_solver_step => rk54_step
            solver_substepped = .true.
        else
            if (myid == 1) then
                write(*,*) 'ERROR: Invalid dust_solver_type = ', dust_solver_type
            end if
            call clean_stop()
        end if

        ! 13. Print the CALIMA dust properties for the user
        if (myid == 1) then
            call print_dust_parameters
        end if

        ! if (dust_test) then
        !     call run_dust_solver_test()
        ! end if

    end subroutine init_CALIMA_dust

    subroutine read_CALIMA_params(nml_ok,nGroups)
        use amr_commons, only:myid
        implicit none
        logical, intent(inout) :: nml_ok
        integer, intent(in) :: nGroups

        logical :: check_dust

        namelist/calima_params/&
                ! Dust physics flags
                dust_log,dust_debug,dust_solver_type,dust_only_rtadv,dust_eq_test,dust_SNdest,dust_inSN,dust_inSNIa,dust_inSW,&
                dust_test,test_nH,test_Tk,test_nsteps,test_dt,test_ne,test_mu,&
                dust_coagulation,dust_coagulation_boost,dust_shattering,dust_shattering_all,dust_shattering_dest,dust_shattering_SN,&
                dust_accretion,dust_sputtering,dust_sputtering_charge,dust_acc_coulomb,dust_ratd,dust_coll_cooling,dust_coll_lowT,dust_coll_charge,&
                dust_sublimation,dust_pe_heating,dust_pe_heating_isrf,ratd_only_rtadv,poppe_ice_enhancement,H2ondust,dust_turbulent_model,&
                ! PAH physics flags
                pah_accretion,pah_acc_spu,pah_coalescence,pah_freezing,pah_desorption,pah_photolysis,pah_sn_destruction,pah_cluster_evaporation,&
                pah_AGBwinds,pah_sputtering,pah_pe_heating,pah_pe_heating_isrf,pah_pe_nolyman,H2onpah,&
                ! Dust dynamics flags
                dust_tva, dust_radpressure, use_w_drift_test, w_drift_test,drag_coefficient,&
                tva_wmax_cs,drag_model,&
                ! Dust modelling options
                sputtering_model,accretion_model,shattering_model,coagulation_model,dust_velocity_model,charging_model,nZmix,ice_model,&
                dust_rtgroups_debug,dust_rtgroups_debug_max,dust_rtgroups_verify,dust_charging_timer,dust_ion_recombination,&
                rtg_predict_tol,rtg_predict_tol_shape,&
                ! PAH modelling options
                photolysis_model,peh_attach_model,coalescence_model,pah_h2_model,pah_growth_model,pah_sputtering_model,&
                cluster_evaporation_model,&
                ! Efficiency parameters
                Sconstant,max_accretion_rate,nh_coa,nhmax_acc,nhmax_coa,nhmax_sha,&
                dust_SNdest_eff,dust_SNsha_eff,dust_SNII_cond_eff,dust_SNIa_cond_eff,dust_AGB_cond_eff,&
                Coulomb_enhance,tensile_strength,Youngs_modulus,Poisson_ratio,surf_energy,work_function,band_gap,e_escape_length,&
                separate_refractive_index,slope_frag_func,errmax,countmax,GDinit,DTMinit,fpah_ini,smallr_dust,&
                ! Dust grain and PAHs bin properties
                dust_composition,dustbins_per_chemtype,&
                asize,sgrain,amin,amax,fmass_ej,&
                pah_nc,pah_nc_min,pah_nc_max,spah,pah_SNdest_eff,fpah_inwind,pah_nc,pah_ncharge_states,&
                pah_is_cluster, &
                ! ISM depletion factors
                fDust_depletions,fCDust_inPAH,GD_solar,DTM_solar,fdustmass_ini,fpahmass_ini,&
                ! Radiation parameters
                fixed_rad_ani,fixed_lambda_mean,&
                ! External dust files
                dust_tables_dir

        ! Initialise as fine
        nml_ok = .true.

        ! 1. Read namelist file
        rewind(1)
        read(1,NML=calima_params,END=301)
301     continue ! No harm if no CALIMA namelist

        ! 2. Check that the parameters are consistent and set any internal variables
        check_dust = check_params_dust(myid)
        if(.not.check_dust) then
            nml_ok = .false.
            return
        end if

        ! 3. If everything is fine, we initialise CALIMA dust parameters and tables
        call init_CALIMA_dust(nGroups)
    end subroutine read_CALIMA_params

    subroutine cmp_lim_elem(dust_index,n_el,el_density,lim_index)
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
        lim_index = minloc(el_density_lim,1)
    end subroutine cmp_lim_elem

    subroutine init_dust_collisional_tables
        ! This subroutine reads at the initialisation of dust parameters
        ! the collisional cooling of gas species with dust grains. This is
        ! based on the pre-computed tables by Rodriguez Montero et al. (2024)
        ! using the stopping power of graphite and SiO2 solid materials.
        ! Tables are organised per dust bin and per element.
        ! NOTE: Keep in mind that these tables are already in log10, such that easy
        ! linear interpolation in log-log space can be computed on the fly!
        use amr_commons, only: myid
        implicit none

        logical :: ok, ok_all
        integer :: nT, nphi, istat, i, j, ii, Zi, nmax, iphi0, n
        character(len=20) :: dustlabel
        character(len=7) :: Zi_str
        character(len=128) :: collisional_filename
        real(dp), allocatable :: phi_grid(:), T_grid(:)

        ! 1. Check first that all files are in the expected place
        ok_all = .true.
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            do i = 1, n_elements
#ifdef RTZ
                if (elements(i)%atomic_number <= 0) cycle
                Zi = elements(i)%atomic_number
#else
                Zi = i
#endif
                write(Zi_str, '(I0)') Zi
                write(collisional_filename, '(A,A,A,A,A,A)') trim(dust_tables_dir), &
                    'cooling_', trim(dustlabel), '_Z_', trim(Zi_str)
                inquire(file=collisional_filename, exist=ok)
                ok_all = ok_all .and. ok
            end do
        end do

        if (.not. ok_all) then
            if (myid .eq. 1) then
                write(*, *) 'ERROR IN DUST COLLISIONAL TABLES'
                write(*, *) 'Cannot access dust directory ', TRIM(dust_tables_dir)
                write(*, *) 'Directory ' // TRIM(dust_tables_dir) // ' not found'
                write(*, *) 'You need to set this correctly for dust_tables_dir in the namelist.'
            end if
            call clean_stop
        end if

        ! 2. Read the files for each dust bin and element
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            do i = 1, n_elements
#ifdef RTZ
                if (elements(i)%atomic_number <= 0) cycle
                Zi = elements(i)%atomic_number
#else
                Zi = i
#endif
                write(Zi_str, '(I0)') Zi
                write(collisional_filename, '(A,A,A,A,A,A)') trim(dust_tables_dir), &
                    'cooling_', trim(dustlabel), '_Z_', trim(Zi_str)

                open(25, file=trim(collisional_filename), status='old', action='read', iostat=istat)
                if (istat /= 0) then
                    write(*, *) 'Error opening file: ', trim(collisional_filename)
                    call clean_stop
                end if

                ! Skip header lines
                do n = 1, 6
                    read(25, *)
                end do

                ! Read the number of temperature points and number of phi values
                read(25, *) nT, nphi

                ! Allocate the DustTable structure
                if (allocated(dustbins_props(ii)%collisional_tab(i)%npts)) then
                    deallocate(dustbins_props(ii)%collisional_tab(i)%npts)
                end if
                allocate(dustbins_props(ii)%collisional_tab(i)%npts(1:2))
                dustbins_props(ii)%collisional_tab(i)%name = 'dust_collisional_'//trim(dustlabel)//'_Z_'//trim(Zi_str)
                dustbins_props(ii)%collisional_tab(i)%ndim = 2
                dustbins_props(ii)%collisional_tab(i)%npts(1) = nT
                dustbins_props(ii)%collisional_tab(i)%npts(2) = nphi
                if (allocated(dustbins_props(ii)%collisional_tab(i)%ipos_zero)) then
                    deallocate(dustbins_props(ii)%collisional_tab(i)%ipos_zero)
                end if
                allocate(dustbins_props(ii)%collisional_tab(i)%ipos_zero(1:2))
                dustbins_props(ii)%collisional_tab(i)%ipos_zero(:) = 1

                ! Allocate table axes and rates
                nmax = max(nT, nphi)
                if (allocated(dustbins_props(ii)%collisional_tab(i)%tab1d)) then
                    deallocate(dustbins_props(ii)%collisional_tab(i)%tab1d)
                end if
                allocate(dustbins_props(ii)%collisional_tab(i)%tab1d(1:nmax, 1:2))
                dustbins_props(ii)%collisional_tab(i)%tab1d = 0d0
                if (allocated(dustbins_props(ii)%collisional_tab(i)%tab2d)) then
                    deallocate(dustbins_props(ii)%collisional_tab(i)%tab2d)
                end if
                allocate(dustbins_props(ii)%collisional_tab(i)%tab2d(1:nT, 1:nphi, 1:1))

                ! Allocate temporary arrays for phi and temperature grids
                if (allocated(phi_grid)) deallocate(phi_grid)
                if (allocated(T_grid)) deallocate(T_grid)
                allocate(phi_grid(1:nphi))
                allocate(T_grid(1:nT))

                ! Read the phi grid (line 2)
                read(25, *) phi_grid(1:nphi)
                dustbins_props(ii)%collisional_tab(i)%tab1d(1:nphi, 2) = phi_grid(1:nphi)
                iphi0 = minloc(abs(phi_grid(1:nphi)), 1)
                dustbins_props(ii)%collisional_tab(i)%ipos_zero(2) = iphi0

                ! Read the temperature values and collisional rates
                do j = 1, nT
                    read(25, *) T_grid(j), dustbins_props(ii)%collisional_tab(i)%tab2d(j, 1:nphi, 1)
                end do
                dustbins_props(ii)%collisional_tab(i)%tab1d(1:nT, 1) = T_grid(1:nT)

                call dustbins_props(ii)%collisional_tab(i)%init()

                close(25)

                deallocate(phi_grid, T_grid)
            end do

            ! Ensure electron collisional table (index 0) is present for this dust bin.
            ! Try to read an explicit electron table file named with Z_0; if not found,
            ! fall back to copying the first initialized element table (usually H).
            write(Zi_str, '(I0)') 0
            write(collisional_filename, '(A,A,A,A,A,A)') trim(dust_tables_dir), &
                'cooling_', trim(dustlabel), '_Z_', trim(Zi_str)
            inquire(file=trim(collisional_filename), exist=ok)
            if (.not. ok) then
                write(*,*) 'ERROR: electron collisional table not found for dust bin:', trim(dustlabel)
                write(*,*) '  Expected file:', trim(collisional_filename)
                stop 1
            end if

            open(25, file=trim(collisional_filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*,*) 'ERROR: could not open electron collisional file for', trim(dustlabel)
                write(*,*) '  File:', trim(collisional_filename), ' iostat=', istat
                stop 1
            end if

            ! Skip header lines
            do n = 1, 6
                read(25, *)
            end do

            read(25, *, iostat=istat) nT, nphi
            if (istat /= 0) then
                write(*,*) 'ERROR: failed reading header (nT,nphi) from', trim(collisional_filename), ' iostat=', istat
                close(25)
                stop 1
            end if

            if (allocated(dustbins_props(ii)%collisional_tab(0)%npts)) then
                deallocate(dustbins_props(ii)%collisional_tab(0)%npts)
            end if
            allocate(dustbins_props(ii)%collisional_tab(0)%npts(1:2))
            dustbins_props(ii)%collisional_tab(0)%ndim = 2
            dustbins_props(ii)%collisional_tab(0)%npts(1) = nT
            dustbins_props(ii)%collisional_tab(0)%npts(2) = nphi
            if (allocated(dustbins_props(ii)%collisional_tab(0)%ipos_zero)) then
                deallocate(dustbins_props(ii)%collisional_tab(0)%ipos_zero)
            end if
            allocate(dustbins_props(ii)%collisional_tab(0)%ipos_zero(1:2))
            dustbins_props(ii)%collisional_tab(0)%ipos_zero(:) = 1
            nmax = max(nT, nphi)
            if (allocated(dustbins_props(ii)%collisional_tab(0)%tab1d)) then
                deallocate(dustbins_props(ii)%collisional_tab(0)%tab1d)
            end if
            allocate(dustbins_props(ii)%collisional_tab(0)%tab1d(1:nmax, 1:2))
            dustbins_props(ii)%collisional_tab(0)%tab1d = 0d0
            if (allocated(dustbins_props(ii)%collisional_tab(0)%tab2d)) then
                deallocate(dustbins_props(ii)%collisional_tab(0)%tab2d)
            end if
            allocate(dustbins_props(ii)%collisional_tab(0)%tab2d(1:nT, 1:nphi, 1:1))
            allocate(phi_grid(1:nphi)); allocate(T_grid(1:nT))

            read(25, *, iostat=istat) phi_grid(1:nphi)
            if (istat /= 0) then
                write(*,*) 'ERROR: failed reading phi grid from', trim(collisional_filename), ' iostat=', istat
                close(25)
                stop 1
            end if

            dustbins_props(ii)%collisional_tab(0)%tab1d(1:nphi, 2) = phi_grid(1:nphi)
            iphi0 = minloc(abs(phi_grid(1:nphi)), 1)
            dustbins_props(ii)%collisional_tab(0)%ipos_zero(2) = iphi0

            do j = 1, nT
                read(25, *, iostat=istat) T_grid(j), dustbins_props(ii)%collisional_tab(0)%tab2d(j, 1:nphi, 1)
                if (istat /= 0) then
                    write(*,*) 'ERROR: failed reading temperature/row', j, 'from', trim(collisional_filename), ' iostat=', istat
                    close(25)
                    stop 1
                end if
            end do

            dustbins_props(ii)%collisional_tab(0)%tab1d(1:nT, 1) = T_grid(1:nT)

            call dustbins_props(ii)%collisional_tab(0)%init()
            close(25)
            if (allocated(phi_grid)) deallocate(phi_grid)
            if (allocated(T_grid)) deallocate(T_grid)
        end do

    end subroutine init_dust_collisional_tables

    subroutine init_thermal_sputtering_tables
        ! This subroutine reads at the initialisation of dust parameters
        ! the pre-computed rate constants for thermal sputtering of regular
        ! carbonaceous and silicate grains for each element. This is based on the model built
        ! by Kirchschlager and collaborators, based on the original modelling
        ! by Tielens and Nozawa. A version of this model is summarised by the fitting
        ! functions to the sputtering yields by Chia-Yu Hu to the results of Nozawa et
        ! al. (2006). For further details read Rodriguez Montero et al. (2024)
        ! Tables are now organised per dust bin and per element.
        ! NOTE: Keep in mind that these yields are already in log10, such that easy
        ! linear interpolation in log-log space can be computed on the fly!
        use amr_commons, only: myid
        implicit none

        logical :: ok, ok_all
        integer :: nT, nphi, istat, i, j, ii, Zi, nmax, iphi0, n
        character(len=20) :: dustlabel
        character(len=7) :: Zi_str
        character(len=128) :: sputtering_filename
        real(dp), allocatable :: phi_grid(:), T_grid(:)

        ! 1. Check first that all files are in the expected place
        ok_all = .true.
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            do i = 1, n_elements
#ifdef RTZ
                if (elements(i)%atomic_number <= 0) cycle
                Zi = elements(i)%atomic_number
#else
                Zi = i
#endif
                write(Zi_str, '(I0)') Zi
                write(sputtering_filename, '(A,A,A,A,A,A)') trim(dust_tables_dir), &
                    'sputtering_', trim(dustlabel), '_Z_', trim(Zi_str)
                inquire(file=sputtering_filename, exist=ok)
                ok_all = ok_all .and. ok
            end do
        end do

        if (.not. ok_all) then
            if (myid .eq. 1) then
                write(*, *) 'ERROR IN THERMAL SPUTTERING TABLES'
                write(*, *) 'Cannot access dust directory ', TRIM(dust_tables_dir)
                write(*, *) 'Directory ' // TRIM(dust_tables_dir) // ' not found'
                write(*, *) 'You need to set this correctly for dust_tables_dir in the namelist.'
            end if
            call clean_stop
        end if

        ! 2. Read the files for each dust bin and element
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            do i = 1, n_elements
#ifdef RTZ
                if (elements(i)%atomic_number <= 0) cycle
                Zi = elements(i)%atomic_number
#else
                Zi = i
#endif
                write(Zi_str, '(I0)') Zi
                write(sputtering_filename, '(A,A,A,A,A,A)') trim(dust_tables_dir), &
                    'sputtering_', trim(dustlabel), '_Z_', trim(Zi_str)

                open(25, file=trim(sputtering_filename), status='old', action='read', iostat=istat)
                if (istat /= 0) then
                    write(*, *) 'Error opening file: ', trim(sputtering_filename)
                    call clean_stop
                end if

                ! Skip header lines
                do n = 1, 6
                    read(25, *)
                end do

                ! Read the number of temperature points and number of phi values
                read(25, *) nT, nphi

                ! Allocate the DustTable structure
                if (allocated(dustbins_props(ii)%sputtering_tab(i)%npts)) then
                    deallocate(dustbins_props(ii)%sputtering_tab(i)%npts)
                end if
                allocate(dustbins_props(ii)%sputtering_tab(i)%npts(1:2))
                dustbins_props(ii)%sputtering_tab(i)%name = 'sputtering_'//trim(dustlabel)//'_Z_'//trim(Zi_str)
                dustbins_props(ii)%sputtering_tab(i)%ndim = 2
                dustbins_props(ii)%sputtering_tab(i)%npts(1) = nT
                dustbins_props(ii)%sputtering_tab(i)%npts(2) = nphi
                if (allocated(dustbins_props(ii)%sputtering_tab(i)%ipos_zero)) then
                    deallocate(dustbins_props(ii)%sputtering_tab(i)%ipos_zero)
                end if
                allocate(dustbins_props(ii)%sputtering_tab(i)%ipos_zero(1:2))
                dustbins_props(ii)%sputtering_tab(i)%ipos_zero(:) = 1

                ! Allocate table axes and rates
                nmax = max(nT, nphi)
                if (allocated(dustbins_props(ii)%sputtering_tab(i)%tab1d)) then
                    deallocate(dustbins_props(ii)%sputtering_tab(i)%tab1d)
                end if
                allocate(dustbins_props(ii)%sputtering_tab(i)%tab1d(1:nmax, 1:2))
                dustbins_props(ii)%sputtering_tab(i)%tab1d = 0d0
                if (allocated(dustbins_props(ii)%sputtering_tab(i)%tab2d)) then
                    deallocate(dustbins_props(ii)%sputtering_tab(i)%tab2d)
                end if
                allocate(dustbins_props(ii)%sputtering_tab(i)%tab2d(1:nT, 1:nphi, 1:1))

                ! Allocate temporary arrays for phi and temperature grids
                if (allocated(phi_grid)) deallocate(phi_grid)
                if (allocated(T_grid)) deallocate(T_grid)
                allocate(phi_grid(1:nphi))
                allocate(T_grid(1:nT))

                ! Read the phi grid (line 2)
                read(25, *) phi_grid(1:nphi)
                dustbins_props(ii)%sputtering_tab(i)%tab1d(1:nphi, 2) = phi_grid(1:nphi)
                iphi0 = minloc(abs(phi_grid(1:nphi)), 1)
                dustbins_props(ii)%sputtering_tab(i)%ipos_zero(2) = iphi0

                ! Read the temperature values and sputtering yields
                do j = 1, nT
                    read(25, *) T_grid(j), dustbins_props(ii)%sputtering_tab(i)%tab2d(j, 1:nphi, 1)
                end do
                dustbins_props(ii)%sputtering_tab(i)%tab1d(1:nT, 1) = T_grid(1:nT)

                call dustbins_props(ii)%sputtering_tab(i)%init()

                close(25)
                deallocate(phi_grid, T_grid)
            end do
        end do

    end subroutine init_thermal_sputtering_tables

    subroutine init_dust_sublimation_tables
        ! Reads the pre-computed thermal sublimation erosion rate tables for
        ! each dust bin. Files are named sublimation_rate_DustBin_xx.dat and
        ! contain a header (lines starting with '#') followed by two columns:
        !   col 1: dust temperature T_d [K]
        !   col 2: fractional erosion rate epsilon = |da/dt|/a [s-1]
        ! The temperature axis is stored as log10(T_d) and the rate as
        ! log10(epsilon) so that linear interpolation can be done on the fly
        ! using the dust temperature saved in the dust chemistry workspace.
        use amr_commons, only: myid
        implicit none

        logical :: ok, ok_all
        integer :: nT, istat, j, ii, ndata
        character(len=20) :: dustlabel
        character(len=256) :: sublimation_filename
        character(len=256) :: line
        real(dp) :: Td, eps
        real(dp), parameter :: log_floor = -300d0

        ! 1. Check first that all files are in the expected place
        ok_all = .true.
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            write(sublimation_filename, '(A,A,A,A)') trim(dust_tables_dir), &
                'sublimation_rate_', trim(dustlabel), '.dat'
            inquire(file=trim(sublimation_filename), exist=ok)
            ok_all = ok_all .and. ok
        end do

        if (.not. ok_all) then
            if (myid .eq. 1) then
                write(*, *) 'ERROR IN THERMAL SUBLIMATION TABLES'
                write(*, *) 'Cannot access dust directory ', TRIM(dust_tables_dir)
                write(*, *) 'Directory ' // TRIM(dust_tables_dir) // ' not found'
                write(*, *) 'You need to set this correctly for dust_tables_dir in the namelist.'
            end if
            call clean_stop
        end if

        ! 2. Read the file for each dust bin
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            write(sublimation_filename, '(A,A,A,A)') trim(dust_tables_dir), &
                'sublimation_rate_', trim(dustlabel), '.dat'

            ! 2.1 First pass: count the number of data rows (skip comments/blanks)
            open(25, file=trim(sublimation_filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(sublimation_filename)
                call clean_stop
            end if
            ndata = 0
            do
                read(25, '(A)', iostat=istat) line
                if (istat /= 0) exit
                line = adjustl(line)
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '#') cycle
                ndata = ndata + 1
            end do
            close(25)

            if (ndata < 2) then
                write(*, *) 'Error: sublimation table has too few rows: ', trim(sublimation_filename)
                call clean_stop
            end if
            nT = ndata

            ! 2.2 Allocate the DustTable structure (1D table)
            if (allocated(dustbins_props(ii)%sublimation_tab%npts)) then
                deallocate(dustbins_props(ii)%sublimation_tab%npts)
            end if
            allocate(dustbins_props(ii)%sublimation_tab%npts(1:1))
            dustbins_props(ii)%sublimation_tab%name = 'sublimation_rate_'//trim(dustlabel)
            dustbins_props(ii)%sublimation_tab%ndim = 1
            dustbins_props(ii)%sublimation_tab%npts(1) = nT
            if (allocated(dustbins_props(ii)%sublimation_tab%tab1d)) then
                deallocate(dustbins_props(ii)%sublimation_tab%tab1d)
            end if
            allocate(dustbins_props(ii)%sublimation_tab%tab1d(1:nT, 1:2))
            dustbins_props(ii)%sublimation_tab%tab1d = 0d0

            ! 2.3 Second pass: read the temperature and erosion rate values.
            !     Store log10(T_d) and log10(epsilon) for on-the-fly interpolation.
            open(25, file=trim(sublimation_filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(sublimation_filename)
                call clean_stop
            end if
            j = 0
            do
                read(25, '(A)', iostat=istat) line
                if (istat /= 0) exit
                line = adjustl(line)
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '#') cycle
                read(line, *, iostat=istat) Td, eps
                if (istat /= 0) then
                    write(*, *) 'Error parsing sublimation table row in ', trim(sublimation_filename)
                    call clean_stop
                end if
                j = j + 1
                dustbins_props(ii)%sublimation_tab%tab1d(j, 1) = log10(Td)
                dustbins_props(ii)%sublimation_tab%tab1d(j, 2) = eps
            end do
            close(25)

            call dustbins_props(ii)%sublimation_tab%init()
            dustbins_props(ii)%T_sublimation_min = 10d0**dustbins_props(ii)%sublimation_tab%tab1d(1, 1)
        end do

    end subroutine init_dust_sublimation_tables

    subroutine init_pah_sputtering_tables
        ! This subroutine reads at the initialisation of dust parameters
        ! the pre-computed rate constants for thermal sputtering of PAHs
        ! using a modified version of the Micelotta et al. (2010b) model
        ! (see Rodriguez Montero et al. (2024) for further details).
        ! These files needs to be saved in the same directory as the oppacity
        ! tables. They can be easily computed using the export_rates function
        ! in the PAHs_sputtering.py script coming with DustRAMSES.
        ! File naming convention: pah_sputtering_pahbin_{i}_Z_{atomic_number}
        ! where Z=0 for electrons, Z=1 for H, Z=2 for He, Z=6 for C, Z=8 for O, etc.
        ! NOTE: Keep in mind that rate values are already in log10, such that
        ! easy linear interpolation in log-log space can be computed 
        ! on the fly!!
        ! NOTE(2): Each file format contains:
        ! - Header: nT (number of temperature points)
        ! - Lines 1 to nT: Temperature values (log10)
        ! - Lines nT+1 to 2*nT: Rate values (log10)
        use amr_commons, only:myid
#ifdef RTZ
        use rtz_module, only:elements
#endif
        implicit none

        logical :: file_exists
        integer :: nT, istat, i, ipahbin, iel, Zi, n
        character(len=256) :: pah_filename
        real(dp), allocatable :: T_grid(:), rate_grid(:)
        character(len=8) :: Z_str, ipah_str

        nT = 0
        ! Read data and allocate DustTable structures for each PAH bin and element.
        ! Missing files are ignored; corresponding tables remain initialized=.false.
        do ipahbin = 1, npah
            write(ipah_str, '(I2.2)') ipahbin

            ! Reset all PAH sputtering tables for this PAH bin.
            do iel = 0, n_elements
                pahbins_props(ipahbin)%sputtering_tab(iel)%name = 'pah_sputtering_pahbin_'//trim(adjustl(ipah_str))//'_Z_'//trim(adjustl(Z_str))
                pahbins_props(ipahbin)%sputtering_tab(iel)%initialised = .false.
                pahbins_props(ipahbin)%sputtering_tab(iel)%ndim = 0
            end do

            ! Electrons are stored in slot 0 with Z=0 in file naming.
            Zi = 0
            write(Z_str, '(I0)') Zi
            write(pah_filename, '(A,A,A,A,A,A,A)') trim(dust_tables_dir), &
                'sputtering_PAHBin_', trim(adjustl(ipah_str)), '_Z_', trim(adjustl(Z_str))
            inquire(file=trim(pah_filename), exist=file_exists)

            if (file_exists) then
                open(10, file=trim(pah_filename), status='old', action='read', iostat=istat)
                if (istat == 0) then
                    do n = 1, 6
                        read(10, *)
                    end do
                    read(10, *) nT

                    if (allocated(pahbins_props(ipahbin)%sputtering_tab(0)%npts)) deallocate(pahbins_props(ipahbin)%sputtering_tab(0)%npts)
                    allocate(pahbins_props(ipahbin)%sputtering_tab(0)%npts(1:1))
                    pahbins_props(ipahbin)%sputtering_tab(0)%ndim = 1
                    pahbins_props(ipahbin)%sputtering_tab(0)%npts(1) = nT

                    if (allocated(pahbins_props(ipahbin)%sputtering_tab(0)%tab1d)) deallocate(pahbins_props(ipahbin)%sputtering_tab(0)%tab1d)
                    allocate(pahbins_props(ipahbin)%sputtering_tab(0)%tab1d(1:nT, 1:2))
                    pahbins_props(ipahbin)%sputtering_tab(0)%tab1d = 0d0

                    if (allocated(T_grid)) deallocate(T_grid)
                    if (allocated(rate_grid)) deallocate(rate_grid)
                    allocate(T_grid(1:nT))
                    allocate(rate_grid(1:nT))

                    do i = 1, nT
                        read(10, *) T_grid(i)
                    end do
                    pahbins_props(ipahbin)%sputtering_tab(0)%tab1d(1:nT, 1) = T_grid(1:nT)

                    do i = 1, nT
                        read(10, *) rate_grid(i)
                    end do
                    pahbins_props(ipahbin)%sputtering_tab(0)%tab1d(1:nT, 2) = rate_grid(1:nT)

                    call pahbins_props(ipahbin)%sputtering_tab(0)%init()

                    close(10)
                    deallocate(T_grid, rate_grid)
                else
                    if (myid.eq.1) write(*, *) 'Warning opening file: ', trim(pah_filename)
                end if
            end if

            ! Ions: loop over tracked elements, using the same Zi mapping as dust tables.
            do iel = 1, n_elements
#ifdef RTZ
                if (elements(iel)%atomic_number <= 0) cycle
                Zi = elements(iel)%atomic_number
#else
                Zi = iel
#endif
                write(Z_str, '(I0)') Zi
                write(pah_filename, '(A,A,A,A,A,A,A)') trim(dust_tables_dir), &
                    'sputtering_PAHBin_', trim(adjustl(ipah_str)), '_Z_', trim(adjustl(Z_str))

                inquire(file=trim(pah_filename), exist=file_exists)
                if (.not. file_exists) cycle

                open(10, file=trim(pah_filename), status='old', action='read', iostat=istat)
                if (istat /= 0) then
                    if (myid.eq.1) write(*, *) 'Warning opening file: ', trim(pah_filename)
                    cycle
                end if

                do n = 1, 6
                    read(10, *)
                end do
                ! Read the number of temperature points
                read(10, *) nT

                if (allocated(pahbins_props(ipahbin)%sputtering_tab(iel)%npts)) then
                    deallocate(pahbins_props(ipahbin)%sputtering_tab(iel)%npts)
                end if
                allocate(pahbins_props(ipahbin)%sputtering_tab(iel)%npts(1:1))
                pahbins_props(ipahbin)%sputtering_tab(iel)%ndim = 1
                pahbins_props(ipahbin)%sputtering_tab(iel)%npts(1) = nT

                ! Allocate table axes and rates
                if (allocated(pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d)) then
                    deallocate(pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d)
                end if
                allocate(pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d(1:nT, 1:2))
                pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d = 0d0

                ! Allocate temporary arrays for temperature and rate grids
                if (allocated(T_grid)) deallocate(T_grid)
                if (allocated(rate_grid)) deallocate(rate_grid)
                allocate(T_grid(1:nT))
                allocate(rate_grid(1:nT))

                ! Read the temperature grid (line 1 to nT)
                do i = 1, nT
                    read(10, *) T_grid(i)
                end do
                pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d(1:nT, 1) = T_grid(1:nT)

                ! Read the sputtering rates (line nT+1 to 2*nT)
                do i = 1, nT
                    read(10, *) rate_grid(i)
                end do
                pahbins_props(ipahbin)%sputtering_tab(iel)%tab1d(1:nT, 2) = rate_grid(1:nT)

                call pahbins_props(ipahbin)%sputtering_tab(iel)%init()

                close(10)

                deallocate(T_grid, rate_grid)
                
            end do
        end do        
    end subroutine init_pah_sputtering_tables

    subroutine init_pah_dissociation_tables
        ! This subroutine reads at the initialisiation of dust parameters
        ! the pre-computed dissociation rates for PAHs due to UV photon
        ! absorption. The modelling is based on the hydrogenation distribution
        ! of Montillaud et al. (2013) and using the microcanonical description
        ! in Rodriguez Montero et al. (2024).
        ! These files need to be saved in the same directory as the oppacity
        ! tables. They can be easily computed using the export_heating function
        ! in PAH_photoelectric_heating.py script in DustRAMSES.
        ! NOTE: Keep in mind that this values are already in log10, such that
        ! easy linear interpolation in log-log space can be computed 
        ! on the fly!!
        ! NOTE(2): Currently only support for the small PAHs (circumcoronene)
        ! TODO: Add support for all PAH sizes
        use amr_commons,only:myid
        implicit none

        logical :: ok_pah
        integer :: n_G0,n_nH,istat,n
        integer :: i,j,ipahbin,nmax
        character(len=7) :: i_str
        character(len=128) :: f_diss_filename

        do ipahbin = 1, npah
            ! Check first that the file is there
            write(i_str, '(I2.2)') ipahbin
            write(f_diss_filename, '(a,a,a,a)') trim(dust_tables_dir), 'dissociation_PAHBin_', trim(i_str), '.dat'
            inquire(file=f_diss_filename,exist=ok_pah)

            if (.not. ok_pah) then
                if(myid.eq.1) then
                    write(*,*)'ERROR IN PAH DISSOCIATION TABLES'
                    write(*,*)'Cannot access file ', TRIM(f_diss_filename)
                    write(*,*)'You need to set this correctly for' // &
                             ' dust_tables_dir in the namelist.'
                endif
                call clean_stop
            end if

            ! Read the data into per-PAH-bin DustTable
            open(111,file=trim(f_diss_filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(f_diss_filename)
                call clean_stop
            end if

            ! Ignore the first 6 lines (comment)
            do n = 1, 6
                read(111,*)
            end do

            ! Read the matrix structure
            read(111,*) n_G0, n_nH
            nmax = max(n_G0,n_nH)

            if (allocated(pahbins_props(ipahbin)%dissociation_tab%npts)) deallocate(pahbins_props(ipahbin)%dissociation_tab%npts)
            allocate(pahbins_props(ipahbin)%dissociation_tab%npts(1:2))
            pahbins_props(ipahbin)%dissociation_tab%name = 'pah_dissociation_pahbin_'//trim(adjustl(i_str))
            pahbins_props(ipahbin)%dissociation_tab%ndim = 2
            pahbins_props(ipahbin)%dissociation_tab%npts(1) = n_G0
            pahbins_props(ipahbin)%dissociation_tab%npts(2) = n_nH
            if (allocated(pahbins_props(ipahbin)%dissociation_tab%tab1d)) deallocate(pahbins_props(ipahbin)%dissociation_tab%tab1d)
            allocate(pahbins_props(ipahbin)%dissociation_tab%tab1d(1:nmax,1:2))
            pahbins_props(ipahbin)%dissociation_tab%tab1d = 0d0
            if (allocated(pahbins_props(ipahbin)%dissociation_tab%tab2d)) deallocate(pahbins_props(ipahbin)%dissociation_tab%tab2d)
            allocate(pahbins_props(ipahbin)%dissociation_tab%tab2d(1:n_G0,1:n_nH,1:1))

            ! Read the G0 array
            do i = 1, n_G0
                read(111,*) pahbins_props(ipahbin)%dissociation_tab%tab1d(i,1)
            end do

            ! Read the nH array
            do i = 1, n_nH
                read(111,*) pahbins_props(ipahbin)%dissociation_tab%tab1d(i,2)
            end do

            ! Read the table
            do i = 1, n_G0
                do j = 1, n_nH
                    read(111, *) pahbins_props(ipahbin)%dissociation_tab%tab2d(i, j, 1)
                end do
            end do

            call pahbins_props(ipahbin)%dissociation_tab%init()
            close(111)
        end do
    end subroutine init_pah_dissociation_tables

    subroutine init_pah_peh_tables
        ! This subroutine reads at the initialisation of dust parameters
        ! the pre-computed photoelectric heating rates for PAHs for the
        ! Draine (1978) ISRF with the extension for long wavelengths.
        use amr_commons,only:myid
        implicit none

        logical :: ok_pah
        integer :: n_gamma,istat,n
        integer :: i,j,istate
        real(dp) :: gamma_i, eff_i, pabs_i, f_anion_i, f_neutral_i, f_cation_i, f_dication_i
        character(len=7) :: i_str
        character(len=128) :: f_peh_filename

        ! Loop over the PAH sizes
        do i = 1, npah
            ! Check first that the file is there
            write(i_str, '(I2.2)') i  ! convert i to string without leading spaces
            write(f_peh_filename, '(a,a,a,a,a,a)')trim(dust_tables_dir),'peh_ISRF_Mathis_Draine_',trim(peh_attach_model),'_PAHBin_',trim(i_str),'.dat'
            inquire(file=f_peh_filename,exist=ok_pah)

            if (.not. ok_pah) then
                if(myid.eq.1) then 
                    write(*,*)'ERROR IN PAH PHOTOELECTRIC HEATING TABLES'
                    write(*,*)'Cannot access dust directory ',TRIM(dust_tables_dir)
                    write(*,*)'Directory '//TRIM(dust_tables_dir)//' not found'
                    write(*,*)'You need to set this correctly for' // &
                             ' dust_tables_dir in the namelist.'
                    write(*,*)'Missing file: ', trim(f_peh_filename)
                endif
                call clean_stop
            end if

            ! Read the data into arrays
            open(111,file=trim(f_peh_filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(f_peh_filename)
                call clean_stop
            end if

            ! Ignore the first 6 lines (comment)
            do n = 1, 6
                read(111,*)
            end do

            ! Read the number of gamma points
            read(111,*) n_gamma


            ! Fill PAH per-bin PEH tables used by interpolation routines
            if (allocated(pahbins_props(i)%peh_eff_tab%npts)) deallocate(pahbins_props(i)%peh_eff_tab%npts)
            allocate(pahbins_props(i)%peh_eff_tab%npts(1:1))
            pahbins_props(i)%peh_eff_tab%name = 'pah_photoelectric_heating_pahbin_'//trim(adjustl(i_str))
            pahbins_props(i)%peh_eff_tab%ndim = 1
            pahbins_props(i)%peh_eff_tab%npts(1) = n_gamma
            if (allocated(pahbins_props(i)%peh_eff_tab%tab1d)) deallocate(pahbins_props(i)%peh_eff_tab%tab1d)
            allocate(pahbins_props(i)%peh_eff_tab%tab1d(1:n_gamma,1:2))

            if (allocated(pahbins_props(i)%peh_pabs_tab%npts)) deallocate(pahbins_props(i)%peh_pabs_tab%npts)
            allocate(pahbins_props(i)%peh_pabs_tab%npts(1:1))
            pahbins_props(i)%peh_pabs_tab%name = 'pah_photoelectric_absorption_pahbin_'//trim(adjustl(i_str))
            pahbins_props(i)%peh_pabs_tab%ndim = 1
            pahbins_props(i)%peh_pabs_tab%npts(1) = n_gamma
            if (allocated(pahbins_props(i)%peh_pabs_tab%tab1d)) deallocate(pahbins_props(i)%peh_pabs_tab%tab1d)
            allocate(pahbins_props(i)%peh_pabs_tab%tab1d(1:n_gamma,1:2))

            if (allocated(pahbins_props(i)%fcharge_tab)) deallocate(pahbins_props(i)%fcharge_tab)
            allocate(pahbins_props(i)%fcharge_tab(1:pahbins_props(i)%ncharge_states))
            do istate = 1, pahbins_props(i)%ncharge_states
                if (allocated(pahbins_props(i)%fcharge_tab(istate)%npts)) deallocate(pahbins_props(i)%fcharge_tab(istate)%npts)
                allocate(pahbins_props(i)%fcharge_tab(istate)%npts(1:1))
                pahbins_props(i)%fcharge_tab(istate)%ndim = 1
                pahbins_props(i)%fcharge_tab(istate)%npts(1) = n_gamma
                if (allocated(pahbins_props(i)%fcharge_tab(istate)%tab1d)) deallocate(pahbins_props(i)%fcharge_tab(istate)%tab1d)
                allocate(pahbins_props(i)%fcharge_tab(istate)%tab1d(1:n_gamma,1:2))
            end do

            ! Read table data directly into per-bin tables (and keep legacy arrays synced)
            do j = 1, n_gamma
                read(111,*) gamma_i, eff_i, pabs_i, f_anion_i, f_neutral_i, f_cation_i, f_dication_i

                pahbins_props(i)%peh_eff_tab%tab1d(j,1) = gamma_i
                pahbins_props(i)%peh_eff_tab%tab1d(j,2) = eff_i
                pahbins_props(i)%peh_pabs_tab%tab1d(j,1) = gamma_i
                pahbins_props(i)%peh_pabs_tab%tab1d(j,2) = pabs_i

                do istate = 1, pahbins_props(i)%ncharge_states
                    pahbins_props(i)%fcharge_tab(istate)%tab1d(j,1) = gamma_i
                end do
                pahbins_props(i)%fcharge_tab(1)%tab1d(j,2) = f_anion_i
                pahbins_props(i)%fcharge_tab(2)%tab1d(j,2) = f_neutral_i
                pahbins_props(i)%fcharge_tab(3)%tab1d(j,2) = f_cation_i
                pahbins_props(i)%fcharge_tab(4)%tab1d(j,2) = f_dication_i
            end do

            call pahbins_props(i)%peh_eff_tab%init()
            call pahbins_props(i)%peh_pabs_tab%init()
            do istate = 1, pahbins_props(i)%ncharge_states
                call pahbins_props(i)%fcharge_tab(istate)%init()
            end do
            close(111)
        end do
    end subroutine init_pah_peh_tables

    subroutine init_dust_IR_emission_tables
        use amr_commons, only: myid
        implicit none
        logical :: ok, ok_all
        integer :: nT, istat, j, ii, ndata, nbands, iband
        character(len=20) :: dustlabel
        character(len=256) :: filename
        character(len=2048) :: line
        character(len=64), dimension(200) :: band_names
        real(dp) :: Td
        real(dp), dimension(200) :: band_luminosities

        ! 1. Check first that all files are in the expected place
        ok_all = .true.
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            write(filename, '(A,A,A,A)') trim(dust_tables_dir), &
                'band_luminosity_', trim(dustlabel), '.txt'
            inquire(file=trim(filename), exist=ok)
            ok_all = ok_all .and. ok
        end do

        if (.not. ok_all) then
            if (myid .eq. 1) then
                write(*, *) 'WARNING: Dust IR emission files not found in ', TRIM(dust_tables_dir)
                write(*, *) 'Skipping IR emission tables initialization.'
            end if
            return
        end if

        ! 2. Read the file for each dust bin
        do ii = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', ii
            write(filename, '(A,A,A,A)') trim(dust_tables_dir), &
                'band_luminosity_', trim(dustlabel), '.txt'

            ! 2.1 First pass: count the number of data rows (skip comments/blanks)
            open(25, file=trim(filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(filename)
                call clean_stop
            end if
            ndata = 0
            do
                read(25, '(A)', iostat=istat) line
                if (istat /= 0) exit
                line = adjustl(line)
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '#') cycle
                ndata = ndata + 1
            end do
            close(25)

            if (ndata < 2) then
                write(*, *) 'Error: IR emission table has too few rows: ', trim(filename)
                call clean_stop
            end if
            nT = ndata

            ! 2.2 Read column names from the header line containing "T_dust" or "log10(T_dust)"
            open(25, file=trim(filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(filename)
                call clean_stop
            end if
            nbands = 0
            do
                read(25, '(A)', iostat=istat) line
                if (istat /= 0) exit
                line = adjustl(line)
                if (index(line, 'T_dust') > 0) then
                    call parse_header_line(line, band_names, nbands)
                    exit
                end if
            end do
            close(25)

            if (nbands == 0) then
                write(*, *) 'Error parsing header line in: ', trim(filename)
                call clean_stop
            end if

            ! 2.3 Allocate the IRemission_tab structure
            if (allocated(dustbins_props(ii)%IRemission_tab)) then
                deallocate(dustbins_props(ii)%IRemission_tab)
            end if
            allocate(dustbins_props(ii)%IRemission_tab(1:nbands))

            do iband = 1, nbands
                allocate(dustbins_props(ii)%IRemission_tab(iband)%npts(1:1))
                dustbins_props(ii)%IRemission_tab(iband)%name = band_names(iband)
                dustbins_props(ii)%IRemission_tab(iband)%ndim = 1
                dustbins_props(ii)%IRemission_tab(iband)%npts(1) = nT
                allocate(dustbins_props(ii)%IRemission_tab(iband)%tab1d(1:nT, 1:2))
                dustbins_props(ii)%IRemission_tab(iband)%tab1d = 0d0
            end do

            ! 2.4 Second pass: read the temperature and luminosity values.
            open(25, file=trim(filename), status='old', action='read', iostat=istat)
            if (istat /= 0) then
                write(*, *) 'Error opening file: ', trim(filename)
                call clean_stop
            end if
            j = 0
            do
                read(25, '(A)', iostat=istat) line
                if (istat /= 0) exit
                line = adjustl(line)
                if (len_trim(line) == 0) cycle
                if (line(1:1) == '#') cycle
                
                read(line, *, iostat=istat) Td, (band_luminosities(iband), iband=1, nbands)
                if (istat /= 0) then
                    write(*, *) 'Error parsing IR emission table row in ', trim(filename)
                    call clean_stop
                end if
                j = j + 1
                do iband = 1, nbands
                    dustbins_props(ii)%IRemission_tab(iband)%tab1d(j, 1) = Td
                    dustbins_props(ii)%IRemission_tab(iband)%tab1d(j, 2) = band_luminosities(iband)
                end do
            end do
            close(25)

            do iband = 1, nbands
                call dustbins_props(ii)%IRemission_tab(iband)%init()
            end do
        end do
    end subroutine init_dust_IR_emission_tables

    subroutine parse_header_line(line, names, n)
        implicit none
        character(len=*), intent(in) :: line
        character(len=64), dimension(:), intent(out) :: names
        integer, intent(out) :: n
        integer :: next_pos
        character(len=2048) :: temp_line
        character(len=64) :: word

        n = 0
        temp_line = adjustl(line)
        
        ! Skip '#'
        if (temp_line(1:1) == '#') then
            temp_line = adjustl(temp_line(2:))
        end if
        
        ! Skip 'log10(T_dust)' or 'T_dust'
        if (temp_line(1:13) == 'log10(T_dust)') then
            temp_line = adjustl(temp_line(14:))
        else if (temp_line(1:6) == 'T_dust') then
            temp_line = adjustl(temp_line(7:))
        end if
        
        do
            temp_line = adjustl(temp_line)
            if (len_trim(temp_line) == 0) exit
            
            next_pos = index(temp_line, ' ')
            if (next_pos == 0) then
                word = temp_line
                temp_line = ''
            else
                word = temp_line(1:next_pos-1)
                temp_line = temp_line(next_pos:)
            end if
            
            n = n + 1
            names(n) = trim(word)
        end do
    end subroutine parse_header_line
end module dust_init