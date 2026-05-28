module dust_dynamics
    use amr_parameters
    use constants, only: twopi, pi, e2instatC, kB, mH
    use hydro_parameters, only: n_elements,nvar
    use dust_commons

    implicit none


    contains

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

        if (trim(model).eq.'Ormel2007') then
            ! This is based on the formulation presented in Kawasaki & Machida (2023)
            ! which is basically the analytical model of Ormel & Cuzzi (2007)

            ! 1. Contribution to relative velocity from thermal (Brownian) motion
            dV_thermal = sqrt(8.d0 * kB * T * (target_m + projectile_m)/(target_m * projectile_m))

            ! Gas sound speed (assumed gas with adiabatic constant of 5d0/3d0)
            cs_gas = sqrt(5d0/3d0 * kB * T / (mH * local_mu))

            ! Thermal velocity (Maxwelian distribution)
            v_th = sqrt(8d0/pi) * cs_gas

            ! 2. Assume that the injection scale of turbulence is a cell size of inject_L length and
            ! the velocity is given by the largest size eddie velocity

            ! Assume the closure equations by Braginskii (1965), based on the Chapman-Enskog scheme
            ! which is based in the assumption that the macroscopic scale of the plasma is large
            ! compared to the mean free path or the gyro-radii of the electrons and the ions. In this
            ! case, the viscosity is dominated by the hydrogen viscosity parallel to the magnetic field
            ! (Braginskii 1965). This is because the ions carry the majority of the momemtum

            ! Distance of closest particle approach (ionised)
            rc = e2instatC / (kB * T)
            ! Particle mean free path
            mfp = 1d0 / (nH * rc**2d0)

            ! Eddie injection timescale
            tau_L = inject_L / v_turb

            ! Reynolds number (ratio of inertial to viscous forces)
            Re = 3d0 * v_turb * inject_L / (cs_gas * mfp)

            ! Disipation timescale
            tau_eta = tau_L / sqrt(Re)
            
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

            ! Gas sound speed (assumed gas with adiabatic constant of 5d0/3d0)
            cs_gas = sqrt(5d0/3d0 * kB * T / (mH * local_mu))

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

end module dust_dynamics