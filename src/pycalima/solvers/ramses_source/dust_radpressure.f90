!=======================================================================
! CALIMA Dust and Gas Radiation Pressure Force Module
! This module computes the radiation pressure force vectors for the gas
! and individual dust/PAH bins directly in code units for the RTZ network.
!=======================================================================
module dust_radpressure_module
    use amr_parameters, only: dp, ndim
    use dust_commons, only: tva_test_mode, TVA_TEST_SPRESS
    use amr_commons, only: cosmo, aexp
    use constants, only: c_cgs, eV2erg, mCO, e2instatC, kB, pi
    use dust_charging_rtgroups, only: dust_charge_moments
    ! rt_pressBoost / rt_isoPress / rt_isIR / iIR / rt_isIRtrap / iIRtrapVar are
    ! declared unconditionally in rt_parameters (see the note at their declaration),
    ! so they need no RTZ fork. isH2_rtz, rtz_UV_background_G0 and the rank of
    ! group_csn do: group_csn is (nGroups,1:27,1:27) under RTZ and (nGroups,nIons)
    ! without it.
    use rt_parameters, only: nGroups, iGroups, group_egy, rt_pressBoost, &
                            rt_isoPress, rt_isIR, iIR, rt_c, group_csn, &
                            iIons, isLW, rt_advect, rt_c_cgs, &
                            rt_n_source, rt_isIRtrap, iIRtrapVar, rt_c_fraction, smallNp
    ! n_elements is the same entity either way: rtz_module re-exports
    ! hydro_parameters' parameter (rtz_module.f90:3).
    use hydro_parameters, only: ndust, npah, idust, ipah, imetal, smallr, smallc, gamma, &
                                neul, nener, gamma_rad, nhydro, n_elements
#ifdef RTZ
    use rt_parameters, only: isH2_rtz, rtz_UV_background_G0
    use rtz_module, only: elements, getNe, getMu_RTZ
    use molecules_module, only: comp_Sd, comp_SH2
#else
    ! Plain RT: no element network, no RTZ H2/CO chemistry, no packed metal block.
    ! mu and ne are formed inline below from the H/He ionisation fractions rather
    ! than via rt_cooling_module's getMu, because rt_cooling_module.o is linked
    ! after this file (see MODOBJ) so its .mod is not available here.
    use rt_parameters, only: isH2, isHe, nIons, ixHI, ixHII, ixHeII, ixHeIII
    use cooling_module, only: X, Y
#endif
#ifdef CO
    use hydro_parameters, only: iCO
#endif
    use dust_commons, only: drag_model, draine_drag, dustbins_props, pahbins_props, group_csr_dust, group_csr_pah, ncharge_pah_max, GD_solar
    use pah_photoelectric_heating, only: interpolate_pah_charge_equilibrium
    use dust_optics, only: get_IR_mean_cross_sections

    implicit none

    public :: compute_gas_dust_radpressure_force, &
              compute_gas_dust_radpressure_acc

contains

    subroutine compute_gas_dust_radpressure_acc(cell_state, cell_rt_state, ilevel, dx, gas_acc, dust_acc, pah_acc, irtrap_share, &
                                                drag_state)
        implicit none
        real(dp), dimension(:), intent(in) :: cell_state
        real(dp), dimension(:), intent(in) :: cell_rt_state
        integer, intent(in) :: ilevel
        real(dp), intent(in) :: dx
        real(dp), dimension(1:ndim), intent(out) :: gas_acc
        ! Shapes match the caller's a_rad_d/a_rad_pah sections, which are
        ! declared (...,1:ndust,1:ndim) / (...,1:npah,1:ndim) in dust_dynamics
        ! and read back as (jbin,idim). Declaring these (ndim,nbin) transposed
        ! them for any run with ndim>1 and ndust>1.
        real(dp), dimension(max(1, ndust), 1:ndim), intent(out) :: dust_acc
        real(dp), dimension(max(1, npah), 1:ndim), intent(out) :: pah_acc
        ! Fraction of the trapped-IR Rosseland extinction carried by each dust
        ! bin, chi_R,k / chi_R,tot. Zero unless rt_isIR (see Part D of the
        ! trapped-IR scheme). Optional so non-TVA callers need not supply it.
        real(dp), dimension(max(1, ndust)), intent(out), optional :: irtrap_share
        ! Local state for the Draine (2011) drag (drag_model='draine2011'):
        ! (1) n_H [cm^-3], (2) T [K], (2+k) Coulomb coefficient of dust bin k,
        ! (n_H+/n_H) phi_k^2 ln(Lambda_k), with phi_k = Z_k e^2/(a_k kT) from
        ! charging_model and Lambda from Draine & Salpeter (1979); with drag_model='draine2011_pz',
        ! (n_H+/n_H) <phi^2 ln(Lambda)> over the charge distribution P(Z).
        real(dp), dimension(1:ndust+2), intent(out), optional :: drag_state

        ! Local variables for cell state extraction
        real(dp) :: scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2
        real(dp) :: scale_Np, scale_Fp
        real(dp) :: dtg_mw
        real(dp) :: Tk, ne, G0, f_shd
        real(dp), dimension(n_elements) :: nElement
        real(dp), dimension(n_elements, 27) :: xion
        real(dp), dimension(max(1, ndust)) :: rho_dust
        real(dp), dimension(max(1, npah)) :: rho_pah
        real(dp), dimension(1:ndim) :: gas_force
        real(dp), dimension(1:ndim, max(1, ndust)) :: dust_force
        real(dp), dimension(1:ndim, max(1, npah)) :: pah_force
        real(dp) :: rho_loc, P_gas, eps_tot, rho_gas, mu
        real(dp) :: E_IR_tot_cgs, T_rad_loc, chi_R_tot
        real(dp), dimension(max(1, ndust))  :: sig_R_dust, sig_P_dust, chi_R_dust
        real(dp), dimension(max(1, 2*npah)) :: sig_R_pah,  sig_P_pah
        integer :: id, iion, counter, e_counter, jbin
        real(dp) :: nH_drag, nHp_drag, Z_drag, phi_drag, Lambda_drag
        real(dp) :: nHep_drag, nHepp_drag, G0_bg_drag, Zsig_drag, phi1_drag, lnL1_drag, coul_drag
        real(dp), dimension(1:nGroups) :: Np_drag
#ifndef RTZ
        real(dp), dimension(max(1,nIons)) :: xion_rt
        real(dp) :: nH_loc, nHe_loc, xHI_loc, xH2_loc
        real(dp) :: xHeI_loc, xHeII_loc, xHeIII_loc
#endif
#ifdef CO
        real(dp) :: nCO
#endif

        ! Get scale factors
        call units(scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)
        call rt_units(scale_Np, scale_Fp)

        rho_loc = max(cell_state(1), smallr)

#ifdef RTZ
        ! 1. Extract elements and ionization fractions
        counter = 0
        e_counter = 0
        xion = 0d0
        do id=1,n_elements
           if (elements(id)%atomic_number.gt.0) then
              do iion=1,elements(id)%n_ions
                 xion(id,iion) = cell_state(iIons+counter) / rho_loc
                 counter = counter + 1
              end do
              elements(id)%scale_n = scale_d / elements(id)%atomic_mass_g
              nElement(id) = cell_state(imetal+e_counter) * elements(id)%scale_n
              e_counter = e_counter + 1
           else
              nElement(id) = 0d0
           end if
        end do
        if (elements(1)%atomic_number.gt.0 .and. isH2_rtz) then
           xion(1,3) = cell_state(iIons+counter) / rho_loc
           counter = counter + 1
        end if
#else
        ! 1. Plain RT: no element network. Read the H/He ionisation fractions
        !    here; the species densities the gas-opacity sum needs are packed
        !    below, once rho_gas is known.
        xion     = 0d0
        nElement = 0d0
        xion_rt(1:nIons) = cell_state(iIons:iIons+nIons-1) / rho_loc
#endif

        ! 2. Extract dust and PAH densities
        eps_tot = 0.0_dp
        if (ndust > 0) then
           do jbin = 1, ndust
              rho_dust(jbin) = cell_state(idust+jbin-1)
              eps_tot = eps_tot + (rho_dust(jbin) / rho_loc)
           end do
        end if
#if NPAH>0
        if (npah > 0) then
           do jbin = 1, npah
              rho_pah(jbin) = cell_state(ipah+jbin-1)
           end do
        end if
#endif
        eps_tot = min(max(eps_tot, 0.0_dp), 1.0_dp - smallr)
        rho_gas = max((1.0_dp - eps_tot) * rho_loc, smallr)
#ifndef RTZ
        ! Pre-ionized ("neutral") species densities in ion-slot order -- the same
        ! nN(:) vector rt_cooling_module builds for phAbs. group_csn is
        ! (nGroups,nIons) without RTZ, so the gas opacity in _force is simply
        ! SUM(nElement(1:nIons)*group_csn(igroup,1:nIons)). nElement keeps its
        ! RTZ shape so the _force interface is unchanged; only slots 1..nIons are
        ! meaningful and xion stays zero (unused without RTZ).
        nH_loc  = rho_gas * scale_nH
        nHe_loc = 0.25d0 * nH_loc * Y / X
        xHI_loc = max(1d0 - xion_rt(ixHII), 0d0)
        xH2_loc = 0d0
        if (isH2) then
           xHI_loc = max(xion_rt(ixHI), 0d0)
           xH2_loc = max(0.5d0*(1d0 - xion_rt(ixHI) - xion_rt(ixHII)), 0d0)
           nElement(ixHI) = nH_loc * xH2_loc                       !    nH2
        end if
        nElement(ixHII) = nH_loc * xHI_loc                         !    nHI
        ! ixHeII/ixHeIII are 0 when helium is not tracked (isHe=.false.), so the
        ! helium fractions must be read ONLY under the guard and cached here --
        ! indexing xion_rt(0) further down is an out-of-bounds access.
        xHeI_loc   = 0d0
        xHeII_loc  = 0d0
        xHeIII_loc = 0d0
        if (isHe) then
           xHeII_loc  = xion_rt(ixHeII)
           xHeIII_loc = xion_rt(ixHeIII)
           xHeI_loc   = max(1d0 - xHeII_loc - xHeIII_loc, 0d0)
           nElement(ixHeII)  = nHe_loc * xHeI_loc                  !   nHeI
           nElement(ixHeIII) = nHe_loc * xHeII_loc                 !  nHeII
        end if
#endif

        ! 3. H2 shielding
        f_shd = 1.d0
#ifdef RTZ
        if (isH2_rtz) then
           dtg_mw = sum(rho_dust(1:ndust)) / rho_loc * GD_solar
           f_shd = comp_SH2(0.5d0*nElement(1)*xion(1,3),dx) * &
                   comp_Sd(nElement(1)*xion(1,1),0.5d0*nElement(1)*xion(1,3),dx,dtg_mw)
        end if
#else
        ! No RTZ H2 network, so no self-shielding factor. f_shd only enters the
        ! LW-band gas opacity in _force, which is inert unless isH2 and some
        ! group is flagged LW.
#endif

        ! 4. Electron density
#ifdef RTZ
        ne = getNe(xion, nElement)
#else
        ne = nH_loc * xion_rt(ixHII)
        if (isHe) ne = ne + nHe_loc*(xHeII_loc + 2d0*xHeIII_loc)
#endif

        ! 5. Temperature
        P_gas = (gamma - 1) * (cell_state(neul) - 0.5_dp * sum(cell_state(2:ndim+1)**2) / cell_state(1))
#if NENER > 0
        do counter = 1, nener
           P_gas = P_gas - (gamma_rad(counter) - 1.0_dp) * cell_state(nhydro+counter)
        end do
#endif
        P_gas = max(P_gas, rho_gas * (smallc**2/gamma))

#ifdef RTZ
#ifdef CO
        nCO = cell_state(iCO) * scale_d / mCO
        mu = getMu_RTZ(ne, nElement, xion, isH2_rtz, nCO)
#else
        mu = getMu_RTZ(ne, nElement, xion, isH2_rtz)
#endif
#else
        ! Same H/He expression as rt_cooling_module's getMu, without its
        ! is_kIR_T/is_mu_H2 molecular correction (Tmu_dissoc is private there).
        ! mu feeds Tk, which is used only for the PAH charge equilibrium below.
        mu = 1d0/(X*(0.5d0+0.5d0*xHI_loc+1.5d0*xion_rt(ixHII)) &
                + 0.25d0*Y*(1d0+xHeII_loc+2d0*xHeIII_loc))
#endif
        Tk = P_gas / rho_loc * scale_T2 * mu

        ! 6. Habing band radiation field G0
#ifdef RTZ
        G0 = rtz_UV_background_G0
#else
        G0 = 0d0   ! no plain-RT equivalent of rtz_UV_background_G0
#endif
        if (rt_advect) then
           do id = 1, nGroups
              if (group_egy(id).gt.5.6d0 .and. group_egy(id).lt.13.6d0) then
                 G0 = G0 + scale_Np * cell_rt_state(iGroups(id)) * rt_c_cgs(ilevel) * group_egy(id) * eV2erg / 1.6d-3
              end if
           end do
        end if

        ! 6b. Total IR energy density [erg/cm^3], streaming + trapped, on the
        ! same footing as dNp in rtz_cooling_module (i.e. the code value, still
        ! inflated by 1/f_c). Outside cooling_fine the two components are stored
        ! separately: streaming in rtuold, trapped in the NENER slot. Np2Ep
        ! folded f_c and rt_pressBoost into the stored trapped energy, so divide
        ! them back out before adding.
        E_IR_tot_cgs = 0.0_dp
        if (rt_isIR) then
           E_IR_tot_cgs = cell_rt_state(iGroups(iIR)) * scale_Np &
                        * group_egy(iIR) * eV2erg
#if NENER>0
           if (rt_isIRtrap) then
              E_IR_tot_cgs = E_IR_tot_cgs + cell_state(iIRtrapVar) &
                           * scale_d * scale_v**2                 &
                           / max(rt_c_fraction(ilevel) * rt_pressBoost, 1d-40)
           end if
#endif
        end if

        ! 6c. IR-group mean cross sections at the local radiation temperature.
        ! Computed once here and handed to both the streaming force below and
        ! the trapped-IR share in step 9, so the two channels cannot disagree
        ! at the tau_c ~ 1 crossover.
        sig_R_dust = 0d0; sig_P_dust = 0d0
        sig_R_pah  = 0d0; sig_P_pah  = 0d0
        T_rad_loc  = 0d0
        if (rt_isIR) then
           call get_IR_mean_cross_sections(E_IR_tot_cgs, rt_c_fraction(ilevel), &
                                           sig_R_dust, sig_P_dust,              &
                                           sig_R_pah,  sig_P_pah, T_rad_loc)
        end if

        ! 7. Call compute_gas_dust_radpressure_force
        call compute_gas_dust_radpressure_force(cell_rt_state, ilevel, sig_R_dust, sig_R_pah, &
            gas_force, dust_force, pah_force, &
            nElement, xion, rho_dust, rho_pah, &
            Tk, ne, G0, f_shd)

        ! 8. Calculate accelerations
        gas_acc = gas_force / rho_gas

        if (ndust > 0) then
           do jbin = 1, ndust
              if (rho_dust(jbin) > 0d0) then
                 dust_acc(jbin, :) = dust_force(:, jbin) / rho_dust(jbin)
              else
                 dust_acc(jbin, :) = 0d0
              end if
           end do
        end if

#if NPAH>0
        if (npah > 0) then
           do jbin = 1, npah
              if (rho_pah(jbin) > 0d0) then
                 pah_acc(jbin, :) = pah_force(:, jbin) / rho_pah(jbin)
              else
                 pah_acc(jbin, :) = 0d0
              end if
           end do
        end if
#endif

        ! 9. Trapped-IR opacity share per dust bin, s_k = chi_R,k / chi_R,tot.
        ! In the diffusion limit the force density on species k is
        ! -(chi_k/chi_tot) grad(E_t)/3 (RT15 eq. 48 summed over species gives
        ! -grad(P_trap)), so s_k is the fraction of the trapped-IR pressure
        ! gradient that bin k feels. PAHs contribute to chi_tot but do not
        ! drift, so their share simply stays with the barycenter; the TVA
        ! closure conserves barycentric momentum regardless.
        if (present(irtrap_share)) then
           irtrap_share = 0d0
           if (rt_isIR .and. ndust > 0) then
              chi_R_tot = 0d0
              do jbin = 1, ndust
                 chi_R_dust(jbin) = sig_R_dust(jbin) * rho_dust(jbin) * scale_d &
                                  / dustbins_props(jbin)%mgrain
                 chi_R_tot = chi_R_tot + chi_R_dust(jbin)
              end do
#if NPAH>0
              ! Charge-state mean, as in the trapping optical depth in
              ! cooling_fine; PAH IR opacity is a small correction.
              do jbin = 1, npah
                 chi_R_tot = chi_R_tot                                        &
                    + 0.5d0 * (sig_R_pah(2*(jbin-1)+1) + sig_R_pah(2*(jbin-1)+2)) &
                    * rho_pah(jbin) * scale_d / pahbins_props(jbin)%mpah
              end do
#endif
              if (chi_R_tot > 1d-40) then
                 do jbin = 1, ndust
                    irtrap_share(jbin) = chi_R_dust(jbin) / chi_R_tot
                 end do
              end if
           end if
        end if

        ! 10. Drag state for the Draine (2011) drag law
        if (present(drag_state)) drag_state = 0d0
        if (present(drag_state) .and. draine_drag()) then
#ifdef RTZ
           nH_drag  = nElement(1)
           nHp_drag = nElement(1) * xion(1,2)
           nHep_drag  = nElement(2) * xion(2,2)
           nHepp_drag = nElement(2) * xion(2,3)
           G0_bg_drag = rtz_UV_background_G0
#else
           nH_drag  = nH_loc
           nHp_drag = nH_loc * xion_rt(ixHII)
           nHep_drag  = nHe_loc * xHeII_loc
           nHepp_drag = nHe_loc * xHeIII_loc
           G0_bg_drag = 0d0
#endif
           ! Local photon densities [cm^-3], as the RTZ cooling step passes them to the charging
           Np_drag = smallNp
           if (rt_advect) then
              do id = 1, nGroups
                 Np_drag(id) = max(scale_Np * cell_rt_state(iGroups(id)), smallNp)
              end do
           end if
           drag_state(1) = nH_drag
           drag_state(2) = Tk
           do jbin = 1, ndust
              if (trim(drag_model) == 'draine2011_pz') then
                 ! The charge fluctuates much faster than the grain is stopped, so the drift
                 ! feels the Coulomb force averaged over P(Z): with phi = Z phi_1 and
                 ! Lambda = Lambda_1/|Z|, phi^2 ln(Lambda) -> phi_1^2 <Z^2 ln+(Lambda_1/|Z|)>
                 if (ne > 0d0 .and. nH_drag > 0d0) then
                    phi1_drag = e2instatC / (dustbins_props(jbin)%asize_cm * kB * Tk)
                    lnL1_drag = log(3d0 / (2d0 * dustbins_props(jbin)%asize_cm * sqrt(e2instatC) * phi1_drag) &
                                    * sqrt(kB * Tk / (pi * ne)))
                    call dust_charge_moments(jbin, Np_drag, group_egy, rt_c_cgs(ilevel), G0_bg_drag, G0, Tk, ne, &
                                             nHp_drag, nHep_drag, nHepp_drag, Z_drag, Zsig_drag, lnL1_drag, coul_drag)
                    drag_state(2+jbin) = nHp_drag / nH_drag * phi1_drag**2 * coul_drag
                 end if
                 cycle
              end if
              call dust_charge_moments(jbin, Np_drag, group_egy, rt_c_cgs(ilevel), G0_bg_drag, G0, Tk, ne, &
                                       nHp_drag, nHep_drag, nHepp_drag, Z_drag, Zsig_drag)
              phi_drag = abs(Z_drag) * e2instatC / (dustbins_props(jbin)%asize_cm * kB * Tk)
              if (phi_drag > 0d0 .and. ne > 0d0 .and. nH_drag > 0d0) then
                 Lambda_drag = 3d0 / (2d0 * dustbins_props(jbin)%asize_cm * sqrt(e2instatC) * phi_drag) &
                             * sqrt(kB * Tk / (pi * ne))
                 drag_state(2+jbin) = nHp_drag / nH_drag * phi_drag**2 * log(max(Lambda_drag, 1d0))
              end if
           end do
        end if

    end subroutine compute_gas_dust_radpressure_acc

    subroutine compute_gas_dust_radpressure_force(cell_rt_state, ilevel, sig_R_dust, sig_R_pah, &
        gas_force_code, dust_force_code, pah_force_code, &
        nElement, dXion, rho_dust, rho_pah, &
        Tk, ne, G0, f_shd)

        ! Input/Output declarations
        real(dp), dimension(:), intent(in) :: cell_rt_state
        integer, intent(in) :: ilevel
        ! IR-group Rosseland mean cross sections [cm^2] at the local radiation
        ! temperature, supplied by the caller so the streaming and trapped
        ! channels are guaranteed to use the same opacity.
        real(dp), dimension(:), intent(in) :: sig_R_dust  ! (1:ndust)
        real(dp), dimension(:), intent(in) :: sig_R_pah   ! (1:2*npah), interlaced
        
        real(dp), dimension(1:ndim), intent(out) :: gas_force_code
        real(dp), dimension(:,:), intent(out) :: dust_force_code
        real(dp), dimension(:,:), intent(out) :: pah_force_code
        
        real(dp), dimension(n_elements), intent(in) :: nElement
        real(dp), dimension(n_elements, 27), intent(in) :: dXion
        real(dp), dimension(:), intent(in) :: rho_dust
        real(dp), dimension(:), intent(in) :: rho_pah
        real(dp), intent(in) :: Tk, ne, G0, f_shd

        ! Local variables
        real(dp) :: scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2
        real(dp) :: c_code
        real(dp) :: scale_Np, scale_Fp
        real(dp) :: scale_E, rt_c_code
        integer :: igroup, idim, iNp, ii, jj
        real(dp) :: fluxMag_code, Np_code
        real(dp), dimension(1:ndim) :: Fp_code
        real(dp), dimension(nGroups) :: opacity_gas_cgs
        real(dp), dimension(nGroups) :: opacity_gas_code
        real(dp), dimension(1:ndust, nGroups) :: opacity_dust_code
        real(dp), dimension(1:npah, nGroups) :: opacity_pah_code
        real(dp), dimension(nGroups) :: group_egy_erg
        real(dp), dimension(nGroups) :: group_egy_code
        real(dp) :: mom_fact_gas, mom_fact_dust, mom_fact_pah
        real(dp) :: pah_ion_fraction
        real(dp), dimension(max(1, ncharge_pah_max), max(1, npah)) :: fcharge_pah_local
        real(dp) :: cs_loc, cs_loc_n, cs_loc_i
        integer  :: idx_n, idx_i

        ! 1. Initialize forces to 0
        gas_force_code = 0d0
        if (ndust > 0) dust_force_code = 0d0
        if (npah > 0) pah_force_code = 0d0

        ! 2. Get scaling factors
        call units(scale_l, scale_t, scale_d, scale_v, scale_nH, scale_T2)
        call rt_units(scale_Np, scale_Fp)
        scale_E = scale_d * (scale_l**2) / (scale_t**2)

        rt_c_code = rt_c(ilevel)
        ! TRUE c in code units. RT15 eq. 27's momentum transfer is
        ! chi * E_gamma * Fp / c with the TRUE speed of light. The cooling path
        ! in rtz_cooling_module gets this right because its rate already carries
        ! a factor rt_c_cgs (sigcr_dust = group_csr_dust*rt_c_cgs) which the
        ! extra one_over_rt_c_cgs cancels. Here opacity_*_code is a RAW opacity
        ! with no rt_c_cgs in it, so dividing by rt_c_code made the force
        ! 1/rt_c_fraction too large -- 333x at rt_c_fraction = 3e-3.
        ! (Np_code below still uses rt_c_code: Np = F/c_red is correct there.)
        c_code = c_cgs / scale_v

        ! 3. Calculate gas opacities (CGS)
        opacity_gas_cgs = 0d0

        ! Calculate opacity_gas_cgs for gas directly using group_csn (in cm2)
#ifdef RTZ
        do ii=1,n_elements
            if (elements(ii)%atomic_number.gt.0) then
                do jj=1,elements(ii)%n_ions-1
                    do igroup=1,nGroups
                        opacity_gas_cgs(igroup) = opacity_gas_cgs(igroup) + nElement(ii) * dXion(ii, jj) * group_csn(igroup, ii, jj)
                    end do
                end do
            end if
        end do
#else
        ! Plain RT: group_csn is (nGroups,nIons) and nElement(1:nIons) holds the
        ! pre-ionized species densities packed by _acc. Same sum rt_cooling_module
        ! forms for phAbs. dXion is unused here.
        do ii=1,nIons
            do igroup=1,nGroups
                opacity_gas_cgs(igroup) = opacity_gas_cgs(igroup) &
                     + nElement(ii) * group_csn(igroup, ii)
            end do
        end do
#endif

#ifdef RTZ
        if (elements(1)%atomic_number.gt.0 .and. isH2_rtz) then
            do igroup=1,nGroups
                if (isLW(igroup).eq.1) then
                    opacity_gas_cgs(igroup) = opacity_gas_cgs(igroup) + 0.5d0 * nElement(1) * dXion(1, 3) * group_csn(igroup, 1, 3) * f_shd
                else
                    opacity_gas_cgs(igroup) = opacity_gas_cgs(igroup) + 0.5d0 * nElement(1) * dXion(1, 3) * group_csn(igroup, 1, 3)
                end if
            end do
        end if
#endif

        ! Convert gas opacity to code units
        do igroup=1,nGroups
            opacity_gas_code(igroup) = opacity_gas_cgs(igroup) * scale_l
        end do

        ! 4. Calculate dust/PAH opacities in code units
        opacity_dust_code = 0d0
        opacity_pah_code = 0d0

        ! For the IR group the flux-mean opacity is the Rosseland mean at the
        ! local radiation temperature, not the band-weighted group_csr value
        ! (RT15 eqs. 12, 21). It must match the cross section used for the
        ! barycentric force in rtz_cooling_module, or the a_rad_mix subtraction
        ! in the TVA driver would no longer cancel.
        if (ndust > 0) then
            do igroup = 1, nGroups
                do ii = 1, ndust
                    if (rt_isIR .and. igroup == iIR) then
                        cs_loc = sig_R_dust(ii)
                    else
                        cs_loc = group_csr_dust(ii, igroup)
                    end if
                    opacity_dust_code(ii, igroup) = cs_loc * rho_dust(ii) * &
                                                    scale_d * scale_l / dustbins_props(ii)%mgrain
                end do
            end do
        end if

#if NPAH>0
        if (npah > 0) then
            do ii = 1, npah
                ! Calculate PAH charge equilibrium fraction
                ! Use PAH photoelectric heating solver for charge state
                call interpolate_pah_charge_equilibrium(ii, G0, ne, Tk, fcharge_pah_local(:,ii))
                pah_ion_fraction = fcharge_pah_local(2, ii)
                ! group_cs*_pah is laid out interlaced, neutral then ion for
                ! each bin (see initialize_cross_sections_from_blackbody_dust_pah
                ! and rad_pah_rate). The previous (ii)/(npah+ii) block indexing
                ! only coincided with that for npah==1.
                idx_n = 2*(ii-1) + 1
                idx_i = 2*(ii-1) + 2
                do igroup = 1, nGroups
                    if (rt_isIR .and. igroup == iIR) then
                        cs_loc_n = sig_R_pah(idx_n)
                        cs_loc_i = sig_R_pah(idx_i)
                    else
                        cs_loc_n = group_csr_pah(idx_n, igroup)
                        cs_loc_i = group_csr_pah(idx_i, igroup)
                    end if
                    opacity_pah_code(ii, igroup) = (cs_loc_n * (1d0 - pah_ion_fraction) + &
                                                    cs_loc_i * pah_ion_fraction) * &
                                                   rho_pah(ii) * scale_d * scale_l / &
                                                   pahbins_props(ii)%mpah
                end do
            end do
        end if
#endif

        ! Calculate group energies in ergs and convert to code units
        group_egy_erg = group_egy * eV2erg
        do igroup = 1, nGroups
            group_egy_code(igroup) = group_egy_erg(igroup) / scale_E
        end do

        ! 5. Calculate radiation pressure forces directly in code units
        do igroup = 1, nGroups
            iNp = iGroups(igroup)
            if (tva_test_mode == TVA_TEST_SPRESS) then
                Fp_code = 0d0
                if (ndim >= 1) Fp_code(1) = rt_n_source(igroup) / scale_Fp
                Np_code = rt_n_source(igroup) / rt_c_code / scale_Np
                fluxMag_code = sqrt(sum(Fp_code**2))
            else
                Np_code = cell_rt_state(iNp)
                Fp_code = cell_rt_state(iNp+1 : iNp+ndim)
                fluxMag_code = sqrt(sum(Fp_code**2))
            end if

            ! --- Gas Force ---
            mom_fact_gas = opacity_gas_code(igroup) * group_egy_code(igroup)

            if (rt_isoPress .and. .not. (rt_isIR .and. igroup == iIR)) then
                if (fluxMag_code > 0d0) then
                    mom_fact_gas = mom_fact_gas * Np_code / fluxMag_code
                else
                    mom_fact_gas = 0d0
                end if
            else
                mom_fact_gas = mom_fact_gas / c_code
            end if

            do idim = 1, ndim
                gas_force_code(idim) = gas_force_code(idim) + Fp_code(idim) * mom_fact_gas * rt_pressBoost
            end do

            ! --- Dust Bin Forces ---
            if (ndust > 0) then
                do ii = 1, ndust
                    mom_fact_dust = opacity_dust_code(ii, igroup) * group_egy_code(igroup)

                    if (rt_isoPress .and. .not. (rt_isIR .and. igroup == iIR)) then
                        if (fluxMag_code > 0d0) then
                            mom_fact_dust = mom_fact_dust * Np_code / fluxMag_code
                        else
                            mom_fact_dust = 0d0
                        end if
                    else
                        mom_fact_dust = mom_fact_dust / c_code
                    end if

                    do idim = 1, ndim
                        dust_force_code(idim, ii) = dust_force_code(idim, ii) + Fp_code(idim) * mom_fact_dust * rt_pressBoost
                    end do
                end do
            end if

            ! --- PAH Bin Forces ---
#if NPAH>0
            if (npah > 0) then
                do ii = 1, npah
                    mom_fact_pah = opacity_pah_code(ii, igroup) * group_egy_code(igroup)

                    if (rt_isoPress .and. .not. (rt_isIR .and. igroup == iIR)) then
                        if (fluxMag_code > 0d0) then
                            mom_fact_pah = mom_fact_pah * Np_code / fluxMag_code
                        else
                            mom_fact_pah = 0d0
                        end if
                    else
                        mom_fact_pah = mom_fact_pah / c_code
                    end if

                    do idim = 1, ndim
                        pah_force_code(idim, ii) = pah_force_code(idim, ii) + Fp_code(idim) * mom_fact_pah * rt_pressBoost
                    end do
                end do
            end if
#endif

        end do

        if (tva_test_mode == TVA_TEST_SPRESS) then
            gas_force_code(:) = 0d0
        end if

    end subroutine compute_gas_dust_radpressure_force

end module dust_radpressure_module
