module dust_surface_chemistry
    use amr_parameters, only:dp
    use constants, only:pi,kB,mH,amu2g,mC_amu
    use dust_commons

    implicit none

    private

    ! Parameters from Cazaux & Spaans (2004)
    ! (https://iopscience.iop.org/article/10.1086/422087/pdf)
    ! TODO: Code for all dust compositions
    real(dp),dimension(2),parameter :: EH2 = (/540d0,340d0/) ! in K
    real(dp),dimension(2),parameter :: mu = (/4d-1,3d-1/)
    real(dp),dimension(2),parameter :: Es = (/250d0,200d0/) ! in K
    real(dp),dimension(2),parameter :: EHp = (/800d0,650d0/) ! in K
    real(dp),dimension(2),parameter :: EHc = (/3d4,3d4/) ! in K
    real(dp),dimension(2),parameter :: nuH2 = (/3d12,2d12/) ! in 1/s
    real(dp),dimension(2),parameter :: nuHc = (/2d13,1d13/) ! in 1/s

    real(dp),parameter :: Ns = 2D15 ! Fixed number of sites per cm2 on the surface of the grain

    public :: grain_h2_formation_rate,ice_sticking_coefficient
    contains

    function h2_sticking_coef(Tgas,Td)
        ! Sticking coefficient from Hollenbach & McKee (1979) - Eq 3.7
        ! (https://articles.adsabs.harvard.edu/pdf/1979ApJS...41..555H)
        implicit none
        real(dp) :: Tgas,Td
        real(dp) :: h2_sticking_coef

        h2_sticking_coef = 1D0 / (1D0 + 4D-1 * sqrt((Tgas+Td)/1D2) + 2D-1 * (Tgas/1D2) + 8D-2 * (Tgas/1D2)**2D0)
    end function h2_sticking_coef

    function beta_h2(Tgas,Td,dust_index)
        ! Desorption rate of H2 from Cazaux & Tielens (2002) - Page 2, beginning of leftmost last paragraph
        ! https://iopscience.iop.org/article/10.1086/342607/pdf
        implicit none
        real(dp) :: Tgas,Td
        real(dp) :: beta_h2
        integer   :: dust_index
        ! NOTE: EH2 is in K, not erg, so the kB factor is not needed
        beta_h2 = nuH2(dust_index) * exp(-EH2(dust_index) / (Td))
    end function beta_h2

    function high_temp_correction(Tgas,Td,F,dust_index)
        ! High temperature correction from Cazaux & Tielens (2002) - Eq 16 in the Erratum
        ! (https://iopscience.iop.org/article/10.1086/342607/pdf)
        implicit none
        real(dp) :: Tgas,Td,F
        real(dp) :: high_temp_correction
        integer   :: dust_index

        real(dp) :: a1,a2,a3

        a1 = nuHc(dust_index) / (2D0 * F)
        a2 = exp(-1.5D0 * EHc(dust_index) / (Td))
        a3 = (1D0 + sqrt( (EHc(dust_index) - Es(dust_index)) / (EHp(dust_index) - Es(dust_index)) ))**2D0

        high_temp_correction = 1D0 / (1D0 + (a1 * a2 * a3))
    end function high_temp_correction

    function beta_hp_over_alphapc(Tgas,Td,dust_index)
        ! Eq 17 from Cazaux & Tielens (2002)
        implicit none
        real(dp) :: Tgas,Td
        real(dp) :: beta_hp_over_alphapc
        integer   :: dust_index

        real(dp) :: a1,a2

        a1 = (1D0 + sqrt( (EHc(dust_index) - Es(dust_index)) / (EHp(dust_index) - Es(dust_index)) ))**2D0
        a2 = exp(-1D0 * Es(dust_index) / (Td))
        beta_hp_over_alphapc =  2.5D-1 * a1 * a2
    end function beta_hp_over_alphapc

    function h_flux(nH,vH)
        ! Eqn 1 of Cazaux & Spaans (2004) 
        ! (https://iopscience.iop.org/article/10.1086/422087/pdf)

        ! One can see from Figure 1 of https://iopscience.iop.org/article/10.1086/342607/pdf 
        ! that the recombination efficiency is essentially independent of flux over 10 orders 
        ! of magnitude. Taking a constant Ns is probably ok
        implicit none
        real(dp) :: nH,vH
        real(dp) :: h_flux

        h_flux = nH * vH / Ns
    end function h_flux

    function recombination_efficiency(Tgas,Td,vH,F,dust_index)
        ! Recombination effiency from Cazaux & Tielens (2002) -- Eq 15
        ! (https://iopscience.iop.org/article/10.1086/342607/pdf)
        implicit none
        real(dp) :: Tgas,Td,vH,F
        real(dp) :: recombination_efficiency
        integer   :: dust_index

        real(dp) :: a1,a2

        a1 = 1D0 / (1D0 + (mu(dust_index)*F) / (2D0*beta_h2(Tgas,Td,dust_index)) + beta_hp_over_alphapc(Tgas,Td,dust_index))
        a2 = high_temp_correction(Tgas,Td,F,dust_index)
        recombination_efficiency = a1 * a2
    end function recombination_efficiency

    function grain_h2_formation_rate(nHI,nH,Tgas,rho_dust,T_dust)
        ! H2 formation rate on dust grains following the formalism of
        ! Cazaux & Spaans (2004) and Cazaux & Tielens (2002).
        ! nHI --> neutral hydrogen density [cm-3]
        ! nH --> total hydrogen density [cm-3]
        ! Tgas --> gas temperature [K]
        ! rho_dust --> dust mass density for each dust bin [g/cm3]
        ! T_dust --> dust temperature for each dust bin [K]
        ! h2_formation_rate <--- H2 formation rate in cm3/s
        implicit none

        ! ---- Input parameters ----
        real(dp) :: nHI,nH,Tgas
        real(dp),dimension(1:ndust) :: rho_dust,T_dust

        ! ---- Local variables ----
        integer :: j,ilow,ihigh
        real(dp) :: R_H2,vH,F,sdust

        ! ---- Output ----
        real(dp) :: grain_h2_formation_rate

        grain_h2_formation_rate = 0d0

        ! Formation rate of H2 onto dust grains from Cazaux & Spaans (2004)
        ! (https://iopscience.iop.org/article/10.1086/422087/pdf)
        ! Mean thermal speed of H atoms, sqrt(8kT/(pi m_H)): the v_H of Cazaux & Spaans (2004),
        ! 1.45e5 (T/100)^0.5 cm/s
        vH = sqrt(8D0 * kB * Tgas / (pi * mH))
        F = h_flux(nHI,vH)
        ! Guard: no atomic H flux → no H2 formation on grains (also prevents NaN in high_temp_correction)
        if (F <= 0d0) return

        ! Add the contribution from each dust grain
        do j = 1, ndust
            ! n_d * sigma_d with the geometric cross section pi a^2 (Cazaux & Spaans 2004, eqs. 2 and 4)
            sdust = (rho_dust(j)/dustbins_props(j)%mgrain) * pi * (dustbins_props(j)%asize_cm)**2D0 ! in cm-1
            R_H2 = sdust * recombination_efficiency(Tgas,T_dust(j),vH,F,dustbins_props(j)%interact_group) ! in cm-1
            grain_h2_formation_rate = grain_h2_formation_rate + R_H2 * h2_sticking_coef(Tgas,T_dust(j)) ! in cm-1
        end do
        grain_h2_formation_rate = 5D-1 * vH / nH * grain_h2_formation_rate ! in cm3*s-1
    end function grain_h2_formation_rate

    subroutine compute_dehydrogenated_fraction(G0,nH,f_dh)
        implicit none
        real(dp), intent(in) :: G0,nH
        real(dp), intent(inout) :: f_dh
        real(dp) :: lognH,logG0,distance,y_intercept
        lognH = log10(nH)
        logG0 = log10(G0)
        ! Compute the G0 and nH relation based on the fitting
        ! to the results of Montillaud et al. (2013) for circumcoronene
        y_intercept = 1.542d0 * logG0 + 0.646d0

        ! Distance from point to the line
        distance = abs(y_intercept - lognH) / 1.838d0 ! denominator is sqrt(1+1.54184841^2)

        if (lognH .ge. y_intercept) then
            f_dh = 0.5d0 - 0.5d0/(1d0+1d-1*distance**(-2d0))
        else
            f_dh = 0.5d0 + 0.5d0/(1d0+1d-1*distance**(-2d0))
        end if
        f_dh = max(f_dh,0d0)
    end subroutine compute_dehydrogenated_fraction

    subroutine compute_superhydrogenated_fraction(G0,nH,f_sh)
        implicit none
        real(dp), intent(in) :: G0,nH
        real(dp), intent(inout) :: f_sh
        real(dp) :: lognH,logG0,distance,y_intercept
        lognH = log10(nH)
        logG0 = log10(G0)
        ! Compute the G0 and nH relation based on the fitting
        ! to the results of Andrews et al. (2016) for
        ! circumcircumcoronene (Nc=96)
        y_intercept = 0.995d0 * lognH - 1.945d0

        ! Distance from point to the line
        distance = abs(y_intercept - logG0) / 1.410d0 ! denominator is sqrt(1+0.99466057^2)

        if (logG0 .ge. y_intercept) then
            f_sh = 0.5d0 - 0.5d0/(1d0+5d-2*distance**(-2d0))
        else
            f_sh = 0.5d0 + 0.5d0/(1d0+5d-2*distance**(-2d0))
        end if
        f_sh = max(f_sh,0d0)
    end subroutine compute_superhydrogenated_fraction

    subroutine Hollenbach2009_icemodel(G0,nO,Tgas,S)
        implicit none

        real(dp),intent(in) :: G0,nO,Tgas
        real(dp),intent(inout) :: S

        real(dp) :: vth, N_mono

        ! 1. Compute the number of monomers based on Eq. 22 in
        ! Hollenbach et al. (2019) assuming that 50% of the mantle
        ! is H20 ice and a desorption yield of 3d-3.
        ! F0 = 1e8 photons/cm2/s
        vth = 3624.65d0 * sqrt(Tgas)
        N_mono = 2d0 * nO * vth / (max(G0,1d-5) * 3d5)

        ! 2. Compute the effective sticking coefficient
        ! assumming that the bare grain coefficient
        ! is already coming from the original model
        ! and the ice coefficient is set to S=0.1
        S = 0.1d0 + (1.d0 - 0.1d0) * exp(-N_mono)
    end subroutine Hollenbach2009_icemodel

    function ice_sticking_coefficient(G0,nO,Tgas,Td)
        implicit none
        real(dp),intent(in) :: G0,nO,Tgas,Td
        real(dp) :: ice_sticking_coefficient

        if (Td >= 100d0) then
            ! If grain is hotter than the sublimation of H2O ice
            ! we assume that there is no mantle formation
            ice_sticking_coefficient = 1d0
            return
        end if

        ! 1. Select the ice model
        select case (ice_model)
        
        case ('Hollenbach2009')
            call Hollenbach2009_icemodel(G0,nO,Tgas,ice_sticking_coefficient)
        case default
            call Hollenbach2009_icemodel(G0,nO,Tgas,ice_sticking_coefficient)
        end select

        ! 2. Cap the sticking coefficient to a maximum of 1 and minimum of 0.1
        ice_sticking_coefficient = max(0.1d0, min(ice_sticking_coefficient, 1.0d0))
    end function ice_sticking_coefficient
end module dust_surface_chemistry