module dust_yields
    use hydro_parameters, only:ndust,ndchemtype,nvar,\
                                imetal,idust,ipah
    use hydro_commons, only:nmetals
    use constants, only:yr2sec,Myr2sec,amu2g,mC_amu,mO_amu,mH_amu
    use dust_commons

    contains

    function pah_from_winds(Mcarbon,Moxygen,dep_metals)
        implicit none
        real(dp),intent(inout) :: Mcarbon, Moxygen
        logical,intent(in) :: dep_metals
        real(dp) :: pah_from_winds
        real(dp) :: pah_eff
        integer :: i_pah

        ! This assumes that the C/O ratio is used
        ! to maximise the CO production

        pah_eff = 0d0
        do i_pah = 1, npah
            pah_eff = pah_eff + pahbins_props(i_pah)%AGB_cond_eff
        end do

        if (Mcarbon/(mC_amu*amu2g) >= Moxygen/(mO_amu*amu2g)) then
            pah_from_winds = pah_eff * (Mcarbon - (mC_amu/mO_amu)*Moxygen)
            if (dep_metals) Mcarbon = max(Mcarbon - pah_from_winds,0d0)
        else
            pah_from_winds = 0d0
        end if

    end function pah_from_winds

    function condensed_AGB_dustmass(dust_index,metal_masses,dep_metals)
        implicit none
        integer,intent(in) :: dust_index
        real(dp),dimension(1:nmetals) :: metal_masses
        logical,intent(in) :: dep_metals
        real(dp) :: condensed_AGB_dustmass
        real(dp),dimension(:),allocatable :: M_el

        integer :: i,ilim

        ! 1. Allocate the masses of the elements tracked in the dust bin
        allocate(M_el(1:dustbins_props(dust_index)%nelements))

        ! 2. Loop over the elements tracked in the dust bin
        do i = 1, dustbins_props(dust_index)%nelements
            M_el(i) = metal_masses(dustbins_props(dust_index)%el_index(i))
        end do

        ! 3. Figure out what is the limiting element
        call cmp_lim_elem(dust_index,dustbins_props(dust_index)%nelements,M_el,ilim)
        condensed_AGB_dustmass = dustbins_props(dust_index)%AGB_cond_eff * M_el(ilim) / &
            dustbins_props(dust_index)%el_mfractions(ilim)

        ! 4. Deplete the elements if requested
        if (dep_metals) then
            do i = 1, dustbins_props(dust_index)%nelements
                metal_masses(dustbins_props(dust_index)%el_index(i)) = max(metal_masses(dustbins_props(dust_index)%el_index(i)) - &
                    (condensed_AGB_dustmass * dustbins_props(dust_index)%el_mfractions(i)),0d0)
            end do
        end if

    end function condensed_AGB_dustmass

    function condensed_SNII_dustmass(dust_index,metal_masses,dep_metals)
        implicit none
        integer,intent(in) :: dust_index
        real(dp),dimension(1:nmetals) :: metal_masses
        logical,intent(in) :: dep_metals
        real(dp) :: condensed_SNII_dustmass
        real(dp),dimension(:),allocatable :: M_el

        integer :: i,ilim

        ! 1. Allocate the masses of the elements tracked in the dust bin
        allocate(M_el(1:dustbins_props(dust_index)%nelements))

        ! 2. Loop over the elements tracked in the dust bin
        do i = 1, dustbins_props(dust_index)%nelements
            M_el(i) = metal_masses(dustbins_props(dust_index)%el_index(i))
        end do

        ! 3. Figure out what is the limiting element
        call cmp_lim_elem(dust_index,dustbins_props(dust_index)%nelements,M_el,ilim)
        condensed_SNII_dustmass = dustbins_props(dust_index)%SNII_cond_eff * M_el(ilim) / &
            dustbins_props(dust_index)%el_mfractions(ilim)

        ! 4. Deplete the elements if requested
        if (dep_metals) then
            do i = 1, dustbins_props(dust_index)%nelements
                metal_masses(dustbins_props(dust_index)%el_index(i)) = max(metal_masses(dustbins_props(dust_index)%el_index(i)) - &
                    (condensed_SNII_dustmass * dustbins_props(dust_index)%el_mfractions(i)),0d0)
            end do
        end if

    end function condensed_SNII_dustmass

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

    subroutine add_sink_metal_from_dust_species(ivar,accreted_mass,sink_metallicity,isink)
        implicit none
        integer,intent(in) :: ivar
        integer,intent(in) :: isink
        real(dp),intent(in) :: accreted_mass
        real(dp),dimension(:,:),intent(inout) :: sink_metallicity

        integer :: iel,metal_index,dust_index,pah_index
        real(dp) :: carbon_mass

        if (accreted_mass <= 0d0) return

        if (ivar.ge.idust .and. ivar.lt.idust+ndust) then
            dust_index = ivar - idust + 1
            do iel = 1, dustbins_props(dust_index)%nelements
                metal_index = dustbins_props(dust_index)%el_index(iel)
                if (metal_index >= 1 .and. metal_index <= size(sink_metallicity,2)) then
                    sink_metallicity(isink,metal_index) = sink_metallicity(isink,metal_index) + &
                        accreted_mass * dustbins_props(dust_index)%el_mfractions(iel)
                end if
            end do
        else if (ivar.ge.ipah .and. ivar.lt.ipah+npah) then
            pah_index = ivar - ipah + 1
            metal_index = pahbins_props(pah_index)%C_index
            if (metal_index >= 1 .and. metal_index <= size(sink_metallicity,2)) then
                carbon_mass = accreted_mass * &
                    (pahbins_props(pah_index)%nc * mC_amu) / &
                    max(pahbins_props(pah_index)%nc * mC_amu + pahbins_props(pah_index)%n * mH_amu, tiny(1d0))
                sink_metallicity(isink,metal_index) = sink_metallicity(isink,metal_index) + carbon_mass
            end if
        end if
    end subroutine add_sink_metal_from_dust_species

end module dust_yields