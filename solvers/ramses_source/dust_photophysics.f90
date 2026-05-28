! Dust photophysics module for Dusty-PRISM
! For details, see Rodriguez Montero et al. (2025)
! By: Curro Rodriguez Montero (Original: 27 May 2025)

module dust_optics
    use hydro_parameters, only:ndust,ndchemtype,npah
    use constants
    use dust_commons
    use dust_utils

    contains
    subroutine init_dust_efficiency_tables(nGroups)
        ! Read dust and PAH optical properties from header-rich text files.
        ! Data lines are parsed after skipping blank/comment lines.
        !-------------------------------------------------------------------------
        use amr_commons,only:myid
        implicit none
        integer, intent(in) :: nGroups

        logical :: ok_dust,ok_pah
        logical :: ok_dust_all,ok_pah_all
        integer :: nwav
        integer :: i,j,k,isize,ios
        character(len=7) :: i_str
        character(len=256),dimension(1:ndust) :: fDust
        character(len=256),dimension(1:npah ) :: fPAH
        character(len=512) :: line
        real(dp) :: wav_ref
        real(dp) :: wav_read,cabs_read,csca_read,crp_read
        real(dp) :: wav_neu,cabs_neu,csca_neu,crp_neu
        real(dp) :: wav_ion,cabs_ion,csca_ion,crp_ion

        ! Check first that all files are in the expected place.
        ok_dust_all = .true.
        ok_pah_all = .true.
        do i=1,ndust
            write(i_str, '(I2.2)') i
            write(fDust(i),'(a,a,a,a)')trim(dust_tables_dir),'averaged_cross_section_DustBin_',trim(i_str),'.txt'
            inquire(file=fDust(i),exist=ok_dust)
            ok_dust_all = ok_dust_all .and. ok_dust
        end do
        if (.not. ok_dust_all) then
            if(myid.eq.1) then
                write(*,*)'ERROR in READ CROSS SECTION TABLES'
                do i = 1,ndust
                    write(*,*)fDust(i)
                end do
                write(*,*)'Cannot access dust directory ',TRIM(dust_tables_dir)
                write(*,*)'Directory '//TRIM(dust_tables_dir)//' not found'
                write(*,*)'You need to set this correctly for' // &
                         ' dust_tables_dir in the namelist.'
            endif
            call clean_stop
        end if

        if (npah > 0) then
            do i=1,npah
                write(i_str, '(I2.2)') i
                write(fPAH(i),'(a,a,a,a)')trim(dust_tables_dir),'averaged_cross_section_PAHBin_',trim(i_str),'.txt'
                inquire(file=fPAH(i),exist=ok_pah)
                ok_pah_all = ok_pah_all .and. ok_pah
            end do
            if (.not. ok_pah_all) then
                if(myid.eq.1) then
                    write(*,*)'ERROR in READ CROSS SECTION TABLES'
                    do i = 1,npah
                        write(*,*)fPAH(i)
                    end do
                    write(*,*)'Cannot access dust directory ',TRIM(dust_tables_dir)
                    write(*,*)'Directory '//TRIM(dust_tables_dir)//' not found'
                    write(*,*)'You need to set this correctly for' // &
                             ' dust_tables_dir in the namelist.'
                end if
                call clean_stop
            end if
        end if

        ! Read dust optical properties.
        do i = 1, ndust
            open(unit=20,file=fDust(i),status='old',form='formatted',action='read',iostat=ios)
            if (ios /= 0) then
                if (myid.eq.1) then
                    write(*,*) 'ERROR in READ CROSS SECTION TABLES'
                    write(*,*) 'Could not open file: ', trim(fDust(i))
                end if
                call clean_stop
            end if

            call read_next_data_line(20,line,ios)
            if (ios /= 0) then
                if (myid.eq.1) write(*,*) 'ERROR reading NWAV in file ', trim(fDust(i))
                call clean_stop
            end if
            read(line,*,iostat=ios) nwav
            if (ios /= 0) then
                if (myid.eq.1) write(*,*) 'ERROR parsing NWAV in file ', trim(fDust(i))
                call clean_stop
            end if

            if (allocated(dustbins_props(i)%cs_abs_tab%npts)) deallocate(dustbins_props(i)%cs_abs_tab%npts)
            allocate(dustbins_props(i)%cs_abs_tab%npts(1:1))
            dustbins_props(i)%cs_abs_tab%ndim = 1
            dustbins_props(i)%cs_abs_tab%npts(1) = nwav
            if (allocated(dustbins_props(i)%cs_abs_tab%ipos_zero)) deallocate(dustbins_props(i)%cs_abs_tab%ipos_zero)
            allocate(dustbins_props(i)%cs_abs_tab%ipos_zero(1:1))
            dustbins_props(i)%cs_abs_tab%ipos_zero(1) = 1
            if (allocated(dustbins_props(i)%cs_abs_tab%tab1d)) deallocate(dustbins_props(i)%cs_abs_tab%tab1d)
            allocate(dustbins_props(i)%cs_abs_tab%tab1d(1:nwav,1:1))
            if (allocated(dustbins_props(i)%cs_abs_tab%tab2d)) deallocate(dustbins_props(i)%cs_abs_tab%tab2d)
            allocate(dustbins_props(i)%cs_abs_tab%tab2d(1:nwav,1:1,1:1))

            if (allocated(dustbins_props(i)%cs_scat_tab%npts)) deallocate(dustbins_props(i)%cs_scat_tab%npts)
            allocate(dustbins_props(i)%cs_scat_tab%npts(1:1))
            dustbins_props(i)%cs_scat_tab%ndim = 1
            dustbins_props(i)%cs_scat_tab%npts(1) = nwav
            if (allocated(dustbins_props(i)%cs_scat_tab%ipos_zero)) deallocate(dustbins_props(i)%cs_scat_tab%ipos_zero)
            allocate(dustbins_props(i)%cs_scat_tab%ipos_zero(1:1))
            dustbins_props(i)%cs_scat_tab%ipos_zero(1) = 1
            if (allocated(dustbins_props(i)%cs_scat_tab%tab1d)) deallocate(dustbins_props(i)%cs_scat_tab%tab1d)
            allocate(dustbins_props(i)%cs_scat_tab%tab1d(1:nwav,1:1))
            if (allocated(dustbins_props(i)%cs_scat_tab%tab2d)) deallocate(dustbins_props(i)%cs_scat_tab%tab2d)
            allocate(dustbins_props(i)%cs_scat_tab%tab2d(1:nwav,1:1,1:1))

            if (allocated(dustbins_props(i)%cs_ext_tab%npts)) deallocate(dustbins_props(i)%cs_ext_tab%npts)
            allocate(dustbins_props(i)%cs_ext_tab%npts(1:1))
            dustbins_props(i)%cs_ext_tab%ndim = 1
            dustbins_props(i)%cs_ext_tab%npts(1) = nwav
            if (allocated(dustbins_props(i)%cs_ext_tab%ipos_zero)) deallocate(dustbins_props(i)%cs_ext_tab%ipos_zero)
            allocate(dustbins_props(i)%cs_ext_tab%ipos_zero(1:1))
            dustbins_props(i)%cs_ext_tab%ipos_zero(1) = 1
            if (allocated(dustbins_props(i)%cs_ext_tab%tab1d)) deallocate(dustbins_props(i)%cs_ext_tab%tab1d)
            allocate(dustbins_props(i)%cs_ext_tab%tab1d(1:nwav,1:1))
            if (allocated(dustbins_props(i)%cs_ext_tab%tab2d)) deallocate(dustbins_props(i)%cs_ext_tab%tab2d)
            allocate(dustbins_props(i)%cs_ext_tab%tab2d(1:nwav,1:1,1:1))

            ! ISRF-averaged cross sections: C_abs_ISRF C_sca_ISRF C_rp_ISRF.
            call read_next_data_line(20,line,ios)
            if (ios /= 0) then
                if (myid.eq.1) write(*,*) 'ERROR reading ISRF averages in file ', trim(fDust(i))
                call clean_stop
            end if
            read(line,*,iostat=ios) dustbins_props(i)%Pabs_isrf, dustbins_props(i)%Psc_isrf, dustbins_props(i)%Prp_isrf
            if (ios /= 0) then
                if (myid.eq.1) write(*,*) 'ERROR parsing ISRF averages in file ', trim(fDust(i))
                call clean_stop
            end if

            dustbins_props(i)%Pabs_isrf = dustbins_props(i)%Pabs_isrf * c_cgs * u_Mathis1983
            dustbins_props(i)%Psc_isrf  = dustbins_props(i)%Psc_isrf  * c_cgs * u_Mathis1983
            dustbins_props(i)%Prp_isrf  = dustbins_props(i)%Prp_isrf  * c_cgs * u_Mathis1983

            do k = 1, nwav
                call read_next_data_line(20,line,ios)
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR reading wavelength row ', k, ' in file ', trim(fDust(i))
                    call clean_stop
                end if
                read(line,*,iostat=ios) wav_read, cabs_read, csca_read, crp_read
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR parsing wavelength row ', k, ' in file ', trim(fDust(i))
                    call clean_stop
                end if
                dustbins_props(i)%cs_abs_tab%tab1d(k,1) = wav_read
                dustbins_props(i)%cs_scat_tab%tab1d(k,1) = wav_read
                dustbins_props(i)%cs_ext_tab%tab1d(k,1) = wav_read

                dustbins_props(i)%cs_abs_tab%tab2d(k,1,1) = cabs_read
                dustbins_props(i)%cs_scat_tab%tab2d(k,1,1) = csca_read
                dustbins_props(i)%cs_ext_tab%tab2d(k,1,1) = crp_read
            end do
            close(20)
            dustbins_props(i)%cs_abs_tab%initialised = .true.
            dustbins_props(i)%cs_scat_tab%initialised = .true.
            dustbins_props(i)%cs_ext_tab%initialised = .true.
        end do

        if (ndust>0) then
            allocate(group_csa_dust(1:nGroups,1:ndust),&
                     group_css_dust(1:nGroups,1:ndust),&
                     group_csr_dust(1:nGroups,1:ndust),&
                     group_csrat_dust(1:nGroups,1:ndust))
            allocate(sigca_dust(1:nGroups,1:ndust),&
                     sigcs_dust(1:nGroups,1:ndust),&
                     sigcr_dust(1:nGroups,1:ndust),&
                     sigcrat_dust(1:nGroups,1:ndust))
        end if

        if (npah > 0) then
            do isize=1,npah
                open(unit=12,file=fPAH(isize),status='old',form='formatted',action='read',iostat=ios)
                if (ios /= 0) then
                    if (myid.eq.1) then
                        write(*,*) 'ERROR in READ CROSS SECTION TABLES'
                        write(*,*) 'Could not open file: ', trim(fPAH(isize))
                    end if
                    call clean_stop
                end if

                call read_next_data_line(12,line,ios)
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR reading NWAV in file ', trim(fPAH(isize))
                    call clean_stop
                end if
                read(line,*,iostat=ios) nwav
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR parsing NWAV in file ', trim(fPAH(isize))
                    call clean_stop
                end if

                if (allocated(pahbins_props(isize)%cs_abs_tab%npts)) deallocate(pahbins_props(isize)%cs_abs_tab%npts)
                allocate(pahbins_props(isize)%cs_abs_tab%npts(1:2))
                pahbins_props(isize)%cs_abs_tab%ndim = 2
                pahbins_props(isize)%cs_abs_tab%npts(1) = nwav
                pahbins_props(isize)%cs_abs_tab%npts(2) = 2
                if (allocated(pahbins_props(isize)%cs_abs_tab%ipos_zero)) deallocate(pahbins_props(isize)%cs_abs_tab%ipos_zero)
                allocate(pahbins_props(isize)%cs_abs_tab%ipos_zero(1:2))
                pahbins_props(isize)%cs_abs_tab%ipos_zero = (/1,1/)
                if (allocated(pahbins_props(isize)%cs_abs_tab%tab1d)) deallocate(pahbins_props(isize)%cs_abs_tab%tab1d)
                allocate(pahbins_props(isize)%cs_abs_tab%tab1d(1:nwav,1:1))
                if (allocated(pahbins_props(isize)%cs_abs_tab%tab2d)) deallocate(pahbins_props(isize)%cs_abs_tab%tab2d)
                allocate(pahbins_props(isize)%cs_abs_tab%tab2d(1:nwav,1:1,1:2))

                if (allocated(pahbins_props(isize)%cs_scat_tab%npts)) deallocate(pahbins_props(isize)%cs_scat_tab%npts)
                allocate(pahbins_props(isize)%cs_scat_tab%npts(1:2))
                pahbins_props(isize)%cs_scat_tab%ndim = 2
                pahbins_props(isize)%cs_scat_tab%npts(1) = nwav
                pahbins_props(isize)%cs_scat_tab%npts(2) = 2
                if (allocated(pahbins_props(isize)%cs_scat_tab%ipos_zero)) deallocate(pahbins_props(isize)%cs_scat_tab%ipos_zero)
                allocate(pahbins_props(isize)%cs_scat_tab%ipos_zero(1:2))
                pahbins_props(isize)%cs_scat_tab%ipos_zero = (/1,1/)
                if (allocated(pahbins_props(isize)%cs_scat_tab%tab1d)) deallocate(pahbins_props(isize)%cs_scat_tab%tab1d)
                allocate(pahbins_props(isize)%cs_scat_tab%tab1d(1:nwav,1:1))
                if (allocated(pahbins_props(isize)%cs_scat_tab%tab2d)) deallocate(pahbins_props(isize)%cs_scat_tab%tab2d)
                allocate(pahbins_props(isize)%cs_scat_tab%tab2d(1:nwav,1:1,1:2))

                if (allocated(pahbins_props(isize)%cs_ext_tab%npts)) deallocate(pahbins_props(isize)%cs_ext_tab%npts)
                allocate(pahbins_props(isize)%cs_ext_tab%npts(1:2))
                pahbins_props(isize)%cs_ext_tab%ndim = 2
                pahbins_props(isize)%cs_ext_tab%npts(1) = nwav
                pahbins_props(isize)%cs_ext_tab%npts(2) = 2
                if (allocated(pahbins_props(isize)%cs_ext_tab%ipos_zero)) deallocate(pahbins_props(isize)%cs_ext_tab%ipos_zero)
                allocate(pahbins_props(isize)%cs_ext_tab%ipos_zero(1:2))
                pahbins_props(isize)%cs_ext_tab%ipos_zero = (/1,1/)
                if (allocated(pahbins_props(isize)%cs_ext_tab%tab1d)) deallocate(pahbins_props(isize)%cs_ext_tab%tab1d)
                allocate(pahbins_props(isize)%cs_ext_tab%tab1d(1:nwav,1:1))
                if (allocated(pahbins_props(isize)%cs_ext_tab%tab2d)) deallocate(pahbins_props(isize)%cs_ext_tab%tab2d)
                allocate(pahbins_props(isize)%cs_ext_tab%tab2d(1:nwav,1:1,1:2))

                ! Read neutral ISRF averages and store in PAHBin (index 1).
                call read_next_data_line(12,line,ios)
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR reading neutral ISRF averages in file ', trim(fPAH(isize))
                    call clean_stop
                end if
                read(line,*,iostat=ios) cabs_neu, csca_neu, crp_neu
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR parsing neutral ISRF averages in file ', trim(fPAH(isize))
                    call clean_stop
                end if
                pahbins_props(isize)%Pabs_isrf(1) = cabs_neu * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983
                pahbins_props(isize)%Psc_isrf(1) = csca_neu * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983
                pahbins_props(isize)%Prp_isrf(1) = crp_neu * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983

                ! Read ionised ISRF averages and store in PAHBin (index 2).
                call read_next_data_line(12,line,ios)
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR reading ionised ISRF averages in file ', trim(fPAH(isize))
                    call clean_stop
                end if
                read(line,*,iostat=ios) cabs_ion, csca_ion, crp_ion
                if (ios /= 0) then
                    if (myid.eq.1) write(*,*) 'ERROR parsing ionised ISRF averages in file ', trim(fPAH(isize))
                    call clean_stop
                end if
                pahbins_props(isize)%Pabs_isrf(2) = cabs_ion * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983
                pahbins_props(isize)%Psc_isrf(2) = csca_ion * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983
                pahbins_props(isize)%Prp_isrf(2) = crp_ion * pi * pahbins_props(isize)%apah_cm**2d0 * c_cgs * u_Mathis1983

                do j = 1, nwav
                    call read_next_data_line(12,line,ios)
                    if (ios /= 0) then
                        if (myid.eq.1) write(*,*) 'ERROR reading wavelength row ', j, ' in file ', trim(fPAH(isize))
                        call clean_stop
                    end if
                    call replace_pipe_with_space(line)
                    read(line,*,iostat=ios) wav_neu, cabs_neu, csca_neu, crp_neu, &
                                            wav_ion, cabs_ion, csca_ion, crp_ion
                    if (ios /= 0) then
                        if (myid.eq.1) write(*,*) 'ERROR parsing wavelength row ', j, ' in file ', trim(fPAH(isize))
                        call clean_stop
                    end if

                    wav_ref = wav_neu
                    if (abs(wav_ion-wav_ref) > 1d-10 * max(abs(wav_ref),1d0)) then
                        if (myid.eq.1) then
                            write(*,*) 'ERROR in READ CROSS SECTION TABLES'
                            write(*,*) 'PAH neutral/ion wavelength mismatch for bin ', isize
                        end if
                        call clean_stop
                    end if

                    pahbins_props(isize)%cs_abs_tab%tab1d(j,1) = wav_neu
                    pahbins_props(isize)%cs_scat_tab%tab1d(j,1) = wav_neu
                    pahbins_props(isize)%cs_ext_tab%tab1d(j,1) = wav_neu

                    pahbins_props(isize)%cs_abs_tab%tab2d(j,1,1) = cabs_neu
                    pahbins_props(isize)%cs_scat_tab%tab2d(j,1,1) = csca_neu
                    pahbins_props(isize)%cs_ext_tab%tab2d(j,1,1) = crp_neu

                    pahbins_props(isize)%cs_abs_tab%tab2d(j,1,2) = cabs_ion
                    pahbins_props(isize)%cs_scat_tab%tab2d(j,1,2) = csca_ion
                    pahbins_props(isize)%cs_ext_tab%tab2d(j,1,2) = crp_ion
                end do
                close(12)

                pahbins_props(isize)%cs_abs_tab%initialised = .true.
                pahbins_props(isize)%cs_scat_tab%initialised = .true.
                pahbins_props(isize)%cs_ext_tab%initialised = .true.
            end do
        end if

        if (npah>0) then
            allocate(group_csa_pah(1:nGroups,1:2*npah),&
                     group_css_pah(1:nGroups,1:2*npah),&
                     group_csr_pah(1:nGroups,1:2*npah))
            allocate(sigca_pah(1:nGroups,1:2*npah),&
                     sigcs_pah(1:nGroups,1:2*npah),&
                     sigcr_pah(1:nGroups,1:2*npah))
        end if

    end subroutine init_dust_efficiency_tables

    subroutine init_dust_dielectric_tables
        use amr_commons,only:myid
        implicit none

        logical :: ok_per,ok_par,ok_iso
        integer :: i,j,istat,ntables
        integer, dimension(1:2) :: nwav
        real(dp) :: wav_read,val_read
        character(len=128) :: f_per,f_par,f_iso,dustlabel

        ! Read dielectric tables by dust bin and store them in dustbins_props(i)%Im_n.
        do i = 1, ndust
            write(dustlabel, '(A,I2.2)') 'DustBin_', i
            write(f_iso,'(a,a,a)') trim(dust_tables_dir), 'Im_n_', trim(dustlabel)
            write(f_per,'(a,a,a)') trim(dust_tables_dir), 'Im_n_', trim(dustlabel)//'_pe'
            write(f_par,'(a,a,a)') trim(dust_tables_dir), 'Im_n_', trim(dustlabel)//'_pa'

            inquire(file=f_per,exist=ok_per)
            inquire(file=f_par,exist=ok_par)
            inquire(file=f_iso,exist=ok_iso)

            if (ok_per .and. ok_par) then
                ntables = 2
            else if (ok_iso) then
                ntables = 1
            else
                if (myid.eq.1) then
                    write(*,*) 'ERROR in READ DIELECTRIC TABLES'
                    write(*,*) 'Missing dielectric table(s) for ', trim(dustlabel)
                    write(*,*) 'Expected either: ', trim(f_iso)
                    write(*,*) 'or both: ', trim(f_per), ' and ', trim(f_par)
                end if
                call clean_stop
            end if

            if (allocated(dustbins_props(i)%Im_n)) deallocate(dustbins_props(i)%Im_n)
            allocate(dustbins_props(i)%Im_n(1:ntables))

            if (ntables == 1) then
                open(unit=20,file=f_iso,status='old',action='read',iostat=istat)
                if (istat /= 0) then
                    if (myid.eq.1) write(*,*) 'Error opening file ', trim(f_iso)
                    call clean_stop
                end if
                read(20,'(i8)') nwav(1)

                if (allocated(dustbins_props(i)%Im_n(1)%npts)) deallocate(dustbins_props(i)%Im_n(1)%npts)
                allocate(dustbins_props(i)%Im_n(1)%npts(1:1))
                dustbins_props(i)%Im_n(1)%ndim = 1
                dustbins_props(i)%Im_n(1)%npts(1) = nwav(1)
                if (allocated(dustbins_props(i)%Im_n(1)%ipos_zero)) deallocate(dustbins_props(i)%Im_n(1)%ipos_zero)
                allocate(dustbins_props(i)%Im_n(1)%ipos_zero(1:1))
                dustbins_props(i)%Im_n(1)%ipos_zero(1) = 1
                if (allocated(dustbins_props(i)%Im_n(1)%tab1d)) deallocate(dustbins_props(i)%Im_n(1)%tab1d)
                allocate(dustbins_props(i)%Im_n(1)%tab1d(1:nwav(1),1:1))
                if (allocated(dustbins_props(i)%Im_n(1)%tab2d)) deallocate(dustbins_props(i)%Im_n(1)%tab2d)
                allocate(dustbins_props(i)%Im_n(1)%tab2d(1:nwav(1),1:1,1:1))

                do j=1,nwav(1)
                    read(20,*) wav_read,val_read
                    dustbins_props(i)%Im_n(1)%tab1d(j,1) = log10(wav_read)
                    dustbins_props(i)%Im_n(1)%tab2d(j,1,1) = log10(val_read)
                end do
                close(20)
                dustbins_props(i)%Im_n(1)%initialised = .true.
                dustbins_props(i)%separate_refractive_index = .false.
            else
                open(unit=20,file=f_per,status='old',action='read',iostat=istat)
                if (istat /= 0) then
                    if (myid.eq.1) write(*,*) 'Error opening file ', trim(f_per)
                    call clean_stop
                end if
                read(20,'(i8)') nwav(1)

                open(unit=21,file=f_par,status='old',action='read',iostat=istat)
                if (istat /= 0) then
                    if (myid.eq.1) write(*,*) 'Error opening file ', trim(f_par)
                    call clean_stop
                end if
                read(21,'(i8)') nwav(2)

                do j = 1, 2
                    if (allocated(dustbins_props(i)%Im_n(j)%npts)) deallocate(dustbins_props(i)%Im_n(j)%npts)
                    allocate(dustbins_props(i)%Im_n(j)%npts(1:1))
                    dustbins_props(i)%Im_n(j)%ndim = 1
                    dustbins_props(i)%Im_n(j)%npts(1) = nwav(j)
                    if (allocated(dustbins_props(i)%Im_n(j)%ipos_zero)) deallocate(dustbins_props(i)%Im_n(j)%ipos_zero)
                    allocate(dustbins_props(i)%Im_n(j)%ipos_zero(1:1))
                    dustbins_props(i)%Im_n(j)%ipos_zero(1) = 1
                    if (allocated(dustbins_props(i)%Im_n(j)%tab1d)) deallocate(dustbins_props(i)%Im_n(j)%tab1d)
                    allocate(dustbins_props(i)%Im_n(j)%tab1d(1:nwav(j),1:1))
                    if (allocated(dustbins_props(i)%Im_n(j)%tab2d)) deallocate(dustbins_props(i)%Im_n(j)%tab2d)
                    allocate(dustbins_props(i)%Im_n(j)%tab2d(1:nwav(j),1:1,1:1))
                end do

                do j=1,nwav(1)
                    read(20,*) wav_read,val_read
                    dustbins_props(i)%Im_n(1)%tab1d(j,1) = log10(wav_read)
                    dustbins_props(i)%Im_n(1)%tab2d(j,1,1) = log10(val_read)
                end do
                close(20)
                dustbins_props(i)%Im_n(1)%initialised = .true.

                do j=1,nwav(2)
                    read(21,*) wav_read,val_read
                    dustbins_props(i)%Im_n(2)%tab1d(j,1) = log10(wav_read)
                    dustbins_props(i)%Im_n(2)%tab2d(j,1,1) = log10(val_read)
                end do
                close(21)
                dustbins_props(i)%Im_n(2)%initialised = .true.
                dustbins_props(i)%separate_refractive_index = .true.
            end if
        end do

    end subroutine init_dust_dielectric_tables

    function photon_attenuation_length(wav,Im_n_1,use_separate_refractive_index,Im_n_2) result(la)
        ! General photon attenuation length from dielectric properties.
        ! If separate_refractive_index is true, use anisotropic expression
        ! with Im_n_1 = Im_perp and Im_n_2 = Im_par (Eq. 15 WD01).
        ! Otherwise, use isotropic expression with Im_n_1 only (Eq. 14 WD01).
        ! wav --> photon wavelength in nm
        ! Im_n_1 --> imaginary refractive index (or perpendicular component)
        ! use_separate_refractive_index --> switch between anisotropic/isotropic methods
        ! Im_n_2 --> parallel component (required when separate_refractive_index=true)
        ! la <-- photon attenuation length in nm
        use amr_commons, only: myid
        implicit none
        real(dp), intent(in) :: wav, Im_n_1
        logical, intent(in) :: use_separate_refractive_index
        real(dp), intent(in), optional :: Im_n_2
        real(dp) :: la, inv_la, Im_n_par

        if (use_separate_refractive_index) then
            if (.not.present(Im_n_2)) then
                if (myid.eq.1) then
                    write(*,*) 'ERROR in photon_attenuation_length'
                    write(*,*) 'separate_refractive_index=.true. requires both dielectric tables (Im_n_1 and Im_n_2).'
                end if
                call clean_stop
            end if
            Im_n_par = Im_n_2
            inv_la = 4d0 * pi / wav * (2d0/3d0 * Im_n_1 + 1d0/3d0 * Im_n_par)
            la = 1d0 / inv_la
        else
            la = wav / (4d0 * pi * Im_n_1)
        end if
    end function photon_attenuation_length

    function getla_dustbin(lambda,isize)
        ! Compute the photon attenuation length for a dust bin using its
        ! dielectric table(s). If separate_refractive_index is true, use
        ! the graphite anisotropic expression, otherwise use the isotropic one.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> index of dust bin
        ! getla_dustbin <-- attenuation length [cm]
        implicit none
        real(dp) :: getla_dustbin
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: Im_n_iso, Im_n_per, Im_n_par
        real(dp) :: lambdasize_cm

        if (.not.allocated(dustbins_props(isize)%Im_n)) then
            getla_dustbin = huge(1d0)
            return
        end if
        lambdasize_cm = lambda * 1d-8  ! convert from angstrom to cm
        if (dustbins_props(isize)%separate_refractive_index) then
            if (size(dustbins_props(isize)%Im_n) < 2) then
                getla_dustbin = huge(1d0)
                return
            end if
            if ((.not. dustbins_props(isize)%Im_n(1)%initialised) .or. &
                (.not. dustbins_props(isize)%Im_n(2)%initialised)) then
                getla_dustbin = huge(1d0)
                return
            end if

            npts = dustbins_props(isize)%Im_n(1)%npts(1)
            call interpolate1D(dustbins_props(isize)%Im_n(1)%tab1d(1:npts,1), &
                               dustbins_props(isize)%Im_n(1)%tab2d(1:npts,1,1), &
                               npts, log10(lambda), Im_n_per)
            call interpolate1D(dustbins_props(isize)%Im_n(2)%tab1d(1:npts,1), &
                               dustbins_props(isize)%Im_n(2)%tab2d(1:npts,1,1), &
                               npts, log10(lambda), Im_n_par)
            Im_n_per = 10d0**Im_n_per
            Im_n_par = 10d0**Im_n_par
            getla_dustbin = photon_attenuation_length(lambdasize_cm, Im_n_per, .true., Im_n_par)
        else
            if (.not. dustbins_props(isize)%Im_n(1)%initialised) then
                getla_dustbin = huge(1d0)
                return
            end if

            npts = dustbins_props(isize)%Im_n(1)%npts(1)
            call interpolate1D(dustbins_props(isize)%Im_n(1)%tab1d(1:npts,1), &
                               dustbins_props(isize)%Im_n(1)%tab2d(1:npts,1,1), &
                               npts, log10(lambda), Im_n_iso)
            Im_n_iso = 10d0**Im_n_iso
            getla_dustbin = photon_attenuation_length(lambdasize_cm, Im_n_iso, .false.)
        end if
    end function getla_dustbin

    subroutine rosseland_mean(wavelength, cross_section, T, rosseland_cs)
        ! Compute the Rosseland mean of the cross-section
        ! wavelength     --> wavelength array [cm]
        ! cross_section  --> cross-section array [cm2]
        ! T              --> temperature [K]
        ! rosseland_cs   <-- Rosseland mean cross-section [cm2]
        implicit none
        ! Input arguments
        real(dp), intent(in) :: wavelength(:)       ! Wavelength array (cm)
        real(dp), intent(in) :: cross_section(:)    ! Cross-section array (cm^2)
        real(dp), intent(in) :: T                   ! Temperature (K)
        ! Output
        real(dp), intent(out) :: rosseland_cs       ! Rosseland mean cross-section (cm^2)
        ! Local variables
        integer :: i, n
        real(dp) :: planck_derivative_i, planck_derivative_ip1
        real(dp) :: weight_i, weight_ip1
        real(dp) :: numerator_integral, denominator_integral
        real(dp) :: delta_lambda

        n = size(wavelength)
        numerator_integral = 0.0d0
        denominator_integral = 0.0d0

        ! Trapezoidal integration
        do i = 1, n - 1
            delta_lambda = wavelength(i+1) - wavelength(i)

            ! Compute the Planck derivative at points i and i+1
            planck_derivative_i   = planck_function_derivative(wavelength(i), T)
            planck_derivative_ip1 = planck_function_derivative(wavelength(i+1), T)

            ! Compute weights for Rosseland mean
            weight_i   = planck_derivative_i / cross_section(i)
            weight_ip1 = planck_derivative_ip1 / cross_section(i+1)

            ! Add contributions using trapezoidal rule
            denominator_integral   = denominator_integral + 0.5d0 * delta_lambda * (weight_i + weight_ip1)
            numerator_integral = numerator_integral + 0.5d0 * delta_lambda * (planck_derivative_i + planck_derivative_ip1)
        end do

        rosseland_cs = numerator_integral / denominator_integral
    end subroutine rosseland_mean

    subroutine planck_mean(wavelength, cross_section, T, planck_cs)
        ! Compute the Planck mean of the cross-section
        ! wavelength     --> wavelength array [cm]
        ! cross_section  --> cross-section array [cm2]
        ! T              --> temperature [K]
        ! planck_cs      <-- Planck mean cross-section [cm2]
        implicit none
        ! Input arguments
        real(dp), intent(in) :: wavelength(:)       ! Wavelength array (cm)
        real(dp), intent(in) :: cross_section(:)    ! Cross-section array (cm^2)
        real(dp), intent(in) :: T                   ! Temperature (K)
        ! Output
        real(dp), intent(out) :: planck_cs          ! Planck mean cross-section (cm^2)
        ! Local variables
        integer :: i, n
        real(dp) :: planck_i, planck_ip1
        real(dp) :: weight_i, weight_ip1
        real(dp) :: numerator_integral, denominator_integral
        real(dp) :: delta_lambda

        n = size(wavelength)
        numerator_integral = 0.0d0
        denominator_integral = 0.0d0

        ! Trapezoidal integration
        do i = 1, n - 1
            delta_lambda = wavelength(i+1) - wavelength(i)

            ! Compute the Planck function at points i and i+1
            planck_i   = planck_function(wavelength(i), T)
            planck_ip1 = planck_function(wavelength(i+1), T)

            ! Compute weights for Planck mean
            weight_i   = planck_i * cross_section(i)
            weight_ip1 = planck_ip1 * cross_section(i+1)

            ! Add contributions using trapezoidal rule
            numerator_integral   = numerator_integral + 0.5d0 * delta_lambda * (weight_i + weight_ip1)
            denominator_integral = denominator_integral + 0.5d0 * delta_lambda * (planck_i + planck_ip1)
        end do

        planck_cs = numerator_integral / denominator_integral
    end subroutine planck_mean

    subroutine planck_mean_derivative(wavelength, cross_section, T, dplanck_cs_dT)
        ! Compute temperature derivative of the Planck-mean cross section
        ! wavelength     --> wavelength array [cm]
        ! cross_section  --> cross-section array [cm2]
        ! T              --> temperature [K]
        ! dplanck_cs_dT  <-- d<sigma>_Planck/dT [cm2/K]
        implicit none
        real(dp), intent(in) :: wavelength(:)
        real(dp), intent(in) :: cross_section(:)
        real(dp), intent(in) :: T
        real(dp), intent(out) :: dplanck_cs_dT

        integer :: i, n
        real(dp) :: B_i, B_ip1, dB_i, dB_ip1
        real(dp) :: num, den, dnum, dden, delta_lambda

        n = size(wavelength)
        num = 0d0
        den = 0d0
        dnum = 0d0
        dden = 0d0

        do i = 1, n - 1
            delta_lambda = wavelength(i+1) - wavelength(i)

            B_i = planck_function(wavelength(i), T)
            B_ip1 = planck_function(wavelength(i+1), T)
            dB_i = planck_function_derivative(wavelength(i), T)
            dB_ip1 = planck_function_derivative(wavelength(i+1), T)

            num = num + 0.5d0 * delta_lambda * (B_i * cross_section(i) + B_ip1 * cross_section(i+1))
            den = den + 0.5d0 * delta_lambda * (B_i + B_ip1)
            dnum = dnum + 0.5d0 * delta_lambda * (dB_i * cross_section(i) + dB_ip1 * cross_section(i+1))
            dden = dden + 0.5d0 * delta_lambda * (dB_i + dB_ip1)
        end do

        if (den <= tiny(1d0)) then
            dplanck_cs_dT = 0d0
        else
            dplanck_cs_dT = (dnum * den - num * dden) / (den**2d0)
        end if
    end subroutine planck_mean_derivative
    
    subroutine init_dust_mean_cross_sections(sed_dir_in)

        use amr_commons,only:myid
        implicit none
        character(len=*), intent(in) :: sed_dir_in
        character(len=256) :: sed_dir_loc
        real(dp) :: log_Tmin, log_Tmax, log_step, T_val
        real(dp) :: dplanck_abs_dT, P_emit, dP_emit_dT
        integer :: i,j,k
        character(len=7) :: i_str

        ! 1. Pre-compute temperature grid parameters
        !    Temperatures range from 1K to 10^3 K in log10 steps of 0.1
        log_Tmin = 0d0
        log_Tmax = 3d0
        log_step = (log_Tmax - log_Tmin) / (100.d0 - 1.d0)

        ! 2. Loop over grain bins and compute mean cross-sections directly
        do i = 1, ndust
            k = dustbins_props(i)%cs_abs_tab%npts(1)

            ! 2.a Allocate and initialize Rosseland mean table
            if (allocated(dustbins_props(i)%Rosseland_tab%npts)) deallocate(dustbins_props(i)%Rosseland_tab%npts)
            allocate(dustbins_props(i)%Rosseland_tab%npts(1:2))
            dustbins_props(i)%Rosseland_tab%ndim = 2
            dustbins_props(i)%Rosseland_tab%npts(1) = 100
            dustbins_props(i)%Rosseland_tab%npts(2) = 3
            if (allocated(dustbins_props(i)%Rosseland_tab%ipos_zero)) deallocate(dustbins_props(i)%Rosseland_tab%ipos_zero)
            allocate(dustbins_props(i)%Rosseland_tab%ipos_zero(1:2))
            dustbins_props(i)%Rosseland_tab%ipos_zero = (/1,1/)
            if (allocated(dustbins_props(i)%Rosseland_tab%tab1d)) deallocate(dustbins_props(i)%Rosseland_tab%tab1d)
            allocate(dustbins_props(i)%Rosseland_tab%tab1d(1:100,1:1))
            if (allocated(dustbins_props(i)%Rosseland_tab%tab2d)) deallocate(dustbins_props(i)%Rosseland_tab%tab2d)
            allocate(dustbins_props(i)%Rosseland_tab%tab2d(1:100,1:1,1:3))

            ! 2.b Allocate and initialize Planck mean table
            if (allocated(dustbins_props(i)%Planck_tab%npts)) deallocate(dustbins_props(i)%Planck_tab%npts)
            allocate(dustbins_props(i)%Planck_tab%npts(1:2))
            dustbins_props(i)%Planck_tab%ndim = 2
            dustbins_props(i)%Planck_tab%npts(1) = 100
            dustbins_props(i)%Planck_tab%npts(2) = 3
            if (allocated(dustbins_props(i)%Planck_tab%ipos_zero)) deallocate(dustbins_props(i)%Planck_tab%ipos_zero)
            allocate(dustbins_props(i)%Planck_tab%ipos_zero(1:2))
            dustbins_props(i)%Planck_tab%ipos_zero = (/1,1/)
            if (allocated(dustbins_props(i)%Planck_tab%tab1d)) deallocate(dustbins_props(i)%Planck_tab%tab1d)
            allocate(dustbins_props(i)%Planck_tab%tab1d(1:100,1:1))
            if (allocated(dustbins_props(i)%Planck_tab%tab2d)) deallocate(dustbins_props(i)%Planck_tab%tab2d)
            allocate(dustbins_props(i)%Planck_tab%tab2d(1:100,1:1,1:3))

            ! 2.c Allocate and initialize Planck power table (for temperature equilibrium in log-log space)
            if (allocated(dustbins_props(i)%Planck_power_tab%npts)) deallocate(dustbins_props(i)%Planck_power_tab%npts)
            allocate(dustbins_props(i)%Planck_power_tab%npts(1:1))
            dustbins_props(i)%Planck_power_tab%ndim = 1
            dustbins_props(i)%Planck_power_tab%npts(1) = 100
            if (allocated(dustbins_props(i)%Planck_power_tab%ipos_zero)) deallocate(dustbins_props(i)%Planck_power_tab%ipos_zero)
            allocate(dustbins_props(i)%Planck_power_tab%ipos_zero(1:1))
            dustbins_props(i)%Planck_power_tab%ipos_zero(1) = 1
            if (allocated(dustbins_props(i)%Planck_power_tab%tab1d)) deallocate(dustbins_props(i)%Planck_power_tab%tab1d)
            allocate(dustbins_props(i)%Planck_power_tab%tab1d(1:100,1:1))
            if (allocated(dustbins_props(i)%Planck_power_tab%tab2d)) deallocate(dustbins_props(i)%Planck_power_tab%tab2d)
            allocate(dustbins_props(i)%Planck_power_tab%tab2d(1:100,1:1,1:1))

            ! 2.d Allocate and initialize Planck derivative table (dP_emit/dT)
            if (allocated(dustbins_props(i)%Planckderiv_tab%npts)) deallocate(dustbins_props(i)%Planckderiv_tab%npts)
            allocate(dustbins_props(i)%Planckderiv_tab%npts(1:1))
            dustbins_props(i)%Planckderiv_tab%ndim = 1
            dustbins_props(i)%Planckderiv_tab%npts(1) = 100
            if (allocated(dustbins_props(i)%Planckderiv_tab%ipos_zero)) deallocate(dustbins_props(i)%Planckderiv_tab%ipos_zero)
            allocate(dustbins_props(i)%Planckderiv_tab%ipos_zero(1:1))
            dustbins_props(i)%Planckderiv_tab%ipos_zero(1) = 1
            if (allocated(dustbins_props(i)%Planckderiv_tab%tab1d)) deallocate(dustbins_props(i)%Planckderiv_tab%tab1d)
            allocate(dustbins_props(i)%Planckderiv_tab%tab1d(1:100,1:1))
            if (allocated(dustbins_props(i)%Planckderiv_tab%tab2d)) deallocate(dustbins_props(i)%Planckderiv_tab%tab2d)
            allocate(dustbins_props(i)%Planckderiv_tab%tab2d(1:100,1:1,1:1))

            ! 2.e Compute all mean cross-sections and store directly in dustbin tables
            do j = 1, 100
                T_val = 10**(log_Tmin + log_step * (j - 1))

                ! Store temperature in linear space (for Rosseland and Planck tables)
                dustbins_props(i)%Rosseland_tab%tab1d(j,1) = T_val
                dustbins_props(i)%Planck_tab%tab1d(j,1) = T_val

                ! Compute and store Rosseland means (absorption, scattering, extinction)
                call rosseland_mean(dustbins_props(i)%cs_abs_tab%tab1d(1:k,1)*1.0d-8, &
                                    dustbins_props(i)%cs_abs_tab%tab2d(1:k,1,1), &
                                    T_val, dustbins_props(i)%Rosseland_tab%tab2d(j,1,1))
                call rosseland_mean(dustbins_props(i)%cs_scat_tab%tab1d(1:k,1)*1.0d-8, &
                                    dustbins_props(i)%cs_scat_tab%tab2d(1:k,1,1), &
                                    T_val, dustbins_props(i)%Rosseland_tab%tab2d(j,1,2))
                call rosseland_mean(dustbins_props(i)%cs_ext_tab%tab1d(1:k,1)*1.0d-8, &
                                    dustbins_props(i)%cs_ext_tab%tab2d(1:k,1,1), &
                                    T_val, dustbins_props(i)%Rosseland_tab%tab2d(j,1,3))

                ! Compute and store Planck means (absorption, scattering, extinction)
                call planck_mean(dustbins_props(i)%cs_abs_tab%tab1d(1:k,1)*1.0d-8, &
                                 dustbins_props(i)%cs_abs_tab%tab2d(1:k,1,1), &
                                 T_val, dustbins_props(i)%Planck_tab%tab2d(j,1,1))
                call planck_mean(dustbins_props(i)%cs_scat_tab%tab1d(1:k,1)*1.0d-8, &
                                 dustbins_props(i)%cs_scat_tab%tab2d(1:k,1,1), &
                                 T_val, dustbins_props(i)%Planck_tab%tab2d(j,1,2))
                call planck_mean(dustbins_props(i)%cs_ext_tab%tab1d(1:k,1)*1.0d-8, &
                                 dustbins_props(i)%cs_ext_tab%tab2d(1:k,1,1), &
                                 T_val, dustbins_props(i)%Planck_tab%tab2d(j,1,3))

                ! Store temperature in log-space (for Planck power table)
                dustbins_props(i)%Planck_power_tab%tab1d(j,1) = log10(max(T_val, tiny(1d0)))
                dustbins_props(i)%Planckderiv_tab%tab1d(j,1) = dustbins_props(i)%Planck_power_tab%tab1d(j,1)

                ! Compute and store Planck power for temperature equilibrium
                ! P_emit(T) = 4*sb*<Cabs>_Planck(T)*T^4, stored in log-space
                P_emit = 4d0 * sb * dustbins_props(i)%Planck_tab%tab2d(j,1,1) * T_val**4d0
                dustbins_props(i)%Planck_power_tab%tab2d(j,1,1) = log10(max(P_emit, tiny(1d0)))

                ! Analytic derivative of P_emit(T) using dB/dT in planck_function_derivative
                call planck_mean_derivative(dustbins_props(i)%cs_abs_tab%tab1d(1:k,1)*1e-8, &
                                            dustbins_props(i)%cs_abs_tab%tab2d(1:k,1,1), &
                                            T_val, dplanck_abs_dT)
                dP_emit_dT = 4d0 * sb * (dplanck_abs_dT * T_val**4d0 + 4d0 * dustbins_props(i)%Planck_tab%tab2d(j,1,1) * T_val**3d0)
                dustbins_props(i)%Planckderiv_tab%tab2d(j,1,1) = log10(max(dP_emit_dT, tiny(1d0)))
            end do

            dustbins_props(i)%Rosseland_tab%initialised = .true.
            dustbins_props(i)%Planck_tab%initialised = .true.
            dustbins_props(i)%Planck_power_tab%initialised = .true.
            dustbins_props(i)%Planckderiv_tab%initialised = .true.
        end do

        ! 3. Write the tables to files for verification
        if (myid==1) then
            ! Resolve SED directory: prefer provided argument, then envvar
            if (trim(sed_dir_in).eq.'') then
                call get_environment_variable('RAMSES_SED_DIR', sed_dir_loc)
            else
                sed_dir_loc = trim(sed_dir_in)
            end if
            if (trim(sed_dir_loc).eq.'') then
                sed_dir_loc = './SEDtables'
            end if

            ! Create subdirectory for dust tables
            do i = 1, ndust
                write(i_str, '(I2.2)') i  ! convert i to string without leading spaces
                open(unit=20+i,file=trim(sed_dir_loc)//'/rosseland_mean_DustBin_'//trim(i_str)//'.list',status='unknown')
                open(unit=30+i,file=trim(sed_dir_loc)//'/planck_mean_DustBin_'// trim(i_str)//'.list',status='unknown')
                do j = 1, 100
                    write(20+i,'(2e14.6)') dustbins_props(i)%Rosseland_tab%tab1d(j,1), dustbins_props(i)%Rosseland_tab%tab2d(j,1,1)
                    write(20+i,'(2e14.6)') dustbins_props(i)%Rosseland_tab%tab1d(j,1), dustbins_props(i)%Rosseland_tab%tab2d(j,1,2)
                    write(20+i,'(2e14.6)') dustbins_props(i)%Rosseland_tab%tab1d(j,1), dustbins_props(i)%Rosseland_tab%tab2d(j,1,3)
                    write(30+i,'(2e14.6)') dustbins_props(i)%Planck_tab%tab1d(j,1), dustbins_props(i)%Planck_tab%tab2d(j,1,1)
                    write(30+i,'(2e14.6)') dustbins_props(i)%Planck_tab%tab1d(j,1), dustbins_props(i)%Planck_tab%tab2d(j,1,2)
                    write(30+i,'(2e14.6)') dustbins_props(i)%Planck_tab%tab1d(j,1), dustbins_props(i)%Planck_tab%tab2d(j,1,3)
                end do
                close(20+i)
                close(30+i)
            end do
        end if
    end subroutine init_dust_mean_cross_sections

    subroutine initialize_cross_sections_from_blackbody_dust_pah(T, group_L0, group_L1, nGroups)
        ! This subroutine initializes per-group dust and PAH cross-sections
        ! using a blackbody spectrum at temperature T, following the same strategy
        ! as initialize_cross_sections_from_blackbody for gas/ions.
        ! For each RT group, a blackbody spectrum is generated and per-bin 
        ! cross-sections are integrated over the group's energy range [L0, L1].
        !
        ! T         --> reference temperature for blackbody [K]
        ! group_L0  --> lower energy bound of each RT group [eV] (nGroups)
        ! group_L1  --> upper energy bound of each RT group [eV] (nGroups)
        ! nGroups   --> number of RT groups
        !
        ! Populates:
        !   group_csa_dust(nGroups, ndust) - absorption cross-section per group
        !   group_css_dust(nGroups, ndust) - scattering cross-section per group
        !   group_csr_dust(nGroups, ndust) - rad-pressure cross-section per group
        !   group_csrat_dust(nGroups, ndust) - RAT cross-section per group
        !   att_len_dust(nGroups, ndust)   - photon attenuation length per group
        !   group_csa_pah(nGroups, npah)   - PAH absorption cross-section per group
        !   group_css_pah(nGroups, npah)   - PAH scattering cross-section per group
        !   group_csr_pah(nGroups, npah)   - PAH rad-pressure cross-section per group
        !
        use constants, only: c_cgs, hplanck, eV2erg
        implicit none

        real(kind=8), intent(in) :: T
        real(kind=8), intent(in) :: group_L0(nGroups), group_L1(nGroups)
        integer, intent(in) :: nGroups

        real(kind=8) :: lambda_min, lambda_max, delta_lambda, tmp
        real(kind=8) :: X(1000), Y(1000)
        real(kind=8) :: B_lam, norm, result
        integer :: ip, ii, isize, idx_n, idx_i
        logical :: skip_group

        ! Allocate group arrays if not already done
        if (.not. allocated(group_csa_dust)) allocate(group_csa_dust(nGroups, ndust))
        if (.not. allocated(group_css_dust)) allocate(group_css_dust(nGroups, ndust))
        if (.not. allocated(group_csr_dust)) allocate(group_csr_dust(nGroups, ndust))
        if (.not. allocated(att_len_dust)) allocate(att_len_dust(nGroups, ndust))
        ! RAT cross-sections per group/dust-bin
        if (.not. allocated(group_csrat_dust)) allocate(group_csrat_dust(nGroups, ndust))

        if (npah > 0) then
            ! PAH group arrays store neutral and ion interlaced: (nGroups, 2*npah)
            if (.not. allocated(group_csa_pah)) allocate(group_csa_pah(nGroups, 2*npah))
            if (.not. allocated(group_css_pah)) allocate(group_css_pah(nGroups, 2*npah))
            if (.not. allocated(group_csr_pah)) allocate(group_csr_pah(nGroups, 2*npah))
        end if

        ! Initialize to zero
        group_csa_dust = 0.d0
        group_css_dust = 0.d0
        group_csr_dust = 0.d0
        att_len_dust = 0.d0
        group_csrat_dust = 0.d0
        if (npah > 0) then
            group_csa_pah = 0.d0
            group_css_pah = 0.d0
            group_csr_pah = 0.d0
        end if

        ! Loop over groups
        do ip = 1, nGroups
            ! Skip non-SED groups (where L0 <= 0 or L1 <= 0 or L0 >= L1)
            if (group_L0(ip) <= 0.d0 .or. group_L1(ip) <= 0.d0 .or. &
                group_L0(ip) >= group_L1(ip)) cycle

            ! Convert group energy bounds [eV] to wavelengths [Angstrom]
            ! lambda [A] = hc / E [eV] -> hc / E * 1e8
            lambda_max = (hplanck * c_cgs / (group_L0(ip) * eV2erg)) * 1d8  ! [A]
            lambda_min = (hplanck * c_cgs / (group_L1(ip) * eV2erg)) * 1d8  ! [A]
            delta_lambda = (lambda_max - lambda_min) / 999.d0

            ! Fill wavelength grid X and blackbody spectrum Y
            do ii = 1, 1000
                X(ii) = lambda_min + (delta_lambda * (dble(ii) - 1.d0))
                tmp = X(ii) * 1.d-8  ! convert A to cm
                Y(ii) = planck_function(tmp, T)
            end do

            ! Compute normalization factor (integral of Y over the wavelength range)
            norm = integrate_spectrum_simple(X, Y, 1000)

            ! Process each dust bin
            do isize = 1, ndust
                ! Absorption cross-section: integral of f*Y*lambda*sigma_abs over wavelength
                result = integrate_dust_absorbtion(X, Y, 1000, isize)
                group_csa_dust(ip, isize) = result / norm
                
                ! Scattering cross-section: integral of f*Y*lambda*sigma_scat over wavelength
                result = integrate_dust_scattering(X, Y, 1000, isize)
                group_css_dust(ip, isize) = result / norm
                
                ! Radiation pressure cross-section: integral of f*Y*lambda*sigma_rp over wavelength
                result = integrate_dust_radpressure(X, Y, 1000, isize)
                group_csr_dust(ip, isize) = result / norm
                
                ! Attenuation length: integral of f*Y*lambda*la over wavelength
                result = integrate_dust_attenuationlength(X, Y, 1000, isize)
                att_len_dust(ip, isize) = result / norm

                ! RAT cross-section: use helper trapezoidal integrator
                result = integrate_dust_RAT(X, Y, 1000, isize)
                group_csrat_dust(ip, isize) = result / norm
            end do

            ! Process each PAH bin
            if (npah > 0) then
                do isize = 1, npah
                    idx_n = 2*(isize-1) + 1  ! neutral index
                    idx_i = 2*(isize-1) + 2  ! ion index

                    ! Neutral PAHs (ion=1)
                    result = integrate_pah_absorbtion(X, Y, 1000, isize, 1)
                    group_csa_pah(ip, idx_n) = result / norm

                    result = integrate_pah_scattering(X, Y, 1000, isize, 1)
                    group_css_pah(ip, idx_n) = result / norm

                    result = integrate_pah_radpressure(X, Y, 1000, isize, 1)
                    group_csr_pah(ip, idx_n) = result / norm

                    ! Ionised PAHs (ion=2)
                    result = integrate_pah_absorbtion(X, Y, 1000, isize, 2)
                    group_csa_pah(ip, idx_i) = result / norm

                    result = integrate_pah_scattering(X, Y, 1000, isize, 2)
                    group_css_pah(ip, idx_i) = result / norm

                    result = integrate_pah_radpressure(X, Y, 1000, isize, 2)
                    group_csr_pah(ip, idx_i) = result / norm
                end do
            end if
        end do  ! end loop over groups

    end subroutine initialize_cross_sections_from_blackbody_dust_pah

    ! Helper integration functions
    function integrate_spectrum_simple(X, Y, N) result(integral)
        implicit none
        integer, intent(in) :: N
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            integral = integral + 0.5d0 * (Y(i) + Y(i+1)) * (X(i+1) - X(i))
        end do
    end function integrate_spectrum_simple

    function integrate_dust_absorbtion(X, Y, N, isize) result(integral)
        implicit none
        integer, intent(in) :: N, isize
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            sigma = getAbsCrosssection(X(i), isize)
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*getAbsCrosssection(X(i+1), isize)) &
                                  * (X(i+1) - X(i))
        end do
    end function integrate_dust_absorbtion

    function integrate_dust_scattering(X, Y, N, isize) result(integral)
        implicit none
        integer, intent(in) :: N, isize
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            sigma = getScCrosssection(X(i), isize)
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*getScCrosssection(X(i+1), isize)) &
                                  * (X(i+1) - X(i))
        end do
    end function integrate_dust_scattering

    function integrate_dust_radpressure(X, Y, N, isize) result(integral)
        implicit none
        integer, intent(in) :: N, isize
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            sigma = getRpCrosssection(X(i), isize)
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*getRpCrosssection(X(i+1), isize)) &
                                  * (X(i+1) - X(i))
        end do
    end function integrate_dust_radpressure

    function integrate_dust_attenuationlength(X, Y, N, isize) result(integral)
        implicit none
        integer, intent(in) :: N, isize
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, la
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            la = getla_dustbin(X(i), isize)
            if (la < huge(1.d0)) then
                integral = integral + 0.5d0 * (Y(i)*X(i)*la + Y(i+1)*X(i+1)*getla_dustbin(X(i+1), isize)) &
                                      * (X(i+1) - X(i))
            end if
        end do
    end function integrate_dust_attenuationlength

    function integrate_dust_RAT(X, Y, N, isize) result(integral)
        implicit none
        integer, intent(in) :: N, isize
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i

        integral = 0.d0
        do i = 1, N - 1
            sigma = getRATCrosssection(X(i), isize)
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*getRATCrosssection(X(i+1), isize)) &
                                  * (X(i+1) - X(i))
        end do
    end function integrate_dust_RAT

    function integrate_pah_absorbtion(X, Y, N, isize, ion) result(integral)
        implicit none
        integer, intent(in) :: N, isize, ion
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            if (ion == 1) then
                sigma = getAbsCrosssection_pah_n(X(i), isize)
            else
                sigma = getAbsCrosssection_pah_i(X(i), isize)
            end if
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*sigma) * (X(i+1) - X(i))
        end do
    end function integrate_pah_absorbtion

    function integrate_pah_scattering(X, Y, N, isize, ion) result(integral)
        implicit none
        integer, intent(in) :: N, isize, ion
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            if (ion == 1) then
                sigma = getScCrosssection_pah_n(X(i), isize)
            else
                sigma = getScCrosssection_pah_i(X(i), isize)
            end if
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*sigma) * (X(i+1) - X(i))
        end do
    end function integrate_pah_scattering

    function integrate_pah_radpressure(X, Y, N, isize, ion) result(integral)
        implicit none
        integer, intent(in) :: N, isize, ion
        real(kind=8), intent(in) :: X(N), Y(N)
        real(kind=8) :: integral, sigma
        integer :: i
        
        integral = 0.d0
        do i = 1, N - 1
            if (ion == 1) then
                sigma = getRpCrosssection_pah_n(X(i), isize)
            else
                sigma = getRpCrosssection_pah_i(X(i), isize)
            end if
            integral = integral + 0.5d0 * (Y(i)*X(i)*sigma + Y(i+1)*X(i+1)*sigma) * (X(i+1) - X(i))
        end do
    end function integrate_pah_radpressure


    ! getla_graphite removed: use getla_dustbin() for all dust bins.

    FUNCTION flaLambda_dust(lambda,f,species,ion)
        implicit none
        real(kind=8):: flaLambda_dust, lambda, f
        integer :: species, ion
        integer :: isize

        if (npah > 0) then
            isize = species - npah*2
        else
            isize = species
        end if

        if (isize < 1 .or. isize > ndust) then
            flaLambda_dust = huge(1d0)
            return
        end if

        ! Use unified getla_dustbin for all grain types; it handles
        ! single- and double-refractive-index tables via
        ! dustbins_props(isize)%separate_refractive_index.
        flaLambda_dust = f * lambda * getla_dustbin(lambda,isize)
    END FUNCTION flaLambda_dust

    function getAbsCrosssection(lambda,isize)
        ! Compute absorption cross section for regular grains
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size grain array
        ! getAbsCrosssection <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getAbsCrosssection
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs

        npts = dustbins_props(isize)%cs_abs_tab%npts(1)
        call interpolate1D(log10(dustbins_props(isize)%cs_abs_tab%tab1d(1:npts,1)), &
                   log10(dustbins_props(isize)%cs_abs_tab%tab2d(1:npts,1,1)), &
                   npts,log10(lambda),cs)
        getAbsCrosssection = 10**cs

    end function getAbsCrosssection

    function getScCrosssection(lambda,isize)
        ! Compute scattering cross section for regular grains
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size grain array
        ! getScCrosssection <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getScCrosssection
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs

        npts = dustbins_props(isize)%cs_scat_tab%npts(1)
        call interpolate1D(log10(dustbins_props(isize)%cs_scat_tab%tab1d(1:npts,1)), &
                   log10(dustbins_props(isize)%cs_scat_tab%tab2d(1:npts,1,1)), &
                   npts, log10(lambda),cs)
        getScCrosssection = 10**cs

    end function getScCrosssection

    function getRpCrosssection(lambda,isize)
        ! Compute radiation pressure cross section for regular grains
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size grain array
        ! getRpCrosssection <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getRpCrosssection
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs

        npts = dustbins_props(isize)%cs_ext_tab%npts(1)
        call interpolate1D(log10(dustbins_props(isize)%cs_ext_tab%tab1d(1:npts,1)), &
                   log10(dustbins_props(isize)%cs_ext_tab%tab2d(1:npts,1,1)), &
                   npts, log10(lambda),cs)
        getRpCrosssection = 10**cs

    end function getRpCrosssection

    function getAbsCrosssection_pah_n(lambda,isize)
        ! Compute absorption cross section for neutral PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getAbsCrosssection_pah_n <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getAbsCrosssection_pah_n
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_abs_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_abs_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_abs_tab%tab2d(1:npts,1,1)), &
                   npts, log10(lambda),cs)
        getAbsCrosssection_pah_n = 10**cs

    end function getAbsCrosssection_pah_n

    function getScCrosssection_pah_n(lambda,isize)
        ! Compute scattering cross section for neutral PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getScCrosssection_pah_n <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getScCrosssection_pah_n
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_scat_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_scat_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_scat_tab%tab2d(1:npts,1,1)), &
                   npts, log10(lambda),cs)
        getScCrosssection_pah_n = 10**cs

    end function getScCrosssection_pah_n

    function getRpCrosssection_pah_n(lambda,isize)
        ! Compute radiation pressure cross section for neutral PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getRpCrosssection_pah_n <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getRpCrosssection_pah_n
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_ext_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_ext_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_ext_tab%tab2d(1:npts,1,1)), &
                   npts, log10(lambda),cs)
        getRpCrosssection_pah_n = 10**cs

    end function getRpCrosssection_pah_n

    function getAbsCrosssection_pah_i(lambda,isize)
        ! Compute absorption cross section for ionised PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getAbsCrosssection_pah_i <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getAbsCrosssection_pah_i
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_abs_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_abs_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_abs_tab%tab2d(1:npts,1,2)), &
                   npts, log10(lambda),cs)
        getAbsCrosssection_pah_i = 10**cs

    end function getAbsCrosssection_pah_i

    function getScCrosssection_pah_i(lambda,isize)
        ! Compute scattering cross section for ionised PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getScCrosssection_pah_i <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getScCrosssection_pah_i
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_scat_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_scat_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_scat_tab%tab2d(1:npts,1,2)), &
                   npts, log10(lambda),cs)
        getScCrosssection_pah_i = 10**cs

    end function getScCrosssection_pah_i

    function getRpCrosssection_pah_i(lambda,isize)
        ! Compute radiation pressure cross section for ionised PAH
        ! via log-log interpolation.
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getRpCrosssection_pah_i <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getRpCrosssection_pah_i
        real(dp) :: lambda
        integer  :: isize
        integer  :: npts
        real(dp) :: cs
        
        npts = pahbins_props(isize)%cs_ext_tab%npts(1)
        call interpolate1D(log10(pahbins_props(isize)%cs_ext_tab%tab1d(1:npts,1)), &
                   log10(pahbins_props(isize)%cs_ext_tab%tab2d(1:npts,1,2)), &
                   npts, log10(lambda),cs)
        getRpCrosssection_pah_i = 10**cs

    end function getRpCrosssection_pah_i

    function getRATCrosssection(lambda,isize)
        ! Compute RAT (radiative torque) cross section.
        ! (https://iopscience.iop.org/article/10.3847/1538-4357/abb1b4)
        ! lambda --> radiation wavelength [angstrom]
        ! isize  --> integer of size PAH array
        ! getRATCrosssection <-- cross section [cm2]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: getRATCrosssection
        real(dp) :: lambda,l2AA
        integer  :: isize

        real(dp) :: agrain,atrans

        agrain = dustbins_props(isize)%asize_cm
        l2AA   = lambda / 1d8 ! Wavelength from Angstrom to cm
        atrans = l2AA / 1.8d0

        ! These are the approximations suggested by the numerical calculations
        ! by Lazarian and Hoang (2007)
        if (agrain < atrans) then
            getRATCrosssection = 2.33d0 * (lambda / agrain)**(-3d0)
        else
            getRATCrosssection = 4d-1
        end if
        getRATCrosssection = pi * (agrain)**2 * getRATCrosssection
    end function getRATCrosssection

    FUNCTION fRATLambda_dust(lambda, f, species, ion)
        implicit none
        real(kind=8):: fRATLambda_dust, lambda, f
        integer :: species, ion
        integer :: isize

        isize = species - npah*2
        fRATLambda_dust = f * lambda * getRATCrosssection(lambda,isize)
    END FUNCTION fRATLambda_dust

    FUNCTION fAbsLambda_pah(lambda, f, species, ion)
        implicit none
        real(kind=8):: fAbsLambda_pah, lambda, f
        integer :: species, ion
        integer :: isize

        isize = int(species/2) + mod(species,2)
        if (mod(species,2) .ne. 0) then
            ! Neutral PAHs are the first
            fAbsLambda_pah = f * lambda * getAbsCrosssection_pah_n(lambda,isize)
        else
            ! While ionised PAHs are the second
            fAbsLambda_pah = f * lambda * getAbsCrosssection_pah_i(lambda,isize)
        end if
    END FUNCTION fAbsLambda_pah

    FUNCTION fScLambda_pah(lambda, f, species, ion)
        implicit none
        real(kind=8):: fScLambda_pah, lambda, f
        integer :: species, ion
        integer :: isize

        isize = int(species/2) + mod(species,2)
        if (mod(species,2) .ne. 0) then
            ! Neutral PAHs are the first
            fScLambda_pah = f * lambda * getScCrosssection_pah_n(lambda,isize)
        else
            ! While ionised PAHs are the second
            fScLambda_pah = f * lambda * getScCrosssection_pah_i(lambda,isize)
        end if
    END FUNCTION fScLambda_pah

    FUNCTION fRpLambda_pah(lambda, f, species, ion)
        implicit none
        real(kind=8):: fRpLambda_pah, lambda, f
        integer :: species, ion
        integer :: isize

        isize = int(species/2) + mod(species,2)
        if (mod(species,2) .ne. 0) then
            ! Neutral PAHs are the first
            fRpLambda_pah = f * lambda * getRpCrosssection_pah_n(lambda,isize)
        else
            ! While ionised PAHs are the second
            fRpLambda_pah = f * lambda * getRpCrosssection_pah_i(lambda,isize)
        end if
    END FUNCTION fRpLambda_pah
    
    FUNCTION fAbsLambda_dust(lambda, f, species, ion)
        implicit none
        real(kind=8):: fAbsLambda_dust, lambda, f
        integer :: species, ion
        integer :: isize
        isize = species
        fAbsLambda_dust = f * lambda * getAbsCrosssection(lambda,isize)
    END FUNCTION fAbsLambda_dust

    FUNCTION fScLambda_dust(lambda, f, species, ion)
        implicit none
        real(kind=8):: fScLambda_dust, lambda, f
        integer :: species, ion
        integer :: isize
        isize = species
        fScLambda_dust = f * lambda * getScCrosssection(lambda,isize)
    END FUNCTION fScLambda_dust

    FUNCTION fRpLambda_dust(lambda, f, species, ion)
        implicit none
        real(kind=8):: fRpLambda_dust, lambda, f
        integer :: species, ion
        integer :: isize
        isize = species
        fRpLambda_dust = f * lambda * getRpCrosssection(lambda,isize)
    END FUNCTION fRpLambda_dust
end module dust_optics

module dust_radiation
    use constants
    use amr_parameters, only:aexp,ndim,dp
    use dust_commons
    use hydro_parameters, only:ndust,npah

    contains

    function rad_dust_rate(cross_sec,rho_dust)
        ! Obtain the total local interaction rate given by the dust densities
        ! and provided averaged cross sections for a particular radiation group
        ! Actually, given that signc_dust and all those arrays are updates and
        ! multiplied already by rt_c_cgs, the actuall units of the cross_sec is
        ! [cm3/s] which means that when this function is called it should not be
        ! multiplied again by rt_c_cgs!!
        ! cross_sec --> grain cross section array for all dust types [cm3/s]
        ! rho_dust  --> ndust dimension array holding dust density [g/cm3]
        ! rate      <-- interaction rate [1/s]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: rad_dust_rate
        real(dp),dimension(1:ndust) :: cross_sec
        real(dp),dimension(1:ndust)   :: rho_dust
        integer :: i
        rad_dust_rate = 0D0
        do i=1,ndust
            rad_dust_rate = rad_dust_rate + cross_sec(i) * (rho_dust(i) / dustbins_props(i)%mgrain)
        end do
    end function rad_dust_rate

    function rad_pah_rate(cross_sec,rho_pah,fcharge_pahs)
        ! Obtain the total local interaction rate given by the PAH densities
        ! and provided averaged cross sections for a particular radiation group
        ! Actually, given that signc_dust and all those arrays are updates and
        ! multiplied already by rt_c_cgs, the actuall units of the cross_sec is
        ! [cm3/s] which means that when this function is called it should not be
        ! multiplied again by rt_c_cgs!!
        ! cross_sec --> PAH cross section array for all PAH types [cm3/s]
        ! rho_pah  --> npah dimension array holding PAH density [g/cm3]
        ! fcharge_pahs --> allocatable/assumed-shape array with the fraction
        !                 of each PAH charge state (column-wise per PAH bin)
        ! rate      <-- interaction rate [1/s]
        !-------------------------------------------------------------------------
        implicit none
        real(dp) :: rad_pah_rate
        real(dp),dimension(1:npah*2) :: cross_sec
        real(dp),dimension(1:npah)    :: rho_pah
        real(dp),dimension(:,:) :: fcharge_pahs
        real(dp) :: pah_ion_fraction
        integer :: i, cation_start, nstates
        rad_pah_rate = 0D0
        do i=1,npah
            nstates = pahbins_props(i)%ncharge_states
            cation_start = pahbins_props(i)%cation_start_idx
            if (cation_start <= nstates) then
                pah_ion_fraction = sum(fcharge_pahs(cation_start:nstates,i))
            else
                pah_ion_fraction = 0d0
            end if
            rad_pah_rate = rad_pah_rate + ((1d0-pah_ion_fraction)*cross_sec((i-1)*2+1) + &
                            & pah_ion_fraction * cross_sec((i-1)*2+2))* (rho_pah(i) / pahbins_props(i)%mpah)
        end do
    end function rad_pah_rate

    subroutine get_Tdust_radiative_eq(j, P_rad, Tmin, T0)
        ! This subroutine computes the local dust temperature by solving the radiative
        ! equilibrium equation: P_abs = P_emit, where P_abs is the absorbed power by
        ! the dust grain and P_emit is the emitted power. Both are computed using Planck
        ! mean cross sections. The solution is obtained via interpolation in log-log space
        ! using the pre-computed Planck power tables.
        ! j     --> integer of dust bin
        ! P_rad --> absorbed power by the dust grain [erg/s]
        ! Tmin  --> minimum allowed dust temperature (e.g. CMB temp) [K]
        ! T0    <-- computed dust temperature [K]
        !-------------------------------------------------------------------------
        use dust_utils, only: interpolate1D
        implicit none

        integer,intent(in) :: j
        real(dp),intent(in) :: P_rad, Tmin
        real(dp),intent(out) :: T0

        integer :: nT
        real(dp) :: logP, logT

        if (P_rad <= tiny(1d0)) then
            T0 = Tmin
            return
        end if

        nT = dustbins_props(j)%Planck_power_tab%npts(1)

        logP = log10(P_rad)

        call interpolate1D(dustbins_props(j)%Planck_power_tab%tab2d(1:nT,1,1), &
                        dustbins_props(j)%Planck_power_tab%tab1d(1:nT,1), &
                        nT, logP, logT)

        T0 = 10d0**logT
        T0 = max(T0, Tmin)
    end subroutine get_Tdust_radiative_eq

    subroutine solve_Tdust_fast(i_dust,P_abs,ne,nElement,xelem_ions,Coulomb_factor,nH2,&
                           nCO,Tgas,dust_charge,coll_heat,recomb_heat,pe_heat,P_rad,T,Tmin)
        ! This subroutine computes the local dust temperature by solving the full energy balance
        ! equation: P_rad + H_coll + recomb_heat = P_emit + pe_heat
        ! i_dust         --> integer of dust bin
        ! P_abs          --> absorbed power by the dust grain [erg/s]
        ! ne             --> electron density [cm^-3]
        ! nElement       --> array of elemental densities [cm^-3]
        ! xelem_ions     --> array of ionisation fractions for each element
        ! Coulomb_factor --> array of Coulomb factors
        ! nH2            --> molecular hydrogen density [cm^-3]
        ! nCO            --> carbon monoxide density [cm^-3]
        ! Tgas           --> gas temperature [K]
        ! dust_charge    --> charge of the dust grain (in units of e)
        ! coll_heat      <-- heating rate from collisions with gas particles [erg/s]
        ! recomb_heat    --> heating rate from recombination of electrons on dust [erg/s]
        ! pe_heat        --> cooling rate from photoelectric effect [erg/s]
        ! P_rad          <-- emitted power by the dust grain [erg/s]
        ! T              <-- computed dust temperature [K]
        ! Tmin           --> minimum allowed dust temperature (e.g. CMB temp) [K]
        !-------------------------------------------------------------------------
        use dust_cooling, only: compute_dust_coll_heating
        use dust_commons, only: dust_log_tdust_solver_update
        implicit none

        integer,intent(in) :: i_dust
        real(dp),intent(in) :: P_abs,ne,nH2,nCO,Tgas,dust_charge,Tmin
        real(dp),intent(inout) :: coll_heat,P_rad
        real(dp),intent(in) :: recomb_heat,pe_heat
        real(dp),dimension(1:n_elements),intent(in) :: nElement
        real(dp),dimension(1:n_elements,1:n_elements),intent(in) :: xelem_ions
        real(dp),dimension(-1:n_elements),intent(in) :: Coulomb_factor
        real(dp),intent(inout) :: T

        integer :: iter
        integer :: max_iter=100
        integer :: iter_used
        real(dp) :: H0, H1, dH_dT
        real(dp) :: dP_dT
        real(dp) :: f, fprime
        real(dp) :: T0, eps, T_new, dT

        T0 = T
        eps = 1d-2
        iter_used = 0

        ! =========================================================
        ! 1. Compute Hcoll and derivative once (if collisional cooling is enabled)
        ! =========================================================
        if (dust_coll_cooling) then
            dT = max(T0*eps,1e-6)
            call compute_dust_coll_heating(i_dust,ne,nElement,xelem_ions,&
                                        Coulomb_factor,nH2,nCO,Tgas,T0+dT,&
                                        dust_charge,H1)

            call compute_dust_coll_heating(i_dust,ne,nElement,xelem_ions,&
                                        Coulomb_factor,nH2,nCO,Tgas,max(T0-dT,Tmin),&
                                        dust_charge,H0)

            dH_dT = (H1 - H0) / (2d0*dT)
        else
            H0 = 0d0
            dH_dT = 0d0
        end if

        ! =========================================================
        ! 2. Newton iterations (with linearised approximation)
        ! =========================================================
        do iter = 1, max_iter
            iter_used = iter
            call dust_emission_with_deriv(i_dust,T,P_rad,dP_dT)
            f = P_abs + H0 + dH_dT*(T - T0) + recomb_heat - P_rad - pe_heat
            fprime = dH_dT - dP_dT
            if (f .eq. 0d0) then
                call dust_log_tdust_solver_update(iter_used, .false.)
                return
            end if
            ! if (abs(fprime) < 1d-20) exit

            T_new = T - f/fprime

            if (T_new < Tmin) then
                T = Tmin
                call dust_log_tdust_solver_update(iter_used, .false.)
                return
            end if

            if (abs(T_new - T)/T < 1d-3) then
                T = T_new
                call dust_log_tdust_solver_update(iter_used, .false.)
                return
            end if
            T = T_new
        end do

        ! fallback (rare)
        call solve_Tdust_brent_fast(i_dust,P_abs,H0,dH_dT,T0,recomb_heat,pe_heat,P_rad,Tmin,1d3,T)
        call dust_log_tdust_solver_update(iter_used, .true.)

        ! Save the final collisional heating rate
        if (dust_coll_cooling) then
            coll_heat = H0 + dH_dT*(T - T0)
        else
            coll_heat = 0d0
        end if

    end subroutine solve_Tdust_fast

    subroutine dust_emission_with_deriv(j,T,P,dP_dT)
        ! This subroutine computes the Planck power emitted by a dust grain at a given
        ! temperature T, as well as its derivative with respect to T. Both are obtained
        ! via interpolation in log-log space using the pre-computed Planck power tables.
        ! j     --> integer of dust bin
        ! T     --> dust temperature [K]
        ! P     <-- emitted power [erg/s]
        ! dP_dT <-- derivative of emitted power with respect to T [erg/s/K]
        !-------------------------------------------------------------------------

        use dust_utils, only: interpolate1D
        implicit none

        integer,intent(in) :: j
        real(dp),intent(in) :: T
        real(dp),intent(out) :: P,dP_dT

        integer :: nT
        real(dp) :: logT, logP, logdP_dT

        nT = dustbins_props(j)%Planck_power_tab%npts(1)

        logT = log10(T)

        ! logP
        call interpolate1D(dustbins_props(j)%Planck_power_tab%tab1d(1:nT,1), &
                        dustbins_props(j)%Planck_power_tab%tab2d(1:nT,1,1), &
                        nT, logT, logP)

        ! dP/dT
        call interpolate1D(dustbins_props(j)%Planckderiv_tab%tab1d(1:nT,1), &
                dustbins_props(j)%Planckderiv_tab%tab2d(1:nT,1,1), &
                nT, logT, logdP_dT)

        P = 10d0**logP
        dP_dT = 10d0**logdP_dT

    end subroutine dust_emission_with_deriv

    subroutine dust_emission_power(j,T,P)
        ! Interpolate tabulated Planck emission power for one dust bin.
        ! j --> dust bin index
        ! T --> dust temperature [K]
        ! P <-- emitted power [erg/s]
        use dust_utils, only: interpolate1D
        implicit none

        integer, intent(in) :: j
        real(dp), intent(in) :: T
        real(dp), intent(out) :: P

        integer :: nT
        real(dp) :: logT, logP

        nT = dustbins_props(j)%Planck_power_tab%npts(1)
        logT = log10(max(T, tiny(1d0)))

        call interpolate1D(dustbins_props(j)%Planck_power_tab%tab1d(1:nT,1), &
                           dustbins_props(j)%Planck_power_tab%tab2d(1:nT,1,1), &
                           nT, logT, logP)

        P = 10d0**logP
    end subroutine dust_emission_power

    subroutine solve_Tdust_brent_fast(j,P_abs,H0,dH_dT,T0,recomb_heat,pe_heat,P_rad,Tmin,Tmax,T)
        ! This subroutine computes the local dust temperature by solving the full energy balance
        ! equation: P_rad + H_coll + recomb_heat = P_emit + pe_heat
        ! using the Brent's method for root finding. This is used as a fallback when the
        ! Newton method fails to converge, which can happen in some cases (e.g. very low T).
        ! j             --> integer of dust bin
        ! P_abs         --> absorbed power by the dust grain [erg/s]
        ! H0            --> collisional heating rate at T0 [erg/s]
        ! dH_dT         --> derivative of collisional heating rate with respect to T [erg/s/K]
        ! T0            --> reference temperature for collisional heating [K]
        ! recomb_heat   --> heating rate from recombination of electrons on dust [erg/s]
        ! pe_heat       --> cooling rate from photoelectric effect [erg/s]
        ! P_rad         <-- emitted power by the dust grain [erg/s]
        ! Tmin          --> minimum allowed dust temperature (e.g. CMB temp) [K]
        ! Tmax          --> maximum allowed dust temperature for root finding [K]
        ! T             <-- computed dust temperature [K]
        !-------------------------------------------------------------------------

        implicit none

        integer,intent(in) :: j
        real(dp),intent(in) :: P_abs,H0,dH_dT,T0,recomb_heat,pe_heat,Tmin,Tmax
        real(dp),intent(inout) :: T,P_rad

        integer :: iter
        real(dp) :: a,b,c,fa,fb,fc,P

        a = Tmin
        b = Tmax

        call dust_emission_power(j,a,P_rad)
        fa = P_abs + H0 + dH_dT*(a-T0) + recomb_heat - P_rad - pe_heat

        call dust_emission_power(j,b,P_rad)
        fb = P_abs + H0 + dH_dT*(b-T0) + recomb_heat - P_rad - pe_heat

        if (fa*fb > 0d0) then
            T = a
            return
        end if

        do iter = 1, 40

            c = 0.5d0*(a+b)

            call dust_emission_power(j,c,P_rad)
            fc = P_abs + H0 + dH_dT*(c-T0) + recomb_heat - P_rad - pe_heat

            if (abs(fc) < 1d-6) then
                T = c
                return
            end if

            if (fa*fc < 0d0) then
                b = c
                fb = fc
            else
                a = c
                fa = fc
            end if

        end do

        T = 0.5d0*(a+b)

    end subroutine solve_Tdust_brent_fast

    subroutine update_T_dust(G0_background,coll_heat,&
                            &recomb_heat,pe_heat,P_rad,T_dust,&
                            &ne,nElement,xelem_ions,Coulomb_factor,&
                            &nH2,nCO,Tgas,dust_charge,&
                            &Ep,cs_abs)
        ! This subroutine updates the local dust temperatures by considering
        ! the radiation field conditions. The balance between radiative
        ! heating and radiative cooling is only achieved for the larger grains
        ! which are in LTE. Actually, given that signc_dust and all those arrays 
        ! are updates and multiplied already by rt_c_cgs, the actuall units of 
        ! the cross_sec is [cm3/s].
        ! Additionnaly, this assumes that all absorption is in the UV-optical and
        ! the emission solely in the IR.
        ! NOTE: This computation is not valid for PAHs, as thermal fluctuations
        ! can be very large.
        ! G0_background --> Background G0 (in the case there is no RT)
        ! coll_heat     <-- collisional heating rate [erg/s]
        ! recomb_heat   --> heating rate from electron recombinations [erg/s]
        ! pe_heat       --> photoelectric cooling rate [erg/s]
        ! P_rad         <-- radiative emitted rate [erg/s]
        ! T_dust        <-- ndust length array with dust temperature [K]
        ! ne            --> electron density [cm^-3]
        ! nElement       --> array of elemental densities [cm^-3]
        ! xelem_ions     --> array of ionisation fractions for each element
        ! Coulomb_factor --> array of Coulomb factors
        ! nH2            --> molecular hydrogen density [cm^-3]
        ! nCO            --> carbon monoxide density [cm^-3]
        ! Tgas           --> gas temperature [K]
        ! dust_charge    --> charge of the dust grains (in units of e)
        ! Ep            --> radiation energy density [eV/cm3]
        ! cs_abs        --> grain cross section array for dust types [cm3/s] (no PAHs)
        !-------------------------------------------------------------------------
        use amr_commons, only: myid
        use dust_utils, only: interpolate1D
        use dust_cooling, only: compute_dust_coll_heating
        implicit none
        real(dp),intent(in) :: G0_background
        real(dp),dimension(1:ndust),intent(inout) :: coll_heat,P_rad
        real(dp),dimension(1:ndust),intent(in) :: recomb_heat,pe_heat
        real(dp),dimension(1:ndust),intent(inout) :: T_dust
        real(dp),dimension(1:n_elements),intent(in) :: nElement
        real(dp),dimension(1:n_elements,1:n_elements),intent(in) :: xelem_ions
        real(dp),dimension(1:ndust,-1:n_elements),intent(in) :: Coulomb_factor
        real(dp),intent(in) :: ne,nH2,nCO,Tgas
        real(dp),dimension(1:ndust),intent(in) :: dust_charge
        real(dp),intent(in),optional :: Ep(:),cs_abs(:,:)

        integer :: i,j
        real(dp) :: P_abs, Tmin, T0, H_coll_at_Tgas

        ! Limit dust temp minimum to CMB temp
        Tmin = 2.725d0 * (1.d0/aexp)

        do j = 1, ndust
            ! --- Radiative absorbed power ---
            P_abs = dustbins_props(j)%Pabs_isrf * (G0_background / 1.13d0) ! [erg/s]
            
            if (present(Ep) .and. present(cs_abs)) then
                if (.not.all(Ep.eq.0d0)) then
                    do i = 1, size(Ep)
                        P_abs = P_abs + Ep(i) * cs_abs(i,j) * eV2erg ! [erg/s]
                    end do
                end if
            endif

            ! --- Initial guess: radiative equilibrium ---
            call get_Tdust_radiative_eq(j, P_abs, Tmin, T0)

            ! --- Check collisional heating at Tgas ---
            ! If collisional heating dominates radiation, start closer to Tgas
            if (dust_coll_cooling) then
                call compute_dust_coll_heating(j,ne,nElement,xelem_ions,&
                                            Coulomb_factor(j,:),nH2,nCO,Tgas,T0,&
                                            dust_charge(j),H_coll_at_Tgas)
                if (H_coll_at_Tgas > P_abs) then
                    ! Collisional heating dominates the initial guess
                    call get_Tdust_radiative_eq(j, H_coll_at_Tgas, Tmin, T0)
                    ! print*, 'Rank ', myid, ': Collisional heating dominates for dust bin ', j, &
                    !         ': H_coll at Tgas = ', H_coll_at_Tgas, ' erg/s > P_abs = ', P_abs, ' erg/s. Starting Newton iterations at Tdust = ', T0, ' K'
                else if (H_coll_at_Tgas < 1d-4 * P_abs) then
                    ! Radiative heating is much larger than collisional, so we can just assume that
                    T_dust(j) = max(T0, Tmin)
                    cycle
                end if
            end if

            call solve_Tdust_fast(j,P_abs,ne,nElement,xelem_ions,Coulomb_factor(j,:),&
                                nH2,nCO,Tgas,dust_charge(j),coll_heat(j),&
                                recomb_heat(j),pe_heat(j),P_rad(j),T0,Tmin)

            T_dust(j) = max(T0, Tmin)
        end do
    end subroutine update_T_dust

end module dust_radiation

module dust_radiative_torques

    use amr_parameters, only:ndim
    use dust_commons
    use cooling_module, only: kB, mH
    use hydro_parameters, only:ndust,ndchemtype,npah

    contains

    function total_radiative_torque(rad_anisotropy,dNp,group_egy_erg,nGroups,local_c,csrat_dust)
        ! This function computes the local radiative torque caused
        ! by the different radiation bins on a particular dust grain
        ! based on the RAT model (see Hoang et al. 2020 for a review)

        use constants, only:twopi,hplanck,c_cgs
        implicit none
        integer, intent(in) :: nGroups
        real(dp), dimension(1:nGroups) :: dNp,csrat_dust,group_egy_erg,rad_anisotropy
        real(dp) :: local_c
        real(dp) :: total_radiative_torque

        integer  :: igroup,idim
        real(dp) :: lambda_mean,rad_density

        total_radiative_torque = 0d0

        do igroup = 1, nGroups
            if (dNp(igroup) .lt. 0d0) cycle

            ! 1. Compute mean wavelength [cm] of radiation bin
            lambda_mean = hplanck * c_cgs / (group_egy_erg(igroup))

            ! 2. Compute radiation energy density in [erg/cm^3]
            rad_density = group_egy_erg(igroup) * dNp(igroup)

            ! 3. Get everything together into the formula of radiative torque [erg]
            !    (Eq. 13 in Hoang et al. (2021))
            total_radiative_torque = total_radiative_torque + csrat_dust(igroup) * &
                                    & rad_density * rad_anisotropy(igroup) * (lambda_mean / (local_c * twopi))
        end do
    end function total_radiative_torque

    function IR_damping_factor(U,nH,Tgas,Tdust)
        ! This function computes the damping factor (wrt the hydrogen gas damping)
        ! due to the negative torque casued by IR photons emitted carrying part of
        ! of the grain angular momentum.
        ! NOTE: The scaling relation here used is an approximation given by Eq. 30
        ! in Draine & Lazarian (1998) which assumes simple power laws for the absorption
        ! cross sections of dust grains as well as gas damping dominated by Hydrogen
        implicit none
        real(dp) :: U,nH,Tgas
        real(dp),dimension(1:ndust) :: Tdust
        real(dp),dimension(1:ndust) :: IR_damping_factor

        IR_damping_factor = 59d0 * (1d-3/asize) * U**(2d0/3d0) * (2d1/nH) * sqrt(1d2/Tgas) * (2d1/Tdust)**2d0

    end function IR_damping_factor

    function RAT_frequency(nH,Tk,local_mu,gamma_RAT,FIR)
        ! This function computes the equilibrium radiative torque frequency as defined
        ! in Lazarian & Hoang 2007; Hoang & Lazarian 2009; Hoang & Lazarian 2014
        implicit none
        real(dp) :: nH,Tk,local_mu
        real(dp),dimension(1:ndust) :: gamma_RAT,FIR
        real(dp),dimension(1:ndust) :: RAT_frequency
        real(dp),dimension(1:ndust) :: tau_gas
        real(dp) :: vth

        integer :: i
        ! See Appendix B1 of Draine & Lazarian (1998)
        vth = sqrt(2d0*kB*Tk/(local_mu*mH))
        do i = 1, ndust
            tau_gas(i) = dustbins_props(i)%tau_gas_0 / (nH * local_mu * vth)
        end do

        ! See Eq. 19 in Hoang et al. (2021)
        RAT_frequency = gamma_RAT * tau_gas / (1d0 + FIR)
    end function RAT_frequency
end module dust_radiative_torques

module dust_photoelectric_heating
    use amr_parameters, only:ndim
    use dust_commons
    use hydro_parameters, only:ndust
    use constants, only: pi, eV2erg, e2instatC
    use dust_utils, only: interpolate1D, interpolate2D

    implicit none
    private   ! default
    public:: compute_dust_peh_rate, interpolate_dust_peh_rate,&
            most_negative_allowed_charge


    contains

    ! ====== FUNCTIONS ======
    function ionisation_potential_valence(W, Z, a) result(IP)
        ! Calculate the ionisation potential for the valence electron of a dust grain
        ! Taken from Eq. 2 Weingartner & Draine (2001)
        ! W --> work function (eV)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! IP <-- ionisation potential in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: W
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: IP

        IP = W + e2instatC / a * ((Z + 0.5d0) + (Z + 2d0) * (3d-9 / a)) / eV2erg
    end function ionisation_potential_valence
    
    function electron_afinity(W,E_g,Z,a,use_separate_refractive_index) result(EA)
        ! Calculate the electron affinity for dust grains
        ! Taken from Eq. 4 and 5 Weingartner & Draine (2001)
        ! W --> work function (eV)
        ! E_g --> band gap (eV)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! EA <-- electron affinity in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: W
        real(dp), intent(in) :: E_g
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp) :: EA

        if (use_separate_refractive_index) then
            EA = W - E_g + e2instatC / a * ((Z - 0.5d0) - 4d-8 / (a + 7d-8)) / eV2erg
        else
            EA = W - E_g + e2instatC / a * (Z - 0.5d0) / eV2erg
        end if
    end function electron_afinity

    function min_energy_ejection(Z,a) result(E_min)
        ! Electron energy at which tunneling probability becomes
        ! significant for electron ejection from a dust grain,
        ! evaluated using the WKB formalism
        ! Taken from Eq. 7 Weingartner & Draine (2001)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! E_min <-- minimum energy in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: E_min

        if (Z >= 0d0) then
            E_min = 0d0
        else
            E_min = - (Z + 1.0d0) * e2instatC / a / (1.0d0 + (2.7d-7 / a)**0.75d0) / eV2erg
        end if
    end function min_energy_ejection

    function photodetachment_energy(W,E_g,Z,a,use_separate_refractive_index) result(hnu_pdt)
        ! Minimum photon energy required for photodetachment of an electron
        ! from a negatively charged dust grain
        ! Taken from Eq. 18 Weingartner & Draine (2001)
        ! W --> work function (eV)
        ! E_g --> band gap (eV)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! hnu_pdt <-- minimum photon energy in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: W
        real(dp), intent(in) :: E_g
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp) :: hnu_pdt

        ! Local
        real(dp) :: E_min, EA

        E_min = min_energy_ejection(Z,a)
        EA = electron_afinity(W,E_g,Z+1,a,use_separate_refractive_index)

        hnu_pdt = EA + E_min
    end function photodetachment_energy

    function photodetachment_cross_section(E,E_det,Z) result(sigma_pdt)
        ! Photodetachment cross section for electrons from negatively charged
        ! dust grains following Eq. 20 in Weingartner & Draine (2001)
        ! E --> photon energy in eV
        ! E_det --> photodetachment energy in eV
        ! Z --> grain charge (in units of e)
        ! sigma_pdt <-- photodetachment cross section in cm^2

        implicit none
        ! Inputs
        real(dp), intent(in) :: E
        real(dp), intent(in) :: E_det
        real(dp), intent(in) :: Z
        ! Output
        real(dp) :: sigma_pdt
        ! Local
        real(dp) :: diffx

        diffx = (E - E_det) / 3.0d0
        if (diffx .lt. 0d0) then
            sigma_pdt = 0d0
        else
            sigma_pdt = 1.2d-17 * abs(Z) * diffx / (1. + (diffx**2d0)/3d0)**2d0
        end if        
    end function photodetachment_cross_section

    function min_photon_energy(IPV,Z,a) result(hnu_min)
        ! Minimum photon energy required for photoelectric emission
        ! from a dust grain
        ! Taken from Eq. 6 Weingartner & Draine (2001)
        ! IPV --> ionisation potential of valence electron (in eV)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! hnu_min <-- minimum photon energy in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: IPV
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: hnu_min

        ! Local
        real(dp) :: E_min

        if (Z >= -1d0) then
            hnu_min = IPV
        else
            E_min = min_energy_ejection(Z,a)
            hnu_min = IPV + E_min
        end if
    end function min_photon_energy

    function parameter_theta(E,Emin_ej,Z,a) result(theta)
        ! Parameter theta defined in Eq. 9 Weingartner & Draine (2001)
        ! This is the parametrisation used for the first part of the
        ! photoelectric yield functions
        ! E --> photon energy in eV
        ! Emin_ej --> minimum energy for electron ejection in eV
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! theta <-- parameter theta (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: E
        real(dp), intent(in) :: Emin_ej
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: theta

        if (Z >= 0) then
            theta = E - Emin_ej + e2instatC / a * (Z + 1d0) / eV2erg
        else
            theta = E - Emin_ej
        end if
    end function parameter_theta

    function attempting_electron_integral(hnu,Emin,Emin_ej,Z,a) result(E_avg)
        ! Average energy of attempting electrons that can escape from a dust grain
        ! computed by analytical integration of Eq.12 Weingartner & Draine (2001)
        ! multiplied by the energy of the electron
        ! See also Eq. 39 in Weingartner & Draine (2001)
        ! hnu --> photon energy in eV
        ! Emin --> minimum photon energy for photoelectric emission in eV
        ! Emin_ej --> minimum energy for electron ejection in eV
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! E_avg <-- average energy in eV

        implicit none
        ! Inputs
        real(dp), intent(in) :: hnu
        real(dp), intent(in) :: Emin
        real(dp), intent(in) :: Emin_ej
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: E_avg
        ! Local
        real(dp) :: Ei,Ef
        real(dp) :: Elow,Ehigh
        real(dp) :: E_avg_1,E_avg_2

        if (Z < 0d0) then
            Elow = Emin
            Ehigh = Emin + hnu - Emin_ej
            Ei = Emin
            Ef = Ehigh
        else
            Elow = - e2instatC / a * (Z + 1d0) / eV2erg
            Ehigh = hnu - Emin_ej
            Ei = 0d0
            Ef = Ehigh
        end if

        E_avg_1 = Ef**2d0 * (6d0*Ehigh*Elow - 4d0*Elow*Ef - 4d0*Ehigh*Ef + 3d0*Ef**2d0) / (2d0*(Elow-Ehigh)**3d0)
        E_avg_2 = Ei**2d0 * (6d0*Ehigh*Elow - 4d0*Elow*Ei - 4d0*Ehigh*Ei + 3d0*Ei**2d0) / (2d0*(Elow-Ehigh)**3d0)
        E_avg = E_avg_1 - E_avg_2
    end function attempting_electron_integral

    function escape_fraction_attempting_electrons(hnu,Emin_ej,Z,a) result(f_esc)
        ! Fraction of attempting electrons that can escape from a dust grain
        ! after being photo-ejected from their initial bound state
        ! Taken from Eq. 11 Weingartner & Draine (2001)
        ! hnu --> photon energy in eV
        ! Emin_ej --> minimum photon energy for electron ejection in eV
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! f_esc <-- escape fraction (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: hnu
        real(dp), intent(in) :: Emin_ej
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        ! Output
        real(dp) :: f_esc
        ! Local
        real(dp) :: Elow,Ehigh

        if (Z >= 0d0) then
            Elow = - e2instatC / a * (Z + 1d0) / eV2erg
            Ehigh = hnu - Emin_ej
            f_esc = Ehigh**2d0 * (Ehigh - 3d0 * Elow) / (Ehigh - Elow)**3d0
        else
            f_esc = 1d0
        end if

    end function escape_fraction_attempting_electrons

    function Watson73_y1(a,la,le) result(y1)
        ! Yield enhancement factor for small grains
        ! Taken from Draine (1978) which reproduces the
        ! theoretical results from Watson (1973) based
        ! in Mie theory
        ! a --> grain radius in cm
        ! la --> photon attenuation length in cm
        ! le --> electron escape length in cm
        ! y1 <-- yield enhancement factor (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: a
        real(dp), intent(in) :: la
        real(dp), intent(in) :: le
        ! Output
        real(dp) :: y1
        ! Local
        real(dp) :: beta, alpha

        beta = a / la
        alpha = a / le + a / la
        y1 = (beta / alpha)**2d0 * (alpha**2d0 - 2d0 * alpha + 2d0 - 2d0 * exp(-alpha)) \
                / (beta**2d0 - 2d0 * beta + 2d0 - 2d0 * exp(-beta))
    end function Watson73_y1

    function y0_graphite(theta,W) result(y0)
        ! Bulk yield for photoelectric emission from a flat surface of graphite
        ! This is based on the fitting in Bakes & Tielens (1994)
        ! to the experimental results from Verstraete et al. (1990)
        ! that approximately reproduces the photoelectric yield of
        ! coronene. This is particularly larger compared to the bulk
        ! graphite yields measured by Feuerbacher & Fitton (1972)
        ! NOTE: These are highly uncertain!!
        ! theta --> parameter theta (in eV)
        ! W --> work function (in eV)
        ! y0 <-- bulk yield (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: theta
        real(dp), intent(in) :: W
        ! Output
        real(dp) :: y0

        y0 = 9d-3 * (theta / W)**5d0 / (1d0 + 3.7d-2*(theta / W)**5d0)
    end function y0_graphite

    function y0_silicate(theta,W) result(y0)
        ! Bulk yield for photoelectric emission from a flat surface of silicates
        ! Taken from Eq. 17 Weingartner & Draine (2001) using a fitting
        ! to the scaling measured by Feuerbacher et al. (1972)
        ! NOTE: These are highly uncertain!!
        ! theta --> parameter theta (in eV)
        ! W --> work function (in eV)
        ! y0 <-- bulk yield (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: theta
        real(dp), intent(in) :: W
        ! Output
        real(dp) :: y0

        y0 = 5d-1 * (theta / W) / (1d0 + 5d0 * (theta / W))
    end function y0_silicate

    function autoionisation_potential(a,use_separate_refractive_index) result(AIP)
        ! Autoionisation potential for dust grains
        ! Taken from Eq. 23 Weingartner & Draine (2001)
        ! a --> grain radius in cm
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! AIP <-- autoionisation potential in V

        implicit none
        ! Inputs
        real(dp), intent(in) :: a
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp) :: AIP

        if (use_separate_refractive_index) then
            AIP = 3.9d0 + 1.2d7 * a + 2d-8 / a
        else
            AIP = 2.5d0 + 7d6 * a + 8d-8 / a
        end if

    end function autoionisation_potential

    function most_negative_allowed_charge(a,use_separate_refractive_index) result(Zmin)
        ! Most negative allowed charge for dust grains
        ! Taken from Eq. 24 Weingartner & Draine (2001)
        ! a --> grain radius in cm
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! Zmin <-- most negative allowed charge (in units of e)

        implicit none
        ! Inputs
        real(dp), intent(in) :: a
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp) :: Zmin
        ! Local
        real(dp) :: U_aip

        U_aip = autoionisation_potential(a,use_separate_refractive_index)
        Zmin = floor(U_aip / 14.4d-8 * a) + 1d0
    end function most_negative_allowed_charge

    function DS87_lambda(Z,q,a,T) result(ltilde)
        ! Dimensionless parameter ltilde defined in Draine & Sutin (1987)
        ! This is used to compute the sticking coefficient of electrons
        ! colliding with a dust grain
        ! Z --> grain charge (in units of e)
        ! q --> charge of the colliding particle (in units of e)
        ! a --> grain radius (in cm)
        ! T --> gas temperature in K
        ! ltilde <-- dimensionless parameter
        use cooling_module, only: kB
        implicit none
        ! Inputs
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: q
        real(dp), intent(in) :: a
        real(dp), intent(in) :: T
        ! Output
        real(dp) :: ltilde
        ! Local
        real(dp) :: nu, tau, theta

        nu = Z / q
        tau = a * kB * T / q**2d0 / e2instatC

        if (nu .eq. 0d0) then
            ltilde = 2d0 + 1.5d0 * sqrt(pi/(2d0*tau))
        else if (nu < 0d0) then
            ltilde = (2d0 - nu/tau) * (1d0 + 1d0/sqrt(tau - nu))
        else
            theta = 1d0 / (1d0 + 1d0/sqrt(nu))
            ltilde = (2d0 + nu/tau) * (1d0 + 1d0/sqrt(1.5d0/tau + 3d0*nu)) * exp(-theta*nu/tau)
        end if
    end function DS87_lambda

    function DS87_J(Z,q,a,T) result(J)
        ! Dimensionless parameter J defined in Draine & Sutin (1987)
        ! This is used to compute the collisional rate of electrons
        ! Z --> grain charge (in units of e)
        ! q --> charge of the colliding particle (in units of e)
        ! a --> grain radius (in cm)
        ! T --> gas temperature in K
        ! J <-- dimensionless parameter
        use cooling_module, only: kB
        implicit none
        ! Inputs
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: q
        real(dp), intent(in) :: a
        real(dp), intent(in) :: T
        ! Output
        real(dp) :: J
        ! Local
        real(dp) :: nu, tau, theta

        nu = Z / q
        tau = a * kB * T / q**2d0 / e2instatC

        if (nu .eq. 0d0) then
            J = 1d0 + sqrt(pi/(2d0*tau))
        else if (nu < 0d0) then
            J = (1d0 - nu/tau) * (1d0 + sqrt(2d0/(tau - 2d0*nu)))
        else
            theta = 1d0 / (1d0 + 1d0/sqrt(nu))
            J = ((1d0 + 1d0/sqrt(4d0*tau + 3d0*nu))**2d0) * exp(-theta*nu/tau)
        end if
    end function DS87_J

    function e_sticking_coeff(Z,a,l_e,use_separate_refractive_index) result(s_e)
        ! Sticking coefficient for electrons colliding with dust grains
        ! Taken from Sec. 3 in Weingartner & Draine (2001)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius in cm
        ! l_e --> electron escape length in cm
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! s_e <-- sticking coefficient (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        real(dp), intent(in) :: l_e
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp) :: s_e
        ! Local
        real(dp) :: Nc,Zmin
        integer :: exp_factor

        if (use_separate_refractive_index) then
            exp_factor = 20
        else
            exp_factor = 25
        end if

        if (Z == 0d0) then
            Nc = 468d0 * (a/1d-7)**3d0
            s_e = 5d-1 * (1d0 - exp(-a/l_e)) * 1d0 / (1d0 + exp(real(exp_factor,dp) - Nc))
        else if (Z < 0d0) then
            Zmin = most_negative_allowed_charge(a*10d0,use_separate_refractive_index)
            if (Z > Zmin) then
                Nc = 468d0 * (a/1d-7)**3d0
                s_e = 5d-1 * (1d0 - exp(-a/l_e)) * 1d0 / (1d0 + exp(real(exp_factor,dp) - Nc))
            else
                s_e = 0d0
            end if
        else
            s_e = 5d-1 * (1d0 - exp(-a/l_e))
        end if
    end function e_sticking_coeff

    ! ======= SUBROUTINES ======
    subroutine photoelectric_yield(W,E_g,Z,a,E,l_a,l_e,use_separate_refractive_index,Y,y2)
        ! Photoelectric yield for dust grains
        ! Taken from Eq. 12 Weingartner & Draine (2001)
        ! W --> work function (eV)
        ! E_g --> band gap (eV)
        ! Z --> grain charge (in units of e)
        ! a --> grain radius (in cm)
        ! E --> photon energy (in eV)
        ! l_a --> photon attenuation length in cm
        ! l_e --> electron escape length in cm
        ! use_separate_refractive_index --> .true. for anisotropic (graphite-like), .false. for isotropic (silicate-like)
        ! Y <-- photoelectric yield (dimensionless)
        ! y2 <-- fraction of escaping electrons (dimensionless)

        implicit none
        ! Inputs
        real(dp), intent(in) :: W
        real(dp), intent(in) :: E_g
        real(dp), intent(in) :: Z
        real(dp), intent(in) :: a
        real(dp), intent(in) :: E
        real(dp), intent(in) :: l_a
        real(dp), intent(in) :: l_e
        logical, intent(in) :: use_separate_refractive_index
        ! Output
        real(dp), intent(out) :: Y,y2
        ! Local
        real(dp) :: IPV,Emin_ej,theta,la,y1,y0

        ! 1. Compute IPV, Emin_ej
        IPV = ionisation_potential_valence(W,Z,a)
        Emin_ej = min_photon_energy(IPV,Z,a)

        if (E .lt. Emin_ej) then
            Y = 0d0
            y2 = 0d0
            return
        end if

        ! 2. Compute theta and y0
        theta = parameter_theta(E,Emin_ej,Z,a)
        if (use_separate_refractive_index) then
            y0 = y0_graphite(theta,W)
        else
            y0 = y0_silicate(theta,W)
        end if

        ! 3. Obtain y1 using the radiation averaged l_a
        y1 = Watson73_y1(a,l_a,l_e)

        ! 4. Obtain y2 using the escape fraction of attempting electrons
        y2 = escape_fraction_attempting_electrons(E,Emin_ej,Z,a)

        ! 5. Get everything together into the final yield
        Y = y2 * min(y0 * y1, 1d0)

    end subroutine photoelectric_yield

    subroutine compute_dust_peh_rate(i_dust,rho_dust,csa,l_a,&
                                    nGroups,local_c,solid_angle,Np,E,Zdust,Zsigma,Tgas,ne,Pinj,Prec,debug_flag)
        ! This subroutine computes the photoelectric heating rate
        ! and recombination cooling rate for a given dust species
        ! following the formalism of Weingartner & Draine (2001)
        ! i_dust   --> index of the dust species
        ! rho_dust --> dust mass density [g/cm^3]
        ! csa      --> dust absorption cross section (already multiplied by c) [cm^3/s]
        ! l_a      --> dust photon attenuation length [cm]
        ! nGroups  --> number of radiation groups
        ! local_c  --> speed of light in cm/s
        ! solid_angle --> solid angle subtended by the radiation field in each group [sr]
        ! Np       --> number density photons in #/cm3
        ! E        --> photon energy in eV
        ! Tgas     --> gas temperature in K
        ! ne       --> electron density in cm^-3
        ! Zdust    --> dust grain charge in units of e
        ! Zsigma   --> dust charge distribution sigma
        ! Pinj     <--> photoelectric heating rate [erg/cm^3/s]
        ! Prec     <--> photoelectric recombination cooling rate [erg/cm^3/s]
        use cooling_module, only: kB
        use dust_charging, only: two_point_charge_mix, three_point_charge_mix
        implicit none
        ! Inputs
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: rho_dust
        integer, intent(in) :: nGroups
        real(dp), dimension(1:nGroups), intent(in) :: csa,l_a,Np,E,solid_angle
        real(dp), intent(in) :: Zdust,Zsigma,Tgas,ne,local_c
        logical, intent(in), optional :: debug_flag
        ! Outputs
        real(dp), intent(inout) :: Pinj,Prec

        ! Local
        integer :: i,iq
        integer :: z1,z2,z3,zmin_mix
        integer :: nZmix_eff,zlo,zhi
        real(dp), dimension(1:3) :: Zmix,wmix
        real(dp) :: asize_cm,W,E_g,l_e,Emin,IPV,Emin_ej
        real(dp) :: E_avg,E_pdt,sigma_pdt
        real(dp) :: Y,y2,pinj_mix_raw,pinj_charge_raw,prec_charge
        real(dp) :: w1,w2,w3,wlo,whi,Zcharge
        real(dp) :: rad_ani,ltilde,s_e,rec_pref
        real(dp) :: EA,Jtilde
        real(dp), dimension(1:nGroups) :: pe_pref,pdt_pref
        logical :: dbg_flag, use_separate_refractive_index, used_two_point

        if (all(Np.eq.0d0)) return

        ! Debug
        if (.not.present(debug_flag)) then
            dbg_flag = .false.
        else
            dbg_flag = debug_flag
        end if
        ! 1. Get fixed dust properties and representative charges
        asize_cm = dustbins_props(i_dust)%asize_cm
        W = dustbins_props(i_dust)%work_function
        E_g = dustbins_props(i_dust)%band_gap
        l_e = dustbins_props(i_dust)%e_escape_length
        use_separate_refractive_index = dustbins_props(i_dust)%separate_refractive_index
        zmin_mix = nint(dustbins_props(i_dust)%Zmin)
        nZmix_eff = min(max(nZmix,1),3)
        used_two_point = .false.
        select case (nZmix_eff)
        case (1)
            Zmix = (/Zdust, 0d0, 0d0/)
            wmix = (/1d0, 0d0, 0d0/)
        case (2)
            call two_point_charge_mix(Zdust,zmin_mix,zlo,zhi,wlo,whi)
            Zmix = (/real(zlo,dp), real(zhi,dp), 0d0/)
            wmix = (/wlo, whi, 0d0/)
            used_two_point = .true.
        case (3)
            call three_point_charge_mix(Zdust,Zsigma,zmin_mix,z1,z2,z3,w1,w2,w3,used_two_point)
            Zmix = (/real(z1,dp), real(z2,dp), real(z3,dp)/)
            wmix = (/w1, w2, w3/)
        case default
            call three_point_charge_mix(Zdust,Zsigma,zmin_mix,z1,z2,z3,w1,w2,w3,used_two_point)
            Zmix = (/real(z1,dp), real(z2,dp), real(z3,dp)/)
            wmix = (/w1, w2, w3/)
        end select
        if (dbg_flag .and. nZmix /= nZmix_eff) print*,'DEBUG: nZmix out of range, clamped to',nZmix_eff,' for i_dust=',i_dust
        if (dbg_flag .and. used_two_point) print*,'DEBUG: two_point_charge_mix fallback used for i_dust=',i_dust

        ! 2. Precompute factors that do not depend on grain charge.
        do i = 1, nGroups
            pe_pref(i) = csa(i) * Np(i) * solid_angle(i)
            pdt_pref(i) = Np(i) * solid_angle(i) * local_c
        end do
        rec_pref = 2.69463707d-10 * rho_dust /dustbins_props(i_dust)%mgrain * asize_cm**2d0 * &
                   & sqrt(Tgas) * Tgas * ne

        ! 3. Loop over representative charge states and mix contributions
        pinj_mix_raw = 0.0d0
        do iq = 1, 3
            if (wmix(iq) .le. 0d0) cycle
            Zcharge = Zmix(iq)

            Emin = min_energy_ejection(Zcharge,asize_cm)
            IPV = ionisation_potential_valence(W,Zcharge,asize_cm)
            Emin_ej = min_photon_energy(IPV,Zcharge,asize_cm)

            pinj_charge_raw = 0.0d0
            E_pdt = 1d40 ! Initialize to a large value
            if (Zcharge .lt. 0d0) then
                E_pdt = photodetachment_energy(W,E_g,Zcharge,asize_cm,use_separate_refractive_index)
            end if
            do i = 1, nGroups
                if (E(i) .gt. 13.6d0) cycle ! Ignore ionising photons
                if (Np(i) .le. 0d0) cycle
                ! 3.A Compute the photoelectric yield
                call photoelectric_yield(W,E_g,Zcharge,asize_cm,E(i),l_a(i),l_e,use_separate_refractive_index,Y,y2)
                if (dbg_flag) print*,'DEBUG: i_dust=',i_dust,'i=',i,' Zdust=',Zcharge,' w=',wmix(iq),' Emin_ej=',Emin_ej,' E=',E(i),' Y=',Y,' y2=',y2

                if (Y .le. 0d0) cycle

                ! 3.B Compute the integral of the photo-electron energy distribution
                E_avg = attempting_electron_integral(E(i),Emin,Emin_ej,Zcharge,asize_cm)
                if (E_avg .le. 0d0) cycle
                E_avg = E_avg / y2

                ! 3.C Compute the injected power from photoemission of valence electrons
                pinj_charge_raw = pinj_charge_raw + Y * E_avg * pe_pref(i)
                if (dbg_flag) print*,'DEBUG: i_dust=',i_dust,'i=',i,' Zdust=',Zcharge,' w=',wmix(iq),' E=',E(i),' Y=',Y,' E_avg=',E_avg,' csa=',csa(i),' Np=',Np(i)*E(i)*eV2erg,' contrib=', &
                      & Y * E_avg * pe_pref(i) * eV2erg
                ! 3.D Compute the photo-detachment contribution (for negatively charged grains)
                if (E(i) .lt. E_pdt) cycle
                sigma_pdt = photodetachment_cross_section(E(i),E_pdt,Zcharge)
                pinj_charge_raw = pinj_charge_raw + sigma_pdt * (E(i) - E_pdt + Emin) * pdt_pref(i)
            end do
            pinj_mix_raw = pinj_mix_raw + wmix(iq) * pinj_charge_raw

            ! 4. Compute the recombination cooling rate contribution of this charge
            prec_charge = 0.0d0
            s_e = e_sticking_coeff(Zcharge,asize_cm,l_e,use_separate_refractive_index)
            if (s_e .gt. 0d0) then
                ltilde = DS87_lambda(Zcharge,-1d0,asize_cm,Tgas)
                ! NOTE: The constant prefactor is precomputed in rec_pref.
                ! pi * sqrt(8d0 * kB / pi / m_e) * kB = 2.69463707d-10 [cm**3*g/(K**(3/2)*s**3)]
                prec_charge = rec_pref * s_e * ltilde
            end if
            Prec = Prec + wmix(iq) * prec_charge

            ! 5. Compute the contribution from autoionisation if the grain has reached Zmin
            if (Zcharge .eq. zmin_mix) then
                EA = electron_afinity(W,E_g,Zcharge,asize_cm,use_separate_refractive_index)
                Jtilde = DS87_J(dble(zmin_mix),-1d0,asize_cm,Tgas)
                pinj_mix_raw = pinj_mix_raw + wmix(iq) * rec_pref * Jtilde * EA * eV2erg
            end if
        end do

        ! 5. Convert to volumetric heating rate [erg/cm^3/s]
        Pinj = Pinj + pinj_mix_raw * rho_dust / dustbins_props(i_dust)%mgrain * eV2erg
    end subroutine compute_dust_peh_rate

    subroutine interpolate_dust_peh_rate(i_dust,rho_dust,G0,ne,Tgas,Pinj,Prec)
        ! This subroutine interpolates the dust photoelectric heating and 
        ! electron recombination cooling rates from the equilibrium tables
        ! computed in Rodriguez Montero et al. (2024) based on the modelling
        ! of Weingartner & Draine (2001)
        ! i_dust   --> index of the dust species
        ! rho_dust --> dust mass density [g/cm^3]
        ! G0       --> radiation field in Habing units
        ! ne       --> electron number density [cm^-3]
        ! Tgas     --> gas temperature in K
        ! Pinj     <-- photoelectric heating rate [erg/cm^3/s]
        ! Prec     <-- photoelectric recombination cooling rate [erg/cm^3/s]
        use amr_commons, only: myid
        implicit none

        ! Inputs
        integer, intent(in) :: i_dust
        real(dp), intent(in) :: rho_dust
        real(dp), intent(in) :: G0,ne,Tgas
        ! Outputs
        real(dp), intent(out) :: Pinj,Prec

        ! Local
        real(dp) :: gamma,log_gamma,log_T,peh_rate,cool_rate
        real(dp) :: ngrains
        integer :: ngamma, nT

        ! 1. Compute the ionisation parameter
        gamma = G0 * sqrt(Tgas) / ne
        log_gamma = log10(gamma)
        log_T = log10(Tgas)

        ! 2. Make the interpolation in log-log in 2D using the per-bin tables
        if ((.not. dustbins_props(i_dust)%peh_tab%initialised) .or. &
            (.not. dustbins_props(i_dust)%rec_tab%initialised)) then
            if (myid.eq.1) then
                write(*,*) 'ERROR in interpolate_dust_peh_rate'
                write(*,*) 'peh_tab/rec_tab are not initialised for dust bin ', i_dust
            end if
            call clean_stop
        end if

        ngamma = dustbins_props(i_dust)%peh_tab%npts(1)
        nT = dustbins_props(i_dust)%peh_tab%npts(2)
        call interpolate2D(dustbins_props(i_dust)%peh_tab%tab1d(1:ngamma,1), &
                           dustbins_props(i_dust)%peh_tab%tab1d(1:nT,2), &
                           dustbins_props(i_dust)%peh_tab%tab2d(1:ngamma,1:nT,1), &
                           ngamma,nT,log_gamma,log_T,peh_rate)
        peh_rate = 10d0**peh_rate

        ngamma = dustbins_props(i_dust)%rec_tab%npts(1)
        nT = dustbins_props(i_dust)%rec_tab%npts(2)
        call interpolate2D(dustbins_props(i_dust)%rec_tab%tab1d(1:ngamma,1), &
                           dustbins_props(i_dust)%rec_tab%tab1d(1:nT,2), &
                           dustbins_props(i_dust)%rec_tab%tab2d(1:ngamma,1:nT,1), &
                           ngamma,nT,log_gamma,log_T,cool_rate)
        cool_rate = 10d0**cool_rate

        ! 3. Convert to volumetric rates
        ngrains = rho_dust / dustbins_props(i_dust)%mgrain
        Pinj = peh_rate  * ngrains * (G0 / 1.13d0) ! [erg/cm^3/s]
        Prec = cool_rate * ngrains  ! [erg/cm^3/s]
    end subroutine interpolate_dust_peh_rate
    

end module dust_photoelectric_heating

module pah_photoelectric_heating

    use amr_parameters, only:ndim
    use dust_commons
    use constants, only: pi, eV2erg, e2instatC
    use dust_utils, only: interpolate1D

    implicit none
    private   ! default
    public :: compute_pah_peh_equilibrium,interpolate_pah_peh_equilibrium
    public ::compute_pah_charge_equilibrium,interpolate_pah_charge_equilibrium

    ! ====== CONSTANTS ======
    ! Fitting parameters for the neutral PAH electron attachment
    ! obtained in the Carelli et al. (2013) experiment
    real(dp), parameter :: Carelli13_a = 2.74d-9, Carelli13_b = 0.11d0, Carelli13_c = -1.12d0

    ! Fraction of the energy that, after ionisation, goes into the kinetic energy of the electron
    ! Value estimated from the experiments with coronene by Bréchignac et al. (2014)
    real(dp), parameter :: partition_coeff = 0.46d0

    contains

    function ionisation_potential(Z, a) result(IP)
        ! This function computes the ionisation potential of a PAH
        ! molecule following the empiricial formalism of Weingartner and Draine (2001)
        ! with the updated parameters from Wenzel et al. (2020)
        ! Z --> charge of the PAH molecule (Z=-1 for anion PAHs)
        ! a --> size of the PAH molecule in cm
        ! IP <-- ionisation potential in eV
        implicit none
        integer, intent(in) :: Z
        real(dp), intent(in) :: a
        real(dp) :: IP

        if (Z == -1) then
            IP = 6.0d0
        else
            IP = 3.9d0 + e2instatC / a * ((Z + 0.5d0) + (Z + 2d0) * (3d-9 / a)) / eV2erg
        end if
    end function ionisation_potential

    function beta_factor(Nc) result(beta)
        ! Beta correction for the cation ionisation yield as obtained by
        ! Wenzel et al. (2020)
        ! Nc --> number of carbon atoms in the PAH molecule
        ! beta <-- correction factor
        implicit none
        integer, intent(in) :: Nc
        real(dp) :: beta

        if (Nc >= 32 .and. Nc < 50) then
            beta = 0.59d0 + 8.1d-3 * dble(Nc)
        else
            beta = 1.0d0
        end if
    end function beta_factor

    function ionisation_yield(Nc, Z, photon_energy, IP) result(Y)
        ! PAH molecule ionisation yields
        ! Nc --> number of carbon atoms in the PAH molecule
        ! Z  --> charge of the PAH molecule (Z=-1 for anion PAHs)
        ! photon_energy --> energy of the incident photon in eV
        ! IP --> ionisation potential in eV
        ! Y <-- ionisation yield
        implicit none
        integer, intent(in) :: Nc, Z
        real(dp), intent(in) :: photon_energy, IP
        real(dp) :: Y, beta

        select case (Z)
        case (-1)
            if (photon_energy < IP) then
                Y = 0.0d0
            else
                Y = 1.0d0
            end if

        case (0)
            if (photon_energy < IP) then
                Y = 0.0d0
            else if (photon_energy <= IP + 9.2d0) then
                Y = (photon_energy - IP) / 9.2d0
            else
                Y = 1.0d0
            end if

        case (1)
            if (photon_energy < IP) then
                Y = 0.0d0
            else if (photon_energy <= 11.3d0) then
                Y = 0.3d0 * (photon_energy - IP) / (11.3d0 - IP)
            else if (photon_energy < 12.9d0) then
                Y = 0.3d0
            else if (photon_energy < 15.0d0) then
                beta = beta_factor(Nc)
                Y = ((beta - 0.3d0) / 2.1d0) * (photon_energy - 12.9d0) + 0.3d0
            else
                Y = beta_factor(Nc)
            end if

        case (2)
            Y = 0.0d0

        case default
            Y = 0.0d0
        end select
    end function ionisation_yield

    function recombination_rate_Spitzer(Nc, Z, T) result(k_rec)
        ! Recombination rate following the Spitzer's formalism (Spitzer 2004) modified
        ! for cations by Verstraete et al. (1990) and extended to Z>0 by Berne et al. (2022)
        ! Nc --> number of carbon atoms in the PAH molecule
        ! Z  --> charge of the PAH molecule (does not apply for anions)
        ! T  --> gas temperature in Kelvin
        ! k_rec <-- recombination rate in [cm3/s]

        implicit none
        integer, intent(in) :: Nc, Z
        real(dp), intent(in) :: T
        real(dp) :: phi, k_rec

        phi = 1.85d5 / T / sqrt(dble(Nc))
        k_rec = 1.28d-10 * dble(Nc) * sqrt(T) * (1.0d0 + phi * (1.0d0 + dble(Z)))
    end function recombination_rate_Spitzer

    function recombination_rate_Tielens21(Nc, T) result(k_rec)
        ! Recombination rate following Eq. 8.106 in Tielens (2021), which assumes
        ! a correction factor the the planar geometry of the PAH.
        ! Nc --> number of carbon atoms in the PAH molecule
        ! T  --> gas temperature in Kelvin
        ! k_rec <-- recombination rate in [cm3/s]
        implicit none
        integer, intent(in) :: Nc
        real(dp), intent(in) :: T
        real(dp) :: k_rec

        k_rec = 1.3d-6 * sqrt(dble(Nc)) * sqrt(300.0d0 / T)
    end function recombination_rate_Tielens21

    function attachment_rate_Carelli13(T) result(k_att)
        ! Electron attachment rate to neutral PAH as obtained 
        ! experimentally for small PAHs in Carelli et al. (2013)
        ! T --> gas temperature in Kelvin
        ! k_att <-- attachment rate in [cm3/s]
        implicit none
        real(dp), intent(in) :: T
        real(dp) :: k_att
        
        k_att = Carelli13_a * (T / 300.0d0)**Carelli13_b * exp(-Carelli13_c / T)
    end function attachment_rate_Carelli13

    function attachment_rate_Tielens05(Nc) result(k_att)
        ! Electron attachment rate to neutral PAHs as given
        ! by Tielens (2005)
        implicit none
        integer, intent(in) :: Nc
        real(dp) :: k_att
        real(dp), parameter :: s_e = 1.0d0

        k_att = 1.3d-7 * s_e * sqrt(dble(Nc))
    end function attachment_rate_Tielens05

    function ionisation_rate(IP,sigma_ion,Np,E) result(k_pe)
        ! This function computes the ionisation rate of a PAH molecule
        ! for a given photon energy E and the number density flux
        ! IP --> ionisation potential in eV
        ! sigma_ion --> ionisation cross section in cm3/s
        ! Np --> number density of photons in #/cm3
        ! E  --> photon energy in eV
        ! k_pe <-- ionisation rate in [1/s]
        implicit none
        real(dp), intent(in) :: IP, sigma_ion, Np, E
        real(dp) :: k_pe

        if (E.ge.IP) then
            ! If the photon energy is larger than the ionisation potential,
            ! we can compute the ionisation rate
            k_pe = sigma_ion * Np
        else
            ! If the photon energy is lower than the ionisation potential,
            ! the ionisation rate is zero
            k_pe = 0.0d0
        end if
    end function ionisation_rate

    function power_absorbed(sigma_abs,F_pe) result(P_abs)
        ! This function computes the power absorbed by a PAH molecule
        ! given the absorption cross section and the flux of photons
        ! sigma_abs --> absorption cross section in cm3/s
        ! F_pe --> flux of photons in erg/cm3
        ! P_abs <-- power absorbed in erg/s
        implicit none
        real(dp), intent(in) :: sigma_abs, F_pe
        real(dp) :: P_abs

        P_abs = sigma_abs * F_pe
    end function power_absorbed

    function power_injected(sigma_ion,IP,E,Np) result(P_inj)
        ! This function computes the power injected into the gas by the ionisation
        ! of a PAH molecule by a photon with energy E
        ! sigma_ion --> ionisation cross section in cm3/s
        ! IP --> ionisation potential in eV
        ! E  --> photon energy in eV
        ! Np --> number density of photons in #/cm3
        ! P_inj <-- power injected in erg/s
        implicit none
        real(dp), intent(in) :: sigma_ion, IP, E, Np
        real(dp) :: P_inj

        if (E.ge.IP) then
            P_inj = sigma_ion * Np * (E - IP) * eV2erg
        else
            P_inj = 0.0d0
        end if
    end function power_injected

    subroutine compute_pah_charge_equilibrium(i_pah,csa_anion,csa_neutral,&
                                              csa_cation,csa_dication,nGroups,&
                                              solid_angle,Np,E,local_c,Tgas,ne,fcharge_pahs)
        ! This subroutine computes only the equilibrium PAH charge distribution.
        ! It uses the same rate model as compute_pah_peh_equilibrium but skips
        ! absorbed/injected/radiative/recombination power calculations.
        ! i_pah         --> index of the PAH bin
        ! csa_anion     --> absorption cross section for anion PAHs in cm3/s
        ! csa_neutral   --> absorption cross section for neutral PAHs in cm3/s
        ! csa_cation    --> absorption cross section for cation PAHs in cm3/s
        ! csa_dication  --> absorption cross section for dication PAHs in cm3/s
        ! nGroups       --> number of energy groups
        ! solid_angle    --> solid angle subtended by the radiation field in each group in sr
        ! Np            --> number density of photons in #/cm3
        ! E             --> photon energy in eV
        ! local_c       --> local speed of light in cm/s
        ! Tgas          --> gas temperature in K
        ! ne            --> electron density in #/cm3
        ! fcharge_pahs  <-- charge distribution of PAHs
        implicit none
        integer, intent(in) :: i_pah,nGroups
        real(dp), intent(in) :: local_c, Tgas, ne
        real(dp), dimension(1:nGroups), intent(in) :: Np, E, solid_angle
        real(dp), dimension(1:nGroups), intent(in) :: csa_anion, csa_neutral, csa_cation, csa_dication
        real(dp), dimension(:), intent(out) :: fcharge_pahs

        integer :: i, Nc, nstates
        real(dp) :: a_pah, rad_ani
        real(dp) :: k_det, k_att, k_pe_0, k_pe_1, k_rec_1, k_rec_2
        real(dp) :: IP_anion, IP_neutral, IP_cation
        real(dp) :: f_anion, f_neutral, f_1, f_2, f_total
        real(dp),dimension(1:nGroups) :: yield_anion, yield_neutral, yield_cation

        if (all(Np.eq.0d0)) return

        Nc = pahbins_props(i_pah)%nc
        a_pah = pahbins_props(i_pah)%apah_cm

        IP_anion = ionisation_potential(-1, a_pah)
        k_det = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle
            yield_anion(i) = ionisation_yield(Nc,-1,E(i),IP_anion)
            k_det = k_det + ionisation_rate(IP_anion,solid_angle(i)*yield_anion(i)*csa_anion(i), &
                                            Np(i),E(i))
        end do

        if (trim(peh_attach_model).eq.'Berne') then
            k_att = attachment_rate_Carelli13(Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_att = attachment_rate_Tielens05(Nc)
        else
            write(*,*) 'Error: Unknown electron attachment model for PAHs!'
            stop
        end if

        IP_neutral = ionisation_potential(0, a_pah)
        k_pe_0 = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle
            yield_neutral(i) = ionisation_yield(Nc,0,E(i),IP_neutral)
            k_pe_0 = k_pe_0 + ionisation_rate(IP_neutral,solid_angle(i)*yield_neutral(i)*csa_neutral(i), &
                                              Np(i),E(i))
        end do

        if (trim(peh_attach_model).eq.'Berne') then
            k_rec_1 = recombination_rate_Spitzer(Nc,0,Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_rec_1 = recombination_rate_Tielens21(Nc,Tgas)
        else
            write(*,*) 'Error: Unknown recombination model for PAHs!'
            stop
        end if

        if (trim(peh_attach_model).eq.'Berne') then
            k_rec_2 = recombination_rate_Spitzer(Nc,1,Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_rec_2 = recombination_rate_Tielens21(Nc,Tgas)
        else
            write(*,*) 'Error: Unknown recombination model for PAHs!'
            stop
        end if

        IP_cation = ionisation_potential(1, a_pah)
        k_pe_1 = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle
            yield_cation(i) = ionisation_yield(Nc,1,E(i),IP_cation)
            k_pe_1 = k_pe_1 + ionisation_rate(IP_cation,solid_angle(i)*yield_cation(i)*csa_cation(i), &
                                              Np(i),E(i))
        end do

        f_anion = 1d0 / (1d0 + k_det / (k_att*ne) + &
                    k_det * k_pe_0 / (k_att*k_rec_1*ne**2d0) + &
                    k_det * k_pe_0 * k_pe_1 / (k_att*k_rec_1*k_rec_2*ne**3d0))

        f_neutral = 1d0 / (1d0 + k_att*ne / k_det + k_pe_0 / (k_rec_1*ne) + &
                        k_pe_0 * k_pe_1 / (k_rec_1*k_rec_2*ne**2d0))

        f_1 = 1d0 / (1d0 + k_rec_1*ne / k_pe_0 + k_pe_1 / (k_rec_2*ne) + &
                    k_att*k_rec_1*ne**2d0 / (k_det*k_pe_0))

        f_2 = 1d0 / (1d0 + k_rec_2*ne / k_pe_1 + k_rec_1*k_rec_2*ne**2d0 / (k_pe_0*k_pe_1) + &
                    k_att*k_rec_1*k_rec_2*ne**3d0/(k_det*k_pe_0*k_pe_0))

        f_total = f_anion + f_neutral + f_1 + f_2
        nstates = pahbins_props(i_pah)%ncharge_states
        fcharge_pahs(:) = 0d0
        fcharge_pahs(1) = f_anion / f_total
        fcharge_pahs(2) = f_neutral / f_total
        if (nstates >= 3) fcharge_pahs(3) = f_1 / f_total
        if (nstates >= 4) fcharge_pahs(4) = f_2 / f_total
    end subroutine compute_pah_charge_equilibrium

    subroutine compute_pah_peh_equilibrium(i_pah,rho_pah,csa_anion,csa_neutral,&
                                          csa_cation,csa_dication,nGroups,&
                                          solid_angle,Np,E,local_c,Tgas,ne,fcharge_pahs,&
                                          Pabs_pah,Pinj_pah,Prad_pah,Prec_pah)
        ! This subroutine computes the equilibrium charge distribution of PAHs
        ! based on the ionisation and recombination rates for the local
        ! conditions. From this, it determines the absorbed power and the
        ! injected power into the gas by the photo-electrons.
        ! i_pah          --> index of the PAH molecule in the array
        ! rho_pah        --> density of the PAH molecule in g/cm3
        ! csa_anion      --> absorption cross section for anion PAHs in cm2
        ! csa_neutral    --> absorption cross section for neutral PAHs in cm2
        ! csa_cation     --> absorption cross section for cation PAHs in cm2
        ! csa_dication   --> absorption cross section for dication PAHs in cm2
        ! nGroups        --> number of radiation groups
        ! solid_angle    --> solid angle subtended by the radiation field in each group in sr
        ! Np             --> number density flux of photons in #/cm2/s
        ! E              --> photon energy in eV (for each group)
        ! Tgas           --> gas temperature in Kelvin
        ! ne             --> electron number density in cm-3
        ! fcharge_pahs   <-- allocatable/assumed-shape vector with the
        !                   fraction of PAH mass in each charge state
        ! Pabs_pah       <-- absorbed power by the PAH molecules in erg/cm3/s
        ! Pinj_pah       <-- injected power into the gas by the PAH molecules in erg/cm3/s
        ! Prad_pah       <-- radiative cooling power of the PAH molecules into IR in erg/cm3/s
        ! Prec_pah       <-- recombination cooling power of the PAH molecules in erg/cm3/s

        use constants, only: kB
        implicit none
        integer, intent(in) :: i_pah,nGroups
        real(dp), intent(in) :: rho_pah, local_c, Tgas, ne
        real(dp), dimension(1:nGroups), intent(in) :: Np, E, solid_angle
        real(dp), dimension(1:nGroups), intent(in) :: csa_anion, csa_neutral, csa_cation, csa_dication
        real(dp), dimension(:), intent(out) :: fcharge_pahs
        real(dp), dimension(1:nGroups) :: Pabs_pah
        real(dp) :: Pinj_pah, Prad_pah, Prec_pah

        integer :: i, Nc, nstates
        real(dp) :: a_pah, yield, F_pe, rad_ani, nmolecules
        real(dp) :: k_det, k_att, k_pe_0, k_pe_1, k_rec_1, k_rec_2
        real(dp) :: IP_anion, IP_neutral, IP_cation
        real(dp) :: f_anion, f_neutral, f_1, f_2, f_total
        real(dp) :: f_cat, f_dicat
        real(dp) :: Pinj_local, Prad_local, Prec_local
        real(dp),dimension(1:nGroups) :: Pabs_local
        real(dp),dimension(1:nGroups) :: yield_anion, yield_neutral, yield_cation

        if (all(Np.eq.0d0)) return

        ! 1. Get the PAH molecule details from per-bin properties
        Nc = pahbins_props(i_pah)%nc
        a_pah = pahbins_props(i_pah)%apah
        nmolecules = rho_pah / pahbins_props(i_pah)%mpah ! Number of molecules in the cell [#/cm3]

        ! 2. Compute the electron detachment rate for the anion for each group
        IP_anion = ionisation_potential(-1, a_pah)
        k_det = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle ! Ignore ionising photons
            yield_anion(i) = ionisation_yield(Nc,-1,E(i),IP_anion)
            k_det = k_det + ionisation_rate(IP_anion,solid_angle(i)*yield_anion(i)*csa_anion(i),&
                                            Np(i),E(i))
        end do

        ! 3. Compute the electron attachment rate for the neutral PAH
        if (trim(peh_attach_model).eq.'Berne') then
            k_att = attachment_rate_Carelli13(Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_att = attachment_rate_Tielens05(Nc)
        else
            write(*,*) 'Error: Unknown electron attachment model for PAHs!'
            stop
        end if

        ! 4. Ionisation rate for neutral PAH
        IP_neutral = ionisation_potential(0, a_pah)
        k_pe_0 = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle ! Ignore ionising photons
            yield_neutral(i) = ionisation_yield(Nc,0,E(i),IP_neutral)
            k_pe_0 = k_pe_0 + ionisation_rate(IP_neutral,solid_angle(i)*yield_neutral(i)*csa_neutral(i),&
                                                Np(i),E(i))
        end do

        ! 5. Recombination rate for cation PAH
        if (trim(peh_attach_model).eq.'Berne') then
            k_rec_1 = recombination_rate_Spitzer(Nc,0,Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_rec_1 = recombination_rate_Tielens21(Nc,Tgas)
        else
            write(*,*) 'Error: Unknown recombination model for PAHs!'
            stop
        end if

        ! 6. Recombination rate for dication PAH
        if (trim(peh_attach_model).eq.'Berne') then
            k_rec_2 = recombination_rate_Spitzer(Nc,1,Tgas)
        else if (trim(peh_attach_model).eq.'Tielens') then
            k_rec_2 = recombination_rate_Tielens21(Nc,Tgas)
        else
            write(*,*) 'Error: Unknown recombination model for PAHs!'
            stop
        end if

        ! 7. Ionisation rate for cation PAH
        IP_cation = ionisation_potential(1, a_pah)
        k_pe_1 = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle ! Ignore ionising photons
            yield_cation(i) = ionisation_yield(Nc,1,E(i),IP_cation)
            k_pe_1 = k_pe_1 + ionisation_rate(IP_cation,solid_angle(i)*yield_cation(i)*csa_cation(i),&
                                                Np(i),E(i))
        end do

        ! 8. Compute the equilibrium charge distribution
        f_anion = 1d0 / (1d0 + k_det / (k_att*ne) + &
                    k_det * k_pe_0 / (k_att*k_rec_1*ne**2d0) + &
                    k_det * k_pe_0 * k_pe_1 / (k_att*k_rec_1*k_rec_2*ne**3d0))

        f_neutral = 1d0 / (1d0 + k_att*ne / k_det + k_pe_0 / (k_rec_1*ne) + &
                        k_pe_0 * k_pe_1 / (k_rec_1*k_rec_2*ne**2d0))
        
        f_1 = 1d0 / (1d0 + k_rec_1*ne / k_pe_0 + k_pe_1 / (k_rec_2*ne) + &
                    k_att*k_rec_1*ne**2d0 / (k_det*k_pe_0))
        
        f_2 = 1d0 / (1d0 + k_rec_2*ne / k_pe_1 + k_rec_1*k_rec_2*ne**2d0 / (k_pe_0*k_pe_1) + &
                    k_att*k_rec_1*k_rec_2*ne**3d0/(k_det*k_pe_0*k_pe_0))

        ! 9. Normalise the fractions
        f_total = f_anion + f_neutral + f_1 + f_2
        nstates = pahbins_props(i_pah)%ncharge_states
        fcharge_pahs(:) = 0d0
        fcharge_pahs(1) = f_anion / f_total
        fcharge_pahs(2) = f_neutral / f_total
        if (nstates >= 3) fcharge_pahs(3) = f_1 / f_total
        if (nstates >= 4) fcharge_pahs(4) = f_2 / f_total
        f_cat = 0d0
        f_dicat = 0d0
        if (nstates >= 3) f_cat = fcharge_pahs(3)
        if (nstates >= 4) f_dicat = fcharge_pahs(4)

        ! 10. Compute the absorbed power by the PAH molecule
        Pabs_local(:) = 0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle ! Ignore ionising photons
            F_pe = Np(i) * E(i) * eV2erg ! [erg/cm3]
            Pabs_local(i) = Pabs_local(i) + power_absorbed(solid_angle(i)*csa_anion(i), F_pe) * fcharge_pahs(1)
            Pabs_local(i) = Pabs_local(i) + power_absorbed(solid_angle(i)*csa_neutral(i), F_pe) * fcharge_pahs(2)
            Pabs_local(i) = Pabs_local(i) + power_absorbed(solid_angle(i)*csa_cation(i), F_pe) * f_cat
            Pabs_local(i) = Pabs_local(i) + power_absorbed(solid_angle(i)*csa_dication(i), F_pe) * f_dicat
            Pabs_pah(i) = Pabs_pah(i) + Pabs_local(i) * nmolecules ! Convert to erg/cm3/s
        end do

        ! 11. Compute the injected power into the gas by photo-electrons
        Pinj_local = 0.0d0
        do i = 1, nGroups
            if (E(i) .gt. 13.6d0 .and. (.not.pah_pe_nolyman)) cycle ! Ignore ionising photons
            Pinj_local = Pinj_local + power_injected(solid_angle(i)*yield_anion(i)*csa_anion(i),&
                                                 IP_anion, E(i), Np(i)) * fcharge_pahs(1)
            Pinj_local = Pinj_local + partition_coeff * power_injected(solid_angle(i)*yield_neutral(i)*csa_neutral(i),&
                                                                    IP_neutral, E(i), Np(i)) * fcharge_pahs(2)
            Pinj_local = Pinj_local + partition_coeff * power_injected(solid_angle(i)*yield_cation(i)*csa_cation(i),&
                                                                    IP_cation, E(i), Np(i)) * f_cat
        end do
        Pinj_pah = Pinj_pah + Pinj_local * nmolecules ! Convert to erg/cm3/s

        ! 12. Compute the radiative cooling power of the PAH molecule
        Prad_pah = Prad_pah + max(sum(Pabs_local) - Pinj_local,0d0)

        ! 13. Compute the recombination cooling power of the PAH molecule
        Prec_local = k_att * fcharge_pahs(2) + &
                   k_rec_1 * f_cat + &
                   k_rec_2 * f_dicat
        Prec_pah = Prec_pah + Prec_local * nmolecules * ne * (1.5d0 * kB * Tgas) ! Convert to erg/cm3/s

    end subroutine compute_pah_peh_equilibrium

    subroutine interpolate_pah_charge_equilibrium(i_pah,G0,ne,Tgas,fcharge_pahs)

        ! This subroutine interpolates only the equilibrium PAH charge
        ! distribution for a given PAH molecule based on local G0, ne and Tgas.
        use amr_commons, only: myid
        implicit none
        integer, intent(in) :: i_pah
        real(dp), intent(in) :: G0, ne, Tgas
        real(dp), dimension(:), intent(out) :: fcharge_pahs

        integer :: nstates, nstates_interp, ngamma, istate
        real(dp) :: gamma, f_total

        gamma = G0 * sqrt(Tgas) / ne
        nstates = pahbins_props(i_pah)%ncharge_states
        nstates_interp = min(nstates,4)
        fcharge_pahs(:) = 0d0
        if (.not. allocated(pahbins_props(i_pah)%fcharge_tab)) then
            if (myid == 1) write(*,*) 'Error: PAH fcharge tables not allocated for PAH bin ', i_pah
            call clean_stop
        end if
        do istate = 1, nstates_interp
            if (.not. pahbins_props(i_pah)%fcharge_tab(istate)%initialised) then
                if (myid == 1) write(*,*) 'Error: PAH fcharge table not initialised for PAH bin ', i_pah, ', state ', istate
                call clean_stop
            end if
            ngamma = pahbins_props(i_pah)%fcharge_tab(istate)%npts(1)
            call interpolate1D(pahbins_props(i_pah)%fcharge_tab(istate)%tab1d(1:ngamma,1), &
                               pahbins_props(i_pah)%fcharge_tab(istate)%tab2d(1:ngamma,1,1), &
                               ngamma,log10(gamma),fcharge_pahs(istate))
        end do
        f_total = sum(fcharge_pahs(:))
        if (f_total > 0d0) fcharge_pahs(:) = fcharge_pahs(:) / f_total
    end subroutine interpolate_pah_charge_equilibrium

    subroutine interpolate_pah_peh_equilibrium(i_pah,rho_pah,G0,ne,Tgas,&
                                                fcharge_pahs,Pabs_pah,&
                                                Pinj_pah,Prad_pah,Prec_pah)

        ! This subroutine interpolates the PAH photoelectric heating equilibrium
        ! conditions for a given PAH molecule based on the local G0, ne and Tgas.
        ! i_pah          --> index of the PAH molecule in the array
        ! rho_pah        --> density of the PAH molecule in g/cm3
        ! G0             --> local G0 value (Habing units)
        ! ne             --> electron number density in cm-3
        ! Tgas           --> gas temperature in Kelvin
        ! fcharge_pahs   <-- allocatable/assumed-shape array with the fraction of PAH
        !                     mass in each charge state
        ! Pabs_pah       <-- absorbed power by the PAH molecules in erg/cm3/s
        ! Pinj_pah       <-- injected power into the gas by the PAH molecules
        !                     in erg/cm3/s
        ! Prad_pah       <-- radiative cooling power of the PAH molecules
        !                     into IR in erg/cm3/s
        ! Prec_pah       <-- recombination cooling power of the PAH molecules
        !                     in erg/cm3/s
        use amr_commons, only: myid
        use constants, only: kB
        implicit none
        integer, intent(in) :: i_pah
        real(dp), intent(in) :: rho_pah, G0, ne, Tgas
        real(dp), dimension(:), intent(out) :: fcharge_pahs
        real(dp), intent(out),optional :: Pabs_pah, Pinj_pah, Prad_pah, Prec_pah

        integer :: Nc, nstates, nstates_interp, ngamma, istate
        real(dp) :: eff,gamma,f_total
        real(dp) :: k_att, k_rec_1, k_rec_2
        real(dp) :: nmolecules
        real(dp) :: f_cat, f_dicat

        nmolecules = rho_pah / pahbins_props(i_pah)%mpah

        ! 1. Get the interpolated value for the PAH PE efficiency
        gamma = G0 * sqrt(Tgas) / ne
        if ((.not. pahbins_props(i_pah)%peh_eff_tab%initialised) .or. &
            (.not. pahbins_props(i_pah)%peh_pabs_tab%initialised)) then
            if (myid == 1) write(*,*) 'Error: PAH PEH tables not initialised for PAH bin ', i_pah
            call clean_stop
        end if

        ngamma = pahbins_props(i_pah)%peh_eff_tab%npts(1)
        call interpolate1D(pahbins_props(i_pah)%peh_eff_tab%tab1d(1:ngamma,1), &
                           pahbins_props(i_pah)%peh_eff_tab%tab2d(1:ngamma,1,1), &
                           ngamma,log10(gamma),eff)
        eff = 10.0d0**eff ! Convert back to linear scale

        ! 2. Get the interpolated value for the PAH absorption power (almost independent of gamma)
        ngamma = pahbins_props(i_pah)%peh_pabs_tab%npts(1)
        call interpolate1D(pahbins_props(i_pah)%peh_pabs_tab%tab1d(1:ngamma,1), &
                           pahbins_props(i_pah)%peh_pabs_tab%tab2d(1:ngamma,1,1), &
                           ngamma,log10(gamma),Pabs_pah)
        ! Scale by the value of G0 and the number density of PAHs
        ! NOTE: Convert G0 to the Mathis ISRF by dividing by 1.13 (see Mathis et al. 1983)
        if (present(Pabs_pah)) then
            Pabs_pah = (10d0**Pabs_pah) * nmolecules * (G0 / 1.13d0) ! [erg/cm3/s]
        end if
        ! 3. Get the injected power into the gas by the PAH molecules
        if (present(Pinj_pah)) then
            Pinj_pah = eff * Pabs_pah
        end if

        ! 4. The radiative cooling power of the PAH molecules
        !    is just the absorbed power minus the injected power
        if (present(Prad_pah)) then
            Prad_pah = max(Pabs_pah - Pinj_pah, 0d0)
        end if

        ! 6. Get the interpolated value for the PAH charges
        nstates = pahbins_props(i_pah)%ncharge_states
        nstates_interp = min(nstates,4)
        fcharge_pahs(:) = 0d0
        if (.not. allocated(pahbins_props(i_pah)%fcharge_tab)) then
            if (myid == 1) write(*,*) 'Error: PAH fcharge tables not allocated for PAH bin ', i_pah
            call clean_stop
        end if
        do istate = 1, nstates_interp
            if (.not. pahbins_props(i_pah)%fcharge_tab(istate)%initialised) then
                if (myid == 1) write(*,*) 'Error: PAH fcharge table not initialised for PAH bin ', i_pah, ', state ', istate
                call clean_stop
            end if
            ngamma = pahbins_props(i_pah)%fcharge_tab(istate)%npts(1)
            call interpolate1D(pahbins_props(i_pah)%fcharge_tab(istate)%tab1d(1:ngamma,1), &
                               pahbins_props(i_pah)%fcharge_tab(istate)%tab2d(1:ngamma,1,1), &
                               ngamma,log10(gamma),fcharge_pahs(istate))
        end do
        f_total = sum(fcharge_pahs(:))
        if (f_total > 0d0) fcharge_pahs(:) = fcharge_pahs(:) / f_total
        f_cat = 0d0
        f_dicat = 0d0
        if (nstates >= 3) f_cat = fcharge_pahs(3)
        if (nstates >= 4) f_dicat = fcharge_pahs(4)
        
        ! 7. Compute the recombination cooling power of the PAH molecules
        if (present(Prec_pah)) then
            Nc = pahbins_props(i_pah)%nc
            if (trim(peh_attach_model).eq.'Berne') then
                k_att = attachment_rate_Carelli13(Tgas)
            else if (trim(peh_attach_model).eq.'Tielens') then
                k_att = attachment_rate_Tielens05(Nc)
            else
                write(*,*) 'Error: Unknown electron attachment model for PAHs!'
                stop
            end if

            if (trim(peh_attach_model).eq.'Berne') then
                k_rec_1 = recombination_rate_Spitzer(Nc,0,Tgas)
            else if (trim(peh_attach_model).eq.'Tielens') then
                k_rec_1 = recombination_rate_Tielens21(Nc,Tgas)
            else
                write(*,*) 'Error: Unknown recombination model for PAHs!'
                stop
            end if

            if (trim(peh_attach_model).eq.'Berne') then
                k_rec_2 = recombination_rate_Spitzer(Nc,1,Tgas)
            else if (trim(peh_attach_model).eq.'Tielens') then
                k_rec_2 = recombination_rate_Tielens21(Nc,Tgas)
            else
                write(*,*) 'Error: Unknown recombination model for PAHs!'
                stop
            end if
            Prec_pah = k_att * fcharge_pahs(2) + &
                    k_rec_1 * f_cat + &
                    k_rec_2 * f_dicat
            Prec_pah = Prec_pah * ne * nmolecules * (1.5d0 * kB * Tgas)! Convert to erg/cm3/s
        end if
    end subroutine interpolate_pah_peh_equilibrium
    
end module pah_photoelectric_heating