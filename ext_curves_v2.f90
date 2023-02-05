module val
  integer,parameter :: dp=selected_real_kind(15)
  real(KIND=dp),parameter :: pi=3.14159265358979_dp
  integer,parameter :: N=2000 !integral discretization
  real(KIND=dp),parameter :: as_0=0.005_dp
  real(KIND=dp),parameter :: al_0=0.1_dp
  real(KIND=dp),parameter :: sigma=0.75_dp
  real(KIND=dp),parameter :: mu=1.4_dp
  real(KIND=dp),parameter :: mH=1.66e-24
  real(KIND=dp),parameter :: s_grains=3.0_dp
!!$  real(KIND=dp),parameter :: carb_part=0.6_dp
!!$  real(KIND=dp),parameter :: sil_part=0.4_dp
  real(KIND=dp),parameter :: carb_part=0.46_dp
  real(KIND=dp),parameter :: sil_part=0.54_dp
!!$  integer    ,  parameter :: nang=200
  integer    ,  parameter :: nang=5
  real(KIND=dp),parameter :: n_real=1.0_dp ! (real index of medium)
  real(KIND=dp),parameter :: beg=0.0005_dp !min size of grains
  real(KIND=dp),parameter :: end=2.5_dp  !max size of grains (for integral)
  integer    ,  parameter :: n_lamb=500
  integer    ,  parameter :: n_wave=166
  !real(KIND=dp),dimension(6), parameter ::eps_v=(/2.8566_dp,0.10095_dp,5.4513_dp,1.1933_dp,5.0394_dp,8.5590_dp/)  
  real(KIND=dp),dimension(6), parameter ::eps_v=(/0.69040_dp+1._dp,0.029860_dp,1.3486_dp+1._dp,&
       0.25405_dp,1.736_dp+1._dp,1.5641_dp/) 
  
  contains 
    subroutine eps_file()
      implicit none
      real(KIND=dp), dimension(5):: sil_val,cpa_val,cpe_val
      integer ::i

      open(57, file=' /data101/dubois/DiscDustGuerra/extinction/data/sil_eps.out', status='old')
      open(58, file=' /data101/dubois/DiscDustGuerra/extinction/data/car_pa01_eps.out', status='old')
      open(59, file=' /data101/dubois/DiscDustGuerra/extinction/data/car_pe01_eps.out', status='old')
      open(60, file=" /data101/dubois/DiscDustGuerra/extinction/data/eps.out", status='unknown',action='write')
      do i=1,n_wave
         read(57,*) sil_val
         read(58,*) cpa_val
         read(59,*) cpe_val
         !write(60,*) sil_val(1),sil_val(2)+1._dp,sil_val(3),cpa_val(2)+1._dp,cpa_val(3),cpe_val(2)+1._dp,cpe_val(3)
         !write(60,*) sil_val(1),sil_val(2),sil_val(3),cpa_val(2),cpa_val(3),cpe_val(2),cpe_val(3)
         write(60,*) sil_val(1),sil_val(4)+1._dp,sil_val(5),cpa_val(4)+1._dp,cpa_val(5),cpe_val(4)+1._dp,cpe_val(5)
      enddo
      close(57)
      close(58)
      close(59)
      close(60)
    end subroutine eps_file
end module val

Module Mie
Implicit None
Private
Public :: BHMIE
Contains
  Subroutine BHMIE(X,REFREL,NANG,S1,S2,QEXT,QSCA,QBACK,GSCA)
    Implicit None
    Integer,  Parameter :: wp = Selected_real_kind(15)
    Real(wp), Parameter :: one=1._wp, zero=0._wp, two=2._wp
    ! Declare parameters:
    Integer, Parameter::  MXNANG=1000,NMXX=150000
    ! Arguments:
    Integer  :: NANG
    Real(wp) :: GSCA,QBACK,QEXT,QSCA,X
    Complex(wp) :: REFREL
!!$    Complex(wp), Dimension(2*MXNANG-1) :: S1, S2
    Complex(wp), Dimension(:) :: S1, S2
    ! Local variables:
    Integer :: J,JJ,N,NSTOP,NMX,NN
    Real(wp) :: CHI,CHI0,CHI1,DANG,DX,EN,FN,P,PII,PSI,PSI0,PSI1,&
         THETA,XSTOP,YMOD
    Real(wp) ::  AMU(MXNANG),PI(MXNANG),PI0(MXNANG),PI1(MXNANG),&
         TAU(MXNANG)
    Complex(wp) ::  AN,AN1,BN,BN1,DREFRL,XI,XI1,Y
    Complex(wp), Dimension(NMXX) :: D
    !***********************************************************************
    ! Subroutine BHMIE is the Bohren-Huffman Mie scattering subroutine
    !    to calculate scattering and absorption by a homogenous isotropic
    !    sphere.
    ! Given:
    !    X = 2*pi*a/lambda
    !    REFREL = (complex refr. index of sphere)/(real index of medium)
    !    NANG = number of angles between 0 and 90 degrees
    !           (will calculate 2*NANG-1 directions from 0 to 180 deg.)
    !           if called with NANG<2, will set NANG=2 and will compute
    !           scattering for theta=0,90,180.
    ! Returns:
    !    S1(1 : 2*NANG-1) = -i*f_22 (incid. E perp. to scatt. plane,
    !                                scatt. E perp. to scatt. plane)
    !    S2(1 : 2*NANG-1) = -i*f_11 (incid. E parr. to scatt. plane,
    !                                scatt. E parr. to scatt. plane)
    !    QEXT = C_ext/pi*a**2 = efficiency factor for extinction
    !    QSCA = C_sca/pi*a**2 = efficiency factor for scattering
    !    QBACK = (dC_sca/domega)/pi*a**2
    !          = backscattering efficiency [NB: this is (1/4*pi) smaller
    !            than the "radar backscattering efficiency"; see Bohren &
    !            Huffman 1983 pp. 120-123]
    !    GSCA = <cos(theta)> for scattering
    !
    ! Original program taken from Bohren and Huffman (1983), Appendix A
    ! Modified by B.T.Draine, Princeton Univ. Obs., 90/10/26
    ! in order to compute <cos(theta)>
    ! 91/05/07 (BTD): Modified to allow NANG=1
    ! 91/08/15 (BTD): Corrected error (failure to initialize P)
    ! 91/08/15 (BTD): Modified to enhance vectorizability.
    ! 91/08/15 (BTD): Modified to make NANG=2 if called with NANG=1
    ! 91/08/15 (BTD): Changed definition of QBACK.
    ! 92/01/08 (BTD): Converted to full double precision and double complex
    !                 eliminated 2 unneed lines of code
    !                 eliminated redundant variables (e.g. APSI,APSI0)
    !                 renamed RN -> EN = double precision N
    !                 Note that DOUBLE COMPLEX and DCMPLX are not part
    !                 of f77 standard, so this version may not be fully
    !                 portable.  In event that portable version is
    !                 needed, use src/bhmie_f77.f
    ! 93/06/01 (BTD): Changed AMAX1 to generic function MAX
    !***********************************************************************
    !*** Safety checks
    If(NANG.Gt.MXNANG) Stop '***Error: NANG > MXNANG in bhmie'
    If(NANG.Lt.2)NANG=2
    !*** Obtain pi:
    PII = 4._wp*Atan(1._wp)
    DX = X
    DREFRL = REFREL
    Y = X*DREFRL
    YMOD = Abs(Y)
    !
    !*** Series expansion terminated after NSTOP terms
    !    Logarithmic derivatives calculated from NMX on down
    XSTOP = X + 4._wp*X**(one/3._wp) + two
    NMX = Max(XSTOP,YMOD) + 15
    ! BTD experiment 91/1/15: add one more term to series and compare results
    !      NMX=AMAX1(XSTOP,YMOD)+16
    ! test: compute 7001 wavelengths between .0001 and 1000 micron
    ! for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    ! computed number changed (out of 4*7001) and it only changed by 1/8387
    ! conclusion: we are indeed retaining enough terms in series!
    NSTOP = XSTOP

    If(NMX.Gt.NMXX)Then
       Write(0,*)'Error: NMX > NMXX=',NMXX,' for |m|x=',YMOD
       Stop
    Endif
    !*** Require NANG.GE.1 in order to calculate scattering intensities
    DANG = zero
    If(NANG.Gt.1)DANG=.5_wp*PII/Real(NANG-1,wp)
    Do J=1,NANG
       THETA = Real(J-1,wp)*DANG
       AMU(J) = Cos(THETA)
    End Do

    PI0(1:NANG) = zero
    PI1(1:NANG) = one

    NN=2*NANG-1 
    S1(1:NN) = (zero,zero)
    S2(1:NN) = (zero,zero)

    !*** Logarithmic derivative D(J) calculated by downward recurrence
    !    beginning with initial value (0.,0.) at J=NMX

    D(NMX)=(zero,zero)
    NN=NMX-1
    Do N=1,NN
       EN = NMX-N+1
       D(NMX-N) = (EN/Y) - (one/(D(NMX-N+1)+EN/Y))
    End Do

    !*** Riccati-Bessel functions with real argument X
    !    calculated by upward recurrence

    PSI0 = Cos(DX)
    PSI1 = Sin(DX)
    CHI0 =-Sin(DX)
    CHI1 = Cos(DX)
    XI1 = Cmplx(PSI1,-CHI1,wp)
    QSCA = zero
    GSCA = zero
    P = -one
    Do N=1,NSTOP
       EN = N
       FN = (two*EN+one)/(EN*(EN+one))
       ! for given N, PSI  = psi_n        CHI  = chi_n
       !              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
       !              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
       ! Calculate psi_n and chi_n
       PSI = (two*EN-one)*PSI1/DX-PSI0
       CHI = (two*EN-one)*CHI1/DX-CHI0
       XI=DCMPLX(PSI,-CHI)

       !*** Store previous values of AN and BN for use
       !    in computation of g=<cos(theta)>
       If(N.Gt.1)Then
          AN1 = AN
          BN1 = BN
       End If

       !*** Compute AN and BN:
       AN = (D(N)/DREFRL+EN/DX)*PSI-PSI1
       AN = AN/((D(N)/DREFRL+EN/DX)*XI-XI1)
       BN = (DREFRL*D(N)+EN/DX)*PSI-PSI1
       BN = BN/((DREFRL*D(N)+EN/DX)*XI-XI1)

       !*** Augment sums for Qsca and g=<cos(theta)>
       QSCA = QSCA+(two*EN+one)*(Abs(AN)**2+Abs(BN)**2)
       GSCA = GSCA+((two*EN+one)/(EN*(EN+one)))* &
            (Real(AN,wp)*Real(BN,wp)+IMAG(AN)*IMAG(BN))
       If(N.Gt.1)Then
          GSCA=GSCA+((EN-1.)*(EN+1.)/EN)* &
               (Real(AN1,wp)*Real(AN,wp)+Aimag(AN1)*Aimag(AN)+ &
               Real(BN1,wp)*Real(BN,wp)+Aimag(BN1)*Aimag(BN))
       End If

       !*** Now calculate scattering intensity pattern
       !    First do angles from 0 to 90
       Do J=1,NANG
          JJ = 2*NANG-J
          PI(J) = PI1(J)
          TAU(J) = EN*AMU(J)*PI(J)-(EN+one)*PI0(J)
          S1(J) = S1(J) + FN*(AN*PI(J)+BN*TAU(J))
          S2(J) = S2(J) + FN*(AN*TAU(J)+BN*PI(J))
       End Do

       !*** Now do angles greater than 90 using PI and TAU from
       !    angles less than 90.
       !    P=1 for N=1,3,...; P=-1 for N=2,4,...
       P = -P
       Do J=1,NANG-1
          JJ = 2*NANG-J
          S1(JJ) = S1(JJ)+FN*P*(AN*PI(J)-BN*TAU(J))
          S2(JJ) = S2(JJ)+FN*P*(BN*PI(J)-AN*TAU(J))
       End Do
       PSI0 = PSI1
       PSI1 = PSI
       CHI0 = CHI1
       CHI1 = CHI
       XI1 = Cmplx(PSI1,-CHI1,wp)

       !*** Compute pi_n for next value of n
       !    For each angle J, compute pi_n+1
       !    from PI = pi_n , PI0 = pi_n-1
       Do J=1,NANG
          PI1(J) = ((two*EN+one)*AMU(J)*PI(J)-(EN+one)*PI0(J))/EN
          PI0(J) = PI(J)
       End Do
    End Do

    !*** Have summed sufficient terms.
    !    Now compute QSCA,QEXT,QBACK,and GSCA
    GSCA = two*GSCA/QSCA
    QSCA = (two/(DX*DX))*QSCA
    QEXT = (4._wp/(DX*DX))*Real(S1(1),wp)
    QBACK = (Abs(S1(2*NANG-1))/DX)**2/PII
  End Subroutine BHMIE
End Module Mie

module math
  use val
  use Mie
  implicit none
  contains

  function dist(Cs,Cl,a)
    real(KIND=dp),intent(in)::Cs,Cl,a
    real(KIND=dp) :: n_small,n_large
    real(KIND=dp) :: dist
    n_small=Cs/a**4*exp(-(log(a/as_0))**2./(2._dp*sigma**2))
    n_large=Cl/a**4*exp(-(log(a/al_0))**2./(2._dp*sigma**2))
    dist=(n_small+n_large)
  end function dist

  subroutine integral(Cs,Cl,eps,lambda,chem_clas,I)
    integer, intent(in) :: chem_clas
    real(KIND=dp), intent(in)  :: lambda,Cs,Cl
    real(KIND=dp), dimension(6),intent(in):: eps !eps_sil1,eps_sil2,eps_Cpa1,eps_Cpa2,eps_Cpe1,eps_Cpe2 
    complex(dp)  :: refrel,refrelpar,refrelper
    real(KIND=dp), intent(out) :: I
    real(KIND=dp) :: h
    integer :: j
    real(KIND=dp) :: GSCA,QBACK,QEXT,QEXTpar,QEXTper,QSCA,X,a,Xs,Xl
    complex(dp), dimension(2*nang-1) :: S1, S2

    open(71, file=" /data101/dubois/DiscDustGuerra/extinction/data/qext_carb_0005.dat", status='unknown')
    open(72, file=" /data101/dubois/DiscDustGuerra/extinction/data/qext_carb_01.dat", status='unknown')
    open(73, file=" /data101/dubois/DiscDustGuerra/extinction/data/qext_carb_2000ang.dat", status='unknown')
    open(74, file=" /data101/dubois/DiscDustGuerra/extinction/data/qext_sil_0005.dat", status='unknown')
    open(75, file=" /data101/dubois/DiscDustGuerra/extinction/data/qext_sil_01.dat", status='unknown')
    
    Xl=2*pi*al_0/lambda
    Xs=2*pi*as_0/lambda
    if (chem_clas==0) then !carboneus       
       
       refrelpar=dcmplx(eps(3),eps(4))
       refrelper=dcmplx(eps(5),eps(6))
       
       call BHMIE(Xs,refrelpar,nang,S1,S2,QEXTpar,QSCA,QBACK,GSCA)
       call BHMIE(Xs,refrelper,nang,S1,S2,QEXTper,QSCA,QBACK,GSCA)
       QEXT=(QEXTpar+2.0_dp*QEXTper)/3.0_dp
       write(71,*) lambda, QEXTpar ,QEXTper
       call BHMIE(Xl,refrelpar,nang,S1,S2,QEXTpar,QSCA,QBACK,GSCA)
       call BHMIE(Xl,refrelper,nang,S1,S2,QEXTper,QSCA,QBACK,GSCA)
       QEXT=(QEXTpar+2.0_dp*QEXTper)/3.0_dp
       write(72,*) lambda, QEXTpar ,QEXTper

       
       X=2*pi*beg/lambda       
       ! refrelpar=sqrt(dcmplx(eps(3),eps(4))/n)
       ! refrelper=sqrt(dcmplx(eps(5),eps(6))/n)
     
       call BHMIE(X,refrelpar,nang,S1,S2,QEXTpar,QSCA,QBACK,GSCA)
       call BHMIE(X,refrelper,nang,S1,S2,QEXTper,QSCA,QBACK,GSCA)
       QEXT=(QEXTpar+2.0_dp*QEXTper)/3.0_dp
       
       if (lambda>1.99 .and. lambda<2.1) write(73,*) beg, QEXTpar ,QEXTper

       I=0.5*dist(Cs,Cl,beg)*pi*beg**2*QEXT
       !I=0.5*beg**(-3.5)*pi*beg**2*QEXT

       h=(end-beg)/N
       do j=1,N-1
          a=beg+j*h
          X=2*pi*a/lambda
          call BHMIE(X,refrelpar,nang,S1,S2,QEXTpar,QSCA,QBACK,GSCA)
          call BHMIE(X,refrelper,nang,S1,S2,QEXTper,QSCA,QBACK,GSCA)
          QEXT=(QEXTpar+2.0_dp*QEXTper)/3.0_dp
          if (lambda>1.99 .and. lambda<2.1) write(73,*) a, QEXTpar ,QEXTper

          I=I+dist(Cs,Cl,a)*pi*a**2*QEXT
          !I=I+a**(-3.5)*pi*a**2*QEXT
       enddo

       X=2*pi*end/lambda
       call BHMIE(X,refrelpar,nang,S1,S2,QEXTpar,QSCA,QBACK,GSCA)
       call BHMIE(X,refrelper,nang,S1,S2,QEXTper,QSCA,QBACK,GSCA)
       QEXT=(QEXTpar+2.0_dp*QEXTper)/3.0_dp
       if (lambda>1.99 .and. lambda<2.1) write(73,*) a, QEXTpar ,QEXTper

       I=I+0.5*dist(Cs,Cl,end)*pi*end**2*QEXT   
       !I=I+0.5*end**(-3.5)*pi*end**2*QEXT 
       I=I*h
       

    else if (chem_clas==1) then !silicate

       refrel=dcmplx(eps(1),eps(2))
       call BHMIE(Xs,refrel,nang,S1,S2,QEXT,QSCA,QBACK,GSCA)
       write(74,*) lambda, QEXT
       call BHMIE(Xl,refrel,nang,S1,S2,QEXT,QSCA,QBACK,GSCA)        
       write(75,*) lambda, QEXT
       
       X=2*pi*beg/lambda
       !refrel=sqrt(dcmplx(eps(1),eps(2))/n)       
       call BHMIE(X,refrel,nang,S1,S2,QEXT,QSCA,QBACK,GSCA)
       I=0.5*dist(Cs,Cl,beg)*pi*beg**2*QEXT
       !I=0.5*beg**(-3.5)*pi*beg**2*QEXT      

       h=(end-beg)/N
       do j=1,N-1
          a=beg+j*h
          X=2*pi*a/lambda
          call BHMIE(X,refrel,nang,S1,S2,QEXT,QSCA,QBACK,GSCA)          
          I=I+dist(Cs,Cl,a)*pi*a**2*QEXT
          !I=I+a**(-3.5)*pi*a**2*QEXT          
       enddo

       X=2*pi*end/lambda
       call BHMIE(X,refrel,nang,S1,S2,QEXT,QSCA,QBACK,GSCA)
       I=I+0.5*dist(Cs,Cl,end)*pi*end**2*QEXT   
       !I=I+0.5*end**(-3.5)*pi*end**2*QEXT
       
       I=I*h
      
    endif

  end subroutine integral

  subroutine calc_norm(dtg_s,dtg_l,Cs,Cl)
    real(KIND=dp), intent(in)  :: dtg_s,dtg_l
    real(KIND=dp), intent(out) :: Cs,Cl
    real(KIND=dp) :: h,Is,Il,a
    integer :: j

    Is=0.5*4._dp/3._dp*pi/beg*s_grains*exp(-(log(beg/as_0))**2/(2*sigma**2))
    Il=0.5*4._dp/3._dp*pi/beg*s_grains*exp(-(log(beg/al_0))**2/(2*sigma**2))

    h=(end-beg)/N
    do j=1,N-1
       a=beg+j*h
       Is=Is+4._dp/3._dp*pi/a*s_grains*exp(-(log(a/as_0))**2/(2*sigma**2))
       Il=Il+4._dp/3._dp*pi/a*s_grains*exp(-(log(a/al_0))**2/(2*sigma**2))
    enddo
    
    Is=Is+0.5*4._dp/3._dp*pi/end*s_grains*exp(-(log(end/as_0))**2/(2*sigma**2))
    Il=Il+0.5*4._dp/3._dp*pi/end*s_grains*exp(-(log(end/al_0))**2/(2*sigma**2))

    Cs=mu*mH*dtg_s/(Is*h)
    Cl=mu*mH*dtg_l/(Il*h)
  end subroutine calc_norm
end module math

program extinction_curves  
  use Mie
  use val
  use math
  implicit none

  real(KIND=dp) :: lambda,I_carb,I_sil,I_carb_v,I_sil_v,test
  real(KIND=dp) :: I_carb_s,I_sil_s,I_carb_l,I_sil_l
  real(KIND=dp) :: A_v_dif,A_lambda_dif,A_v_den,A_lambda_den
  real(KIND=dp),allocatable, dimension(:) :: A_v,A_lambda
  real(KIND=dp) :: lambda_v=0.5500621_dp
  real(KIND=dp) :: dtg_s,dtg_l,dtg_s_dif,dtg_l_dif,dtg_s_den,dtg_l_den
  real(KIND=dp) :: dtg_C_s,dtg_C_l,dtg_Sil_s,dtg_Sil_l
  real(KIND=dp) :: time,Cs,Cl,Cs_dif,Cl_dif,Cs_den,Cl_den
  real(KIND=dp) :: Cs_C,Cl_C,Cs_Sil,Cl_Sil
  !real(KIND=dp) :: carb_part, sil_part
  integer :: i,j,n_rays
  real(KIND=dp), dimension(6):: eps
  integer :: io
  real(KIND=dp), dimension(:), allocatable :: wave,eps1,eps2
  real,dimension(100)::tt
  logical::time_this=.false.
  real(KIND=dp) :: mgas,dummy
  integer :: icount
  character(LEN=500) :: inputfile,outputfile

  n_rays=0
  open(1,file='params.txt')
  do
     read(1,*,iostat=io)
     if(io < 0)exit
     n_rays=n_rays+1
  enddo
  close(1)
  write(*,*) 'Number of rays :', n_rays

!!$  open(1,file='params.txt')
!!$  read(1,*,iostat=io)inputfile,outputfile
!!$  close(1)
!!$  write(*,*)TRIM(inputfile)
!!$  write(*,*)TRIM(outputfile)
!!$  n_rays=1
!!$  allocate(A_v(n_rays))
!!$  allocate(A_lambda(n_rays))
  
  ! open(60, file=" /data101/dubois/DiscDustGuerra/extinction/data/eps.out", status='old')
  ! open(56, file=' /data101/dubois/DiscDustGuerra/extinction/data/dtg_Zsun_rays_1d12.txt', status='old')
  ! read(56,*) time, dtg_s, dtg_l
  ! close(56)
  ! call calc_norm(dtg_s,dtg_l,Cs,Cl)
  ! do i=1,n_wave
  !    read(60,*) lambda,eps 
  !    write(*,*) i, '/', n_wave
  !    call integral(Cs,Cl,eps,lambda,0,I_carb) !carboneus
  !    call integral(Cs,Cl,eps,lambda,1,I_sil)  !silicate
  ! enddo
  ! close(60)
  
  !call eps_file()
  !stop
 
!!$  open(70, file=" /data101/dubois/DiscDustGuerra/extinction/data/data_extinction_curve_1d12_rays_v2.dat", status="unknown")

  open(1,file='params.txt')
  do j=1,n_rays
     read(1,*,iostat=io)inputfile,outputfile
     write(*,*)TRIM(inputfile)
     write(*,*)TRIM(outputfile)
     
     open(56, file=inputfile, status='old')
     read(56,*) time, mgas, dummy, dummy, dummy, dummy, dummy, dummy ,dummy, dummy, dummy, dummy, dtg_C_s, dtg_C_l, dtg_Sil_s, dtg_Sil_l
     close(56)

     dtg_C_s  =dtg_C_s/mgas
     dtg_C_l  =dtg_C_l/mgas
     dtg_Sil_s=dtg_Sil_s/mgas/0.163d0
     dtg_Sil_l=dtg_Sil_l/mgas/0.163d0
     
     if(time_this)then
        icount=0
        icount=icount+1
        call cpu_time(tt(icount))
     endif
     call calc_norm(dtg_C_s,dtg_C_l,Cs_C,Cl_C)
     if(time_this)then
        icount=icount+1
        call cpu_time(tt(icount))
        write(*,'(A,es13.5)')'calc norm        ',tt(icount)-tt(icount-1)
     endif
     call calc_norm(dtg_Sil_s,dtg_Sil_l,Cs_sil,Cl_sil)
     if(time_this)then
        icount=icount+1
        call cpu_time(tt(icount))
        write(*,'(A,es13.5)')'calc norm        ',tt(icount)-tt(icount-1)
     endif

     open(60, file=" /data101/dubois/DiscDustGuerra/extinction/data/eps.out", status='old')
     open(70, file=outputfile, status="unknown")
     do i=1,n_wave

        read(60,*) lambda,eps 
!!$        write(*,*) i, '/', n_wave

!!$        write(*,*)Cs_C,Cl_C,Cs_sil,Cl_sil

        call integral(Cs_C,Cl_C,eps_v,lambda_v,0,I_carb_v) !carboneus
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_C_v',tt(icount)-tt(icount-1)
        endif
        call integral(Cs_sil,Cl_sil,eps_v,lambda_v,1,I_sil_v)  !silicate 
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_S_v',tt(icount)-tt(icount-1)
        endif
        call integral(Cs_C,0.0d0,eps,lambda,0,I_carb_s) !carboneus small
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_C  ',tt(icount)-tt(icount-1)
        endif
        call integral(Cs_sil,0.0d0,eps,lambda,1,I_sil_s)  !silicate small
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_S  ',tt(icount)-tt(icount-1)
        endif
        call integral(0.0d0,Cl_C,eps,lambda,0,I_carb_l) !carboneus large
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_C  ',tt(icount)-tt(icount-1)
        endif
        call integral(0.0d0,Cl_sil,eps,lambda,1,I_sil_l)  !silicate large
        if(time_this)then
           icount=icount+1
           call cpu_time(tt(icount))
           write(*,'(A,es13.5)')'integral Qext_S  ',tt(icount)-tt(icount-1)
        endif

        I_carb_v=I_carb_v*s_grains/2.24d0
        I_sil_v =I_sil_v *s_grains/3.5d0
        I_carb=I_carb*s_grains/2.24d0
        I_sil =I_sil *s_grains/3.5d0
        I_carb_s=I_carb_s*s_grains/2.24d0
        I_sil_s =I_sil_s *s_grains/3.5d0
        I_carb_l=I_carb_l*s_grains/2.24d0
        I_sil_l =I_sil_l *s_grains/3.5d0
        I_carb=I_carb_s+I_carb_l
        I_sil=I_sil_s+I_sil_l
        !carb_part = 0.1+j*0.2
        !sil_part  = 0.9-j*0.2
!!$        A_lambda(j+1)= sil_part*I_sil+carb_part*I_carb
!!$        A_v(j+1)= sil_part*I_sil_v+carb_part*I_carb_v       
        ! call integral(Cs_dif,Cl_dif,eps,lambda,0,I_carb) !carboneus
        ! call integral(Cs_dif,Cl_dif,eps,lambda,1,I_sil)  !silicate
        ! A_lambda_dif= sil_part*I_sil+carb_part*I_carb
        ! call integral(Cs_den,Cl_den,eps,lambda,0,I_carb) !carboneus
        ! call integral(Cs_den,Cl_den,eps,lambda,1,I_sil)  !silicate
        ! A_lambda_den= sil_part*I_sil+carb_part*I_carb
       
!!$     write(70,*) lambda, A_lambda/A_v !, A_lambda_dif/A_v, A_lambda_den/A_v
        write(70,'(10es13.5)') lambda,I_sil,I_carb,I_sil_v,I_carb_v,I_sil_s,I_carb_s,I_sil_l,I_carb_l
     enddo
     close(60)
     close(70)
  enddo  
  close(1)

end program extinction_curves
