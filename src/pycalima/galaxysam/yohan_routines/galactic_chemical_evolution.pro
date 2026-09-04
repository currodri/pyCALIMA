;+
; NAME:
; galactic_chemical_evolution
;
; PURPOSE:
; This routine computes the galactic chemical evolution for a given
; set of input SSP yields (SNII+AGB+SNIa) and a given set of galactic
; parameters (accretion rate, SFR, outflows, etc.)
;
; CALLING SEQUENCE:
; galactic_chemical_evolution,
; dir=dir, readyields=readyields, fileyieldevol=fileyieldevol,
; vel=vel, zyield=zyield, asnia=asnia, fileyieldsniaevol=fileyieldsniaevol,
; tscale_infall=tscale_infall, tscale_sfr=tscale_sfr,
; windmodel=windmodel, minmload=minmload, maxmload=maxmload, 
; wind_loading=wind_loading,
; cosmic=cosmic, startz=startz, startm=startm, accmodel=accmodel,
; prefactor=prefactor, 
; alphaks=alphaks, starburst=starburst
; nbint=nbint,
; printps=printps, th=th, suffix=suffix,
; nosnia=nosnia, mass_return=mass_return, yield=yield,
; zsun=zsun, pristineDe=pristineDe,
; recycling=recycling,
; Mgrenorm=Mgrenorm, 
; stop=stop, help=help
;
; INPUTS:
;
;       dir: Your input SSP yield directory. Default is
;       '/home/dubois/StellarYields/ResultingYields'. You better hack
;       the code right now and change to your corresponding directory
;       where the SSP yields are located.
;
;       readyields: Parameter to trigger the reading of the input SSP
;       (SNII+AGB) yield evolution. 
;
;       fileyieldevol: List of files containing the SSP yield
;       evolution. Give only the file names. Their location is
;       provided by the dir parameter (see above).
;
;       vel: Rotational velocity of the stars (string of
;       characters). It is only useful for the LC18 set of stellar
;       yields, and avoids specifying the input file name
;       fileyieldevol for SSP yields. This will directly pick up the
;       appropriate LC18 file. Possible values are 'v0', 'v25', 'v50',
;       'v75', ..., 'v300'. Default: 'v0'. 
;
;       zyield: The set of zero age stellar metallicities of the SSP
;       used for the SSP yield in log(Z/Zsun). It must be consistent
;       with the list of files given in fileyieldevol. Default:
;       [-3.0,-2.0,-1.0,-0.6,-0.3,0.0,0.3] 
;
;       fileyieldsniaevol: The single file containing the SNIa yield
;       evolution. Give only the file name. Its location is
;       provided by the dir parameter (see above).
;
;       asnia: the fraction of stars of the IMF that turn into type Ia
;       SNe. Default: 0.05.
;
;       tscale_infall: time scale for the gas accretion rate onto the
;       galaxy in Gyr. It could take one value exp(-t/tinf) or two
;       values t/tinf2*exp(-t/tinf1)
;       (e.g. tscale_infall=[1.,10.]. Default: 7.
;
;       tscale_sfr: time scale for the SFR in the Schmidt-Kennicutt
;       law for star formation in Gyr. Default: 2.2
;
;       wind_model: Chose the wind model if any through the wind mass
;       loading law (W=\dotM_W/SFR). Available choices are:
;          - no wind model if parameter not specified (Default)
;          - 1: wind mass loading scales with MinW<sqrt(2.5d10/Mstar)<MaxW
;          - 2: Hayward & Hopkins 17 wind mass loading
;          - >2: wind mass loading scales with MinW<sqrt(2.5d10/Mstar)<MaxW
;
;       minmload: Minimum mass loading of the wind (MinW see
;       above). Default: 0.
;
;       maxmload: Maximum mass loading of the wind (MaxW see
;       above). Default: 10.
;
;       wind_loading: A constant value for the wind mass
;       loading. Default is 0. If non-zero, this must be used without
;       specifying the wind_model.
;
;       cosmic: A parameter that triggers cosmological accretion rates
;       based on the fitting formula of from Dekel+09 (see their
;       Appendix) for DM halos assuming a constant baryon fraction of
;       0.165. 
;
;       startz: Starting redshift for the accretion rate. Only applies
;       if using the cosmic accretion law (see above). Default: 20.
;
;       startm: Starting mass of the halo in Msun at starting redshift
;       startz for calculation of the accretion rate in the Dekel+09
;       relation. Only applies if using the cosmic accretion law (see
;       above). Default: 3.75d9. 
;
;       accmodel: Choose between various model for gas accretion onto
;       the galaxy. Available choices are:
;          - exp(-t/tinf) if not specified or not equals to 2 or 3
;          - 2: exp(-t/tinf) or t/tinf2*exp(-t/tinf1) depending on
;              the number of parameters given to tscale_infall (see
;              above)
;          - 3: no accretion. Starts with an initial amount of gas
;            equals to value given with the prefactor paramerer (see
;            below) 
;
;       prefactor: This parameter is extremely tricky. In several GCE
;       models there is a rescaling to operate to reach the desired
;       final mass contents. However in most cases the rescaling
;       cannot be performed at the end by renormalising all quantities
;       (because employed differential equations are non-linear). Therefore, a
;       prefactor for gas accretion rate (in Msun/Gyr) must be
;       chosen if accmodel allows for accretion or prefactor becomes
;       the initial gas mass (in Msun) if there is no accretion (see
;       explanations above for the accmodel parameter). To find the
;       exact value, it is simply trial and error to reach the desired
;       final masses.
;
;       alphaks: Power in the Schmidt-Kennicutt SF law:
;       SFR\proptoMgas^alphaks/tscale_sfr. Default: alphaks=1.
;
;       starburst: Parameter to trigger the starburst mode of SFR. The
;       standard SFR is boosted by MIN(Mgas/Mstar,10) when
;       Mgas/Mstar>1.
;
;       nbint: Number of integration time step. Default: 1000. Do not
;       decrease too much the default value or the low metallicty
;       evolution will be poorly resolved. This corresponds to a
;       uniform time step of 13.5Gyr/1000=13.5 Myr
;
;       ASNIa: The fraction of type Ia SNe. this amount is removed
;       from the contribution of intermediate + massive
;       stars. The contribution from type Ia SNe is NOT computed
;       here. Default: 3.5d-2
;
;       pristineDe: Initial ratio of Deuterium to Hydrogen. Default:
;       2.45d-5.
;
; INPUTS (Experimental):
;
;       nosnia: Parameter to prevent considering the yields from
;       SNIa. Purely experimental to see the effect of SNIa on various
;       elements 
;
;       mass_return: A fixed mass return fraction that is returned
;       instantaneously to the gas. Only used w/o the exact SSP yield
;       evolution. Useful to check whether the galactic metallicity is
;       correct for a simple model, when you think your result is
;       bugged.
;
;       yield: A fixed yield. Only used w/o the exact SSP yield
;       evolution. Default: 0.1. Useful to check whether the
;       galactic metallicity is correct for a simple model, when you
;       think your result is bugged.
;
;       recycling: A parameter to trigger the CGM component. Very
;       experimental. Use it at your own risks (= do not use it).
;
;       Mgrenorm: Renormalisation factor for the amount of Mg (in
;       deficit in LC18 stellar yields). This renormalisation applies
;       to AGB and SNII but not SNIa. You should rather use the
;       fudgemg parameter of cmp_yield_release.pro (only SNII) rather
;       than this one here. Default: 1.
;
; OUTPUTS:
;
;       printps: Parameter to turn the plots into postscript files.
;
;       th: Line thickness value for plots. Default=1. if X-window and
;       5. if postscript file.
;
;       suffix: String of characters for the identification of the postscript
;       file names.
;
;       zsun: Solar metallicity. Only used for plots. Default: 0.01345
;       (Asplund).
;
;       stop: Parameter to force the code to stop right before the
;       end. Useful to debug or play with arrays, do additional plots,
;       etc.
;
;       help: This help file. But if you see this help you might know
;       it already :-)
;
; EXAMPLES:
;
; Default:
;       galactic_chemical_evolution
;
; MODIFICATION HISTORY:
; Written by:Yohan Dubois, 01/09/2019.
;                       e-mail: dubois@iap.fr
;December, 2020:Comments and header added by Yohan Dubois.
;-
;###################################################
;###################################################
;###################################################
pro galactic_chemical_evolution,tscale_infall=tscale_infall,tscale_sfr=tscale_sfr,mass_return=mass_return,wind_loading=wind_loading,nbint=nbint,yield=yield,cosmic=cosmic,accmodel=accmodel,alphaks=alphaks,startz=startz,startm=startm,starburst=starburst,zsun=zsun,windmodel=windmodel,prefactor=prefactor,pristineDe=pristineDe,readyields=readyields,fileyieldevol=fileyieldevol,fileyieldsniaevol=fileyieldsniaevol,vel=vel,nosnia=nosnia,asnia=asnia,printps=printps,th=th,suffix=suffix,minmload=minmload,maxmload=maxmload,zyield=zyield,recycling=recycling,Mgrenorm=Mgrenorm,stop=stop,help=help

IF keyword_set(help) THEN BEGIN
   DOC_LIBRARY,'galactic_chemical_evolution'
   RETURN
ENDIF

thubble=13.5d0
if not keyword_set(nbint)then nbint=1000L
if not keyword_set(zsun )then zsun=0.01345
dt=thubble/double(nbint)
time=findgen(nbint)*dt
XH=0.76

XAsplund=[1.0d0,0.7381d0,0.2485d0,2.39d-3,6.99d-4,5.78d-3,5.09e-07,1.27d-3,7.16d-4,6.70d-4,3.1d-4,1.30d-3]

if not keyword_set(pristineDe   )then pristineDe=2.55d-5
if not keyword_set(prefactor    )then prefactor=1d9
if not keyword_set(tscale_infall)then tscale_infall=7.0d0 ;; Gyr
if not keyword_set(tscale_sfr   )then tscale_sfr   =2.2d0 ;; Gyr
if not keyword_set(mass_return  )then mass_return=0.0d0
if not keyword_set(wind_loading )then wind_loading=0.0d0
if not keyword_set(yield        )then yield=0.1d0
if not keyword_set(accmodel     )then accmodel=1
if not keyword_set(alphaks      )then alphaks=1.0d0
if not keyword_set(minmload     )then minmload= 0.0d0
if not keyword_set(maxmload     )then maxmload=10.0d0
if not keyword_set(Mgrenorm     )then Mgrenorm=1.0
;; if alphaks ne 1.0 then begin
;;    print,'there is a renormalisation step, which does not tolerate a power law different from 1'
;;    stop
;; endif


;; Read yield tables
if keyword_set(readyields)then begin
   if not keyword_set(fileyieldevol)then begin
      if not keyword_set(vel)then vel='v0'
      if(vel eq 'v50')then begin
         fileyieldevol=['yields_evol_z-3_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z-2_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z-1_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z-0.6_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z-0.3_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z0_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt','yields_evol_z0.3_'+vel+'_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt']
      endif else begin
         fileyieldevol=['yields_evol_z-3_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z-2_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z-1_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z-0.6_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z-0.3_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z0_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt','yields_evol_z0.3_'+vel+'_chabrier_imf0.1-100msun_dustcpopping17.txt']
      endelse
   endif
   if not keyword_set(zyield)then begin
      zyield=[-3.0,-2.0,-1.0,-0.6,-0.3,0.0,0.3]
   endif
   zyield=10d0^zyield*0.01345
   nfileyield=n_elements(fileyieldevol)
   for i=0L,nfileyield-1L do begin
      print,fileyieldevol(i)
   endfor
   if not keyword_set(dir)then dir='/home/dubois/StellarYields/ResultingYields'
   for i=0L,nfileyield-1L do begin
      fileyieldevol(i)=dir+'/'+fileyieldevol(i)
   endfor
   readcol,fileyieldevol(0),ty,my,mHy,mHey,mCy,mNy,mOy,mFy,mNey,mMgy,mSiy,mSy,mFey,mHyD,mHeyD,mCyD,mNyD,m0yD,mFyD,mNeyD,mMgyD,mSiyD,mSyD,mFeyD,/silent
   ntimeyield=n_elements(ty)
   tyield=ty/1d9 ;; Gyr units
   ;; n time steps, total mass return+10 species, nfiles
   ;; (metallicities) to read
   nchem=12L
   myield=dblarr(ntimeyield,nchem,nfileyield)
   for ifile=0L,nfileyield-1L do begin
      readcol,fileyieldevol(ifile),ty,my,mHy,mHey,mCy,mNy,mOy,mFy,mNey,mMgy,mSiy,mSy,mFey,mHyD,mHeyD,mCyD,mNyD,mOyD,mFyD,mNeyD,mMgyD,mSiyD,mSyD,mFeyD,/silent
      ichem=0L     &myield(0L:ntimeyield-1L,ichem,ifile)=my
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mHy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mHey
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mCy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mNy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mOy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mFy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mNey
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mMgy*Mgrenorm
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mSiy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mSy
      ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=mFey
   endfor
   if not keyword_set(nosnia)then begin
      ;; Add the SNIa contribution
      if not keyword_set(fileyieldsniaevol)then fileyieldsniaevol='cr_yields_Ia.txt'
      print,fileyieldsniaevol
      readcol,dir+'/'+fileyieldsniaevol,tyy,mCy,mNy,mOy,mNey,mMgy,mSiy,mSy,mFey,/silent
      if not keyword_set(asnia)then begin
         asnia=5d-2
         print,'Using ASNIa=',asnia,' by default'
      endif
      renorm=asnia
      for ifile=0L,nfileyield-1L do begin
         ichem=3      &myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mCy *renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mNy *renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mOy *renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mFy *renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mNey*renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mMgy*renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mSiy*renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mSy *renorm
         ichem=ichem+1&myield(0L:ntimeyield-1L,ichem,ifile)=myield(0L:ntimeyield-1L,ichem,ifile)+mFey*renorm
      endfor
   endif
endif

if keyword_set(cosmic)then begin
;; Compute the LCDM accretion rate and corresponding halo mass growth
   nbinz = nbint
   omega_m        = 0.3
   omega_l        = 0.7
   omega_k        = 0.0d0
   H0             = 70.0d0
   unsurH0        = 977.8/H0    ; in Gyr
   t_c            = dblarr(nbinz)
   t_c(0)         = -1d-10
   t_c(nbinz-1)   = -10.0       ; maximum (absolute value) of conformal star formation time of a star particle
   delta_tc       = (t_c(0)-t_c(nbinz-1)) / double(nbinz-1)
   for i=1,nbinz-2 do begin
      t_c(i) = t_c(0) - double(i)*delta_tc
   endfor
   print,'min & max conformal times: ',min(t_c),max(t_c)
;convert conformal time to real time with precision prec: 
   prec   = 1d-3
   tforz  = dblarr(nbinz)
   zred   = dblarr(nbinz)
   friedman_2,omega_m,omega_l,omega_k,prec,t_c,tforz,zred,nbinz
   tforz      = tforz*unsurH0   ; to get look back time in Gyr 
   tforz=max(tforz)-tforz
   redshift=dblarr(nbinz)
   time2   =dblarr(nbinz)
   for i=0L,nbinz-1L do begin
      redshift(i)=zred (nbinz-1L-i)
      time2   (i)=tforz(nbinz-1L-i)
   endfor
   if not keyword_set(startz)then startz=20.0
   if not keyword_set(startm)then startm=3.75d9
   ind=where(redshift lt startz,nelems)
   nmin=ind(0L)
   mhalo     =dblarr(nelems)
   gasaccrate=dblarr(nelems)
   mhalo(0)=startm
   print,'nmin=',nmin
   for i=nmin+1L,nmin+nelems-1L do begin
      ii=i-nmin
      accrate=6.6d0*(mhalo(ii-1L)/1d12)^1.15d0*(1d0+redshift(i))^(2.25d0)/0.165d0
      dt=(time2(i)-time2(i-1))*1d9
      dm=accrate*dt
      mhalo(ii)=mhalo(ii-1L)+dm
      ;; print,dt,zred(i),accrate,dm,mhalo(nmax-1L-i)
      ;; print,tforz(i),accrate*0.165d0
      gasaccrate(ii)=accrate*0.165d0
   endfor
   mytime=time2(nmin:nmin+nelems-1L)
;;oh oh
   nbint=nelems
   time=mytime
   mdot_infall=gasaccrate*prefactor ;; Msun/Gyr (prefactor=1d9 by default)
;; print,mdot_infall
   mgas    =dblarr(nbint)
   mgas    (0)=0.0d0
endif else begin
   mgas       =dblarr(nbint)
   mgas    (0)=0.0d0
   mdot_infall=dblarr(nbint)
   if accmodel eq 3 then begin
      mdot_infall(0L:nbint-1L)=0.0d0
      mgas(0)=prefactor
   endif else if accmodel eq 2 then begin
      if(n_elements(tscale_infall) eq 2)then begin
         mdot_infall=time/tscale_infall(1)*exp(-time/tscale_infall(0))*prefactor
      endif else begin
         mdot_infall=time*exp(-time/tscale_infall)*prefactor
      endelse
   endif else begin
      mdot_infall=exp(-time/tscale_infall)*prefactor
   endelse
endelse

minfall =dblarr(nbint)
mstar   =dblarr(nbint)
mcgm    =dblarr(nbint)
sfr     =dblarr(nbint)
fgas    =dblarr(nbint)
mwind   =dblarr(nbint)
mzstar  =dblarr(nbint)
mzgas   =dblarr(nbint)
mDegas  =dblarr(nbint)
mzwind  =dblarr(nbint)
mzcgm   =dblarr(nbint)
mDecgm  =dblarr(nbint)
mDewind =dblarr(nbint)
dmwind  =dblarr(nbint)
dmzwind =dblarr(nbint)
dmDewind=dblarr(nbint)
mywindload=dblarr(nbint)
minfall (0)=0.0d0
mstar   (0)=0.0d0
mcgm    (0)=0.0d0
sfr     (0)=0.0d0
fgas    (0)=1.0d0
mwind   (0)=0.0d0
mzcgm   (0)=0.0d0
mDecgm  (0)=0.0d0
mzwind  (0)=0.0d0
mDewind (0)=0.0d0
mzgas   (0)=0.0d0
mDegas  (0)=0.0d0
mzstar  (0)=0.0d0
dmwind  (0)=0.0d0
dmzwind (0)=0.0d0
dmDewind(0)=0.0d0
if keyword_set(fileyieldevol)then begin
   ff0=dblarr(nchem)
   mchemwind =dblarr(nbint,nchem)
   mchemcgm  =dblarr(nbint,nchem)
   dmchemwind=dblarr(nbint,nchem)
   mchemgas  =dblarr(nbint,nchem)
   mchemstar =dblarr(nbint,nchem)
   mchemwind (0,0:nchem-1)=0.0d0
   mchemcgm  (0,0:nchem-1)=0.0d0
   dmchemwind(0,0:nchem-1)=0.0d0
   mchemstar (0,0:nchem-1)=0.0d0
   ff0(0L)=1.0d0
   ff0(1L)=XH
   ff0(2L)=1d0-XH
   ff0(3L:nchem-1L)=0.0d0
   mchemgas  (0,0:nchem-1)=ff0(0:nchem-1)*mgas(0)
endif
zssp=dblarr(nbint)
zssp (0)=0.0d0
deltam=0.0d0
for i=1L,nbint-1L do begin
   dt=time(i)-time(i-1L)
   mgas  (i)=mgas  (i-1L)+mdot_infall(i)*dt            ;; Msun
   if keyword_set(recycling)then begin
      fDe=pristinede
      if(mcgm(i-1L) gt 0.)then fDe=mDecgm(i-1)/mcgm(i-1L)
      mDegas(i)=mDegas(i-1L)+mdot_infall(i)*dt*fDe ;; Msun
   endif else begin
      mDegas(i)=mDegas(i-1L)+mdot_infall(i)*dt*pristineDe ;; Msun
   endelse

   if keyword_set(fileyieldevol)then begin
      ;; print,'step',i,' out of ',nbint
      ;; Loop over all produced SSP over previous time steps
      for j=0L,i-1L do begin
         dmssp=sfr(j)*dt
         age =time(i)-time(j)
         ind=where(zyield lt zssp(j),nzsel)
         if(nzsel ge 1)then begin
            j1=ind(nzsel-1L)
         endif else begin 
            j1=0L
         endelse
         ind=where(tyield lt age   ,nasel)
         i1b=ind(nasel-1L)
         ind=where(tyield lt age-dt,nasel)
         if(nasel ge 1)then begin
            i1a=ind(nasel-1L)
         endif else begin
            i1a=0L
         endelse
         i2b=i1b+1
         if(i2b gt ntimeyield-1L)then i2b=i1b
         i2a=i1a+1
         if(i2a gt ntimeyield-1L)then i2a=i1a
         j2=j1+1
         if(j2 gt nfileyield-1L)then j2=j1
         zz=zyield(j2)-zyield(j1)
         ;; print,time(i),time(j),zssp(j),zyield(j1),zyield(j2)
         tta=tyield(i2a)-tyield(i1a)
         ttb=tyield(i2b)-tyield(i1b)
         if(zssp(j)ge zyield(nfileyield-1))then begin
            ;; print,zssp(j),zyield(nfileyield-1),age
            j1=nfileyield-1L
            for ichem=0L,nchem-1 do begin
               ;; interpolate the total mass returned at t=age
               mmb= myield(i1b,ichem,j1)+(age-tyield(i1b))/ttb*(myield(i2b,ichem,j1)-myield(i1b,ichem,j1))
               ;; interpolate the total mass returned at t=age-dt
               mma= myield(i1a,ichem,j1)+((age-dt)-tyield(i1a))/tta*(myield(i2a,ichem,j1)-myield(i1a,ichem,j1))
               ;; subtract the two values to get the increment and
               ;; multiply by the correspond SSP mass
               mchemgas(i,ichem)=mchemgas(i,ichem)+(mmb-mma)*dmssp
            endfor
         endif else begin
            for ichem=0L,nchem-1 do begin
               ;; interpolate the total mass returned at t=age
               fy1= (tyield(i2b)-age)/ttb*myield(i1b,ichem,j1)+ $
                    (age-tyield(i1b))/ttb*myield(i2b,ichem,j1)
               fy2= (tyield(i2b)-age)/ttb*myield(i1b,ichem,j2)+ $
                    (age-tyield(i1b))/ttb*myield(i2b,ichem,j2)
               mmb= (zyield(j2)-zssp(j))/zz*fy1+ $
                    (zssp(j)-zyield(j1))/zz*fy2
               ;; if(ichem eq 0)then print,'fy1,fy2 for mmb: ',fy1,fy2,format='(A,2e18.10)'
               ;; interpolate the total mass returned at t=age-dt
               fy1= (tyield(i2a)-(age-dt))/tta*myield(i1a,ichem,j1)+ $
                    ((age-dt)-tyield(i1a))/tta*myield(i2a,ichem,j1)
               fy2= (tyield(i2a)-(age-dt))/tta*myield(i1a,ichem,j2)+ $
                    ((age-dt)-tyield(i1a))/tta*myield(i2a,ichem,j2)
               mma= (zyield(j2)-zssp(j))/zz*fy1+ $
                    (zssp(j)-zyield(j1))/zz*fy2
               ;; if(ichem eq 0)then print,'fy1,fy2 for mma: ',fy1,fy2,format='(A,2e18.10)'
               ;; subtract the two values to get the increment and
               ;; multiply by the correspond SSP mass
               mchemgas(i,ichem)=mchemgas(i,ichem)+(mmb-mma)*dmssp
               ;; if(ichem eq 5)then print,j,mchemgas(i,5),i1b,i2b,myield(i1b,ichem,j1),myield(i2b,ichem,j1),myield(i1b,ichem,j1),myield(i2b,ichem,j2),mmb,i1a,i2a,myield(i1a,ichem,j1),myield(i2a,ichem,j1),mma,mmb-mma,dmssp
               ;; if(ichem eq 0)then print,age,mma,mmb,mmb-mma,i1a,i1b,format='(4e18.10,2i5)'
            endfor
         endelse
      endfor
      ;; print,mstar(i-1),mchemgas(i,0),format='(2e10.2)'
      mgas (i)=mgas (i)   +mchemgas(i,0)
      mstar(i)=mstar(i-1L)-mchemgas(i,0)
      ;; could use total mass released and subtract the H&He, but w/
      ;; interpolations, it is dangerous
      mzgas(i)=mzgas(i-1L)
      if keyword_set(recycling) and (mcgm(i-1L) gt 0.0) then mzgas(i)=mzgas(i)+mdot_infall(i)*dt*mzcgm(i-1L)/mcgm(i-1L)
      mzreturn=0.0d0
      for ichem=3,nchem-1 do begin
         mzreturn=mzreturn+mchemgas(i,ichem)
      endfor
      mzgas(i)=mzgas(i)+mzreturn
      ;; mchemstar(i,0L:nchem-1L)=mchemstar(i-1L,0L:nchem-1L)-mchemgas(i,0L:nchem-1L)
      mzstar(i)=0.0d0
      ;; print,'mstar    (i-1  )=',mstar(i-1)
      ;; print,'mchemstar(i-1,0)=',mchemstar(i-1,0)
      if(mchemstar(i-1L,0) gt 0.0d0)then begin
         for ichem=1,nchem-1L do begin
            xx=mchemstar(i-1L,ichem)/mchemstar(i-1L,0)
            ;; print,'xx=',xx
            mchemstar(i,ichem)=xx*(mchemstar(i-1L,0)-mchemgas(i,0))
            if(ichem ge 3)then mzstar(i)=mzstar(i)+mchemstar(i,ichem)
         endfor
      endif
      mchemstar(i,0)=mchemstar(i-1L,0)-mchemgas(i,0)
      ;; print,'(1)',total(mchemstar(i-1L,3:nchem-1L))/mstar(i-1L)
      ;; print,'(2)',total(mchemstar(i   ,3:nchem-1L))/mstar(i)
      
      if keyword_set(recycling) and (mcgm(i-1L) gt 0.0) then begin
         ff(0L:nchem-1L)=mchemcgm(i-1L,0L:nchem-1L)/mcgm(i-1L)
      endif else begin
         ff=ff0
      endelse
      mchemgas (i,0L:nchem-1L)=mchemgas (i-1L,0L:nchem-1L)+mchemgas(i,0L:nchem-1L)
      ;; print,'Stellar mass loss O:',mchemstar(i-1L,5),mchemstar(i,5),mchemgas(i,5)
      ;; print,'Stellar mass loss:',mstar(i),mchemstar(i,0)
      ;; mchemgas (i,0L)=mchemgas(i,0L)+mdot_infall(i)*dt
      ;; mchemgas (i,1L)=mchemgas(i,1L)+mdot_infall(i)*dt*XH
      ;; mchemgas (i,2L)=mchemgas(i,2L)+mdot_infall(i)*dt*(1d0-XH)
      mchemgas (i,0L)=mchemgas(i,0L)+mdot_infall(i)*dt*ff(0L)
      mchemgas (i,1L)=mchemgas(i,1L)+mdot_infall(i)*dt*ff(1L)
      mchemgas (i,2L)=mchemgas(i,2L)+mdot_infall(i)*dt*ff(2L)
      mchemgas (i,3L:nchem-1L)=mchemgas(i,3L:nchem-1L)+mdot_infall(i)*dt*ff(3L:nchem-1L)
      ;; print,ff(3:nchem-1L)
      ;; print,'====='
   endif else begin
      mzgas (i)=mzgas (i-1L)+yield*deltam*dt*mass_return
      if keyword_set(recycling) and (mcgm(i-1L) gt 0.0) then mzgas(i)=mzgas(i)+mdot_infall(i)*dt*mzcgm(i-1L)/mcgm(i-1L)
      mgas  (i)=mgas  (i   )+      deltam*dt*mass_return  ;; Msun (returned by stellar ejecta)
      mstar (i)=mstar (i-1L)-deltam*dt*mass_return        ;; Msun (returned by stellar ejecta)
      mzstar(i)=mzstar(i-1L)
   endelse
   if keyword_set(starburst) and mgas(i)/mstar(i) gt 1.0 then begin
      boost=min([mgas(i)/mstar(i),10.])
      sfr  (i)=mgas(i)^alphaks*boost/tscale_sfr    ;; Msun/Gyr
   endif else begin
      sfr  (i)=mgas(i)^alphaks/tscale_sfr          ;; Msun/Gyr
   endelse
   minfall(i)=minfall(i-1L)+mdot_infall(i)*dt      ;; Msun

   if(sfr(i)*dt gt mgas(i))then begin
      deltam=mgas(i)
      print,'sfr*dt>mgas!'
   endif else begin
      deltam=sfr(i)
   endelse
   mstar  (i)=mstar(i)+deltam*dt
   if keyword_set(fileyieldevol)then begin
      for ichem=0L,nchem-1L do begin
         if(ichem eq 0)then begin
            zcgas=1.0
         endif else begin
            zcgas=mchemgas(i,ichem)/mgas(i)
         endelse
         ;; print,ichem,zcgas
         mchemstar(i,ichem)=mchemstar(i,ichem)+deltam*dt*zcgas
         mchemgas (i,ichem)=mchemgas (i,ichem)-deltam*dt*zcgas
      endfor
   endif
   ;; Deplete gas through SFR
   zzgas =mzgas (i)/mgas(i)
   zssp(i)=zzgas
   zDegas=mDegas(i)/mgas(i)
   mzstar (i)=mzstar(i)+deltam*dt*zzgas
   mzgas  (i)=mzgas (i)-deltam*dt*zzgas
   mgas   (i)=mgas  (i)-deltam*dt
   mDegas (i)=mDegas(i)-deltam*dt*zDegas
;; print,time(i)*1d9,sfr(i)/1d9,mgas(i),mstar(i),mzgas(i),total(mchemstar(i,3:nchem-1L)),mzgas(i)/mgas(i),total(mchemstar(i,3:nchem-1L))/mstar(i),format='(8e10.3)'
;; if (i eq 5)then stop
   ;; Deplete gas through winds
   if keyword_set(windmodel)then begin
      if(windmodel eq 1)then begin
         wind_loading=max([minmload,min([sqrt(2.5d10/mstar(i-1L)),maxmload])])
      endif else if(windmodel eq 2)then begin
         wind_loading=min([14d0*(fgas(i-1L)*mstar(i-1L)/1d10)^(-0.23d0)*exp(-0.75d0/fgas(i-1d0)),maxmload]) ;; Hayward & Hopkins 17
      endif else begin
         wind_loading=max([minmload,min([sqrt(2.5d10/mstar(i-1L)),maxmload])])
      endelse
      mywindload(i)=wind_loading
      ;; print,time(i),wind_loading,mstar(i)
   endif   
   if(wind_loading*deltam*dt le mgas(i))then begin
      dmwind(i)=wind_loading*deltam*dt
   endif else begin
      dmwind(i)=mgas(i)*0.5d0
      print,'not enough gas for wind'
      ;; if(time(i) gt 0.02)then stop
   endelse
   if keyword_set(fileyieldevol)then begin
      for ichem=0L,nchem-1L do begin
         zcgas=mchemgas(i,ichem)/mgas(i)
         dmchemwind(i,ichem)=dmwind(i)*zcgas
         mchemwind (i,ichem)=mchemwind (i-1L,ichem)+dmchemwind(i,ichem)
         mchemgas  (i,ichem)=mchemgas  (i   ,ichem)-dmchemwind(i,ichem)
         dmchemwind(i,ichem)=dmchemwind(i   ,ichem)/dt
      endfor
   endif
   zzgas =mzgas (i)/mgas(i)
   zDegas=mDegas(i)/mgas(i)
   dmzwind (i)=dmwind(i)*zzgas
   dmDewind(i)=dmwind(i)*zDegas
   mwind  (i)=mwind (i-1L)+dmwind (i)
   mzwind (i)=mzwind (i-1L)+dmzwind (i)
   mDewind(i)=mDewind(i-1L)+dmDewind(i)
   mzgas  (i)=mzgas (i)-dmzwind (i)
   mDegas (i)=mDegas(i)-dmDewind(i)
   mgas   (i)=mgas (i)-dmwind(i)
   dmwind (i)=dmwind (i)/dt
   dmzwind (i)=dmzwind (i)/dt
   dmDewind(i)=dmDewind(i)/dt
   fgas   (i)=mgas(i)/(mstar(i)+mgas(i))
   if keyword_set(recycling)then begin
      ;; This gives the CGM mass by assuming it stands on the Moster
      ;; relation for MW and for a halo mass above 1e11 Msun (shock-heating)
      if(mstar(i)*1d12/6d10 gt 1d11)then begin
         mcgm(i)=mstar(i)*1d12/6d10*0.16
         mzcgm(i)=mzcgm(i-1L)+dmzwind(i)*dt
         if keyword_set(fileyieldevol)then begin
            for ichem=0L,nchem-1L do begin
               mdinfallcgm=ff0(ichem)*(mcgm(i)-mcgm(i-1L)-dmchemwind(i,ichem)*dt)
               mchemcgm(i,ichem)=mchemcgm(i-1L,ichem)+dmchemwind(i,ichem)*dt+mdinfallcgm
            endfor
         endif
      endif else begin
         mcgm(i)=0.0d0
         mzcgm(i)=0.0d0
         mchemcgm(i,0L:nchem-1L)=0.0d0
      endelse
      mdinfallcgm=pristineDe*(mcgm(i)-mcgm(i-1L)-dmwind(i)*dt)
      mDecgm(i)=mDecgm(i-1L)+dmDewind(i)*dt+mdinfallcgm
   endif

   ;; print,'End:',mstar(i),mchemstar(i,0)
   ;; if(i eq 3)then stop
endfor
mwmass=6d10
mrenorm=mwmass/mstar(nbint-1L)
;; print,'mstar before renorm',mstar(nbint-1L),mrenorm
print,'Final mstar: ',mstar(nbint-1L),' Msun'
mrenorm=1.0d0 ;; by-pass the renomr step (very dangerous can be misleading for the interpretation of the results)
mtot=mstar+mgas
;; mrenorm=53.0d0/max(mtot)
mtot  =mtot  *mrenorm
mstar =mstar *mrenorm
mgas  =mgas  *mrenorm
mwind =mwind *mrenorm
mzgas =mzgas *mrenorm
mDegas=mDegas*mrenorm
mzstar=mzstar*mrenorm
mzwind =mzwind *mrenorm
mDewind=mDewind*mrenorm
mdot_infall=mdot_infall*mrenorm/1d9 ;; Msun/yr
sfr=sfr*mrenorm/1d9                 ;; Msun/yr
dmwind =dmwind *mrenorm/1d9
dmzwind =dmzwind *mrenorm/1d9
dmDewind=dmDewind*mrenorm/1d9
if keyword_set(fileyieldevol)then begin
   mchemstar =mchemstar *mrenorm
   mchemgas  =mchemgas  *mrenorm
   mchemwind =mchemwind *mrenorm
   dmchemwind=dmchemwind*mrenorm
endif
if keyword_set(recycling)then begin
   mcgm =mcgm *mrenorm
   mzcgm=mzcgm*mrenorm
   if keyword_set(fileyieldevol)then begin
      mchemcgm=mchemcgm*mrenorm
   endif
   dmcgm=dblarr(nbint)
   dmcgm(0)=0.0d0
   for i=1,nbint-1L do begin
      dmcgm(i)=(mcgm(i)-mcgm(i-1L))/(dt*1d9)
   endfor
endif


if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(suffix)then suffix='canonical'
   device,filename='galactic_properties_'+suffix+'.eps',/encaps,xs=4*15,ys=15,/color
   if not keyword_set(th)then th=5
   !p.charthick=th
   !p.font=0
endif else begin
   iw=0&window,iw,xs=1200,ys=300
endelse
!p.multi=[0,4,1]
tek_color
ymin=1d9&ymax=3d12
!p.charsize=3
plot_io,time,mtot,lines=1,yr=[1d8,3d12],/xs,/ys,xtitle='time (Gyr)',ytitle='Mass (Msun)',xr=[0.,1.05*max(time)],xth=th,yth=th,th=th
oplot,time,mstar,lines=2,th=th
oplot,time,mgas,th=th
oplot,time,mwind,lines=4,th=th
if keyword_set(recycling)then oplot,time,mcgm,lines=5,th=th,color=4
if keyword_set(cosmic)then oplot,mytime,mhalo,color=2,th=th
xx=[max(time)]&yy=[5.84d10]&err=[1.17d10] ;; Côté et al. 2018
errplot,xx,yy-err,yy+err,th=th
xx=[max(time)]&yy=[9.2d9]&err=[5.3d9]     ;; Côté et al. 2018
errplot,xx,yy-err,yy+err,th=th
lymin=alog10(ymin)&lymax=alog10(ymax)
!p.charsize=1.5
dy=(lymax-lymin)*0.05
yy=10d0^(lymax-1.*dy)
plots,[1.0,3.0],[yy,yy],lines=1,th=th
xyouts,3.5,yy,'Mtot'
yy=10d0^(lymax-2.*dy)
plots,[1.0,3.0],[yy,yy],lines=2,th=th
xyouts,3.5,yy,'Mstar'
yy=10d0^(lymax-3.*dy)
plots,[1.0,3.0],[yy,yy],th=th
xyouts,3.5,yy,'Mgas'
if(wind_loading gt 0.)then begin
   yy=10d0^(lymax-4.*dy)
   plots,[1.0,3.0],[yy,yy],lines=4,th=th
   xyouts,3.5,yy,'Mejected'
endif
!p.charsize=3
ymin=1d-1&ymax=1d3
plot_io,time,mdot_infall,lines=1,yr=[ymin,ymax],/xs,/ys,xtitle='time (Gyr)',ytitle='dM/dt (Msun/yr)',xr=[0.,1.05*max(time)],xth=th,yth=th,th=th
oplot,time,sfr,lines=2,th=th
oplot,time,dmwind,lines=4,th=th
if keyword_set(recycling)then oplot,time,dmcgm,lines=5,th=th,color=4
if keyword_set(windmodel)then oplot,time,mywindload,color=2,lines=4,th=th
if keyword_set(cosmic)then oplot,time,gasaccrate,color=2,th=th
xx=[max(time)]&yy=[1.65]&err=[0.19]       ;; Licquia & Newman 2015
errplot,xx,yy-err,yy+err,th=th
lymin=alog10(ymin)&lymax=alog10(ymax)
!p.charsize=1.5
dy=(lymax-lymin)*0.05
yy=10d0^(lymax-1.*dy)
plots,[1.0,3.0],[yy,yy],lines=1,th=th
xyouts,3.5,yy,'Acc. rate'
yy=10d0^(lymax-2.*dy)
plots,[1.0,3.0],[yy,yy],lines=2,th=th
xyouts,3.5,yy,'SFR'
if(wind_loading gt 0.)then begin
   yy=10d0^(lymax-3.*dy)
   plots,[1.0,3.0],[yy,yy],lines=4,th=th
   xyouts,3.5,yy,'Out. rate'
endif
!p.charsize=3
ymin=1d-3&ymax=2.5
plot_io,time,mzstar/mstar/zsun,lines=2,yr=[1d-3,4.],/xs,/ys,xtitle='time (Gyr)',ytitle='Z/Zsun',xr=[0.,1.05*max(time)],xth=th,yth=th,th=th
oplot,time,mzgas/mgas/zsun,th=th
oplot,time,mzwind/mwind/zsun,lines=4,th=th
oplot,time,mzcgm/mcgm/zsun,lines=5,th=th,color=4
;; oplot,time,dmzwind/dmwind/zsun,lines=4,color=4,th=th
lymin=alog10(ymin)&lymax=alog10(ymax)
!p.charsize=1.5
dy=(lymax-lymin)*0.05
yy=10d0^(lymin+3.*dy)
plots,[5.0,7.0],[yy,yy],lines=2,th=th
xyouts,7.5,yy,'Zstar'
yy=10d0^(lymin+2.*dy)
plots,[5.0,7.0],[yy,yy],th=th
xyouts,7.5,yy,'Zgas'
if(wind_loading gt 0.)then begin
   yy=10d0^(lymin+1.*dy)
   plots,[5.0,7.0],[yy,yy],lines=4,th=th
   xyouts,7.5,yy,'Zwind'
endif
!p.charsize=3

ymin=0.0d0&ymax=1.0d0
plot,time,mDegas/mgas/pristineDe,lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='time (Gyr)',ytitle='X!dD!n',xr=[0.,1.05*max(time)],/nodata,xth=th,yth=th,th=th
oplot,time,mDegas/mgas/pristineDe,th=th
oplot,time,mDewind/mwind/pristineDe,lines=4,th=th
!p.charsize=1.5
dy=(ymax-ymin)*0.05
yy=ymin+2.*dy
plots,[5.0,7.0],[yy,yy],th=th
xyouts,7.5,yy,'X!dDgas!n'
if(wind_loading gt 0.)then begin
   yy=ymin+1.*dy
   plots,[5.0,7.0],[yy,yy],lines=4,th=th
   xyouts,7.5,yy,'X!dDwind!n'
endif
xx=[max(time)*0.950]&yy=[2.00d-5/pristineDe]&err=[0.10d-5/pristineDe] ;; Prodanovic+12
errplot,xx,yy-err,yy+err,th=th
;; xx=[max(time)*0.950]&yy=[1.52d-5/pristineDe]&err=[0.08d-5/pristineDe] ;; Moos+ 02 (very local ISM~100pc)
;; errplot,xx,yy-err,yy+err,th=th
xx=[max(time)*0.975]&yy=[2.31d-5/pristineDe]&err=[0.24d-5/pristineDe] ;; Linsky+ 06
errplot,xx,yy-err,yy+err,th=th
xx=[max(time)*1.000]&yy=[2.10d-5/pristineDe]&err1=[0.60d-5/pristineDe]&err2=[0.80d-5/pristineDe] ;; Savage+ 07
errplot,xx,yy-err,yy+err,th=th
!p.charsize=3
!p.multi=0

if keyword_set(printps)then device,/close

if keyword_set(readyields)then begin
   readcol,'ObservationnalData/Bensby2014.txt',ab14,bb14,cb14,db14,eb14,fb14,hipb14,msb14,ageb14,fehb14,ofeb14,nafeb14,mgfeb14,alfeb14,sifeb14,cafeb14,tifeb14,crfeb14,nifeb14,znfeb14,yfeb14,bafeb14,skip=73,/silent
   readcol,'ObservationnalData/Chen2000_tmp.txt',ac00,mc00,agec00,fehc00,ofec00,nafec00,mgfec00,alfec00,sifec00,kfec00,cafec00,tifec00,vfec00,crfec00,nifec00,bafec00,skip=60,/silent
   ;; fehc00=fehc00+(7.67-7.50)&ofec00=ofec00+(8.93-8.69)-(7.67-7.50)&mgfec00=mgfec00+(7.58-7.60)-(7.67-7.50)&sifec00=sifec00+(7.55-7.51)-(7.67-7.50) ;; AG89 -> A09
   readcol,'ObservationnalData/Israelian2004_mine.txt',ai04,fehi04,nhi04,dnhi04,ohi04,dohi04,skip=1,/silent,format='(a,f,f,f,f,f)'
   ;; fehi04=fehi04+(7.67-7.50)&ohi04=ohi04+(8.93-8.69)&nhi04=nhi04+(8.05-7.83) ;; AG89 -> A09
   noi04=nhi04-ohi04 & ofei04=ohi04-fehi04 & nfei04=nhi04-fehi04
   readcol,'ObservationnalData/cgisess_a5feab70ff4f90f6ea35e5e7922c6fab_data.tsv',a,b,zsaga,fehsaga,ofesaga,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_feh_cfe_all.txt',a,b,zsaga,fehsagacfe,cfesagafeh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_feh_nfe_all.txt',a,b,zsaga,fehsaganfe,nfesagafeh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_feh_sfe_all.txt',a,b,zsaga,fehsagasfe,sfesagafeh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_feh_mgfe_all.txt',a,b,zsaga,fehsagamgfe,mgfesagafeh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_feh_sife_all.txt',a,b,zsaga,fehsagasife,sifesagafeh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_oh_ch_all.txt',a,b,zsaga,ohsagach,chsagaoh,skip=1,/silent,format='(a,a,f,f,f)'
   readcol,'ObservationnalData/saga_oh_nh_all.txt',a,b,zsaga,ohsaganh,nhsagaoh,skip=1,/silent,format='(a,a,f,f,f)'
   ;; readcol,'ObservationnalData/saga_feh_cfe.txt',fehsagaofe,cfesagafeh,a,b,c,d,/silent


   if keyword_set(printps)then begin
      device,filename='abundances_vs_time_'+suffix+'.eps',/encaps,xs=4*15,ys=2*15,/color
   endif else begin
      iw=iw+1&window,iw,xs=1500,ys=600
   endelse
   !p.multi=[0,5,2]
   ymin=-3.0d0&ymax=1.0d0
   ytitle=['[all]','[H]','[He]','[C]','[N]','[O]','[F]','[Ne]','[Mg]','[Si]','[S]','[Fe]']
   plot,time,alog10(mzstar/mstar/zsun),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='time (Gyr)',ytitle='[Z]',xr=[0.,1.05*max(time)],xth=th,yth=th,th=th
   mz2star=dblarr(nbint)
   for ichem=3,nchem-1L do begin
      mz2star(0L:nbint-1L)=mz2star(0L:nbint-1L)+mchemstar(0L:nbint-1L,ichem)
   endfor
   ;; oplot,time,alog10(mz2star/mstar/zsun),lines=2,color=5,th=th
   for ichem=3,nchem-1L do begin
      plot,time,alog10(mchemstar(*,ichem)/mstar/XAsplund(ichem)),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='time (Gyr)',ytitle=ytitle(ichem),xr=[0.,1.05*max(time)],xth=th,yth=th,th=th
      oplot,time,alog10(mchemgas(*,ichem)/mgas/XAsplund(ichem)),th=th
      if(wind_loading gt 0.)then begin
         oplot,time,alog10(mchemwind(*,ichem)/mwind/XAsplund(ichem)),lines=4,th=th
         if keyword_set(recycling)then oplot,time,alog10(mchemcgm(*,ichem)/mcgm/XAsplund(ichem)),lines=5,th=th,color=4
      endif
      if(ytitle(ichem) eq '[O]')then begin
         oplot,14.-ageb14,ofeb14+fehb14,psym=3
         oplot,14.-agec00,ofec00+fehc00,psym=3,color=2
      endif
      if(ytitle(ichem) eq '[Mg]')then begin
         oplot,14.-ageb14,mgfeb14+fehb14,psym=3
         oplot,14.-agec00,mgfec00+fehc00,psym=3,color=2
      endif
      if(ytitle(ichem) eq '[Si]')then begin
         oplot,14.-ageb14,sifeb14+fehb14,psym=3
         oplot,14.-agec00,sifec00+fehc00,psym=3,color=2
      endif
   endfor
   !p.multi=0

   if keyword_set(printps)then begin
      device,/close
      device,filename='abundances_vs_iron_'+suffix+'.eps',/encaps,xs=4*15,ys=2*15,/color
   endif else begin
      iw=iw+1&window,iw,xs=1200,ys=900
   endelse
   !p.multi=[0,4,3]
   ymin=-1.3d0&ymax=1.3d0
   ytitle=['[Z/Fe]','[H/Fe]','[He/Fe]','[C/Fe]','[N/Fe]','[O/Fe]','[F/Fe]','[Ne/Fe]','[Mg/Fe]','[Si/Fe]','[S/Fe]','[Fe/Fe]']
   xoverfe =dblarr(nbint,nchem)
   feoverh =dblarr(nbint)
   dxoverfe =dblarr(nbint,nchem)
   dfeoverh =dblarr(nbint)
   xoverfeg=dblarr(nbint,nchem)
   feoverhg=dblarr(nbint)
   if(wind_loading gt 0.)then begin
      xoverfew=dblarr(nbint,nchem)
      feoverhw=dblarr(nbint)
   endif
   if keyword_set(recycling)then begin
      xoverfecgm=dblarr(nbint,nchem)
      feoverhcgm=dblarr(nbint)
   endif
   for i=1L,nbint-1L do begin
      dxoverfe (i,0)=(mzstar(i)-mzstar(i-1L))/(mchemstar(i,nchem-1L)-mchemstar(i-1L,nchem-1L))/(zsun/XAsplund(nchem-1L))
   endfor
   xoverfe (0L:nbint-1L,0)=mzstar(0L:nbint-1L)/mchemstar(0L:nbint-1L,nchem-1L)/(zsun/XAsplund(nchem-1L))
   xoverfeg(0L:nbint-1L,0)=mzgas (0L:nbint-1L)/mchemgas (0L:nbint-1L,nchem-1L)/(zsun/XAsplund(nchem-1L))
   if(wind_loading gt 0.)   then xoverfew  (0L:nbint-1L,0)=mzwind(0L:nbint-1L)/mchemwind(0L:nbint-1L,nchem-1L)/(zsun/XAsplund(nchem-1L))
   if keyword_set(recycling)then xoverfecgm(0L:nbint-1L,0)=mzcgm (0L:nbint-1L)/mchemcgm (0L:nbint-1L,nchem-1L)/(zsun/XAsplund(nchem-1L))
   for ichem=1L,nchem-1L do begin
      xoverfe (0L:nbint-1L,ichem)=mchemstar(0L:nbint-1L,ichem)/mchemstar(0L:nbint-1L,nchem-1L)/(XAsplund(ichem)/XAsplund(nchem-1L))
      for i=1L,nbint-1L do begin
         dxoverfe (i,ichem)=(mchemstar(i,ichem)-mchemstar(i-1L,ichem))/(mchemstar(i,nchem-1L)-mchemstar(i-1L,nchem-1L))/(XAsplund(ichem)/XAsplund(nchem-1L))
      endfor
      xoverfeg(0L:nbint-1L,ichem)=mchemgas (0L:nbint-1L,ichem)/mchemgas (0L:nbint-1L,nchem-1L)/(XAsplund(ichem)/XAsplund(nchem-1L))
      if(wind_loading gt 0.)   then xoverfew  (0L:nbint-1L,ichem)=mchemwind(0L:nbint-1L,ichem)/mchemwind(0L:nbint-1L,nchem-1L)/(XAsplund(ichem)/XAsplund(nchem-1L))
      if keyword_set(recycling)then xoverfecgm(0L:nbint-1L,ichem)=mchemcgm (0L:nbint-1L,ichem)/mchemcgm (0L:nbint-1L,nchem-1L)/(XAsplund(ichem)/XAsplund(nchem-1L))
   endfor
   feoverh (0L:nbint-1L)=mchemstar(0L:nbint-1L,nchem-1L)/mchemstar(0L:nbint-1L,1L)/(XAsplund(nchem-1L)/XAsplund(1L))
   for i=1L,nbint-1L do begin
      dfeoverh (i)=(mchemstar(i,nchem-1L)-mchemstar(i-1L,nchem-1L))/(mchemstar(i,1L)-mchemstar(i-1L,1L))/(XAsplund(nchem-1L)/XAsplund(1L))
   endfor
   feoverhg(0L:nbint-1L)=mchemgas (0L:nbint-1L,nchem-1L)/mchemgas (0L:nbint-1L,1L)/(XAsplund(nchem-1L)/XAsplund(1L))
   if(wind_loading gt 0.)   then feoverhw  (0L:nbint-1L)=mchemwind(0L:nbint-1L,nchem-1L)/mchemwind(0L:nbint-1L,1L)/(XAsplund(nchem-1L)/XAsplund(1L))
   if keyword_set(recycling)then feoverhcgm(0L:nbint-1L)=mchemcgm (0L:nbint-1L,nchem-1L)/mchemcgm (0L:nbint-1L,1L)/(XAsplund(nchem-1L)/XAsplund(1L))
   plot,alog10(feoverh),alog10(xoverfe(*,0)),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='[Fe/H]',ytitle=ytitle(0),xr=[-3.,1.],xth=th,yth=th,th=th
   ;; oplot,alog10(dfeoverh),alog10(dxoverfe(*,0)),lines=2,color=4,th=th
   oplot,alog10(feoverhg),alog10(xoverfeg(*,0)),th=th
   if(wind_loading gt 0.)   then oplot,alog10(feoverhw  ),alog10(xoverfew  (*,0)),lines=3,th=th
   if keyword_set(recycling)then oplot,alog10(feoverhcgm),alog10(xoverfecgm(*,0)),lines=5,color=4,th=th
   for ichem=3,nchem-2L do begin
      plot,alog10(feoverh),alog10(xoverfe(*,ichem)),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='[Fe/H]',ytitle=ytitle(ichem),xr=[-3.,1.],xth=th,yth=th,th=th
      ;; oplot,alog10(dfeoverh),alog10(dxoverfe(*,ichem)),lines=2,color=4,th=th
      oplot,alog10(feoverhg),alog10(xoverfeg(*,ichem)),th=th
      if(wind_loading gt 0.)   then oplot,alog10(feoverhw  ),alog10(xoverfew  (*,ichem)),lines=3,th=th
      if keyword_set(recycling)then oplot,alog10(feoverhcgm),alog10(xoverfecgm(*,ichem)),lines=5,color=4,th=th
      if(ytitle(ichem) eq '[C/Fe]' )then begin
         oplot,fehsagacfe,cfesagafeh,psym=1,color=6
      endif
      if(ytitle(ichem) eq '[N/Fe]' )then begin
         ;; oplot,fehi04,nfei04,psym=1,color=3
         oplot,fehsaganfe,nfesagafeh,psym=1,color=6
      endif
      if(ytitle(ichem) eq '[O/Fe]' )then begin
         ;; oplot,fehb14,ofeb14 ,psym=1
         ;; oplot,fehc00,ofec00,psym=1,color=2
         ;; oplot,fehi04,ofei04,psym=1,color=3
         oplot,fehsaga,ofesaga,psym=1,color=6
      endif
      if(ytitle(ichem) eq '[S/Fe]')then begin
         oplot,fehsagasfe,sfesagafeh,psym=1,color=6
      endif
      if(ytitle(ichem) eq '[Mg/Fe]')then begin
         ;; oplot,fehb14,mgfeb14,psym=1
         ;; oplot,fehc00,mgfec00,psym=1,color=2
         oplot,fehsagamgfe,mgfesagafeh,psym=1,color=6
      endif
      if(ytitle(ichem) eq '[Si/Fe]')then begin
         ;; oplot,fehb14,sifeb14,psym=1
         ;; oplot,fehc00,sifec00,psym=1,color=2
         oplot,fehsagasife,sifesagafeh,psym=1,color=6
      endif
      ;; oplot,time,alog10(mchemgas(*,ichem)/mgas/XAsplund(ichem)),th=th
      ;; if(wind_loading gt 0.)then begin
      ;;    oplot,time,alog10(mchemwind(*,ichem)/mwind/XAsplund(ichem)),lines=4,th=th
      ;; endif
   endfor
   covero =mchemstar(0L:nbint-1L,3)/mchemstar(0L:nbint-1L,5)/(XAsplund(3)/XAsplund(5))
   novero =mchemstar(0L:nbint-1L,4)/mchemstar(0L:nbint-1L,5)/(XAsplund(4)/XAsplund(5))
   ooverh =mchemstar(0L:nbint-1L,5)/mchemstar(0L:nbint-1L,1)/(XAsplund(5)/XAsplund(1))
   coverog=mchemgas (0L:nbint-1L,3)/mchemgas (0L:nbint-1L,5)/(XAsplund(3)/XAsplund(5))
   noverog=mchemgas (0L:nbint-1L,4)/mchemgas (0L:nbint-1L,5)/(XAsplund(4)/XAsplund(5))
   ooverhg=mchemgas (0L:nbint-1L,5)/mchemgas (0L:nbint-1L,1)/(XAsplund(5)/XAsplund(1))
   if(wind_loading gt 0.)then begin
      coverow=mchemwind(0L:nbint-1L,3)/mchemwind(0L:nbint-1L,5)/(XAsplund(3)/XAsplund(5))
      noverow=mchemwind(0L:nbint-1L,4)/mchemwind(0L:nbint-1L,5)/(XAsplund(4)/XAsplund(5))
      ooverhw=mchemwind(0L:nbint-1L,5)/mchemwind(0L:nbint-1L,1)/(XAsplund(5)/XAsplund(1))
   endif
   plot,alog10(ooverh),alog10(novero),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='[O/H]',ytitle='[N/O]',xr=[-3.,1.],xth=th,yth=th,th=th
   oplot,alog10(ooverhg),alog10(noverog),th=th
   if(wind_loading gt 0.)then oplot,alog10(ooverhw),alog10(noverow),lines=3,th=th
   ind=where(dohi04 ne -10.0 and dnhi04 ne -10.)
   oplot,ohi04(ind),noi04(ind),psym=3,color=3
   oplot,ohsaganh,nhsagaoh-ohsaganh,psym=1,color=6

   plot,alog10(ooverh),alog10(covero),lines=2,yr=[ymin,ymax],/xs,/ys,xtitle='[O/H]',ytitle='[C/O]',xr=[-3.,1.],xth=th,yth=th,th=th
   oplot,alog10(ooverhg),alog10(coverog),th=th
   if(wind_loading gt 0.)then oplot,alog10(ooverhw),alog10(coverow),lines=3,th=th
   oplot,ohsagach,chsagaoh-ohsagach,psym=1,color=6

   !p.multi=0
   !p.charsize=2
   if keyword_set(printps)then begin
      device,/close
      device,filename='deuterium_vs_oxygen_'+suffix+'.eps',/encaps,xs=15,ys=15,/color
   endif else begin
      iw=iw+1&window,iw,xs=512,ys=512
   endelse
   plot,alog10(ooverhg),mDegas/mgas/pristineDe,yr=[0.4,1.0],/xs,/ys,xtitle='[O/H]',ytitle='X!dD!n',xr=[-2.,0.5],xth=th,yth=th,th=th
   if(wind_loading gt 0.)then oplot,alog10(ooverhw),mDewind/mwind/pristineDe,lines=3,th=th
   if keyword_set(printps)then device,/close

   ;; iw=iw+1&window,iw,xs=1024,ys=512
   dfeoverh=dblarr(nbint)
   for i=1L,nbint-1L do begin
      dfe=mchemstar(i,nchem-1L)-mchemstar(i-1L,nchem-1L)
      dh =mchemstar(i,1L      )-mchemstar(i-1L,1L      )
      dfeoverh(i)=alog10( (dfe/dh) / (XAsplund(nchem-1L)/XAsplund(1)) )
   endfor
   histoplot,dfeoverh,pdffe,xfe,weight=sfr,nbin=50,min=-2.5,max=1.0
   plot,xfe,pdffe/max(pdffe),psym=10,xr=[-1.5,1.0],yr=[0.,1.1],/xs,/ys,xtitle='[Fe/H]',ytitle='dN/d[Fe/H]'
   ;; APOGEE from Haywood+18
   xx=(findgen(100)+0.5d0)/100.0d0*2.0-1.5
   ff=exp(-(xx-0.19)^2d0/(2d0*0.09^2d0))*200.+exp(-(xx+0.31)^2d0/(2d0*0.24^2d0))*65.0
   hh=exp(-(xx-0.20)^2d0/(2d0*0.08^2d0))*90.+exp(-(xx+0.28)^2d0/(2d0*0.26^2d0))*40.0
   gg=exp(-(xx-0.19)^2d0/(2d0*0.09^2d0))*130.+exp(-(xx+0.32)^2d0/(2d0*0.25^2d0))*60.0   
   oplot,xx,(ff+gg+hh)/max(ff+gg+hh),lines=2

endif


if keyword_set(stop)then stop

end
