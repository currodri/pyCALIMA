;+
; NAME:
; cmp_yield_release
;
; PURPOSE:
; This routine computes the chemical mass (and energy) returned for a
; single stellar population (only intermediate and massive stars, not
; SNIa). For type Ia SNe chemical returned masses -> snia.pro
;
; CALLING SEQUENCE:
; cmp_yield_release, rootdir=rootdir, fileage=fileage,
; fileintermediate=fileintermediate, filemassive=filemassive,
; filesecondary=filesecondary, zmet=zmet, vel=ve,
; constantomega=constantomega, alpha_slope=alpha_slope,
; bound_slope=bound_slope, chabrier=chabrier,
; revisedchabrier=revisedchabrier, mass_separatrix=mass_separatrix,
; mfailed=mfailed, failed_fraction=failed_fraction, ASNIa=ASNIa,
; fudgemg=fudgemg, fudgene=fudgene
; lc13rot=lc13rot, lc13norot=lc13norot, fileesn=fileesn,
; dwek=dwek, 
; hnfraction=hnfraction, kobayashi=kobayashi, 
; maxage=maxage, dlt=dlt, fileout=fileout, noplot=noplot,
; th=th, printps=printps, suffix=suffix
; fileeps=fileeps, title=title, elements2plot=elements2plot
; plotagb=plotagb, plotdust=plotdust, 
; mreturnmin=mreturnmin, mreturnmax=mreturnmax,
; stop=stop, help=help
;
; INPUTS:
;
;       rootdir: Your working directory. Default is
;       '/home/dubois'. You better hack the code right now and change
;       to your corresponding directory where the StellarYields
;       directory is. 
;
;       fileage: This file contains the final age of stars as a
;       function of their mass. Default: Padova stellar tracks
;       depending on the input metallicity. Warning: if you change
;       this input file, make sure it is consistent with the input
;       metallicity.
;
;       fileintermediate: This file contains the stellar yields of the
;       intermediate mass stars as a function of their mass. Default:
;       Karakas+10 stellar yields. Warning: if you change this input
;       file, make sure it is consistent with the input metallicity.
;
;       filemassive: This file contains the stellar yields of the
;       massive stars as a function of their mass. Default:
;       Limongi&Chieffi+18 stellar yields. Warning: if you change this
;       input file, make sure it is consistent with the input metallicity.
;
;       filesecondary: This file contains another set of stellar
;       yields of massive stars in addition to filemassive. This is
;       used to linearly interpolate between the failed and non-failed
;       sets of SNe, or between SNII and HN yields. Alternatively this
;       filesecondary can be used as an array of three files for the
;       three standard LC18 velocities (0,150,300 km/s) to interpolate
;       arbitrary values of velocities (see vel) or rotational support
;       (see constantomega). Default: none.
;
;       zmet: Metallicity in logZ/Zsun of the SSP that can be chosen
;       to avoid specifying the input file names for stellar yields
;       and age. This will directly pick up the appropriate files:
;       Padova stellar tracks, Karakas+10 and
;       Limongi&Chieffi+18. Possible values are -3.,-2.,-1.,-0.6,
;       -0.3, 0.0, 0.3. Default: none.
;
;       vel: Rotational velocity of the stars in km/s. It is only
;       useful for the LC18 set of stellar yields, and avoids
;       specifying the input file name for stellar yields of massive
;       stars. This will directly pick up the appropriate LC18
;       file. Possible values are 0, 25, 50, 75, ..., 300. Default: 0.
;
;       constantomega: Alternatively to specifying the rotational
;       velocity of stars, one can give the fraction of critical velocity
;       of the star (velocity above which the star breaks up by
;       centrifugal forces). This value must be below 1.0 and must be
;       used in conjuction with filesecondary. Still a bit
;       experimental.
;
;       alpha_slope: Slopes of the power-law of the IMF
;       phi^alpha_slope. This can contain several values (broken power
;       laws), e.g. alpha_slope=[-1.0,-1.5,-2.0]. Defaut: [-2.3]
;       (Salpeter IMF by default).
;
;       bound_slope: Boundary values of the IMF slopes in Msun. This
;       contains as many values as the number of values for
;       alpha_slope plus 1, e.g. if alpha_slope=[-1.0,-1.5,-2.0], then
;       bound_slope can be =[0.1,1.0,10.,100.0]. The first and final
;       values define the min and max cutoff of the IMF. Default:
;       [0.1,100.0] (Salpeter IMF by default).
;
;       chabrier: Parameter to trigger the functional form of the
;       Chabrier (2003) IMF that is not directly represented by a broken
;       power law. If used, the bound_slope must have only two extreme
;       values. One can also change the slope of the high-mass end
;       with alpha_slope.
;
;       revisedchabrier: Parameter to trigger the revised 2005 version of
;       the Chabrier IMF. Same remarks than for chabrier.
;
;       mass_separatrix: Stellar mass that separates between the
;       intermediate mass stars and the massive stars in
;       Msun. Default: 8.0
;
;       mfailed: The stellar mass in Msun above which massive stars are
;       failed, i.e. do not explode into SNII. Default: 30. 
;       WARNING: if Kobayashi yields are used, it is considered that
;       all stars are SNII failed above 40 Msun (hard-coded from their
;       stellar yields).
;
;       failed_fraction: The fraction of stars above mfailed that fail
;       to undergo SNII. Default: 1.0.
;       WARNING: if Kobayashi yields are used, it is considered that
;       all stars are SNII failed above 40 Msun (hard-coded from their
;       stellar yields).
;
;       ASNIa: The fraction of type Ia SNe. this amount is removed
;       from the contribution of intermediate + massive
;       stars. The contribution from type Ia SNe is NOT computed
;       here. Default: 3.5d-2
;
;       fudgemg: Rescaling factor for the Mg yields of massive
;       stars. For instance using LC18, this factor should 2 to 3 in
;       order to get appropriate final abundances of Mg. Default: 1.
;
;       fudgene: Rescaling factor for the Ne yields of massive
;       stars. For instance using LC18, this factor should 2 in
;       order to get appropriate final abundances of Ne. Default: 1.
;
;       lc13rot: Parameter to define the energy released by SNII as a
;       function of their stellar mass following the star with
;       rotation of Limongi & Chieffi 13. If not specified, all
;       non-failed SNII release 1e51 erg. 
;
;       lc13norot: Parameter to define the energy released by SNII as a
;       function of their stellar mass following the star without
;       rotation of Limongi & Chieffi 13. If not specified, all
;       non-failed SNII release 1e51 erg. 
;
;       fileesn: This file contains the energy released by SNII as a
;       function of their stellar mass following the star for LC18. If
;       not specified, all non-failed SNII release 1e51 erg.
;
;       dwek: Parameter to trigger the use of dust condensation
;       efficiencies from Dwek 98. If not used, condensation
;       efficiencies from Popping+17 are employed. These values do NOT
;       affect the computed output chemical elements released by stars
;       (i.e. they are for gas+dust).
;
;       compound: chemical composition of the dust silicate grain
;       Default: [1,1,1,4] for MgFeSiO4
;
; INPUTS (Experimental):
;
;       hnfraction: Fraction of HNe above 20 Msun. This parameter is
;       only used to compute the energy released (not for yields)
;       assuming a 1e52 erg energy for HNe. For yields, see filesecondary.
;     
;       kobayashi: Use this parameter to trigger the use of
;       Kobayashi+13 stellar yields for Mg. This parameter is used to
;       correct for the deficit of Mg in the LC18 yields. Its use is
;       not recommend: just experimental. For using the complete K13
;       stellar yields specify the appropriate files in filemassive.
;
; OUTPUTS:
;
;       maxage: Maximum age in years up to which the output chemical
;       elements released by the SSP are computed. Default: 2d10.
;
;       dlt: time bin size in dex. Default: 0.1 dex
;
;       fileout: Name of the output files that will contain the
;       computed data (mass and energy released as a function of
;       time). Only written if name is specified. 
;
;       noplot: Parameter to prevent plots from appearing. Very useful
;       when this routine is linked to another command to loop over
;       the available parameters (e.g. with run_all_yield_release.pro)
;       and you do not want to have plots jumping all over the screen.
;
;       th: Line thickness value for plots. Default=1. if X-window and
;       5. if postscript file.
;
;       printps: Parameter to turn the plots into postscript files.
;
;       suffix: String of characters for the identification of the postscript
;       file names.
;
;       fileeps: Postscript file name for the mass returned into various elements
;       as a function of time.
;
;       title: String of characters for the title to be written on top
;       of the mass return as a function of time plot.
;
;       elements2plot: Elements to be ploted in the figures. It could
;       be helpful for a better reading of the the plots. By default,
;       all elements are plotted. (one can use
;       e.g. elements2plot=['C','N','O','Mg'])
;
;       plotagb: Parameter to plot separately the contribution from
;       intermediate (AGB) stars.
;
;       plotdust: Parameter to plot separately the dust released.
;
;       mreturnmin: Minimum value for the ratio of mass returned as a
;       function of time in the corresponding plot. Default: 1d-7
;
;       mreturnmax: Maximum value for the ratio of mass returned as a
;       function of time in the corresponding plot. Default: 1d0
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
;       cmp_yield_release
;
; Use latest Chabrier IMF, change the fraction of SNIa and the mass
; scale separating intermediate and massive stars:
;       cmp_yield_release,/revisedchabrier,asnia=5d-2,mass_separatrix=7.0
;
; Change the initial metallicity and rotation of massive stars:
;       cmp_yield_release,zmet=-0.6,vel=150.
;
; Use a given fraction of failed SNII:
;       cmp_yield_release,zmet=-0.3,vel=50.,filemassive='./limongichieffi_modelm_z-0.3_vel50_simplified.txt',filesecondary='./limongichieffi_z-0.3_vel50_simplified.txt',mfailed=35.,failed_fraction=0.5
;
; MODIFICATION HISTORY:
; Written by:Yohan Dubois, 01/01/2019.
;                       e-mail: dubois@iap.fr
;December, 2020:Comments and header added by Yohan Dubois.
;-
;###################################################
;###################################################
;###################################################
pro interpolate_files,mzas_m,zzas_m,vel_m,mfin_m,el_m,mlos_m,mla_m,filemassive=filemassive,vel=vel,constantomega=constantomega
  
if(n_elements(filemassive) eq 1)then begin
   readcol,filemassive     ,mzas_m,zzas_m,vel_m,mfin_m,el_m,mlos_m,mla_m,format='(f,f,f,f,a,f,f)',/silent
endif else begin
   readcol,filemassive(0),mzas_m1,zzas_m1,vel_m1,mfin_m1,el_m1,mlos_m1,mla_m1,format='(f,f,f,f,a,f,f)',/silent
   readcol,filemassive(1),mzas_m2,zzas_m2,vel_m2,mfin_m2,el_m2,mlos_m2,mla_m2,format='(f,f,f,f,a,f,f)',/silent
   readcol,filemassive(2),mzas_m3,zzas_m3,vel_m3,mfin_m3,el_m3,mlos_m3,mla_m3,format='(f,f,f,f,a,f,f)',/silent
   if not keyword_set(constantomega)then begin
      if(vel ge 0. and vel lt 150.)then begin
         yy=vel/150.0
         mzas_m=mzas_m1+yy*(mzas_m2-mzas_m1)
         zzas_m=zzas_m1+yy*(zzas_m2-zzas_m1)
         vel_m =vel_m1 +yy*(vel_m2 -vel_m1 )
         mfin_m=mfin_m1+yy*(mfin_m2-mfin_m1)
         el_m  =el_m1
         mlos_m=mlos_m1+yy*(mlos_m2-mlos_m1)
         mla_m =mla_m1 +yy*(mla_m2 -mla_m1 )
      endif
      if(vel ge 150. and vel lt 300.)then begin
         yy=(vel-150.)/150.0
         mzas_m=mzas_m2+yy*(mzas_m3-mzas_m2)
         zzas_m=zzas_m2+yy*(zzas_m3-zzas_m2)
         vel_m =vel_m2 +yy*(vel_m3 -vel_m2 )
         mfin_m=mfin_m2+yy*(mfin_m3-mfin_m2)
         el_m  =el_m2
         mlos_m=mlos_m2+yy*(mlos_m3-mlos_m2)
         mla_m =mla_m2 +yy*(mla_m3 -mla_m2 )
      endif
      if(vel ge 300.)then begin
         mzas_m=mzas_m3
         zzas_m=zzas_m3
         vel_m =vel_m3
         mfin_m=mfin_m3
         el_m  =el_m3
         mlos_m=mlos_m3
         mla_m =mla_m3
      endif
   endif else begin
      momega    =[13. ,15. ,20. ,25. ,30. ,40. ,60. ,80. ,120.];;Zsun & 150 km/s
      ratioomega=[0.35,0.34,0.32,0.30,0.29,0.28,0.25,0.24,0.22]
      vel_omega=150.0*constantomega/ratioomega
      print,momega
      print,vel_omega
      vel_m=fltarr(n_elements(mzas_m1))
      for i=0L,n_elements(mzas_m1)-1L do begin
         vel_m(i)=vel_omega(where(momega eq mzas_m1(i),nmatch))
         if(nmatch ne 1)then begin
            print,'nmatch not equal 1',nmatch
            stop
         endif
         if(vel_m(i) ge 0. and vel_m(i) lt 150.)then begin
            yy=vel_m(i)/150.0
            mzas_m=mzas_m1+yy*(mzas_m2-mzas_m1)
            zzas_m=zzas_m1+yy*(zzas_m2-zzas_m1)
            mfin_m=mfin_m1+yy*(mfin_m2-mfin_m1)
            el_m  =el_m1
            mlos_m=mlos_m1+yy*(mlos_m2-mlos_m1)
            mla_m =mla_m1 +yy*(mla_m2 -mla_m1 )
         endif
         if(vel_m(i) ge 150. and vel_m(i) lt 300.)then begin
            yy=(vel_m(i)-150.)/150.0
            mzas_m=mzas_m2+yy*(mzas_m3-mzas_m2)
            zzas_m=zzas_m2+yy*(zzas_m3-zzas_m2)
            mfin_m=mfin_m2+yy*(mfin_m3-mfin_m2)
            el_m  =el_m2
            mlos_m=mlos_m2+yy*(mlos_m3-mlos_m2)
            mla_m =mla_m2 +yy*(mla_m3 -mla_m2 )
         endif
         if(vel_m(i) ge 300.)then begin
            mzas_m=mzas_m3
            zzas_m=zzas_m3
            mfin_m=mfin_m3
            el_m  =el_m3
            mlos_m=mlos_m3
            mla_m =mla_m3
         endif
      endfor
   endelse
endelse


end
;;=================================================================
pro interpolate_files_esn,mesn_sampled,zzas_m,vel_m,esn_m,fileesn=fileesn,vel=vel,constantomega=constantomega
  
if(n_elements(fileesn) eq 1)then begin
   readcol,fileesn     ,mesn_sampled,zzas_m,vel_m,esn_m,format='(f,f,f,f)',/silent
endif else begin
   readcol,fileesn(0),mesn_sampled1,zzas_m1,vel_m1,esn_m1,format='(f,f,f,f)',/silent
   readcol,fileesn(1),mesn_sampled2,zzas_m2,vel_m2,esn_m2,format='(f,f,f,f)',/silent
   readcol,fileesn(2),mesn_sampled3,zzas_m3,vel_m3,esn_m3,format='(f,f,f,f)',/silent
   if not keyword_set(constantomega)then begin
      if(vel ge 0. and vel lt 150.)then begin
         yy=vel/150.0
         mesn_sampled=mesn_sampled1+yy*(mesn_sampled2-mesn_sampled1)
         zzas_m=zzas_m1+yy*(zzas_m2-zzas_m1)
         vel_m =vel_m1 +yy*(vel_m2 -vel_m1 )
         esn_m=esn_m1+yy*(esn_m2-esn_m1)
      endif
      if(vel ge 150. and vel lt 300.)then begin
         yy=(vel-150.)/150.0
         mesn_sampled=mesn_sampled2+yy*(mesn_sampled3-mesn_sampled2)
         zzas_m=zzas_m2+yy*(zzas_m3-zzas_m2)
         vel_m =vel_m2 +yy*(vel_m3 -vel_m2 )
         esn_m=esn_m2+yy*(esn_m3-esn_m2)
      endif
      if(vel ge 300.)then begin
         mesn_sampled=mesn_sampled3
         zzas_m=zzas_m3
         vel_m =vel_m3
         esn_m=esn_m3
      endif
   endif else begin
      momega    =[13. ,15. ,20. ,25. ,30. ,40. ,60. ,80. ,120.];;Zsun & 150 km/s
      ratioomega=[0.35,0.34,0.32,0.30,0.29,0.28,0.25,0.24,0.22]
      vel_omega=150.0*constantomega/ratioomega
      print,momega
      print,vel_omega
      vel_m=fltarr(n_elements(mesn_sampled1))
      for i=0L,n_elements(mesn_sampled1)-1L do begin
         vel_m(i)=vel_omega(where(momega eq mesn_sampled1(i),nmatch))
         if(nmatch ne 1)then begin
            print,'nmatch not equal 1',nmatch
            stop
         endif
         if(vel_m(i) ge 0. and vel_m(i) lt 150.)then begin
            yy=vel_m(i)/150.0
            mesn_sampled=mesn_sampled1+yy*(mesn_sampled2-mesn_sampled1)
            zzas_m=zzas_m1+yy*(zzas_m2-zzas_m1)
            esn_m=esn_m1+yy*(esn_m2-esn_m1)
         endif
         if(vel_m(i) ge 150. and vel_m(i) lt 300.)then begin
            yy=(vel_m(i)-150.)/150.0
            mesn_sampled=mesn_sampled2+yy*(mesn_sampled3-mesn_sampled2)
            zzas_m=zzas_m2+yy*(zzas_m3-zzas_m2)
            esn_m=esn_m2+yy*(esn_m3-esn_m2)
         endif
         if(vel_m(i) ge 300.)then begin
            mesn_sampled=mesn_sampled3
            zzas_m=zzas_m3
            esn_m=esn_m3
         endif
      endfor
   endelse
endelse


end
;;=================================================================
;;=================================================================
;; Main routine
pro cmp_yield_release,rootdir=rootdir,fileage=fileage,fileintermediate=fileintermediate,filemassive=filemassive,filesecondary=filesecondary,alpha_slope=alpha_slope,bound_slope=bound_slope,mass_separatrix=mass_separatrix,maxage=maxage,dlt=dlt,th=th,zmet=zmet,vel=vel,printps=printps,fileeps=fileeps,title=title,chabrier=chabrier,fileout=fileout,ASNIa=ASNIa,elements2plot=elements2plot,dwek=dwek,plotagb=plotagb,plotdust=plotdust,mreturnmin=mreturnmin,mreturnmax=mreturnmax,kobayashi=kobayashi,hnfraction=hnfraction,fudgemg=fudgemg,fudgene=fudgene,suffix=suffix,constantomega=constantomega,mfailed=mfailed,failed_fraction=failed_fraction,revisedchabrier=revisedchabrier,lc13rot=lc13rot,lc13norot=lc13norot,stop=stop,fileesn=fileesn,noplot=noplot,olivine=olivine,pyroxene=pyroxene,compound=compound,help=help

IF keyword_set(help) THEN BEGIN
    DOC_LIBRARY,'cmp_yield_release'
    RETURN
ENDIF

if not keyword_set(compound)then compound=[1d0,1d0,1d0,4d0] ;; MgFeSiO4
if not keyword_set(hnfraction)then hnfraction=0.0d0
;; Z=0.02 Kobayashi+060
mkobayashi   =[13.   ,15.     ,18.    ,20.    ,25.    ,30.    ,40.    ]
mg24kobayashi=[2.52d-2,3.79d-2,1.03d-1,7.16d-2,2.18d-1,1.88d-1,3.10d-1]
mg25kobayashi=[2.56d-3,1.47d-3,7.08d-3,1.44d-2,3.13d-2,3.12d-2,7.28d-2]
mg26kobayashi=[2.18d-3,1.73d-3,5.90d-3,8.87d-3,2.73d-2,2.80d-2,7.34d-2]
mgkobayashi=mg24kobayashi+mg25kobayashi+mg26kobayashi

;; list_elements=['p','He','C','N','O','Ne','Mg','Si','S','Fe']
list_elements=['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
print,list_elements
nel2follow=n_elements(list_elements)
if not keyword_set(mass_separatrix)then begin
   mass_separatrix=8d0
   print,'Mass separatrix between intermediate and massive star behaviors not specified'
   print,'-> Use: ',mass_separatrix,' Msun by default',format='(A,f8.2,A)'
endif

if not keyword_set(fudgemg)then fudgemg=1.0d0
if not keyword_set(fudgene)then fudgene=1.0d0

if not keyword_set(rootdir)then rootdir='/home/dubois'
if not keyword_set(vel)then vel=0
vel=strcompress(vel,/remove_all)
if keyword_set(zmet)then begin
   if(zmet eq 0.3)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.0269.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.02_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z0_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq 0.0)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.01345.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.01345_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z0_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq -0.3)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.006725.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.006725_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z-0.3_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq -0.6)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.003362.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.003362_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z-0.6_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq -1.)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.001345.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.001345_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z-1_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq -2.)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.0001345.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.0001345_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z-2_vel'+vel+'_simplified.txt'
   endif
   if(zmet eq -3.)then begin
      if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.00001345.txt'
      if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.00001345_simplified.txt'
      if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z-3_vel'+vel+'_simplified.txt'
   endif
endif else begin
   if not keyword_set(fileage         )then fileage         =rootdir+'/StellarYields/PadovaStellarTracks/ages_padova_z0.01345.txt'
   if not keyword_set(fileintermediate)then fileintermediate=rootdir+'/StellarYields/karakas_z0.01345_simplified.txt'
   if not keyword_set(filemassive     )then filemassive     =rootdir+'/StellarYields/limongichieffi_z0_vel'+vel+'_simplified.txt'
endelse

readcol,fileage,mzas_fromage,logage,/silent
readcol,fileintermediate,mzas_i,zzas_i,mfin_i,el_i,y_i,mlos_i,mzer_i,mla_i,format='(f,f,f,a,f,f,f,f)',/silent

interpolate_files,mzas_m,zzas_m,vel_m,mfin_m,el_m,mlos_m,mla_m,filemassive=filemassive,vel=vel,constantomega=constantomega
if keyword_set(filesecondary)then begin
   interpolate_files,mzas_m_b,zzas_m_b,vel_m_b,mfin_m_b,el_m_b,mlos_m_b,mla_m_b,filemassive=filesecondary,vel=vel,constantomega=constantomega
   if not keyword_set(mfailed)then begin
      mfailed=40.0
      print,'failed SN mass by default is:',mfailed,' Msun',format='(A,f8.2,A)'
   endif
   if not keyword_set(failed_fraction)then begin
      failed_fraction=1.0
      print,'failed SN fraction by default is:',failed_fraction,format='(A,f8.2)'
   endif
   ind=where(mzas_m gt mfailed,nfailed)
   if(nfailed gt 0)then begin
      mlos_m(ind)=mlos_m(ind)*(1d0-failed_fraction)+mlos_m_b(ind)*failed_fraction
      mla_m (ind)=mla_m (ind)*(1d0-failed_fraction)+mla_m_b (ind)*failed_fraction
      mfin_m(ind)=mfin_m(ind)*(1d0-failed_fraction)+mfin_m_b(ind)*failed_fraction
   endif
endif

indmg=where(el_m eq 'Mg')
mlos_m(indmg)=mlos_m(indmg)*fudgemg
indne=where(el_m eq 'Ne')
mlos_m(indne)=mlos_m(indne)*fudgene
for i=0L,n_elements(mzas_m)-1L do begin
   indmg=where(el_m eq 'Mg' and mzas_m eq mzas_m(i))
   img=indmg(0)
   mla_m(i)=mla_m(i)-mlos_m(img)/fudgemg+mlos_m(img)
   indne=where(el_m eq 'Ne' and mzas_m eq mzas_m(i))
   ine=indne(0)
   mla_m(i)=mla_m(i)-mlos_m(ine)/fudgemg+mlos_m(ine)
endfor


MdotWR=1d-5        ;; Msun/yr Values from Crowther 07
MdotWR=MdotWR*2d33 ;; g/yr
vWR=2d3            ;; km/s

if keyword_set(revisedchabrier)then chabrier=1

if not keyword_set(alpha_slope)then begin
   if not keyword_set(chabrier)then begin
      print,'IMF not specified'
      print,'-> Use Salpeter by default'
      nslopes=1L
      alpha_slope=[-2.3]
      if not keyword_set(bound_slope)then begin
         bound_slope=[0.1,100.0]
         print,'Slope boundaries not specified'
         print,'-> Use: [',bound_slope(0),',',bound_slope(1),'] Msun by default',format='(A,f8.2,A,f8.2,A)'
      endif else if (n_elements(bound_slope) ne nslopes+1)then begin
         print,'The number of slope boundaries must be equal to '
         print,'the number of slope power laws plus one'
         print,'Stopping...'
         stop
      endif
   endif else begin
      if not keyword_set(bound_slope)then begin
         bound_slope=[0.1,100.0]
         print,'Slope boundaries not specified'
         print,'-> Use: [',bound_slope(0),',',bound_slope(1),'] Msun by default',format='(A,f8.2,A,f8.2,A)'
         alpha_slope=[-2.3]
         print,'High mass-end slope  not specified'
         print,'-> Use: [',alpha_slope[0],'] by default',format='(A,f8.2,A)'
      endif else begin
         print,'-> Use: [',bound_slope(0),',',bound_slope(1),'] Msun ',format='(A,f8.2,A,f8.2,A)'
         alpha_slope=[-2.3]
         print,'High mass-end slope  not specified'
         print,'-> Use: [',alpha_slope[0],'] by default',format='(A,f8.2,A)'
      endelse
   endelse
endif else begin
   nslopes=n_elements(alpha_slope)
   if(n_elements(bound_slope) ne nslopes+1)then begin
      print,'The number of slope boundaries must be equal to '
      print,'the number of slope power laws plus one'
      print,'Stopping...'
      stop
   endif
endelse

nbound_slope=n_elements(bound_slope)
mmin=bound_slope(0)
mmax=bound_slope(nbound_slope-1L)
nmass_sampling=5000L
msampling=mmin+findgen(nmass_sampling)*(mmax-mmin)/double(nmass_sampling-1L)

ni=n_elements(mzas_i)
nm=n_elements(mzas_m)
mla =dblarr(nmass_sampling)
mlos=dblarr(nmass_sampling,nel2follow)
;; interpolate the yields values on the sampled mass data
for i=0L,nmass_sampling-1L do begin
   ;; case: intermediate mass
   if(msampling(i) lt mass_separatrix)then begin
      mbase=mzas_i
      mbase_los=mlos_i
      mbase_all=mla_i
      nlast=ni
      elbase=el_i
   endif else begin
      mbase=mzas_m
      mbase_los=mlos_m
      mbase_all=mla_m
      nlast=nm
      elbase=el_m
      if keyword_set(kobayashi)then begin
         ind=where(elbase eq 'Mg',nfound)
         mbase_los(ind)=0.0d0
         for j=0L,n_elements(mkobayashi)-1L do begin
            ind=where(elbase eq 'Mg' and mbase eq mkobayashi(j),nfound)
            ;; if(nfound eq 0L)then print,mkobayashi(j),' not found'
            if(nfound eq 1L)then begin
               i0=ind(0)
               ;; print,mkobayashi(j),'     found'
               mbase_los(i0)=mgkobayashi(j)
            endif
         endfor
      endif
   endelse

   ind=where(mbase ge msampling(i),nind)
   if(nind eq 0)then begin
      ;; case where the sampling mass is larger than max(mbase)
      ;; -> extrapolate up from last mbase
      mla(i)=mbase_all(nlast-1L)*msampling(i)/mbase(nlast-1L)
      for j=0L,nel2follow-1L do begin
         indel=where(elbase eq list_elements(j),nel)
         ii=indel(nel-1L)
         mlos(i,j)=mbase_los(ii)*msampling(i)/mbase(nlast-1L)
      endfor
   endif else begin
      i1=ind(0)
      if(i1 eq 0L)then begin
         ;; case where the sampling mass is smaller than min(mbase)
         ;; -> extrapolate down from first mbase
         mla(i)=mbase_all(0)*msampling(i)/mbase(0)
         for j=0L,nel2follow-1L do begin
            indel=where(elbase eq list_elements(j),nel)
            ii=indel(0)
            mlos(i,j)=mbase_los(ii)*msampling(i)/mbase(0)
         endfor
      endif else begin
         ;; case where the sampling mass is within the range of
         ;; mbase values
         ;; -> interpolate
         indel=where(mbase lt msampling(i) and elbase eq 'p',nel)
         ii0=indel(nel-1L)
         indel=where(mbase gt msampling(i) and elbase eq 'p',nel)
         ii1=indel(0L)
         mratio=(msampling(i)-mbase(ii0))/(mbase(ii1)-mbase(ii0))
         mla(i)=mbase_all(ii0)+(mbase_all(ii1)-mbase_all(ii0))*mratio
         for j=0L,nel2follow-1L do begin
            indel=where(mbase lt msampling(i) and elbase eq list_elements(j),nel)
            ii0=indel(nel-1L)
            indel=where(mbase ge msampling(i) and elbase eq list_elements(j),nel)
            ii1=indel(0L)
            mlos(i,j)=mbase_los(ii0)+(mbase_los(ii1)-mbase_los(ii0))*mratio
         endfor
      endelse
   endelse
endfor

esn_table=dblarr(nmass_sampling)
if keyword_set(lc13rot) or keyword_set(lc13norot) then begin
   ;; Chieffi & Limongi 2013 SN kinetic energy
   mesn_sampled     =[13.0,15.0,20.0,25.0,30.0,40.0,60.0,80.0,120.]
   ecl13rot  =[1.00,1.00,1.17,1.91,2.05,2.21,7.39,6.55,2.91]
   ecl13norot=[1.00,1.00,1.00,1.15,1.63,1.77,4.22,4.32,5.48]
   if keyword_set(lc13rot  )then esnsampled=ecl13rot
   if keyword_set(lc13norot)then esnsampled=ecl13norot
endif else if(keyword_set(fileesn))then begin
   ;; Limongi & Chieffi 2018
   interpolate_files_esn,mesn_sampled,zzas_m_esn,vel_m_esn,esnsampled,fileesn=fileesn,vel=vel,constantomega=constantomega
   print,mesn_sampled
endif else begin
   esn_table(0L:nmass_sampling-1L)=1d0
endelse
ncl13=n_elements(mesn_sampled)
if keyword_set(lc13rot) or keyword_set(lc13norot) or keyword_set(fileesn)then begin
;; interpolate the SN energy values
for i=0L,nmass_sampling-1L do begin
   ;; case: intermediate mass
   if(msampling(i) lt mass_separatrix)then begin
      esn_table(i)=0.0d0
   endif else begin
      ind=where(mesn_sampled ge msampling(i),nind)
      if(nind eq 0)then begin
         ;; case where the sampling mass is larger than max(mesn_sampled)
         ;; -> Use the last value
         esn_table(i)=esnsampled(ncl13-1L)
      endif else begin
         i1=ind(0)
         if(i1 eq 0L)then begin
            ;; case where the sampling mass is smaller than min(mesn_sampled)
            ;; -> Use the lowest value
            esn_table(i)=esnsampled(0)
         endif else begin
            ;; case where the sampling mass is within the range of
            ;; mesn_sampled values
            ;; -> interpolate
            indel=where(mesn_sampled lt msampling(i),nel)
            ii0=indel(nel-1L)
            indel=where(mesn_sampled gt msampling(i),nel)
            ii1=indel(0L)
            mratio=(msampling(i)-mesn_sampled(ii0))/(mesn_sampled(ii1)-mesn_sampled(ii0))
            esn_table(i)=esnsampled(ii0)+(esnsampled(ii1)-esnsampled(ii0))*mratio
         endelse
      endelse
   endelse
endfor
endif
;; Use the failed fraction
for i=0L,nmass_sampling-1L do begin
   if keyword_set(mfailed)then begin
      if(msampling(i) gt mfailed)then esn_table(i)=(1d0-failed_fraction)*esn_table(i)
   endif
endfor
esn_table=esn_table*1d51


;; ==========================================================
;; Compute the mass in dust for each element
;; ==========================================================
mlos_dust=dblarr(nmass_sampling,nel2follow)
mlos_dust_olivine =dblarr(nmass_sampling,nel2follow) ;; Following the Gjergo et al 18 framework (MgSiFeO4)
mlos_dust_pyroxene=dblarr(nmass_sampling,nel2follow) ;; Following the Gjergo et al 18 framework (MgFeSi2O6)
;; mlos_dust(0L:nmass_sampling-1L,0L:nel2follow-1L)=1d-10
mlos_dust(0L:nmass_sampling-1L,0L:nel2follow-1L)=0.0d0
ind=where(list_elements eq 'C') &jC =ind(0)
ind=where(list_elements eq 'O') &jO =ind(0)
ind=where(list_elements eq 'S') &jS =ind(0)
ind=where(list_elements eq 'Mg')&jMg=ind(0)
ind=where(list_elements eq 'Si')&jSi=ind(0)
ind=where(list_elements eq 'Fe')&jFe=ind(0)
if keyword_set(dwek)then begin
;; Dwek 98
   dC_AGB_COgt1 =0.99d0 ;; instead of 1.0 (to avoid num. issues)
   dO_AGB_COlt1 =0.8d0
   dS_AGB_COlt1 =0.8d0
   dMg_AGB_COlt1=0.8d0
   dSi_AGB_COlt1=0.8d0
   dFe_AGB_COlt1=0.8d0
   dC_SN =0.5d0
   dO_SN =0.8d0
   dS_SN =0.8d0
   dMg_SN=0.8d0
   dSi_SN=0.8d0
   dFe_SN=0.8d0
   dOli_AGB=0.99d0 ;; instead of 1.0 (to avoid num. issues)
   dOli_SN =0.8d0
   dPyr_AGB=0.99d0 ;; instead of 1.0 (to avoid num. issues)
   dPyr_SN =0.8d0
   ;; dOli_AGB=0.8d0
   ;; dOli_SN =0.8d0
   ;; dPyr_AGB=0.8d0
   ;; dPyr_SN =0.8d0
endif else begin
;; Popping et al. 17
   dC_AGB_COgt1 =0.2d0
   dO_AGB_COlt1 =0.2d0
   dS_AGB_COlt1 =0.2d0
   dMg_AGB_COlt1=0.2d0
   dSi_AGB_COlt1=0.2d0
   dFe_AGB_COlt1=0.2d0
   dC_SN =0.15d0
   dO_SN =0.15d0
   dS_SN =0.15d0
   dMg_SN=0.15d0
   dSi_SN=0.15d0
   dFe_SN=0.15d0
   dOli_AGB=0.20d0
   dOli_SN =0.15d0
   dPyr_AGB=0.20d0
   dPyr_SN =0.15d0
endelse
muC =12.0107d0
muO =15.9990d0
muS =32.0650d0
muMg=24.3050d0
muSi=28.0855d0
muFe=55.8450d0
for i=0L,nmass_sampling-1L do begin
   ;; AGB winds
   if(msampling(i) lt mass_separatrix)then begin
      ;; C/O > 1 (number ratio, not mass)
      c2oratio=mlos(i,jC)/mlos(i,jO)*muO/muC
      if(c2oratio gt 1.0)then begin
         mlos_dust(i,jC)=dC_AGB_COgt1*(mlos(i,jC)-0.75d0*mlos(i,jO))
         mlos_dust_olivine (i,jC )=mlos_dust(i,jC) ;; dumb stuff to have all dust  in the same array
         mlos_dust_pyroxene(i,jC )=mlos_dust(i,jC) ;; dumb stuff to have all dust  in the same array
      ;; C/O < 1
      endif else begin
         for j=0L,nel2follow-1L do begin
            ll=list_elements(j)
            if(ll eq 'O')then begin
               mlos_dust(i,j)=muO*(dS_AGB_COlt1*mlos(i,jS)/muS+dMg_AGB_COlt1*mlos(i,jMg)/muMg+dSi_AGB_COlt1*mlos(i,jSi)/muSi+dFe_AGB_COlt1*mlos(i,jFe)/muFe)
               ;; if(dO_AGB_COlt1*mlos(i,jO) lt mlos_dust(i,j))then mlos_dust(i,j)=dO_AGB_COlt1*mlos(i,jO)
            endif
            if(ll eq 'S' )then mlos_dust(i,j)=dS_AGB_COlt1 *mlos(i,j)
            if(ll eq 'Mg')then mlos_dust(i,j)=dMg_AGB_COlt1*mlos(i,j)
            if(ll eq 'Si')then mlos_dust(i,j)=dSi_AGB_COlt1*mlos(i,j)
            if(ll eq 'Fe')then mlos_dust(i,j)=dFe_AGB_COlt1*mlos(i,j)
         endfor
         ;; aa=[mlos(i,jO)/(4d0*muO),mlos(i,jMg)/muMg,mlos(i,jSi)/muSi,mlos(i,jFe)/muFe]
         ;; mlos_dust_olivine(i,jO )=dOli_AGB*min(aa)*4d0*muO
         ;; mlos_dust_olivine(i,jMg)=dOli_AGB*min(aa)    *muMg
         ;; mlos_dust_olivine(i,jSi)=dOli_AGB*min(aa)    *muSi
         ;; mlos_dust_olivine(i,jFe)=dOli_AGB*min(aa)    *muFe
         aa=[mlos(i,jO)/(compound(3)*muO),mlos(i,jMg)/(compound(0)*muMg),mlos(i,jSi)/(compound(2)*muSi),mlos(i,jFe)/(compound(1)*muFe)]
         mlos_dust_olivine(i,jO )=dOli_AGB*min(aa)*compound(3)*muO
         mlos_dust_olivine(i,jMg)=dOli_AGB*min(aa)*compound(0)*muMg
         mlos_dust_olivine(i,jSi)=dOli_AGB*min(aa)*compound(2)*muSi
         mlos_dust_olivine(i,jFe)=dOli_AGB*min(aa)*compound(1)*muFe
         aa=[mlos(i,jO)/(6d0*muO),mlos(i,jMg)/muMg,mlos(i,jSi)/(2d0*muSi),mlos(i,jFe)/muFe]
         mlos_dust_pyroxene(i,jO )=dPyr_AGB*min(aa)*6d0*muO
         mlos_dust_pyroxene(i,jMg)=dPyr_AGB*min(aa)    *muMg
         mlos_dust_pyroxene(i,jSi)=dPyr_AGB*min(aa)*2d0*muSi
         mlos_dust_pyroxene(i,jFe)=dPyr_AGB*min(aa)    *muFe
      endelse
   ;; SNII
   endif else begin
      for j=0L,nel2follow-1L do begin
         ll=list_elements(j)
         if(ll eq 'C')then mlos_dust(i,j)=dC_SN*mlos(i,j)
         if(ll eq 'O')then begin
            mlos_dust(i,j)=muO*(dS_SN*mlos(i,jS)/muS+dMg_SN*mlos(i,jMg)/muMg+dSi_SN*mlos(i,jSi)/muSi+dFe_SN*mlos(i,jFe)/muFe)
            ;; if(dO_SN*mlos(i,jO) lt mlos_dust(i,j))then mlos_dust(i,j)=dO_SN*mlos(i,jO)
         endif
         if(ll eq 'S' )then mlos_dust(i,j)=dS_SN *mlos(i,j)
         if(ll eq 'Mg')then mlos_dust(i,j)=dMg_SN*mlos(i,j)
         if(ll eq 'Si')then mlos_dust(i,j)=dSi_SN*mlos(i,j)
         if(ll eq 'Fe')then mlos_dust(i,j)=dFe_SN*mlos(i,j)
      endfor
      mlos_dust_olivine (i,jC )=mlos_dust(i,jC) ;; dumb stuff to have all dust  in the same array
      mlos_dust_pyroxene(i,jC )=mlos_dust(i,jC) ;; dumb stuff to have all dust  in the same array
      ;; aa=[mlos(i,jO)/(4d0*muO),mlos(i,jMg)/muMg,mlos(i,jSi)/muSi,mlos(i,jFe)/muFe]
      ;; mlos_dust_olivine(i,jO )=dOli_SN*min(aa)*4d0*muO
      ;; mlos_dust_olivine(i,jMg)=dOli_SN*min(aa)*muMg
      ;; mlos_dust_olivine(i,jSi)=dOli_SN*min(aa)*muSi
      ;; mlos_dust_olivine(i,jFe)=dOli_SN*min(aa)*muFe
      aa=[mlos(i,jO)/(compound(3)*muO),mlos(i,jMg)/(compound(0)*muMg),mlos(i,jSi)/(compound(2)*muSi),mlos(i,jFe)/(compound(1)*muFe)]
      mlos_dust_olivine(i,jO )=dOli_SN*min(aa)*compound(3)*muO
      mlos_dust_olivine(i,jMg)=dOli_SN*min(aa)*compound(0)*muMg
      mlos_dust_olivine(i,jSi)=dOli_SN*min(aa)*compound(2)*muSi
      mlos_dust_olivine(i,jFe)=dOli_SN*min(aa)*compound(1)*muFe
      aa=[mlos(i,jO)/(6d0*muO),mlos(i,jMg)/muMg,mlos(i,jSi)/(2d0*muSi),mlos(i,jFe)/muFe]
      mlos_dust_pyroxene(i,jO )=dPyr_SN*min(aa)*6d0*muO
      mlos_dust_pyroxene(i,jMg)=dPyr_SN*min(aa)    *muMg
      mlos_dust_pyroxene(i,jSi)=dPyr_SN*min(aa)*2d0*muSi
      mlos_dust_pyroxene(i,jFe)=dPyr_SN*min(aa)    *muFe
   endelse
endfor

;; ==========================================================
;; Interpolate the age of stars
;; ==========================================================
asampling=dblarr(nmass_sampling)
for i=0L,nmass_sampling-1L do begin
   asampling(i)=alog10(interpol(10d0^logage,mzas_fromage,msampling(i)))
endfor

;; ==========================================================
;; Initialize the IMF
;; ==========================================================
xi=findgen(nmass_sampling)
prefactor=1.0d0
if keyword_set(chabrier)then begin
   ind=where(msampling lt 1.0d0,nind)
   lmm=alog10(msampling(ind))
   mi=0.5d0*msampling(ind(nind-1L))
   if keyword_set(revisedchabrier)then begin
      xi(ind)=0.093d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.2d0))^2d0/(2d0*0.55d0^2d0))
      ind=where(msampling ge 1.0d0)
      mi=mi+0.5d0*msampling(ind(0L))
      lmm=alog10(mi)
      xii=0.093d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.2d0))^2d0/(2d0*0.55d0^2d0)) 
   endif else begin
      xi(ind)=0.158d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0))
      ind=where(msampling ge 1.0d0)
      mi=mi+0.5d0*msampling(ind(0L))
      lmm=alog10(mi)
      xii=0.158d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0)) 
   endelse
   ;; xi(ind)=msampling(ind)^(-2.3d0)*(xii/mi^(-2.3d0))
   xi(ind)=msampling(ind)^(alpha_slope(0))*(xii/mi^(alpha_slope(0)))
endif else begin
   for j=0L,nbound_slope-2L do begin
      ind=where(msampling ge bound_slope(j) and msampling lt bound_slope(j+1),nind)
      xi(ind)=msampling(ind)^alpha_slope(j)
      if(j gt 0L)then begin
         prefactor=last_xi/mm^alpha_slope(j)
         xi(ind)=prefactor*xi(ind)
      endif
      if(j lt nbound_slope-2L)then begin
         ii=ind(nind-1L)
         mm=0.5d0*(msampling(ii)+msampling(ii+1L))
         last_xi=prefactor*mm^alpha_slope(j)
      endif
      if(j eq nbound_slope-2L)then begin
         xi(nmass_sampling-1L)=prefactor*msampling(nmass_sampling-1L)^alpha_slope(j)
      endif
   endfor
endelse
if keyword_set(ASNIa)then begin
   mIaLow= 3.0d0
   mIaHigh=16.0d0
   ind=where(msampling gt mIaLow and msampling le mIaHigh)
   print,'SNIa number/mass fraction over the entire mass range:',total(xi(ind))*ASNIa/total(xi),total(msampling(ind)*xi(ind))*ASNIa/total(msampling*xi)
   xi(ind)=xi(ind)*(1d0-ASNIa)
endif
;; renorm=int_tabulated(msampling,xi)
;; print,'renorm=',renorm
renorm=msampling(nmass_sampling-1L)-msampling(0)
print,'renorm=',renorm
xi=xi/renorm
;; ==========================================================
;; Plot the IMF
;; ==========================================================
renorm=int_tabulated(msampling,xi*msampling)
if not keyword_set(noplot)then begin
if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename='imf.eps',/encaps,xs=30,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif else begin
   iw=0
   window,iw,xs=1000,ys=500
   th=1
endelse
plot_oi,msampling,alog10(xi/renorm),/ys,xtitle='M!dZAS!n (M!dsun!n)',ytitle='log IMF',xth=th,yth=th,th=th
if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charthick=1
endif
endif


;; ==========================================================
;; Compute the mass returned as a function of time
;; ==========================================================
if not keyword_set(maxage)then maxage=2d10
lt1=alog10(maxage)
t0=1d6
lt0=alog10(t0)
if not keyword_set(dlt)then dlt=0.1d0 ; dex
nt=(lt1-lt0)/dlt
ltime_evol=lt0+findgen(nt)*(lt1-lt0)/double(nt)
asampling_linear=10d0^asampling
time_evol=10d0^ltime_evol

etasn_return=dblarr(nt)
msn_return  =dblarr(nt)
esnII_return=dblarr(nt)
eWR_return=dblarr(nt)
mass_return_all=dblarr(nt)
mass_return_los=dblarr(nt,nel2follow)
mass_return_los_dust=dblarr(nt,nel2follow)
mass_return_los_dust_olivine=dblarr(nt,nel2follow)
mass_return_los_dust_pyroxene=dblarr(nt,nel2follow)
mass_return_all_agb=dblarr(nt)
mass_return_los_agb=dblarr(nt,nel2follow)
mass_return_los_dust_agb=dblarr(nt,nel2follow)
mass_return_los_dust_olivine_agb=dblarr(nt,nel2follow)
mass_return_los_dust_pyroxene_agb=dblarr(nt,nel2follow)
mass_return_all_snii=dblarr(nt)
mass_return_los_snii=dblarr(nt,nel2follow)
mass_return_los_dust_snii=dblarr(nt,nel2follow)
mass_return_los_dust_olivine_snii=dblarr(nt,nel2follow)
mass_return_los_dust_pyroxene_snii=dblarr(nt,nel2follow)
for i=0L,nt-1L do begin
   ind=where(asampling le ltime_evol(i),nind)
   indagb =where(msampling lt mass_separatrix and asampling le ltime_evol(i),nindagb )
   indsnii=where(msampling ge mass_separatrix and asampling le ltime_evol(i),nindsnii)
   if(nind gt 1)then begin
      xx=msampling(ind)
      yy=xi(ind)*mla(ind)
      mass_return_all(i)=int_tabulated(xx,yy)/renorm
      for j=0L,nel2follow-1L do begin
         yy=xi(ind)*mlos(ind,j)
         mass_return_los(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(ind)*mlos_dust(ind,j)
         mass_return_los_dust(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(ind)*mlos_dust_olivine(ind,j)
         mass_return_los_dust_olivine(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(ind)*mlos_dust_pyroxene(ind,j)
         mass_return_los_dust_pyroxene(i,j)=int_tabulated(xx,yy)/renorm
      endfor
   endif else begin
      mass_return_all(i)=0.0d0
      mass_return_los(i,0L:nel2follow-1L)=0.0d0
      mass_return_los_dust(i,0L:nel2follow-1L)=0.0d0
      mass_return_los_dust_olivine(i,0L:nel2follow-1L)=0.0d0
      mass_return_los_dust_pyroxene(i,0L:nel2follow-1L)=0.0d0
   endelse
   if(nindagb gt 1)then begin
      xx=msampling(indagb)
      yy=xi(indagb)*mla(indagb)
      mass_return_all_agb(i)=int_tabulated(xx,yy)/renorm
      for j=0L,nel2follow-1L do begin
         yy=xi(indagb)*mlos(indagb,j)
         mass_return_los_agb(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indagb)*mlos_dust(indagb,j)
         mass_return_los_dust_agb(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indagb)*mlos_dust_olivine(indagb,j)
         mass_return_los_dust_olivine_agb(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indagb)*mlos_dust_pyroxene(indagb,j)
         mass_return_los_dust_pyroxene_agb(i,j)=int_tabulated(xx,yy)/renorm
      endfor
   endif else begin
      mass_return_all_agb(i)=0.0d0
      mass_return_los_agb(i,0L:nel2follow-1L)=0.0d0
      mass_return_los_dust_agb(i,0L:nel2follow-1L)=0.0d0
   endelse

   if(nindsnii gt 1)then begin
      xx=msampling(indsnii)
      yy=xi(indsnii)*mla(indsnii)
      mass_return_all_snii(i)=int_tabulated(xx,yy)/renorm
      for j=0L,nel2follow-1L do begin
         yy=xi(indsnii)*mlos(indsnii,j)
         mass_return_los_snii(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indsnii)*mlos_dust(indsnii,j)
         mass_return_los_dust_snii(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indsnii)*mlos_dust_olivine(indsnii,j)
         mass_return_los_dust_olivine_snii(i,j)=int_tabulated(xx,yy)/renorm
         yy=xi(indsnii)*mlos_dust_pyroxene(indsnii,j)
         mass_return_los_dust_pyroxene_snii(i,j)=int_tabulated(xx,yy)/renorm
      endfor
   endif else begin
      mass_return_all_snii(i)=0.0d0
      mass_return_los_snii(i,0L:nel2follow-1L)=0.0d0
      mass_return_los_dust_snii(i,0L:nel2follow-1L)=0.0d0
   endelse

   ;; indsnII=where(msampling ge mass_separatrix and asampling le ltime_evol(i),nindsnII)

   indsnall=where(msampling ge mass_separatrix and asampling le ltime_evol(i),nindsnall)
   if(nindsnall ge 1L)then begin
      xx=msampling(indsnall)
      yy=xi(indsnall)*msampling(indsnall)
      etasn_return(i)=int_tabulated(xx,yy)/renorm
      msn_return  (i)=int_tabulated(xx,yy)/int_tabulated(xx,xi(indsnall))
   endif
   ;; indsnII=where(msampling ge mass_separatrix and msampling lt 20.0d0 and asampling le ltime_evol(i),nindsnII)
   indsnII=where(msampling ge mass_separatrix and asampling le ltime_evol(i),nindsnII)
   indsnIIhn=where(msampling ge 20.0d0 and asampling le ltime_evol(i),nindsnIIhn)
   if(nindsnII ge 1L)then begin
      xx=msampling(indsnII)
      yy=xi(indsnII)*msampling(indsnII)
      ;; etasn_return(i)=int_tabulated(xx,yy)/renorm
      ;; msn_return  (i)=int_tabulated(xx,yy)/int_tabulated(xx,xi(indsnII))
      yy=xi(indsnII)*esn_table(indsnII)
      esnII_return(i)=int_tabulated(xx,yy)/renorm
   endif
   ;; print,'renorm=',renorm
   if(nindsnIIhn ge 1L)then begin
      xx=msampling(indsnIIhn)
      yy=xi(indsnIIhn)*msampling(indsnIIhn)
      ;; etasn_return(i)=etasn_return(i)+int_tabulated(xx,yy)/renorm
      ;; msn_return  (i)=msn_return  (i)+int_tabulated(xx,yy)/int_tabulated(xx,xi(indsnIIhn))
      yy=(1d0-hnfraction)*xi(indsnIIhn)*esn_table(indsnIIhn)+hnfraction*xi(indsnIIhn)*1d52
      esnII_return(i)=esnII_return(i)+int_tabulated(xx,yy)/renorm
   endif

   indWR=where(msampling ge 20.0 and 0.9d0*asampling_linear le time_evol(i),nindWR)
   ;; indWR=where(msampling ge 20.0 and asampling le ltime_evol(i),nindWR)
   if(nindWR ge 1L)then begin
      xx=msampling(indWR)
      yy=xi(indWR)*msampling(indWR)
      WRduration=time_evol(i)-(0.9d0*asampling_linear(indWR))
      for j=0L,nindWR-1L do begin
         jj=indWR(j)
         if(WRduration(j) gt 0.1d0*asampling_linear(jj))then begin
            WRduration(j)=0.1d0*asampling_linear(jj)
         endif
      endfor
      ;; WRduration=0.1d0*10d0^asampling(indWR) ;; Assume that WR phase lasts 10% of the age (Meynet & Maeder 05)
      yy=xi(indWR)* WRduration*0.5d0*MdotWR*(vWR*1d5)^2d0
      eWR_return(i)=int_tabulated(xx,yy)/renorm
   endif

endfor

;; ==========================================================
;; Plot the mass return as a function of time
;; ==========================================================
time=10d0^ltime_evol
if not keyword_set(noplot)then begin
if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='toto.eps'
   device,filename=fileeps,/encaps,xs=30,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif else begin
   iw=iw+1
   window,iw,xs=1000,ys=500
   th=1
endelse
y=mass_return_all
x=time/1d9
if not keyword_set(mreturnmin)then mreturnmin=1d-7
if not keyword_set(mreturnmax)then mreturnmax=1d0
plot_oo,x,y,xtitle='time (Gyr)',ytitle='Mlost/Msun',xr=[4d-3,2d1],/xs,yr=[mreturnmin,mreturnmax],/ys,xth=th,yth=th,th=th,title=title
;; y=mass_return_all_agb
;; oplot,x,y,th=th/2.,lines=2
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   last_mass(j)=mass_return_los(nt-1L,j)
endfor
indsort=reverse(sort(last_mass))
plotelem=lonarr(nel2follow)
if keyword_set(elements2plot) then begin
   for jj=0L,nel2follow-1L do begin
      j=indsort(jj)
      ind=where(elements2plot eq list_elements(j),nexist)
      if(nexist ge 1L)then plotelem(j)=1L
   endfor
endif else begin
   plotelem(*)=1L
endelse
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   if(plotelem(j) eq 1L)then begin
   mycolor=j+2
   y=mass_return_los(*,j)
   oplot,x,y,th=th,color=mycolor
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(x)*1.1
   endif else begin
      xpos=max(x)*1.3
   endelse
   xyouts,xpos,y(nt-1L),list_elements(j),color=mycolor
   if keyword_set(plotagb)then begin
      y=mass_return_los_agb(*,j)
      oplot,x,y,th=th/2.,lines=2,color=mycolor
   endif
   if keyword_set(plotdust)then begin
      y=mass_return_los_dust(*,j)
      oplot,x,y,th=th/2.,lines=2,color=mycolor
      if keyword_set(olivine)then begin
         y=mass_return_los_dust_olivine(*,j)
         oplot,x,y,th=th/2.,lines=1,color=mycolor
      endif
      if keyword_set(pyroxene)then begin
         y=mass_return_los_dust_pyroxene(*,j)
         oplot,x,y,th=th/2.,lines=4,color=mycolor
      endif
   endif
   endif
endfor
if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif
endif

print,'eta_sn                      :',etasn_return(nt-1L),format='(A,f8.5)'
print,'m_sn                        :',msn_return  (nt-1L),format='(A,f8.5)'
print,'eSNII (erg/Msun)            :',esnII_return(nt-1L),format='(A,e9.2)'
print,'total mass return           :',mass_return_all(nt-1L),format='(A,f8.5)'
print,'mass return of listed metals:',total(mass_return_los(nt-1L,2L:nel2follow-1L)),format='(A,f8.5)'
print,'mass return of not H&He     :',mass_return_all(nt-1L)-total(mass_return_los(nt-1L,0L:1L)),format='(A,f8.5)'
print,'metallicity of the ejecta   :',total(mass_return_los(nt-1L,2L:nel2follow-1L))/mass_return_all(nt-1L),format='(A,f8.5)'
print,'      ',list_elements,format='(A7,11A8)'
print,'total:',mass_return_los(nt-1L,*),format='(A7,11e8.1)'
print,'AGB  :',mass_return_los_agb(nt-1L,*),format='(A7,11e8.1)'
print,'%AGB :',mass_return_los_agb(nt-1L,*)/mass_return_los(nt-1L,*)*100.0,format='(A7,11f8.1)'
;;1.0 is for Fluor: change this
ref_mass_lost_all=[2.4e-01, 1.5e-01, 7.5e-03, 1.7e-03, 1.7e-02, 1.0, 1.4e-03, 5.1e-04, 1.4e-03, 6.5e-04, 1.4e-03]
ref_mass_lost_agb=[1.7e-01, 7.6e-02, 1.2e-03, 9.1e-04, 1.5e-03, 1.0, 3.8e-04, 1.2e-04, 1.2e-04, 9.9e-05, 2.1e-04]
ref_mass_lost_sn=ref_mass_lost_all-ref_mass_lost_agb
print,'ratio:',mass_return_los(nt-1L,*)/ref_mass_lost_all,' compared to standard set',format='(A7,11f8.2,A)'
print,'ratio:',alog10(mass_return_los(nt-1L,*)/ref_mass_lost_all),' [dex] compared to standard set',format='(A7,11f8.2,A)'



;; Asplun et al 2009
XHAsplund=0.7381d0
lAO =8.69d0
lAF =4.56d0
lAFe=7.50d0
lAC =8.43d0
lAN =7.83d0
lAMg=7.60d0
lASi=7.51d0
lAS =7.12d0
lANe=7.93d0
lAHe=10.925227d0 ; This is the value necessary to get the H, He, Z mass fraction of Asplund+09
lAH =12.0d0
logH =lAH                  +alog10(XHAsplund)
logHe=lAHe+alog10(3.9993d0)+alog10(XHAsplund)
logF =lAF +alog10(18.998d0)+alog10(XHAsplund)
logO =lAO +alog10(15.999d0)+alog10(XHAsplund)
logFe=lAFe+alog10(55.845d0)+alog10(XHAsplund)
logC =lAC +alog10(12.011d0)+alog10(XHAsplund)
logN =lAN +alog10(14.007d0)+alog10(XHAsplund)
logMg=lAMg+alog10(24.305d0)+alog10(XHAsplund)
logSi=lASi+alog10(28.086d0)+alog10(XHAsplund)
logS =lAS +alog10(32.065d0)+alog10(XHAsplund)
logNe=lANe+alog10(20.1797d0)+alog10(XHAsplund)
;; lAlist=[lAH ,lAHe ,lAO ,lAC ,lAFe ,lAN ,lAMg ,lASi ,lAS ,laNe ]
;; llist =[logH,logHe,logO,logC,logFe,logN,logMg,logSi,logS,logNe] !Wrong!!
lAlist=[lAH ,lAHe ,lAC ,lAN ,lAO ,lAF ,lANe ,lAMg ,lASi ,lAS ,laFe ]
llist =[logH,logHe,logC,logN,logO,logF,logNe,logMg,logSi,logS,logFe]
;; plot,mass_return_los_agb(nt-1L,*)/mass_return_los(nt-1L,*)*100.0,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10
if not keyword_set(noplot)then begin
if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename='ejecta_abundances'+suffix+'.eps',/encaps,xs=45,ys=15,/color
   th=5
   !p.charthick=5
endif else begin
   iw=iw+1
   window,iw,xs=512*3,ys=512
endelse
!p.multi=[0,3,1]
!p.charsize=3
yy=alog10(mass_return_los(nt-1L,*)/mass_return_los(nt-1L,0L))-(llist-llist(0))
plot,yy,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10,ytitle='[X/H]!dej!n',yr=[-0.5,2.5],/ys,xth=th,yth=th,th=th
yy=alog10((mass_return_los(nt-1L,*)-mass_return_los_agb(nt-1L,*))/(mass_return_los(nt-1L,0L)-mass_return_los_agb(nt-1L,0L)))-(llist-llist(0))
plot,yy,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10,ytitle='[X/H]!dej,SNII!n',yr=[-0.5,2.5],/ys,xth=th,yth=th,th=th
yy=alog10(mass_return_los_agb(nt-1L,*)/mass_return_los_agb(nt-1L,0L))-(llist-llist(0))
plot,yy,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10,ytitle='[X/H]!dej,AGB!n',yr=[-0.5,2.5],/ys,xth=th,yth=th,th=th
!p.charsize=1.5
!p.multi=0
if keyword_set(printps)then begin
   device,/close
   device,filename='ejecta_masses'+suffix+'.eps',/encaps,xs=30,ys=15,/color
endif else begin
   iw=iw+1
   window,iw,xs=512*2,ys=512
endelse
!p.charsize=2
!p.multi=[0,2,1]
yy=mass_return_los(nt-1L,*)
plot_io,yy,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10,ytitle='M!dX,ej!n/M!dSSP!n',/ys,yr=[7d-5,5d-1],xth=th,yth=th,th=th
yy=mass_return_los_agb(nt-1L,*)
oplot,yy,psym=10,lines=2,th=th
yy=mass_return_los_agb(nt-1L,*)/mass_return_los(nt-1L,*)
plot,yy,xtickname=list_elements,xtitle='element',xstyle=1,xticks=nel2follow-1,psym=10,ytitle='AGB fraction!dX,ej!n',xth=th,yth=th,th=th
!p.charsize=1.5
!p.multi=0
if keyword_set(printps)then device,/close
endif

;; ==========================================================
;; Write useful data
;; ==========================================================
if keyword_set(fileout)then begin
   openw,1,fileout
   ncolumns=2+nel2follow*2L
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   myheader='# time     Mtot     '
   for i=0L,nel2follow-1L do begin
      mystring=list_elements(i)+' total'
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   for i=0L,nel2follow-1L do begin
      if keyword_set(olivine)then begin
         mystring=list_elements(i)+' Odust'
      endif else if keyword_set(pyroxene)then begin
         mystring=list_elements(i)+' Pdust'
      endif else begin
         mystring=list_elements(i)+' dust'
      endelse
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   printf,1,myheader
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   for i=0L,nt-1L do begin
      met_tot=mass_return_all(i)-total(mass_return_los(i,0L:1L))
      myformat='('+strcompress(ncolumns,/remove_all)+'e10.3)'      
      ;; printf,1,time(i),mass_return_all(i),mass_return_los(i,0L:nel2follow-1L),mass_return_los_dust(i,0L:nel2follow-1L),format=myformat
      if keyword_set(olivine)then begin
         printf,1,time(i),total(mass_return_los(i,0L:nel2follow-1L)),mass_return_los(i,0L:nel2follow-1L),mass_return_los_dust_olivine(i,0L:nel2follow-1L),format=myformat
      endif else if keyword_set(pyroxene)then begin
         printf,1,time(i),total(mass_return_los(i,0L:nel2follow-1L)),mass_return_los(i,0L:nel2follow-1L),mass_return_los_dust_pyroxene(i,0L:nel2follow-1L),format=myformat
      endif else begin
         printf,1,time(i),total(mass_return_los(i,0L:nel2follow-1L)),mass_return_los(i,0L:nel2follow-1L),mass_return_los_dust(i,0L:nel2follow-1L),format=myformat
      endelse
   endfor
   close,1

   openw,1,fileout+'.esnII'
   for i=0L,nt-1L do begin
      printf,1,time(i),esnII_return(i),eWR_return(i),format='(3e10.3)'
   endfor
   close,1

   openw,1,fileout+'.agb'
   ncolumns=2+nel2follow*2L
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   myheader='# time     Magb     '
   for i=0L,nel2follow-1L do begin
      mystring=list_elements(i)+' total'
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   for i=0L,nel2follow-1L do begin
      if keyword_set(olivine)then begin
         mystring=list_elements(i)+' Odust'
      endif else if keyword_set(pyroxene)then begin
         mystring=list_elements(i)+' Pdust'
      endif else begin
         mystring=list_elements(i)+' dust'
      endelse
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   printf,1,myheader
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   for i=0L,nt-1L do begin
      met_tot=mass_return_all(i)-total(mass_return_los(i,0L:1L))
      myformat='('+strcompress(ncolumns,/remove_all)+'e10.3)'      
      if keyword_set(olivine)then begin
         printf,1,time(i),total(mass_return_los_agb(i,0L:nel2follow-1L)),mass_return_los_agb(i,0L:nel2follow-1L),mass_return_los_dust_olivine_agb(i,0L:nel2follow-1L),format=myformat
      endif else if keyword_set(pyroxene)then begin
         printf,1,time(i),total(mass_return_los_agb(i,0L:nel2follow-1L)),mass_return_los_agb(i,0L:nel2follow-1L),mass_return_los_dust_pyroxene_agb(i,0L:nel2follow-1L),format=myformat
      endif else begin
         printf,1,time(i),total(mass_return_los_agb(i,0L:nel2follow-1L)),mass_return_los_agb(i,0L:nel2follow-1L),mass_return_los_dust_agb(i,0L:nel2follow-1L),format=myformat
      endelse
   endfor
   close,1

   openw,1,fileout+'.snII'
   ncolumns=2+nel2follow*2L
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   myheader='# time     Msnii    '
   for i=0L,nel2follow-1L do begin
      mystring=list_elements(i)+' total'
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   for i=0L,nel2follow-1L do begin
      if keyword_set(olivine)then begin
         mystring=list_elements(i)+' Odust'
      endif else if keyword_set(pyroxene)then begin
         mystring=list_elements(i)+' Pdust'
      endif else begin
         mystring=list_elements(i)+' dust'
      endelse
      myheader=myheader+string(mystring,format='(A10)')
   endfor
   printf,1,myheader
   myheader='#---------'
   for i=1L,ncolumns-1L do begin
      myheader=myheader+' ---------'
   endfor
   printf,1,myheader
   for i=0L,nt-1L do begin
      met_tot=mass_return_all(i)-total(mass_return_los(i,0L:1L))
      myformat='('+strcompress(ncolumns,/remove_all)+'e10.3)'      
      if keyword_set(olivine)then begin
         printf,1,time(i),total(mass_return_los_snii(i,0L:nel2follow-1L)),mass_return_los_snii(i,0L:nel2follow-1L),mass_return_los_dust_olivine_snii(i,0L:nel2follow-1L),format=myformat
      endif else if keyword_set(pyroxene)then begin
         printf,1,time(i),total(mass_return_los_snii(i,0L:nel2follow-1L)),mass_return_los_snii(i,0L:nel2follow-1L),mass_return_los_dust_pyroxene_snii(i,0L:nel2follow-1L),format=myformat
      endif else begin
         printf,1,time(i),total(mass_return_los_snii(i,0L:nel2follow-1L)),mass_return_los_snii(i,0L:nel2follow-1L),mass_return_los_dust_snii(i,0L:nel2follow-1L),format=myformat
      endelse
   endfor
   close,1


endif


;; ==========================================================
;; Plot the stellar yields
;; ==========================================================
if not keyword_set(noplot)then begin
if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename='stellaryields.eps',/encaps,xs=30,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif else begin
   iw=iw+1
   window,iw,xs=1000,ys=500
   th=1
endelse
yy=alog10(mlos(*,0))
plot_oi,msampling,yy,xtitle='M!dZAS!n (M!dsun!n)',ytitle='M!dlost!n (M!dsun!n)',xth=th,yth=th,th=th,xr=[1.0,150.0],/xs,yr=[-8.,1.3],/ys
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   if(plotelem(j) eq 1L)then begin
   mycolor=j+2
   xx=msampling
   yy=alog10(mlos(*,j))
   oplot,xx,yy,th=th,color=mycolor
   ind=where(xx ge 1.25d0)&ii=ind(0)
   xyouts,msampling(ii),alog10(mlos(ii,j))+0.1,list_elements(j),color=mycolor
   if keyword_set(plotdust)then begin
      yy=alog10(mlos_dust(*,j))
      oplot,xx,yy,th=th/2.,lines=2,color=mycolor
      yy=alog10(mlos_dust_olivine(*,j))
      oplot,xx,yy,th=th/2.,lines=1,color=mycolor
      yy=alog10(mlos_dust_pyroxene(*,j))
      oplot,xx,yy,th=th/2.,lines=4,color=mycolor
   endif
   endif
endfor

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charthick=1
endif
endif

if keyword_set(stop)then stop

end
