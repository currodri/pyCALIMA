;+
; NAME:
; run_all_yield_release
;
; PURPOSE:
; This routine is a wrapper for cmp_yield_release.pro, which computes
; the chemical mass returned for a single stellar population (only
; intermediate and massive stars, not SNIa). Several important
; parameters are simply hidden from the user. You might want to spy
; into this code and for more complete information in
; cmp_yield_release.pro where the real meat is.
;
; CATEGORY:
; Wrapper
;
; CALLING SEQUENCE:
; run_all_yield_release, chabrier=chabrier, salpeter=salpeter
;         , varying=varying, ASNIa=asnIa, mfailed=mfailed,
;         , failed_fraction=failed_fraction, mass_separatrix=mass_separatrix
;
; INPUTS:
;       Chabrier = use the Chabrier IMF
;       Salpeter = use the Salpeter IMF
;       Varying  = use the Varying IMF. The varying IMF is based upon
;       the results of metallicity-dependent IMF of Martin-Navarro+15
;
; OPTIONAL INPUTS:
;
;       ASNIa: The fraction of type Ia SNe. this amount is removed
;       from the contribution of intermediate + massive
;       stars. Default: 3.5d-2
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
;       mass_separatrix: The stellar mass below which stars are
;       considered intermediate stars and above which they are
;       considered massive stars and, hence, use different stellar
;       yields. Default:
;
;       fudgemg: the multiplication factor for Mg (LC18 underpredicts
;       Mg). Default:1
;
; OUTPUTS:
;       The output are those produced by cmp_yield_release.pro (see
;       this routine for details)
;       
; EXAMPLE:
;
;       run_all_yield_release,/chabrier,asnia=5d-2,mfailed=35.,failed_fraction=0.5,mass_separatrix=7.0
;
; MODIFICATION HISTORY:
; Written by:Yohan Dubois, 01/01/2019.
;                       e-mail: dubois@iap.fr
;December, 2020:Comments and header added by Yohan Dubois.
;-
;###################################################
;###################################################
;###################################################
pro run_all_yield_release,chabrier=chabrier,salpeter=salpeter,varying=varying,revised=revised,ASNIa=ASNIa,mfailed=mfailed,failed_fraction=failed_fraction,mass_separatrix=mass_separatrix,kobayashi=kobayashi,fudgemg=fudgemg,pyroxene=pyroxene,olivine=olivine,compound=compound,dust=dust

IF not (keyword_set(chabrier) or keyword_set(salpeter) or keyword_set(varying)) THEN BEGIN
    PRINT, 'Wrong number of arguments'
    PRINT, 'Give at least an IMF (/chabrier,/salpeter,/varying)'
    DOC_LIBRARY,'run_all_yield_release'
    RETURN
ENDIF

if not keyword_set(dust)then begin
   if keyword_set(olivine )then dust='_olivinedwek'
   if keyword_set(pyroxene)then dust='_pyroxenedwek'
endif
if not keyword_set(ASNIa)then ASNIa=3.5d-2
if not keyword_set(mfailed)then mfailed=30
if not keyword_set(fudgemg)then fudgemg=1.0d0
if not keyword_set(failed_fraction)then failed_fraction=1.0
if not keyword_set(mass_separatrix)then mass_separatrix=8.0
myvel    =[ 0 , 25 , 50 , 100 , 150 , 200 , 250 , 300 ]
;; myvel=[0,150,300]
zmet   =[ -3., -2., -1., -0.6, -0.3  , 0., 0.3 ]
zstring=['-3','-2','-1','-0.6','-0.3','0','0.3']
nz=n_elements(zmet)
nvel=n_elements(myvel)


;;Kobayashi yields
if keyword_set(kobayashi)then begin
   zmet   =alog10([ 0.00, 0.001, 0.004, 0.008, 0.02, 0.05 ]/0.01345)
   zstring1=['0' ,'0.001','0.004','0.008','0.02', '0.05']   ;; Lin z
   zstring2=['-3','-1.13','-0.53','-0.23','0.17','0.57']    ;; Log Z
   nz=n_elements(zmet)
   if keyword_set(chabrier)then begin
   for j=0,nz-1L do begin
      fileage='/home/dubois/StellarYields/PadovaStellarTracks/ages_padova_z'+zstring1(j)+'.txt'
      fileintermediate='kobayashi13agb_z'+zstring1(j)+'_simplified.txt'
      filemassive='kobayashi13snii_z'+zstring1(j)+'_simplified.txt'
      if keyword_set(mass_separatrix)then begin
         mstring='msep'+string(mass_separatrix,format='(e7.1)')+'msun'
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_chabrier_imf0.1-100msun_'+mstring+'_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endif else begin
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_chabrier_imf0.1-100msun_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endelse
      print,fileage
      print,fileintermediate
      print,filemassive
      print,fileout
      print,'======='
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,/chabrier,revised=revised,bound=[0.1,100.0],zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,fileage=fileage,fileintermediate=fileintermediate,filemassive=filemassive,mfailed=double(mfailed),mass_separatrix=mass_separatrix,/noplot,failed_fraction=failed_fraction
   endfor
   endif
   if keyword_set(salpeter)then begin
   for j=0,nz-1L do begin
      fileage='/home/dubois/StellarYields/PadovaStellarTracks/ages_padova_z'+zstring1(j)+'.txt'
      fileintermediate='kobayashi13agb_z'+zstring1(j)+'_simplified.txt'
      filemassive='kobayashi13snii_z'+zstring1(j)+'_simplified.txt'
      if keyword_set(mass_separatrix)then begin
         mstring='msep'+string(mass_separatrix,format='(e7.1)')+'msun'
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_salpeter_imf0.1-100msun_'+mstring+'_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endif else begin
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_salpeter_imf0.1-100msun_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endelse
      print,fileage
      print,fileintermediate
      print,filemassive
      print,fileout
      print,'======='
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,/salpeter,alpha=[-2.3],bound=[0.1,100.0],zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,fileage=fileage,fileintermediate=fileintermediate,filemassive=filemassive,mfailed=double(mfailed),mass_separatrix=mass_separatrix,/noplot,failed_fraction=failed_fraction
   endfor
   endif
   if keyword_set(varying)then begin
   for j=0,nz-1L do begin
      aa=2.73d0+3.1d0*zmet(j)
      if(aa lt aa0)then aa=aa0
      alpha_slope=[-aa0,-aa]
      print,zmet(j),alpha_slope,format='(3f10.2)'
      nbound_slope=n_elements(bound_slope)
      fileage='/home/dubois/StellarYields/PadovaStellarTracks/ages_padova_z'+zstring1(j)+'.txt'
      fileintermediate='kobayashi13agb_z'+zstring1(j)+'_simplified.txt'
      filemassive='kobayashi13snii_z'+zstring1(j)+'_simplified.txt'
      if keyword_set(mass_separatrix)then begin
         mstring='msep'+string(mass_separatrix,format='(e7.1)')+'msun'
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_varying_imf0.1-100msun_'+mstring+'_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endif else begin
         fileout='yields_evol_kobayashi_z'+zstring2(j)+'_varying_imf0.1-100msun_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endelse
      print,fileage
      print,fileintermediate
      print,filemassive
      print,fileout
      print,'======='
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,/varying,alpha=alpha_slope,bound=[0.1,0.6,100.0],zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,fileage=fileage,fileintermediate=fileintermediate,filemassive=filemassive,mfailed=double(mfailed),mass_separatrix=mass_separatrix,/noplot,failed_fraction=failed_fraction
   endfor
   endif
endif else begin

;; Chabrier
if keyword_set(chabrier)then begin
   for i=0,nvel-1L do begin
      for j=0,nz-1L do begin
      vstring=strcompress(myvel(i),/remove_all)
      filesecondary='limongichieffi_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      filemassive  ='limongichieffi_modelm_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      if keyword_set(mass_separatrix)then begin
         mstring='msep'+string(mass_separatrix,format='(e7.1)')+'msun'
         fileout='yields_evol_z'+zstring(j)+'_v'+vstring+'_chabrier_imf0.1-100msun_'+mstring+'_mfailed'+strcompress(mfailed,/remove_all)+'msun_ffailed'+string(failed_fraction,format='(f4.2)')+'_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endif else begin
         fileout='yields_evol_z'+zstring(j)+'_v'+vstring+'_chabrier_imf0.1-100msun_mfailed'+strcompress(mfailed,/remove_all)+'msun_ffailed'+string(failed_fraction,format='(f4.2)')+'_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      endelse
      print,filemassive
      print,filesecondary
      print,fileout
      print,'compound=',compound
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,/chabrier,revised=revised,bound=[0.1,100.0],v=myvel(i),zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,filemassive=filemassive,filesecondary=filesecondary,mfailed=double(mfailed),mass_separatrix=mass_separatrix,/noplot,failed_fraction=failed_fraction,olivine=olivine,compound=compound,pyroxene=pyroxene,/dwek
   endfor
endfor
endif

;; Salpeter
if keyword_set(salpeter)then begin
for i=0,nvelp-1L do begin
   for j=0,nz-1L do begin
      vstring=strcompress(myvel(i),/remove_all)
      filesecondary='limongichieffi_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      filemassive  ='limongichieffi_modelm_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      fileout='yields_evol_z'+zstring(j)+'_v'+vstring+'_salpeter_imf0.1-100msun_mfailed'+strcompress(mfailed,/remove_all)+'msun_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      print,filemassive
      print,filesecondary
      print,fileout
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,alpha=[-2.3],bound=[0.1,100.0],v=myvel(i),zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,filemassive=filemassive,filesecondary=filesecondary,mfailed=double(mfailed),/noplot
   endfor
endfor
endif

;; Varying IMF
if keyword_set(varying)then begin
aa0=1.0d0
for i=0,nvel-1L do begin
   for j=0,nz-1L do begin
      aa=2.73d0+3.1d0*zmet(j)
      if(aa lt aa0)then aa=aa0
      alpha_slope=[-aa0,-aa]
      print,zmet(j),alpha_slope,format='(3f10.2)'
      nbound_slope=n_elements(bound_slope)
      vstring=strcompress(myvel(i),/remove_all)
      filesecondary='limongichieffi_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      filemassive  ='limongichieffi_modelm_z'+zstring(j)+'_vel'+strcompress(myvel(i),/remove_all)+'_simplified.txt'
      fileout='yields_evol_z'+zstring(j)+'_v'+vstring+'_varying_imf0.1-100msun_mfailed'+strcompress(mfailed,/remove_all)+'msun_asnia'+string(ASNIa,format='(e7.1)')+dust+'.txt'
      print,filemassive
      print,filesecondary
      print,fileout
      cmp_yield_release,ASNIa=ASNIa,fileout=fileout,alpha=alpha_slope,bound=[0.1,0.6,100.0],v=myvel(i),zmet=zmet(j),maxage=2d10,fudgemg=fudgemg,filemassive=filemassive,filesecondary=filesecondary,mfailed=double(mfailed),/noplot
   endfor
endfor
endif

endelse

end
