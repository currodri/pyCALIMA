pro convert_myyieldsintoramses2,vel=vel,ASNIa=ASNIa,printps=printps,fileeps=fileeps,files=files,zmet=zmet,fileout=fileout,noplot=noplot,legend=legend,xs=xs,nomoto=nomoto,maoz=maoz,pyroxene=pyroxene,compound=compound,dust=dust

nsilMg=1.0d0
nsilFe=1.0d0
nsilSi=1.0d0
nsilO =4.0d0
if not keyword_set(compound)then begin
   if keyword_set(pyroxene)then begin
      nsilMg=1.0d0
      nsilFe=1.0d0
      nsilSi=2.0d0
      nsilO =6.0d0
   endif
endif else begin
   if not keyword_set(dust)then begin
      print,'you need to give me the name of the dust for the file to be read'
      print,'stopping...'
      stop
   endif
   nsilMg=compound(0)
   nsilFe=compound(1)
   nsilSi=compound(2)
   nsilO =compound(3)
endelse
muC =12.0107d0
muO =15.9990d0
muS =32.0650d0
muMg=24.3050d0
muSi=28.0855d0
muFe=55.8450d0
dir='/home/dubois/StellarYields/ResultingYields/'
if not keyword_set(files)then begin
print,'Using vel:',vel
if(vel eq -1.)then begin
   print,'Using IDROV ~Prantzos'
   if not keyword_set(compound)then begin
      if keyword_set(pyroxene)then begin
         files=[  'yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' $
                  ,'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_pyroxenedwek.txt' ]
      endif else begin
         files=[  'yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
                  ,'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' ]
      ;; files=[  'yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
      ;;          ,'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
      ;;          ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
      ;;          ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' ]
      endelse
   endif else begin
      files=[  'yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' $
               ,'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_'+dust+'.txt' ]         
   endelse
   ;; files=[  'yields_evol_z-3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
   ;;         ,'yields_evol_z-2_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
   ;;         ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
   ;;         ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' ]
endif
if(vel eq 0.)then begin
   files=[ 'yields_evol_z-3_v0_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-2_v0_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-1_v0_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z0_v0_chabrier_imf0.01-100msun_dustcpopping17.txt']
endif
if(vel eq 150.)then begin
   files=[ 'yields_evol_z-3_v150_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-2_v150_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-1_v150_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z0_v150_chabrier_imf0.01-100msun_dustcpopping17.txt']
endif
if(vel eq 300.)then begin
   files=[ 'yields_evol_z-3_v300_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-2_v300_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z-1_v300_chabrier_imf0.01-100msun_dustcpopping17.txt' $
           ,'yields_evol_z0_v300_chabrier_imf0.01-100msun_dustcpopping17.txt']
endif
endif
nfiles=n_elements(files)
;; Sanity check: verify that all files have the same number of time steps
for i=0L,nfiles-1L do begin
   filename=dir+files(i)+'.snII'
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
   nsteps=n_elements(time)
   if(i gt 0L)then begin
      if(nsteps ne nsteps0)then begin
         print,'fail in # of time steps ', filename,nsteps,nsteps0,i
         stop
      endif else begin
         nsteps0=nsteps
      endelse
   endif else begin
      nsteps0=nsteps
   endelse
endfor
print,files

;; first things first: write down the time sampling
filename=dir+files(0)+'.snII'
readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
if not keyword_set(fileout)then fileout='ramses.txt'
openw,1,fileout
zsun=0.01345
printf,1,' '
if not keyword_set(zmet)then begin
   if(vel eq -1.)then begin
      zmet=[0.00001345,0.0001345,0.001345,0.0033784874,0.0067409685,0.01345,0.026836279]
      printf,1,'log_Zgrid   = (/',format='(A,$)'
      for i=0L,n_elements(zmet)-1L do begin
         if(i ne n_elements(zmet)-1L)then begin
            printf,1,string(alog10(zmet(i)),format='(f11.7)')+',',format='(A,$)'
         endif else begin
            printf,1,string(alog10(zmet(i)),format='(f11.7)')+' /)',format='(A)'
         endelse
      endfor
   endif else begin
      zmet=[0.00001345,0.0001345,0.001345,0.01345]
      printf,1,'log_Zgrid = (/ 0.00001345,0.0001345,0.001345,0.01345 /)'
   endelse
endif else begin
   printf,1,'log_Zgrid = (/ ',format='(A,$)'
   for i=0L,nfiles-1L do begin
      if(i ne nfiles-1)then begin
         printf,1,string(zmet(i),format='(f12.10)')+',',format='(A,$)'
      endif else begin
         printf,1,string(zmet(i),format='(f12.10)')+'',format='(A,$)'
      endelse
   endfor
   printf,1,' /) ',format='(A)'
endelse   

;; SNII
mmet =dblarr(nfiles,nsteps,12L)
mdust=dblarr(nfiles,nsteps,12L)
for j=0L,nfiles-1L do begin
   filename=dir+files(j)+'.snII'
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
   ;; Ignore fluor so far...

   ;; Compute the integrated mass return from the sum of all elements
   mmet(j,0L:nsteps-1L,2)=h (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,3)=he(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,4)=c (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,5)=n (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,6)=o (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,7)=mg(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,8)=si(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,9)=s (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,10)=fe(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,11)=neon(0L:nsteps-1L)

   mdust(j,0L:nsteps-1L,2)=hd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,3)=hed(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,4)=cd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,5)=nd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,6)=od (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,7)=mgd(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,8)=sid(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,9)=sd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,10)=fed(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,11)=neond(0L:nsteps-1L)

   for i=0L,nsteps-1L do begin
      mmet(j,i,1)=total(mmet(j,i,4L:11L))   ; Total metal mass
      mmet(j,i,0)=total(mmet(j,i,2L:11L))   ; Total gas mass (metal+H+He)
      mdust(j,i,1)=total(mdust(j,i,2L:10L)) ; Total dust mass
   endfor
endfor

;; AGB
mmetagb =dblarr(nfiles,nsteps,12L)
mdustagb=dblarr(nfiles,nsteps,12L)
for j=0L,nfiles-1L do begin
   filename=dir+files(j)+'.agb'
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
   ;; Ignore fluor so far...

   ;; Compute the integrated mass return from the sum of all elements
   mmetagb(j,0L:nsteps-1L,2)=h (0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,3)=he(0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,4)=c (0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,5)=n (0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,6)=o (0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,7)=mg(0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,8)=si(0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,9)=s (0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,10)=fe(0L:nsteps-1L)
   mmetagb(j,0L:nsteps-1L,11)=neon(0L:nsteps-1L)

   mdustagb(j,0L:nsteps-1L,2)=hd (0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,3)=hed(0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,4)=cd (0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,5)=nd (0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,6)=od (0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,7)=mgd(0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,8)=sid(0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,9)=sd (0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,10)=fed(0L:nsteps-1L)
   mdustagb(j,0L:nsteps-1L,11)=neond(0L:nsteps-1L)

   for i=0L,nsteps-1L do begin
      mmetagb(j,i,1)=total(mmetagb(j,i,4L:11L))   ; Total metal mass
      mmetagb(j,i,0)=total(mmetagb(j,i,2L:11L))   ; Total gas mass (metal+H+He)
      mdustagb(j,i,1)=total(mdustagb(j,i,2L:10L)) ; Total dust mass
   endfor
endfor

;; ===============================
;; Plot the data
;; ===============================
if not keyword_set(nomoto)then begin
   ;;Iwamoto et al. (1999) W70 (carbon-deflagration model)
   yield_snIa = [ $  ;;! Msun per SN
                5.08E-02,1.56E-09,3.31E-08,4.13E-07,1.33E-01,3.33E-10,2.69E-10,1.37E-10,$
                2.29E-03,2.81E-08,2.15E-08,1.41E-05,1.58E-02,1.64E-07,1.87E-07,1.13E-04,$
                1.42E-01,5.79E-05,7.12E-05,9.12E-05,9.14E-02,6.07E-05,1.74E-05,3.41E-11,$
                1.06E-05,5.56E-06,1.91E-02,6.60E-07,3.42E-12,1.67E-06,4.83E-07,1.81E-02,$
                1.06E-08,6.17E-08,1.38E-05,1.01E-09,2.47E-09,3.85E-08,3.49E-07,4.08E-07,$
                3.13E-04,2.94E-06,1.04E-04,1.22E-08,4.27E-05,6.65E-05,7.73E-03,5.66E-04,$
                9.04E-04,6.66E-03,7.30E-02,6.80E-01,1.92E-02,2.96E-03,9.68E-04,8.34E-02,$
                1.47E-02,2.15E-04,1.85E-03,1.65E-05,3.00E-06,8.33E-07,7.01E-05,6.26E-06,$
                7.28E-09,1.13E-08]
endif else begin
   ;;Nomoto et al. 1997
   yield_snIa = [ $  ;; ! Msun per SN
                4.83E-02,1.40E-06,1.16E-06,1.32E-09,1.43E-01,3.54E-08,8.25E-10,5.67E-10,$
                2.02E-03,8.46E-06,2.49E-03,6.32E-05,8.50E-03,4.05E-05,3.18E-05,9.86E-04,$
                1.50E-01,8.61E-04,1.74E-03,4.18E-04,8.41E-02,4.50E-04,1.90E-03,3.15E-07,$
                1.34E-04,3.98E-05,1.49E-02,1.06E-03,1.26E-08,8.52E-05,7.44E-06,1.23E-02,$
                3.52E-05,1.03E-07,8.86E-06,1.99E-09,7.10E-12,2.47E-07,1.71E-05,6.04E-07,$
                2.03E-04,1.69E-05,1.26E-05,8.28E-09,5.15E-05,2.71E-04,5.15E-03,7.85E-04,$
                1.90E-04,8.23E-03,1.04E-01,6.13E-01,2.55E-02,9.63E-04,1.02E-03,1.28E-01,$
                1.05E-02,2.51E-04,2.66E-03,1.31E-06,1.79E-06,6.83E-07,1.22E-05,2.12E-05,$
                1.34E-08,1.02E-08]
endelse
chem_list=['C','N','O','Ne','Mg','Si','S','Fe']
nchem=n_elements(chem_list)
mz_Ia=dblarr(nchem+1)
mz_Ia(0)=total(yield_snIa)
for i=0L,nchem-1L do begin
   if(chem_list(i) eq 'C' )then mz_Ia(i+1)=total(yield_snIa(0:1))
   if(chem_list(i) eq 'N' )then mz_Ia(i+1)=total(yield_snIa(2:3))
   if(chem_list(i) eq 'O' )then mz_Ia(i+1)=total(yield_snIa(4:6))
   if(chem_list(i) eq 'Ne')then mz_Ia(i+1)=total(yield_snIa(8:10))
   if(chem_list(i) eq 'Mg')then mz_Ia(i+1)=total(yield_snIa(12:14))
   if(chem_list(i) eq 'Si')then mz_Ia(i+1)=total(yield_snIa(16:18))
   if(chem_list(i) eq 'S' )then mz_Ia(i+1)=total(yield_snIa(20:23))
   if(chem_list(i) eq 'Fe')then mz_Ia(i+1)=total(yield_snIa(50:53))
endfor
for i=0L,nchem-1L do begin
   print,chem_list(i),mz_Ia(i+1),format='(A3,e10.2)'
endfor
readcol,dir+'/cr_Ia.txt',time2,cr_Ia,/silent
if not keyword_set(noplot)then begin
if keyword_set(printps)then begin
   if not keyword_set(xs)then xs=15
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='yield_ejecta_evol.eps'
   device,filename=fileeps,/encaps,xs=5*xs,ys=3*xs,/color
   !p.charsize=4.5
   !p.font=0
   th=5
   !p.charthick=th
endif else begin
   if not keyword_set(xs)then xs=300
   window,0,xs=3*xs,ys=2*xs
endelse
;; !p.multi=[0,5,2]
!p.multi=[0,3,3]
if not keyword_set(ASNIa)then begin
   ASNIa=3.5d-2
   print,'ASNIa not specified'
   print,'-> using ASNIa:',ASNIa
endif
if keyword_set(maoz)then begin
   cmIa=cr_Ia
   tmin=5d-2  ;;Gyr
   tmax=13.7d0;;Gyr
   time2=time2/1d9
   ind0=where(time2 le tmin)
   cmIa(ind0)=0.0d0
   ind0=where(time2 gt tmin)
   cmIa(ind0)=2.35d-3*(alog(time2(ind0))-alog(tmin))*mz_Ia(0)/10d0

   ;; cmIad(= total(yield_snIa(0:1))*0.5d0
   ;; ZDejecta_chem_Ia(1)=Zejecta_chem_Ia(ich)*0.5d0
   
   ;; Zejecta_chem_Ia(ich)=total(yield_snIa(4:6))
   ;; nO_Ia =Zejecta_chem_Ia(ich)/(nsilO *muO )

   ;; Zejecta_chem_Ia(ich)=total(yield_snIa(12:14))
   ;; nMg_Ia=Zejecta_chem_Ia(ich)/(nsilMg*muMg)
   
   ;; Zejecta_chem_Ia(ich)=total(yield_snIa(16:18))
   ;; nSi_Ia=Zejecta_chem_Ia(ich)/(nsilSi*muSi)

   ;; Zejecta_chem_Ia(ich)=total(yield_snIa(50:53))
   ;; nFe_Ia=Zejecta_chem_Ia(ich)/(nsilFe*muFe)
   
   ;; 0.99d0*MIN(nMg_Ia,nFe_Ia,nSi_Ia,nO_Ia)*nsilSi*muSi
endif else begin
   cmIa=ASNIa*cr_Ia*mz_Ia(0)
endelse
ind=where((cmIa+mmet(0,*,0)) ne 0.)
t=time(ind)/1d9
if (vel eq -1.)then begin
   dcolor=20.
endif else begin
   dcolor=15.
endelse

;; Plot total mass release
;; plot_oi,t,cmIa(ind)+mmet(0,ind,0)+mmetagb(0,ind,0),xtitle='time (Gyr)',ytitle='M!dlost!n/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,0.5],/ys,xth=th,yth=th,th=th
;; loadct,1,/silent
;; for i=0L,nfiles-1L do begin
;;    oplot,t,cmIa(ind)+mmet(i,ind,0)+mmetagb(i,ind,0),color=120+i*dcolor,th=th
;; endfor
;; ;; Only AGB
;; loadct,3,/silent
;; if (vel eq -1.)then begin
;;    dcolor=8.
;; endif else begin
;;    dcolor=15.
;; endelse
;; for i=0L,nfiles-1L do begin
;;    ;; oplot,t,mmetagb(i,ind,0),lines=i,color=120+i*dcolor,th=th
;;    oplot,t,mmetagb(i,ind,0),color=120+i*dcolor,th=th
;; endfor
;; ; SNIa
;; tek_color
;; oplot,t,cmIa(ind),color=12,th=th
;; loadct,0,/silent

; Metallicity
;; Chem elements
;; iorder=[6L ,4L ,10L ,5L ,7L  ,8L  ]
;; ytchem=['O','C','Fe','N','Mg','Si']
iorder=[6L ,4L ,10L ,5L ,7L  ,8L  ,9L ,11L  ]
ytchem=['O','C','Fe','N','Mg','Si','S','Ne']
;; ytmax=[0.042,0.017,0.003,0.003,0.003,0.003,0.003,0.003]
ytmax=[0.025,0.008,0.003,0.002,0.002,0.002,0.001,0.003]
nchem=n_elements(ytchem)
;; yy=(cmIa(ind)+mmet(0,ind,1))/(cmIa(ind)+mmet(0,ind,0))
yy=cmIa(ind)+mmet(0,ind,1)+mmetagb(0,ind,1)
plot_oi,t,yy,xtitle='time (Gyr)',ytitle='M!dlost!n(Z)/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,0.9D0*total(ytmax)],/ys,xth=th,yth=th,th=th
loadct,1,/silent
for i=0L,nfiles-1L do begin
   ;; yy=(cmIa(ind)+mmet(i,ind,1))/(cmIa(ind)+mmet(i,ind,0))
   yy=cmIa(ind)+mmet(i,ind,1)+mmetagb(i,ind,1)
   oplot,t,yy,color=120+i*dcolor,th=th
endfor
;; AGB
loadct,3,/silent
for i=0L,nfiles-1L do begin
   yy=mmetagb(i,ind,1)
   oplot,t,yy,color=120+i*dcolor,th=th
endfor
yy=cmIa(ind)
tek_color
oplot,t,yy,color=12,th=th
loadct,8,/silent
for i=0L,nfiles-1L do begin
   ;; ind=where(mmet(0,*,0) ne 0.)
   oplot,t,mdust(i,ind,1)+mdustagb(i,ind,1),color=120+i*dcolor,th=th
endfor
;; for i=0L,nfiles-1L do begin
   ;; yy=(cmIa(ind))/(cmIa(ind)+mmet(i,ind,0))
   ;; oplot,t,yy,lines=i,color=120+i*dcolor,th=th
;; endfor
loadct,0,/silent

aa=dblarr(4)
ic=0
for i=0L,nchem-1L do begin
   if(chem_list(i) eq 'O' )then begin
      aa(ic)=mz_Ia(i+1)/(nsilO*muO)
      ic=ic+1L
   endif
   if(chem_list(i) eq 'Mg')then begin
      aa(ic)=mz_Ia(i+1)/(nsilMg*muMg)
      ic=ic+1L
   endif
   if(chem_list(i) eq 'Si')then begin
      aa(ic)=mz_Ia(i+1)/(nsilSi*muSi)
      ic=ic+1L
   endif
   if(chem_list(i) eq 'Fe')then begin
      aa(ic)=mz_Ia(i+1)/(nsilFe*muFe)
      ic=ic+1L
   endif
endfor
mdsil_key=min(aa)
md_Ia=dblarr(nchem+1L)
for i=0L,nchem-1L do begin
   if(chem_list(i) eq 'C' )then md_Ia(i+1)=mz_Ia(i+1)*0.5d0
   if(chem_list(i) eq 'O' )then md_Ia(i+1)=mdsil_key*nsilO *muO *0.99d0
   if(chem_list(i) eq 'Mg')then md_Ia(i+1)=mdsil_key*nsilMg*muMg*0.99d0
   if(chem_list(i) eq 'Si')then md_Ia(i+1)=mdsil_key*nsilSi*muSi*0.99d0
   if(chem_list(i) eq 'Fe')then md_Ia(i+1)=mdsil_key*nsilFe*muFe*0.99d0
endfor

for ii=0L,nchem-1L do begin
   indIa=where(chem_list eq ytchem(ii))&iIa=indIa(0)+1
   if keyword_set(maoz)then begin
      cmIa_chem=cr_Ia
      ind0=where(time2 le tmin)
      cmIa_chem(ind0)=0.0d0
      ind0=where(time2 gt tmin)
      cmIa_chem(ind0)=2.35d-3*(alog(time2(ind0))-alog(tmin))*mz_Ia(iIa)/10d0
   endif else begin
      cmIa_chem=ASNIa*cr_Ia*mz_Ia(iIa)
   endelse

   ichem=iorder(ii)
   ymax=ytmax(ii)
   yy=cmIa_chem(ind)+mmet(0,ind,ichem)+mmetagb(0,ind,ichem)
   plot_oi,t,yy,xtitle='time (Gyr)',ytitle='M!dlost!n('+ytchem(ii)+')/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,ymax],/ys,xth=th,yth=th,th=th
   for i=0L,nfiles-1L do begin
      yy=cmIa_chem(ind)+mmet(i,ind,ichem)+mmetagb(i,ind,ichem)
      loadct,1,/silent
      oplot,t,yy,color=120+i*dcolor,th=th
      yy=mmetagb(i,ind,ichem)
      loadct,3,/silent
      oplot,t,yy,color=120+i*dcolor,th=th
      loadct,8,/silent
      mdreleased_Ia=md_Ia(iIa)*cmIa_chem/mz_Ia(iIa)
      oplot,t,mdust(i,ind,ichem)+mdustagb(i,ind,ichem)+mdreleased_Ia(ind),color=120+i*dcolor,th=th
      ;; loadct,0,/silent
      ;; ;; mdreleased_Ia=ASNIa*cr_Ia*md_Ia(iIa)
      ;; mdreleased_Ia=md_Ia(iIa)*cmIa_chem/mz_Ia(iIa)
      ;; yy=mdreleased_Ia(ind)
      ;; oplot,t,yy,color=120,th=th
   endfor
   tek_color
   yy=cmIa_chem(ind)
   if(max(yy) gt 1d-5)then begin
      oplot,t,yy,color=12,th=th
   endif
   if keyword_set(legend)then begin
   if(ytchem(ii) eq 'C')then begin
      loadct,0,/silent
      deltay=ymax
      dy=ymax/12.
      for i=0L,nfiles-1L do begin
         plots,[0.1,0.5],[ymax-double(i+1)*dy,ymax-double(i+1)*dy],lines=i,color=120+i*dcolor,th=th
         xyouts,0.8,ymax-double(i+1)*dy,'10!u'+string(alog10(zmet(i)/zsun),format='(f4.1)')+'!n Z!dsun!n',charsize=1.5
      endfor
   endif
   endif
endfor
;; if keyword_set(printps)then begin
;;    device,/close
;;    set_plot,'x'
;;    !p.charsize=1.5
;;    !p.font=0
;;    !p.charthick=1
;; endif
!p.multi=0
;; Now plot the dust ratios of Sil and C
if keyword_set(printps)then begin
   device,/close
   fileeps='dust_ratio_ejecta.eps'
   device,filename=fileeps,/encaps,xs=15,ys=15,/color
   !p.charsize=1.5
endif else begin
   window,1,xs=xs,ys=xs
   !p.charsize=1.5
endelse
for i=0L,nchem-1L do begin
   if(ytchem(i) eq 'C' )then indexC =iorder(i)
   if(ytchem(i) eq 'O' )then indexO =iorder(i)
   if(ytchem(i) eq 'Mg')then indexMg=iorder(i)
   if(ytchem(i) eq 'Si')then indexSi=iorder(i)
   if(ytchem(i) eq 'Fe')then indexFe=iorder(i)
endfor
mdsil=mdust(0,*,indexMg)+mdust(0,*,indexFe)+mdust(0,*,indexSi)+mdust(0,*,indexO)+mdustagb(0,*,indexMg)+mdustagb(0,*,indexFe)+mdustagb(0,*,indexSi)+mdustagb(0,*,indexO)
mdcar=mdust(0,*,indexC)+mdustagb(0,*,indexC)
print,'indexC',indexC
mdtot=mdsil+mdcar
ind=where(mdtot gt 0d0)
tt=time(ind)/1d9
yy=mdcar/mdtot
plot_oi,tt,yy,xtitle='time (Gyr)',ytitle='M!dD,C!n/M!dDtot!n',xr=[1d-3,2.1d1],/xs,yr=[0.,1.0d0],/ys,xth=th,yth=th,th=th
for i=0L,nfiles-1L do begin
   mdsil=mdust(i,*,indexMg)+mdust(i,*,indexFe)+mdust(i,*,indexSi)+mdust(i,*,indexO)+mdustagb(i,*,indexMg)+mdustagb(i,*,indexFe)+mdustagb(i,*,indexSi)+mdustagb(i,*,indexO)
   mdcar=mdust(i,*,indexC)+mdustagb(i,*,indexC)
   mdtot=mdsil+mdcar
   ind=where(mdtot gt 0d0)
   tt=time(ind)/1d9
   yy=mdcar/mdtot
   loadct,8,/silent
   oplot,tt,yy,lines=i,color=120+i*dcolor,th=th
endfor
if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.font=0
   !p.charthick=1
endif
;; ===============================
;; ===============================

;;yy=cmIa(ind)+mmet(0,ind,1)+mmetagb(0,ind,1)
if keyword_set(printps)then begin
   device,/close
   fileeps='sn_mass_ejecta.eps'
   device,filename=fileeps,/encaps,xs=15,ys=15,/color
   !p.charsize=1.5
endif else begin
   window,1,xs=xs,ys=xs
   !p.charsize=1.5
endelse
yy=mmet(0,ind,0)
plot_oi,t,yy,xtitle='time (Gyr)',ytitle='M!dlost!n/M!dSSP!n',xr=[1d-3,2.1d-1],/xs,yr=[0.,0.25],/ys,xth=th,yth=th,th=th
loadct,1,/silent
for i=0L,nfiles-1L do begin
   ;; yy=cmIa(ind)+mmet(i,ind,1)+mmetagb(i,ind,1)
   yy=mmet(i,ind,0)
   oplot,t,yy,color=120+i*dcolor,th=th
endfor


endif ;; no plot



lowmass=1d-7
lowmet =1d-1*lowmass
lowchem=1d-2*lowmet
;; lowdust=1d-2*lowchem
;; for olivine
aa=[lowchem/(4d0*muO),lowchem/muMg,lowchem/muSi,lowchem/muFe]
lowdust=1d-1*min(aa)*muSi ;; Si key element
print,lowchem,lowdust

nsstring=strcompress(nsteps,/remove_all)
nzstring=strcompress(nfiles,/remove_all)
;; print the total mass return rates
for ichem=0L,12L-1L do begin
   counter=0L
   if(ichem eq 0 )then printf,1,'log_SNII_m  = (/',format='(A,$)'
   if(ichem eq 1 )then printf,1,'log_SNII_Z  = (/',format='(A,$)'
   if(ichem eq 2 )then printf,1,'log_SNII_H  = (/',format='(A,$)'
   if(ichem eq 4 )then printf,1,'log_SNII_C  = (/',format='(A,$)'
   if(ichem eq 5 )then printf,1,'log_SNII_N  = (/',format='(A,$)'
   if(ichem eq 6 )then printf,1,'log_SNII_O  = (/',format='(A,$)'
   if(ichem eq 7 )then printf,1,'log_SNII_Mg = (/',format='(A,$)'
   if(ichem eq 8 )then printf,1,'log_SNII_Si = (/',format='(A,$)'
   if(ichem eq 9 )then printf,1,'log_SNII_S  = (/',format='(A,$)'
   if(ichem eq 10)then printf,1,'log_SNII_Fe = (/',format='(A,$)'
   if(ichem eq 11)then printf,1,'log_SNII_Ne = (/',format='(A,$)'
   if(ichem ne 3)then begin ;; skip helium (deduced from the rest)
   for j=0L,nfiles-1L do begin

      if(ichem ne 0)then begin
         mmet0=alog10((mmet(j,nsteps-1L,ichem)+lowchem)/mmet(j,nsteps-1L,0))
      endif else begin
         mmet0=alog10(mmet(j,nsteps-1L,ichem)+lowmass)
      endelse
      if(j ne nfiles-1L)then begin
         printf,1,string(mmet0,format='(f11.7)')+',',format='(A,$)'
      endif else begin
         printf,1,string(mmet0,format='(f11.7)')+' /)',format='(A)'
      endelse
   endfor
   endif
endfor

printf,1,'log_SNII_Dust   = (/',format='(A,$)'
for j=0L,nfiles-1L do begin
   xx=alog10((mdust(j,nsteps-1L,1 )+lowdust)/mmet(j,nsteps-1L,0)) ; DTG ratio
   if(j ne nfiles-1L)then begin
      printf,1,string(xx,format='(f11.7)')+',',format='(A,$)'
   endif else begin
      printf,1,string(xx,format='(f11.7)')+' /)',format='(A)'
   endelse
endfor
printf,1,'log_SNII_C_Dust = (/',format='(A,$)'
for j=0L,nfiles-1L do begin
   xx=alog10((mdust(j,nsteps-1L,4 )+lowdust)/mmet(j,nsteps-1L,0)) ; (D_C)TG ratio
   if(j ne nfiles-1L)then begin
      printf,1,string(xx,format='(f11.7)')+',',format='(A,$)'
   endif else begin
      printf,1,string(xx,format='(f11.7)')+' /)',format='(A)'
   endelse
endfor
printf,1,'log_SNII_Si_Dust= (/',format='(A,$)'
for j=0L,nfiles-1L do begin
   xx=alog10((mdust(j,nsteps-1L,8)+lowdust)/mmet(j,nsteps-1L,0)) ; (D_Si)TG ratio
   if(j ne nfiles-1L)then begin
      printf,1,string(xx,format='(f11.7)')+',',format='(A,$)'
   endif else begin
      printf,1,string(xx,format='(f11.7)')+' /)',format='(A)'
   endelse
endfor
printf,1,'log_SNII_Fe_Dust= (/',format='(A,$)'
for j=0L,nfiles-1L do begin
   xx=alog10((mdust(j,nsteps-1L,10)+lowdust)/mmet(j,nsteps-1L,0)) ; (D_Fe)TG ratio
   if(j ne nfiles-1L)then begin
      printf,1,string(xx,format='(f11.7)')+',',format='(A,$)'
   endif else begin
      printf,1,string(xx,format='(f11.7)')+' /)',format='(A)'
   endelse
endfor


;; Stellar wind files (in that case only intermediate stars, not
;; including the pre-SN phase)
counter=0L
openw,2,'ramses_swind_Sikey.dat',/f77_unformatted
openw,3,'ramses_snII_Sikey.dat',/f77_unformatted
writeu,2,long(nsteps),long(nfiles)
writeu,2,double(alog10(time)) ;; log years
writeu,2,double(alog10(zmet)) ;; log Z (not in solar units)
writeu,3,long(nsteps),long(nfiles)
writeu,3,double(alog10(time)) ;; log years
writeu,3,double(alog10(zmet)) ;; log Z (not in solar units)
for j=0L,nfiles-1L do begin
   writeu,2,double(alog10(mmetagb(j,0L:nsteps-1L,0)+lowmass)) ;; log mass per Msun
   writeu,3,double(alog10(mmet(j,0L:nsteps-1L,0)+lowmass)) ;; log mass per Msun
endfor

filesener=files
for i=0L,nfiles-1L do begin
   filesener(i)=files(i)+'.esnII'
endfor
;; Print the energy input
for j=0L,nfiles-1L do begin
   filename=dir+filesener(j)
   readcol,filename,time,energy,energySW,/silent,format='(f,d,d)'
   lenergy=alog10(energy+1d0)
   lenergyagb=alog10(energySW*1.5d0^2d0+1d0) ;; the factor 1.5^2 is to turn 2e3 km/s WR velocities into 3e3 (closer to reality)
   writeu,2,double(lenergyagb) ;; log energy in ergs
   writeu,3,double(lenergy) ;; log energy in ergs
endfor

for j=0L,nfiles-1L do begin
   writeu,2,double(alog10(mmetagb(j,0L:nsteps-1L,1)+lowmet)) ;; log mass per Msun (not relative to released total mass)
   writeu,3,double(alog10(mmet   (j,0L:nsteps-1L,1)+lowmet)) ;; log mass per Msun (not relative to released total mass)
endfor

for ichem=2,10 do begin
   for j=0L,nfiles-1L do begin
      writeu,2,double(alog10(mmetagb(j,0L:nsteps-1L,ichem)+lowchem)) ;; H,He,C,N,O,Mg,Si,S,Fe
      writeu,3,double(alog10(mmet   (j,0L:nsteps-1L,ichem)+lowchem)) ;; H,He,C,N,O,Mg,Si,S,Fe
   endfor
endfor

;; Dust
for j=0L,nfiles-1L do begin
   writeu,2,double(alog10(mdustagb(j,0L:nsteps-1L,1)+lowdust)) ;; all dust
   writeu,3,double(alog10(mdust   (j,0L:nsteps-1L,1)+lowdust)) ;; all dust
endfor
for j=0L,nfiles-1L do begin
   writeu,2,double(alog10(mdustagb(j,0L:nsteps-1L,4)+lowdust)) ;; C
   writeu,3,double(alog10(mdust   (j,0L:nsteps-1L,4)+lowdust)) ;; C
endfor
for j=0L,nfiles-1L do begin
   writeu,2,double(alog10(mdustagb(j,0L:nsteps-1L,8)+lowdust)) ;; Si (from Si: Mg, Fe and O in Sillicates can be obtained. This should be dealt in Ramses.)
   writeu,3,double(alog10(mdust   (j,0L:nsteps-1L,8)+lowdust)) ;; Si (from Si: Mg, Fe and O in Sillicates can be obtained. This should be dealt in Ramses.)
   for i=0L,nsteps-1L do begin
      print,time(i),double(alog10(mdustagb(j,i,8)+lowdust))
   endfor
   print,'============='
endfor
;; for j=0L,nfiles-1L do begin
;;    writeu,2,double(alog10(mdustagb(j,0L:nsteps-1L,10)+lowdust)) ;; Fe (from Fe: Mg, Si and O in Sillicates can be obtained. This should be dealt in Ramses.)
;; endfor


close,/all
stop



;; print the dust mass return rates
for ichem=0L,12L-1L do begin
   counter=0L
   if(ichem eq 1)then printf,1,'cDwind_Z = reshape( (/ &'
   if(ichem eq 4)then printf,1,'cDwind_C = reshape( (/ &'
   if(ichem eq 6)then printf,1,'cDwind_O = reshape( (/ &'
   if(ichem eq 7)then printf,1,'cDwind_Mg = reshape( (/ &'
   if(ichem eq 8)then printf,1,'cDwind_Si = reshape( (/ &'
   if(ichem eq 9)then printf,1,'cDwind_S = reshape( (/ &'
   if(ichem eq 10)then printf,1,'cDwind_Fe = reshape( (/ &'
   if(ichem eq 1 or ichem eq 4 or ichem eq 6 or ichem eq 7 or ichem eq 8 or ichem eq 9 or ichem eq 10)then begin
   for j=0L,nfiles-1L do begin
      for i=0L,nsteps-1L do begin
         counter=counter+1L
         if(j eq nfiles-1L and i eq nsteps-1L)then begin
            printf,1,string(alog10(mmet(j,i,ichem)+1d-10),format='(f11.7)')+'/), (/'+nsstring+','+nzstring+'/) )',format='(A)'
            printf,1,' '
            counter=0L
         endif
         if(counter eq 1L)then begin
            printf,1,' & '+string(alog10(mmet(j,i,ichem)+1d-10),format='(f11.7)')+',',format='(A,$)'
         endif
         if(counter gt 1L and counter le 4L)then begin
            printf,1,string(alog10(mmet(j,i,ichem)+1d-10),format='(f11.7)')+',',format='(A,$)'
         endif
         if(counter eq 5L)then begin
            printf,1,string(alog10(mmet(j,i,ichem)+1d-10),format='(f11.7)')+', &',format='(A)'
            counter=0L
         endif
      endfor
   endfor
   endif
endfor



end
