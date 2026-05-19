pro convert_myyieldsintoramses,vel=vel,ASNIa=ASNIa,printps=printps,fileeps=fileeps,files=files,zmet=zmet,fileout=fileout

dir='/home/dubois/StellarYields/ResultingYields/'
if not keyword_set(files)then begin
print,'Using vel in convert_myyieldsintoramses:',vel
if(vel eq -1.)then begin
   files=[  'yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' $
           ,'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_olivinedwek.txt' ]
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
   filename=dir+files(i)
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

;; first things first: write down the time sampling
filename=dir+files(0)
readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
if not keyword_set(fileout)then fileout='ramses.txt'
openw,1,fileout
counter=0L
printf,1,'t_wind = (/ &'
for i=0L,nsteps-1L do begin
   counter=counter+1L
   if(i eq nsteps-1L)then begin
      printf,1,string(time(i),format='(e11.5)')+' /)',format='(A)'
      counter=0L
   endif
   if(counter eq 1L)then begin
      printf,1,' & '+string(time(i),format='(e11.5)')+',',format='(A,$)'
   endif
   if(counter gt 1L and counter le 4L)then begin
      printf,1,string(time(i),format='(e11.5)')+',',format='(A,$)'
   endif
   if(i ne nsteps-1L and counter eq 5L)then begin
      printf,1,string(time(i),format='(e11.5)')+', &',format='(A)'
      counter=0L
   endif
endfor

zsun=0.01345
printf,1,' '
if not keyword_set(zmet)then begin
   if(vel eq -1.)then begin
      zmet=[0.00001345,0.0001345,0.001345,0.0033784874,0.0067409685,0.01345,0.026836279]
      printf,1,'Z_wind = (/ 0.00001345,0.0001345,0.001345,0.0033784874,0.0067409685,0.01345,0.026836279 /)'
   endif else begin
      zmet=[0.00001345,0.0001345,0.001345,0.01345]
      printf,1,'Z_wind = (/ 0.00001345,0.0001345,0.001345,0.01345 /)'
   endelse
endif else begin
   printf,1,'Z_wind = (/ ',format='(A,$)'
   for i=0L,nfiles-1L do begin
      if(i ne nfiles-1)then begin
         printf,1,string(zmet(i),format='(f12.10)')+',',format='(A,$)'
      endif else begin
         printf,1,string(zmet(i),format='(f12.10)')+'',format='(A,$)'
      endelse
   endfor
   printf,1,' /) ',format='(A)'
endelse   
printf,1,' '

mmet =dblarr(nfiles,nsteps,12L)
mdust=dblarr(nfiles,nsteps,12L)
for j=0L,nfiles-1L do begin
   filename=dir+files(j)
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

;; ===============================
;; Plot the data
;; ===============================
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
if not keyword_set(xs)then xs=400
if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='yield_ejecta_evol.eps'
   device,filename=fileeps,/encaps,xs=4*15,ys=2*15,/color
   !p.charsize=2.5
   !p.font=0
   th=5
   !p.charthick=th
endif else begin
   window,0,xs=5*xs,ys=2*xs
endelse
!p.multi=[0,5,2]
if not keyword_set(ASNIa)then begin
   ASNIa=5d-2
   print,'ASNIa not specified'
   print,'-> using ASNIa:',ASNIa
endif
cmIa=ASNIa*cr_Ia*mz_Ia(0)
ind=where((cmIa+mmet(0,*,0)) ne 0.)
t=time(ind)/1d9
plot_oi,t,cmIa(ind)+mmet(0,ind,0),xtitle='time (Gyr)',ytitle='M!dlost!n/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,0.65],/ys,xth=th,yth=th,th=th
loadct,1,/silent
if (vel eq -1.)then begin
   dcolor=8.
endif else begin
   dcolor=15.
endelse
for i=0L,nfiles-1L do begin
   ;; print,'*',ASNIa,dir
   ;; print,'x',cr_Ia
   ;; print,'======'
   ;; print,cmIa(ind)
   ;; print,mmet(i,ind,0)
   ;; stop
   oplot,t,cmIa(ind)+mmet(i,ind,0),lines=i,color=120+i*dcolor,th=th
endfor
loadct,3,/silent
for i=0L,nfiles-1L do begin
   oplot,t,cmIa(ind),lines=i,color=120+i*dcolor,th=th
endfor
loadct,0,/silent
; Metallicity
;; Chem elements
;; iorder=[6L ,4L ,10L ,5L ,7L  ,8L  ]
;; ytchem=['O','C','Fe','N','Mg','Si']
iorder=[6L ,4L ,10L ,5L ,7L  ,8L  ,9L ,11L  ]
ytchem=['O','C','Fe','N','Mg','Si','S','Ne']
;;ytmax=[0.042,0.042,0.003,0.003,0.003,0.003]
ytmax=[0.042,0.042,0.003,0.003,0.003,0.003,0.003,0.003]
nchem=n_elements(ytchem)
;; yy=(cmIa(ind)+mmet(0,ind,1))/(cmIa(ind)+mmet(0,ind,0))
yy=cmIa(ind)+mmet(0,ind,1)
plot_oi,t,yy,xtitle='time (Gyr)',ytitle='M!dlost!n(Z)/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,1.5D0*ytmax(0)],/ys,xth=th,yth=th,th=th
loadct,1,/silent
for i=0L,nfiles-1L do begin
   ;; yy=(cmIa(ind)+mmet(i,ind,1))/(cmIa(ind)+mmet(i,ind,0))
   yy=cmIa(ind)+mmet(i,ind,1)
   oplot,t,yy,lines=i,color=120+i*dcolor,th=th
endfor
loadct,3,/silent
yy=cmIa(ind)
tek_color
oplot,t,yy,color=2,th=th
loadct,8,/silent
for i=0L,nfiles-1L do begin
   ind=where(mmet(0,*,0) ne 0.)
   oplot,t,mdust(i,ind,1),lines=i,color=120+i*dcolor,th=th
endfor
;; for i=0L,nfiles-1L do begin
   ;; yy=(cmIa(ind))/(cmIa(ind)+mmet(i,ind,0))
   ;; oplot,t,yy,lines=i,color=120+i*dcolor,th=th
;; endfor
loadct,0,/silent
for ii=0L,nchem-1L do begin
   indIa=where(chem_list eq ytchem(ii))&iIa=indIa(0)+1
   cmIa_chem=ASNIa*cr_Ia*mz_Ia(iIa)

   ichem=iorder(ii)
   ymax=ytmax(ii)
   ;; yy=(cmIa_chem(ind)+mmet(0,ind,ichem))/(cmIa(ind)+mmet(0,ind,0))
   yy=cmIa_chem(ind)+mmet(0,ind,ichem)
   plot_oi,t,yy,xtitle='time (Gyr)',ytitle='M!dlost!n('+ytchem(ii)+')/M!dSSP!n',xr=[1d-3,2.1d1],/xs,yr=[0.,ymax],/ys,xth=th,yth=th,th=th
   for i=0L,nfiles-1L do begin
      ind=where(mmet(0,*,0) ne 0.)
      ;; yy=(cmIa_chem(ind)+mmet(i,ind,ichem))/(cmIa(ind)+mmet(i,ind,0))
      yy=cmIa_chem(ind)+mmet(i,ind,ichem)
      loadct,1,/silent
      oplot,t,yy,lines=i,color=120+i*dcolor,th=th
      loadct,8,/silent
      oplot,t,mdust(i,ind,ichem),lines=i,color=120+i*dcolor,th=th
   endfor
   tek_color
   yy=cmIa_chem(ind)
   oplot,t,yy,color=2,th=th
   ;; for i=0L,nfiles-1L do begin
   ;;    ind=where(mmet(0,*,0) ne 0.)
   ;;    ;; yy=(cmIa_chem(ind))/(cmIa(ind)+mmet(i,ind,0))
   ;;    yy=cmIa_chem(ind)
   ;;    oplot,t,yy,lines=i,color=120+i*dcolor,th=th
   ;; endfor
   if(ytchem(ii) eq 'C')then begin
      loadct,0,/silent
      deltay=ymax
      dy=ymax/12.
      for i=0L,nfiles-1L do begin
         plots,[0.1,0.5],[ymax-double(i+1)*dy,ymax-double(i+1)*dy],lines=i,color=120+i*dcolor,th=th
         xyouts,0.8,ymax-double(i+1)*dy,'10!u'+string(alog10(zmet(i)/zsun),format='(f4.1)')+'!n Z!dsun!n',charsize=1.5
      endfor
   endif
endfor
if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=2.5
   !p.font=0
   !p.charthick=1
endif
!p.multi=0
;; ===============================
;; ===============================


nsstring=strcompress(nsteps,/remove_all)
nzstring=strcompress(nfiles,/remove_all)
;; print the total mass return rates
for ichem=0L,12L-1L do begin
   counter=0L
   if(ichem eq 0)then printf,1,'cMwind = reshape( (/ &'
   if(ichem eq 1)then printf,1,'cMwind_Z = reshape( (/ &'
   if(ichem eq 2)then printf,1,'cMwind_H = reshape( (/ &'
   if(ichem eq 4)then printf,1,'cMwind_C = reshape( (/ &'
   if(ichem eq 5)then printf,1,'cMwind_N = reshape( (/ &'
   if(ichem eq 6)then printf,1,'cMwind_O = reshape( (/ &'
   if(ichem eq 7)then printf,1,'cMwind_Mg = reshape( (/ &'
   if(ichem eq 8)then printf,1,'cMwind_Si = reshape( (/ &'
   if(ichem eq 9)then printf,1,'cMwind_S = reshape( (/ &'
   if(ichem eq 10)then printf,1,'cMwind_Fe = reshape( (/ &'
   if(ichem eq 11)then printf,1,'cMwind_Ne = reshape( (/ &'
   if(ichem ne 3)then begin ;; skip helium (deduced from the rest)
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


for i=0L,nfiles-1L do begin
   files(i)=files(i)+'.esnII'
endfor

printf,1,'if(f_w>0.and.(.not.stellar_wind_kinmode))then'
printf,1,'if(myid==1)write(*,*)''initialise the stellar_wind tabulated energies without SNII contribution'' '
;; print the energy input from winds only
printf,1,'cEwind = reshape( (/ &'
for j=0L,nfiles-1L do begin
   for i=0L,nsteps-1L do begin
      counter=counter+1L
      if(j eq nfiles-1L and i eq nsteps-1L)then begin
         printf,1,string(0.0d0,format='(f11.7)')+'/), (/'+nsstring+','+nzstring+'/) )',format='(A)'
         printf,1,' '
         counter=0L
      endif
      if(counter eq 1L)then begin
         printf,1,' & '+string(0.0d0,format='(f11.7)')+',',format='(A,$)'
      endif
      if(counter gt 1L and counter le 4L)then begin
         printf,1,string(0.0d0,format='(f11.7)')+',',format='(A,$)'
      endif
      if(counter eq 5L)then begin
         printf,1,string(0.0d0,format='(f11.7)')+', &',format='(A)'
         counter=0L
      endif
   endfor
endfor

printf,1,'else'
printf,1,'if(myid==1)write(*,*)''initialise the stellar_wind tabulated energies with all contributions (winds+SNII/Ia)'' '

;; Print the energy input
printf,1,'cEwind = reshape( (/ &'
for j=0L,nfiles-1L do begin
   filename=dir+files(j)
   readcol,filename,time,energy,/silent,format='(f,d)'
   lenergy=alog10(energy+1d0)
   for i=0L,nsteps-1L do begin
      counter=counter+1L
      if(j eq nfiles-1L and i eq nsteps-1L)then begin
         printf,1,string(lenergy(i),format='(f11.7)')+'/), (/'+nsstring+','+nzstring+'/) )',format='(A)'
         printf,1,' '
         counter=0L
      endif
      if(counter eq 1L)then begin
         printf,1,' & '+string(lenergy(i),format='(f11.7)')+',',format='(A,$)'
      endif
      if(counter gt 1L and counter le 4L)then begin
         printf,1,string(lenergy(i),format='(f11.7)')+',',format='(A,$)'
      endif
      if(counter eq 5L)then begin
         printf,1,string(lenergy(i),format='(f11.7)')+', &',format='(A)'
         counter=0L
      endif
   endfor
endfor

printf,1,'endif'

close,/all

end
