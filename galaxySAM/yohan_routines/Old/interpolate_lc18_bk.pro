pro interpolate_lc18,zmet=zmet,vel=vel,fileout=fileout,modelm=modelm

if not keyword_set(vel )then vel=0
vels=strcompress(vel,/remove_all)

zsun=0.01345

if keyword_set(modelm)then begin
   rootname='limongichieffi_modelm'
endif else begin
   rootname='limongichieffi'
endelse
if(zmet lt 3.236d-5)then begin
   spawn,'cp '+rootname+'_fe-3_vel'+vels+'_simplified.txt '+fileout
   stop
endif
if(zmet ge 0.01345)then begin
   spawn,'cp '+rootname+'_fe0_vel'+vels+'_simplified.txt '+fileout
   stop
endif
;; [Fe/H]= 0 <=> Z=1.345d-2
;; [Fe/H]=-1 <=> Z=3.236d-3
;; [Fe/H]=-2 <=> Z=3.236d-4
;; [Fe/H]=-3 <=> Z=3.236d-5
if(zmet ge 3.236d-3 and zmet lt 1.345d-2)then begin
   file1=''+rootname+'_fe-1_vel'+vels+'_simplified.txt'
   file2=''+rootname+'_fe0_vel'+vels+'_simplified.txt'
   z1=3.234d-3
   z2=1.345d-2
endif
if(zmet ge 3.236d-4 and zmet lt 3.236d-3)then begin
   file1=''+rootname+'_fe-2_vel'+vels+'_simplified.txt'
   file2=''+rootname+'_fe-1_vel'+vels+'_simplified.txt'
   z1=3.234d-4
   z2=3.234d-3
endif
if(zmet ge 3.236d-5 and zmet lt 3.236d-4)then begin
   file1=''+rootname+'_fe-3_vel'+vels+'_simplified.txt'
   file2=''+rootname+'_fe-2_vel'+vels+'_simplified.txt'
   z1=3.234d-5
   z2=3.234d-4
endif

print,file1,z1
print,file2,z2
readcol,file1,mzas_1,zzas_1,vel1,mfin_1,el_1,mlos_1,mla_1,format='(f,f,f,f,a,f,f,f,f)',/silent
readcol,file2,mzas_2,zzas_2,vel2,mfin_2,el_2,mlos_2,mla_2,format='(f,f,f,f,a,f,f,f,f)',/silent

dz=(zmet-z1)
deltaz=z2-z1

deltam=mlos_2-mlos_1

nlines1=n_elements(mzas_1)
nlines2=n_elements(mzas_2)
if(nlines1 lt nlines2)then begin
   nlines=nlines1
   target=2
endif else begin
   nlines=nlines2
   target=1
endelse
print,'# of lines',nlines1,nlines2,nlines

ratio=dz/deltaz
if keyword_set(fileout)then begin
   openw,1,fileout
   printf,1,'#---Details of Columns:'
   printf,1,'#    M0 (solMass)          (F6.2)  [1/6.5] Initial mass [ucd=phys.mass]'
   printf,1,'#    [Fe/H]                (F7.2)  Initial metallicity [ucd=phys.abund.Z]'
   printf,1,'#    vel (km/s)            (F7.2)  Initial rotation [ucd=phys.mass]'
   printf,1,'#    M1 (solMass)          (F7.2)  Final mass (1) [ucd=phys.mass]'
   printf,1,'#    El                    (a4)    Species i (2) [ucd=phys.atmol.element]'
   printf,1,'#    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]'
   printf,1,'#    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]'
   printf,1,'#----- ------ ------ ------ ---- --------- -----------'
   printf,1,'#M0                                                                 '
   printf,1,'#(sol         vel    M1 (so      M(i)lost  M(i)lostall'
   printf,1,'#Mass) Z0     (km/s) lMass) El   (solMass) (solMass)  '
   printf,1,'#----- ------ ------ ------ ---- --------- -----------'
endif
for i=0L,nlines-1L do begin
   if(target eq 2)then begin
      ind=where(mzas_2 eq mzas_1(i) and el_2 eq el_1(i),nfound)
      if(nfound eq 0)then begin ;; if mass is missing -> interpolate
         ind_massinterpol=where(mzas_2 lt mzas_1(i) and el_2 eq el_1(i),nget)
         i0=ind_massinterpol(nget-1L)
         ind_massinterpol=where(mzas_2 gt mzas_1(i) and el_2 eq el_1(i))
         i1=ind_massinterpol(0)
         mratio=(mzas_1(i)-mzas_2(i0))/(mzas_2(i1)-mzas_2(i0))
         mremnant2=mfin_2(i0)+(mfin_2(i1)-mfin_2(i0))*mratio
         mlos2    =mlos_2(i0)+(mlos_2(i1)-mlos_2(i0))*mratio
         mla2     =mla_2 (i0)+(mla_2 (i1)-mla_2 (i0))*mratio
         print,mlos_2(i0),mlos_2(i1),mlos2,el_1(i)
         ;; now interpolate in redshift
         mlos    =mlos2 +(mlos_1(i)-mlos2 )*ratio
         mla     =mla2  +(mla_1 (i)-mla2  )*ratio
         mremnant=mzas_1(i)-mla
      endif else begin
         ii=ind(0)
         mremnant=mfin_1(i)+(mfin_2(ii)-mfin_1(i))*ratio
         mlos    =mlos_1(i)+(mlos_2(ii)-mlos_1(i))*ratio
         mla     =mla_1 (i)+(mla_2 (ii)-mla_1 (i))*ratio
         mremnant=mzas_1(i)-mla
      endelse
      print,target,nfound,mzas_1(i),zmet,mremnant,vel,el_1(i),mlos,mla,format='(2i2,f6.2,3f7.2,A5,2e10.2)'
      if keyword_set(fileout)then printf,1,mzas_1(i),alog10(zmet/zsun),mremnant,vel,el_1(i),mlos,mla,format='(f6.2,3f7.2,A5,2e10.2)'
   endif else begin
      ind=where(mzas_1 eq mzas_2(i) and el_1 eq el_2(i),nfound)
      if(nfound eq 0)then begin ;; if mass is missing -> interpolate
         ind_massinterpol=where(mzas_1 lt mzas_2(i) and el_1 eq el_2(i),nget)
         i0=ind_massinterpol(nget-1L)
         ind_massinterpol=where(mzas_1 gt mzas_2(i) and el_1 eq el_2(i))
         i1=ind_massinterpol(0)
         mratio=(mzas_2(i)-mzas_1(i0))/(mzas_1(i1)-mzas_1(i0))
         mremnant1=mfin_1(i0)+(mfin_1(i1)-mfin_1(i0))*mratio
         mlos1    =mlos_1(i0)+(mlos_1(i1)-mlos_1(i0))*mratio
         mla1     =mla_1 (i0)+(mla_1 (i1)-mla_1 (i0))*mratio
         print,mlos_1(i0),mlos_1(i1),mlos1,el_2(i)
         ;; now interpolate in redshift
         mlos    =mlos1 +(mlos_2(i)-mlos1 )*ratio
         mla     =mla1  +(mla_2 (i)-mla1  )*ratio
         mremnant=mzas_2(i)-mla
      endif else begin
         ii=ind(0)
         mremnant=mfin_1(ii)+(mfin_2(i)-mfin_1(ii))*ratio
         mlos    =mlos_1(ii)+(mlos_2(i)-mlos_1(ii))*ratio
         mla     =mla_1 (ii)+(mla_2 (i)-mla_1 (ii))*ratio
         mremnant=mzas_1(ii)-mla
         ;; print,target,nfound,mlos_1(ii),mlos_2(i),mlos,ratio,el_2(i),format='(2i2,4e10.2,A5)'
      endelse
      print,target,nfound,mzas_2(i),zmet,mremnant,vel,el_2(i),mlos,mla,format='(2i2,f6.2,3f7.2,A5,2e10.2)'
      if keyword_set(fileout)then printf,1,mzas_2(i),alog10(zmet/zsun),vel,mremnant,el_2(i),mlos,mla,format='(f6.2,3f7.2,A5,2e10.2)'
   endelse
endfor
if keyword_set(fileout)then close,1

end
