pro interpolate_metallicity_karakas,zmet=zmet,fileout=fileout

if not keyword_set(zmet)then begin
   print,'I need some metallicity value (absolute value, e.g. 0.01345)'
   print,'Stopping...'
   stop
endif

if(zmet lt  0.02 and zmet ge  0.008)then begin
   file1='karakas_z0.008_simplified.txt'
   file2='karakas_z0.02_simplified.txt'
endif
if(zmet lt 0.008 and zmet ge  0.004)then begin
   file1='karakas_z0.004_simplified.txt'
   file2='karakas_z0.008_simplified.txt'
endif
if(zmet lt 0.004)then begin
   file1='karakas_z0.0001_simplified.txt'
   file2='karakas_z0.004_simplified.txt'
endif
print,file1
print,file2

readcol,file1,mzas_1,zzas_1,mfin_1,el_1,y_1,mlos_1,mzer_1,mla_1,format='(f,f,f,a,f,f,f,f)',/silent
readcol,file2,mzas_2,zzas_2,mfin_2,el_2,y_2,mlos_2,mzer_2,mla_2,format='(f,f,f,a,f,f,f,f)',/silent

dz=(zmet-max(zzas_1))
deltaz=max(zzas_2)-max(zzas_1)
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
   printf,1,'#    M0 (solMass)          (F5.2)  [1/6.5] Initial mass [ucd=phys.mass]'
   printf,1,'#    Z0                    (D10.3)  Initial metallicity [ucd=phys.abund.Z]'
   printf,1,'#    M1 (solMass)          (F6.3)  Final mass (1) [ucd=phys.mass]'
   printf,1,'#    El                    (a4)    Species i (2) [ucd=phys.atmol.element]'
   printf,1,'#    Yield (solMass)       (D9.2)  Net yield [ucd=phys.composition.yield]'
   printf,1,'#    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]'
   printf,1,'#    M(i)0 (solMass)       (D9.2)  Mass of species i initially present in the wind [ucd=phys.mass]'
   printf,1,'#    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]'
   printf,1,'#----- --------- ----- ---- --------- --------- --------- -----------'
   printf,1,'#M0                                                                 '
   printf,1,'#(sol            M1 (so     Yield     M(i)lost  M(i)0     M(i)lostall'
   printf,1,'#Mass) Z0        lMass) El  (solMass) (solMass) (solMass) (solMass)  '
   printf,1,'#----- --------- ----- ---- --------- --------- --------- -----------'
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
         yield2   =y_2   (i0)+(y_2   (i1)-y_2   (i0))*mratio
         mlos2    =mlos_2(i0)+(mlos_2(i1)-mlos_2(i0))*mratio
         mzer2    =mzer_2(i0)+(mzer_2(i1)-mzer_2(i0))*mratio
         mla2     =mla_2 (i0)+(mla_2 (i1)-mla_2 (i0))*mratio
         print,mlos_2(i0),mlos_2(i1),mlos2,el_1(i)
         ;; now interpolate in redshift
         yield   =yield2+(y_1   (i)-yield2)*ratio
         mlos    =mlos2 +(mlos_1(i)-mlos2 )*ratio
         mzer    =mzer2 +(mzer_1(i)-mzer2 )*ratio
         mla     =mla2  +(mla_1 (i)-mla2  )*ratio
         mremnant=mzas_1(i)-mla
      endif else begin
         ii=ind(0)
         mremnant=mfin_1(i)+(mfin_2(ii)-mfin_1(i))*ratio
         yield   =y_1   (i)+(y_2   (ii)-y_1   (i))*ratio
         mlos    =mlos_1(i)+(mlos_2(ii)-mlos_1(i))*ratio
         mzer    =mzer_1(i)+(mzer_2(ii)-mzer_1(i))*ratio
         mla     =mla_1 (i)+(mla_2 (ii)-mla_1 (i))*ratio
         mremnant=mzas_1(i)-mla
      endelse
      print,target,nfound,mzas_1(i),zmet,mremnant,el_1(i),yield,mlos,mzer,mla
      if keyword_set(fileout)then printf,1,mzas_1(i),zmet,mremnant,el_1(i),yield,mlos,mzer,mla,format='(f5.2,e11.3,f6.3,A5,4e10.2)'
   endif else begin
      ind=where(mzas_1 eq mzas_2(i) and el_1 eq el_2(i),nfound)
      if(nfound eq 0)then begin ;; if mass is missing -> interpolate
         ind_massinterpol=where(mzas_1 lt mzas_2(i) and el_1 eq el_2(i),nget)
         i0=ind_massinterpol(nget-1L)
         ind_massinterpol=where(mzas_1 gt mzas_2(i) and el_1 eq el_2(i))
         i1=ind_massinterpol(0)
         mratio=(mzas_2(i)-mzas_1(i0))/(mzas_1(i1)-mzas_1(i0))
         mremnant1=mfin_1(i0)+(mfin_1(i1)-mfin_1(i0))*mratio
         yield1   =y_1   (i0)+(y_1   (i1)-y_1   (i0))*mratio
         mlos1    =mlos_1(i0)+(mlos_1(i1)-mlos_1(i0))*mratio
         mzer1    =mzer_1(i0)+(mzer_1(i1)-mzer_1(i0))*mratio
         mla1     =mla_1 (i0)+(mla_1 (i1)-mla_1 (i0))*mratio
         print,mlos_1(i0),mlos_1(i1),mlos1,el_2(i)
         ;; now interpolate in redshift
         yield   =yield1+(y_2   (i)-yield1)*ratio
         mlos    =mlos1 +(mlos_2(i)-mlos1 )*ratio
         mzer    =mzer1 +(mzer_2(i)-mzer1 )*ratio
         mla     =mla1  +(mla_2 (i)-mla1  )*ratio
         mremnant=mzas_2(i)-mla
      endif else begin
         ii=ind(0)
         mremnant=mfin_1(ii)+(mfin_2(i)-mfin_1(ii))*ratio
         yield   =y_1   (ii)+(y_2   (i)-y_1   (ii))*ratio
         mlos    =mlos_1(ii)+(mlos_2(i)-mlos_1(ii))*ratio
         mzer    =mzer_1(ii)+(mzer_2(i)-mzer_1(ii))*ratio
         mla     =mla_1 (ii)+(mla_2 (i)-mla_1 (ii))*ratio
         mremnant=mzas_1(ii)-mla
      endelse
      print,target,nfound,mzas_2(i),zmet,mremnant,el_2(i),yield,mlos,mzer,mla
      if keyword_set(fileout)then printf,1,mzas_2(i),zmet,mremnant,el_2(i),yield,mlos,mzer,mla,format='(f5.2,e11.3,f6.3,A5,4e10.2)'
   endelse
endfor
if keyword_set(fileout)then close,1


end
