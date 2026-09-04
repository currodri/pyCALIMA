pro interpolate_lc18,zmet=zmet,vel=vel,fileout=fileout,modelm=modelm

if not keyword_set(vel )then vel=0
vels=strcompress(vel,/remove_all)

zsun=0.01345

if keyword_set(modelm)then begin
   rootname='limongichieffi_modelm'
endif else begin
   rootname='limongichieffi'
endelse
;; if(zmet lt 3.236d-5)then begin
;;    spawn,'cp '+rootname+'_fe-3_vel'+vels+'_simplified.txt '+fileout
;;    stop
;; endif
;; if(zmet ge 0.01345)then begin
;;    spawn,'cp '+rootname+'_fe0_vel'+vels+'_simplified.txt '+fileout
;;    stop
;; endif
;; [Fe/H]= 0 <=> Z=1.345d-2
;; [Fe/H]=-1 <=> Z=3.236d-3
;; [Fe/H]=-2 <=> Z=3.236d-4
;; [Fe/H]=-3 <=> Z=3.236d-5
if(zmet ge 1.345d-2)then begin
   fes1='fe0'
   fes2=fes1
   z1=1.345d-2
   z2=z1
endif   
if(zmet ge 3.236d-3 and zmet lt 1.345d-2)then begin
   fes1='fe-1'
   fes2='fe0'
   z1=3.236d-3
   z2=1.345d-2
endif
if(zmet ge 3.236d-4 and zmet lt 3.236d-3)then begin
   fes1='fe-2'
   fes2='fe-1'
   z1=3.236d-4
   z2=3.236d-3
endif
if(zmet ge 3.236d-5 and zmet lt 3.236d-4)then begin
   fes1='fe-3'
   fes2='fe-2'
   z1=3.236d-5
   z2=3.236d-4
endif
if(zmet lt 3.236d-5)then begin
   fes1='fe-3'
   fes2=fes1
   z1=3.236d-5
   z2=z1
endif

if(vel ge 300.)then begin
   vels1='vel300'
   vels2=vels1
   v1=300d0
   v2=v1
endif
if(vel ge 150. and vel lt 300.0)then begin
   vels1='vel150'
   vels2='vel300'
   v1=150d0
   v2=300d0
endif
if(vel ge 0. and vel lt 150.0)then begin
   vels1='vel0'
   vels2='vel150'
   v1=0d0
   v2=150d0
endif
file11=''+rootname+'_'+fes1+'_'+vels1+'_simplified.txt'
file21=''+rootname+'_'+fes2+'_'+vels1+'_simplified.txt'
file12=''+rootname+'_'+fes1+'_'+vels2+'_simplified.txt'
file22=''+rootname+'_'+fes2+'_'+vels2+'_simplified.txt'

print,file11,z1,v1
print,file21,z2,v1
print,file12,z1,v2
print,file22,z2,v2
readcol,file11,mzas_11,zzas_11,vel1,mfin_11,el_11,mlos_11,mla_11,format='(f,f,f,f,a,f,f,f,f)',/silent
readcol,file21,mzas_21,zzas_21,vel2,mfin_21,el_21,mlos_21,mla_21,format='(f,f,f,f,a,f,f,f,f)',/silent
readcol,file12,mzas_12,zzas_12,vel1,mfin_12,el_12,mlos_12,mla_12,format='(f,f,f,f,a,f,f,f,f)',/silent
readcol,file22,mzas_22,zzas_22,vel2,mfin_22,el_22,mlos_22,mla_22,format='(f,f,f,f,a,f,f,f,f)',/silent

dz=(zmet-z1)
deltaz=z2-z1
dv=(vel-v1)
deltav=v2-v1

nlines11=n_elements(mzas_11)
nlines21=n_elements(mzas_21)
nlines12=n_elements(mzas_12)
nlines22=n_elements(mzas_22)
nlines=nlines11
target=1
mynlines=[nlines11,nlines21,nlines12,nlines22]
for i=1,3 do begin
   if(nlines gt mynlines(i))then begin
      nlines=mynlines(i)
      target=i
   endif
endfor
print,'# of lines',nlines11,nlines21,nlines12,nlines22,nlines,target,format='(A10,6I5)'

if(deltaz ne 0.)then ratioz=dz/deltaz
if(deltav ne 0.)then ratiov=dv/deltav

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
      ind=where(mzas_21 eq mzas_11(i) and el_21 eq el_11(i),nfound)
      if(nfound eq 0)then begin ;; if mass is missing -> interpolate
         ind_massinterpol=where(mzas_21 lt mzas_11(i) and el_21 eq el_11(i),nget)
         i0=ind_massinterpol(nget-1L)
         ind_massinterpol=where(mzas_21 gt mzas_11(i) and el_21 eq el_11(i))
         i1=ind_massinterpol(0)
         mratio=(mzas_11(i)-mzas_21(i0))/(mzas_21(i1)-mzas_21(i0))
         mremnant2=mfin_21(i0)+(mfin_21(i1)-mfin_21(i0))*mratio
         mlos2    =mlos_21(i0)+(mlos_21(i1)-mlos_21(i0))*mratio
         mla2     =mla_21 (i0)+(mla_21 (i1)-mla_21 (i0))*mratio
         print,mlos_21(i0),mlos_21(i1),mlos2,el_11(i)
         ;; now interpolate in redshift
         mlos    =mlos2 +(mlos_11(i)-mlos2 )*ratioz
         mla     =mla2  +(mla_11 (i)-mla2  )*ratioz
         mremnant=mzas_11(i)-mla
         print,'special case: mass does not exist 1...'
         stop
      endif else begin
         ii=ind(0)
         mremnant=mfin_11(i)+(mfin_21(ii)-mfin_11(i))*ratioz
         mlos    =mlos_11(i)+(mlos_21(ii)-mlos_11(i))*ratioz
         mla     =mla_11 (i)+(mla_21 (ii)-mla_11 (i))*ratioz
         mremnant=mzas_11(i)-mla
         print,'fishy...'
         stop
      endelse
      print,target,nfound,mzas_11(i),zmet,mremnant,vel,el_11(i),mlos,mla,format='(2i2,f6.2,3f7.2,A5,2e10.2)'
      if keyword_set(fileout)then printf,1,mzas_11(i),alog10(zmet/zsun),mremnant,vel,el_11(i),mlos,mla,format='(f6.2,3f7.2,A5,2e10.2)'
   endif else begin
      ind=where(mzas_11 eq mzas_21(i) and el_11 eq el_21(i),nfound)
      if(nfound eq 0)then begin ;; if mass is missing -> interpolate
         ind_massinterpol=where(mzas_11 lt mzas_21(i) and el_11 eq el_21(i),nget)
         i0=ind_massinterpol(nget-1L)
         ind_massinterpol=where(mzas_11 gt mzas_21(i) and el_11 eq el_21(i))
         i1=ind_massinterpol(0)
         mratio=(mzas_21(i)-mzas_11(i0))/(mzas_11(i1)-mzas_11(i0))
         mremnant1=mfin_11(i0)+(mfin_11(i1)-mfin_11(i0))*mratio
         mlos1    =mlos_11(i0)+(mlos_11(i1)-mlos_11(i0))*mratio
         mla1     =mla_11 (i0)+(mla_11 (i1)-mla_11 (i0))*mratio
         print,mlos_11(i0),mlos_11(i1),mlos1,el_21(i)
         ;; now interpolate in redshift
         mlos    =mlos1 +(mlos_21(i)-mlos1 )*ratioz
         mla     =mla1  +(mla_21 (i)-mla1  )*ratioz
         mremnant=mzas_21(i)-mla
         print,'special case: mass does not exist2...'
         stop
      endif else begin
         ii=ind(0)
         ;;i and ii equal...
         if(deltaz eq 0.)then begin
            if(deltav eq 0.)then begin
               mremnant2=mfin_11(ii)
               mlos     =mlos_11(ii)
               mla      =mla_11(ii)
            endif else begin
               f1=mfin_11(ii)
               f2=mfin_12(i )
               mremnant2=(1d0-ratiov)*f1+ratiov*f2
               f1=mlos_11(ii)
               f2=mlos_12(i )
               mlos     =(1d0-ratiov)*f1+ratiov*f2
               f1=mla_11 (ii)
               f2=mla_12 (i )
               mla    =(1d0-ratiov)*f1+ratiov*f2
            endelse
         endif else if(deltav eq 0.)then begin
            mremnant2=(1d0-ratioz)*mfin_11(ii)+ratioz*mfin_21(i)
            mlos     =(1d0-ratioz)*mlos_11(ii)+ratioz*mlos_21(i)
            mla      =(1d0-ratioz)*mla_11 (ii)+ratioz*mla_21 (i)
         endif else begin
            f1=(1d0-ratioz)*mfin_11(ii)+ratioz*mfin_21(i)
            f2=(1d0-ratioz)*mfin_12(i )+ratioz*mfin_22(i)
            mremnant2=(1d0-ratiov)*f1+ratiov*f2
            f1=(1d0-ratioz)*mlos_11(ii)+ratioz*mlos_21(i)
            f2=(1d0-ratioz)*mlos_12(i )+ratioz*mlos_22(i)
            mlos     =(1d0-ratiov)*f1+ratiov*f2
            f1=(1d0-ratioz)*mla_11 (ii)+ratioz*mla_21 (i)
            f2=(1d0-ratioz)*mla_12 (i )+ratioz*mla_22 (i)
            mla      =(1d0-ratiov)*f1+ratiov*f2
         endelse
         mremnant=mzas_11(ii)-mla ;; infer the remnant mass from the interpolated wind mass
         ;; print,'mremnant comparison:',mremnant,mremnant2
         ;; print,target,nfound,mlos_11(ii),mlos_21(i),mlos,ratioz,el_21(i),format='(2i2,4e10.2,A5)'
      endelse
      print,target,nfound,mzas_21(i),zmet,mremnant,vel,el_21(i),mlos,mla,format='(2i2,f7.2,3f7.2,A5,2e10.2)'
      if keyword_set(fileout)then printf,1,mzas_21(i),alog10(zmet/zsun),vel,mremnant,el_21(i),mlos,mla,format='(f6.2,3f7.2,A5,2e10.2)'
   endelse
endfor
if keyword_set(fileout)then close,1

end
