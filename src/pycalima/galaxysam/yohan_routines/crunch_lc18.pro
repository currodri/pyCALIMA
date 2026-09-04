pro crunch_lc18,zmet=zmet,vel=vel,fileout=fileout

if not keyword_set(zmet)then zmet=0.0
if not keyword_set(vel )then vel=0.0

readcol,'limongichieffi18.txt',velzas,zzas,el,yzas13,yzas15,yzas20,yzas25,yzas30,yzas40,yzas60,yzas80,yzas120,format='(f,f,a,f,f,f,f,f,f,f,f,f)',skip=62

mzas=[13.,15.,20.,25.,30.,40.,60.,80.,120.]
;; list_elements =['H','He' ,'C'  ,'N'  ,'O'  ,'Ne'                ,'Mg'                       ,'Si'                ,'S'              ,'Fe']
;; list_elementsp=['p','He' ,'C'  ,'N'  ,'O'  ,'Ne'                ,'Mg'                       ,'Si'                ,'S'              ,'Fe']
;; list_isotopes =['H','He4','C12','N14','O16','Ne20','Ne22','Na22','Mg24','Mg25','Mg26','Al26','Si28','Si29','Si30','S32','S33','S34','Fe54','Fe56','Fe57','Fe58','Ni56','Co56']
list_elements =['H','He','C','N','O','F','Ne','Mg','Si','S','Fe']
list_elementsp=['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
list_isotopes =['H','He4','C12','N14','O16','F19','Ne20','Ne22','Na22','Mg24','Mg25','Mg26','Al26','Si28','Si29','Si30','S32','S33','S34','Fe54','Fe56','Fe57','Fe58','Ni56','Co56']

;; Na22 is a radioactive isotope that decays into Ne22 in 2 yr
;; Al26 is a radioactive isotope that decays into Mg26 in 1 Myr
;; Ni56 and Co56 are radioactive isotopes that decay into resp. Co56 and
;; Fe56 in a matter of days: add them into Fe

nmass=n_elements(mzas)
nel2follow=n_elements(list_elements)
nis2follow=n_elements(list_isotopes)
;; nisotopes_per_element=[1,1,1,1,1,3,4,3,3,6]
nisotopes_per_element=[1,1,1,1,1,1,3,4,3,3,6]
nlines=n_elements(velzas)
mlost_all=fltarr(nlines,nmass)
mlost_all(*,0)=yzas13(*)
mlost_all(*,1)=yzas15(*)
mlost_all(*,2)=yzas20(*)
mlost_all(*,3)=yzas25(*)
mlost_all(*,4)=yzas30(*)
mlost_all(*,5)=yzas40(*)
mlost_all(*,6)=yzas60(*)
mlost_all(*,7)=yzas80(*)
mlost_all(*,8)=yzas120(*)

print,nel2follow,nis2follow

for i=0L,nel2follow-1L do begin
   ii=total(nisotopes_per_element(0L:i))-nisotopes_per_element(i)
   print,list_elements(i),list_isotopes(ii:ii+nisotopes_per_element(i)-1L)
endfor

mlostall_array=fltarr(nmass)
nskip=0L
indall=where(velzas eq vel and zzas eq zmet,nind)
for j=0L,nmass-1L do begin
   mlostmass=mlost_all(indall,j)
   mlostall_array(j)=total(mlostmass)
endfor

nskip=0L
mlost_array=fltarr(nmass,nel2follow)
for i=0L,nel2follow-1L do begin
   for ii=nskip,nskip+nisotopes_per_element(i)-1L do begin
      ind   =where(el eq list_isotopes(ii) and velzas eq vel and zzas eq zmet,nind)
      for j=0L,nmass-1L do begin
         mlost_array(j,i)=mlost_array(j,i)+mlost_all(ind,j)
         print,list_isotopes(ii),mzas(j),vel,zmet,mlost_all(ind,j)
      endfor
   endfor
   nskip=nskip+nisotopes_per_element(i)
endfor

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
   for i=0L,nmass-1L do begin
      for j=0L,nel2follow-1L do begin
         printf,1,mzas(i),zmet,vel,mzas(i)-mlostall_array(i),list_elementsp(j),mlost_array(i,j),mlostall_array(i),format='(f6.2,3f7.2,A5,2e10.2)'
      endfor
   endfor
   close,1
endif


if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='toto.eps'
   device,filename=fileeps,/encaps,xs=15,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif
plot,mzas,alog10(mlostall_array),psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAS!n (M!dsun!n)',ytitle='log M!dlost!n/M!dsun!n',xr=[10.0,150.0],/xs,yr=[-5.,2.3],/ys
xpos=max(mzas)+0.1
xyouts,xpos,alog10(mlostall_array(nmass-1L)),'all'
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   last_mass(j)=mlost_array(nmass-1L,j)
endfor
indsort=reverse(sort(last_mass))
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   mycolor=j+2
   oplot,mzas,alog10(mlost_array(*,j)),psym=-4,th=th,color=mycolor
   ;; oplot,mzas,alog10(mzero_array(*,j)),th=th,color=mycolor,lines=2
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas)+5.
   endif else begin
      xpos=max(mzas)+15
   endelse
   xyouts,xpos,alog10(mlost_array(nmass-1L,j)),list_elements(j),color=mycolor
endfor

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif

end
