pro rewrite_lc18_modelm,fileinput=fileinput,fileout=fileout,feoverh=feoverh,vel=vel

if not keyword_set(feoverh)then feoverh=0
if not keyword_set(vel    )then vel=0
if not keyword_set(fileinput)then begin
   fileinput='./limongichieffi18_modelm_fe'+strcompress(feoverh,/remove_all)+'_v'+strcompress(vel,/remove_all)+'.dec'
endif
if not keyword_set(fileout)then begin
   fileout='./limongichieffi_modelm_fe'+strcompress(feoverh,/remove_all)+'_v'+strcompress(vel,/remove_all)+'_simplified.txt'
endif

mass_separatrix=8.0
;; list_elements =['p','He','C','N','O','Ne','Mg','Si','S','Fe']
;; list_elements2=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
list_elements =['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
list_elements2=['H','He','C','N','O','F','Ne','Mg','Si','S','Fe']
nel2follow=n_elements(list_elements)
mzas=[13.,15.,20.,25.,30.,40.,60.,80.,120.]
nmassread=n_elements(mzas)

myformat='(A,i,i,f'
for i=0L,nmassread-1L do begin
   myformat=myformat+',f'
endfor
myformat=myformat+')'
print,myformat,nmassread,n_elements(mzas)
readcol,fileinput,el,At,dummy1,dummy2,y1,y2,y3,y4,y5,y6,y7,y8,y9,format=myformat
nlines=n_elements(el)
yield_tmp=dblarr(nlines,27)
yield_tmp(*,0)=y1
yield_tmp(*,1)=y2
yield_tmp(*,2)=y3
yield_tmp(*,3)=y4
yield_tmp(*,4)=y5
yield_tmp(*,5)=y6
yield_tmp(*,6)=y7
yield_tmp(*,7)=y8
yield_tmp(*,8)=y9
mlos=fltarr(nmassread,nel2follow)
mla =fltarr(nmassread)
for j=0L,nel2follow-1L do begin
   indel=where(el eq list_elements2(j),nel)
   for i=0L,nmassread-1L do begin
      mlos(i,j)=total(yield_tmp(indel,i))
   endfor
endfor

for i=0L,nmassread-1L do begin
   mla(i)=total(mlos(i,*))
   print,mzas(i),mla(i)
endfor

if not keyword_set(fileout)then fileout='toto.dat'
if not keyword_set(feoverh)then feoverh=0.0
if not keyword_set(vel    )then vel=0.0
openw,2,fileout
printf,2,'#---Details of Columns:'
printf,2,'#    M0 (solMass)          (F6.2)  [1/6.5] Initial mass [ucd=phys.mass]'
printf,2,'#    [Fe/H]                (F7.2)  Initial metallicity [ucd=phys.abund.Z]'
printf,2,'#    vel (km/s)            (F7.2)  Initial rotation [ucd=phys.mass]'
printf,2,'#    M1 (solMass)          (F7.2)  Final mass (1) [ucd=phys.mass]'
printf,2,'#    El                    (a4)    Species i (2) [ucd=phys.atmol.element]'
printf,2,'#    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]'
printf,2,'#    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]'
printf,2,'#----- ------ ------ ------ ---- --------- -----------'
printf,2,'#M0                                                                 '
printf,2,'#(sol         vel    M1 (so      M(i)lost  M(i)lostall'
printf,2,'#Mass) Z0     (km/s) lMass) El   (solMass) (solMass)  '
printf,2,'#----- ------ ------ ------ ---- --------- -----------'
for i=0L,nmassread-1L do begin
   for j=0L,nel2follow-1L do begin
      printf,2,mzas(i),feoverh,vel,mzas(i)-mla(i),list_elements(j),mlos(i,j),mla(i),format='(f6.2,3f7.2,A5,2e10.2)'
   endfor
endfor
close,2

end
