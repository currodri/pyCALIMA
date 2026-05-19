pro crunch_karakas,filename=filename,fileout=fileout,printps=printps,fileeps=fileeps

if not keyword_set(filename)then filename='karakas-subset.txt'
;; filename='karakas_z0.02.txt'
readcol,filename,mzas,dummy,zzas,mfinal,element,A,yield,mlost,mzero,x,x0,f,format='(f,a,f,f,a,i,f,f,f,f,f,f)',/silent

nlines=n_elements(mzas)
ind_m=where(element eq 'p',nmass)

print,'# of lines in file=',nlines
print,'# of different masses in file=',nmass

mzas_array=mzas(ind_m)
print,'list of masses:',mzas_array

;; list_elements           =['p','He','C','N','O','Ne','Mg','Si','S','Fe']
;; reformated_list_elements=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
list_elements           =['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
reformated_list_elements=['H','He','C','N','O','F','Ne','Mg','Si','S','Fe']
nel2follow=n_elements(list_elements)

if keyword_set(fileout)then begin 
   openw,1,fileout
   printf,1,'#---Details of Columns:'
   printf,1,'#    M0 (solMass)          (F5.2)  [1/6.5] Initial mass [ucd=phys.mass]'
   printf,1,'#    Z0                    (F7.4)  Initial metallicity [ucd=phys.abund.Z]'
   printf,1,'#    M1 (solMass)          (F6.3)  Final mass (1) [ucd=phys.mass]'
   printf,1,'#    El                    (a4)    Species i (2) [ucd=phys.atmol.element]'
   printf,1,'#    Yield (solMass)       (D9.2)  Net yield [ucd=phys.composition.yield]'
   printf,1,'#    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]'
   printf,1,'#    M(i)0 (solMass)       (D9.2)  Mass of species i initially present in the wind [ucd=phys.mass]'
   printf,1,'#    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]'
   printf,1,'#----- ----- ----- ---- --------- --------- --------- -----------'
   printf,1,'#M0                                                                 '
   printf,1,'#(sol        M1 (so     Yield     M(i)lost  M(i)0     M(i)lostall'
   printf,1,'#Mass) Z0    lMass) El  (solMass) (solMass) (solMass) (solMass)  '
   printf,1,'#----- ----- ----- ---- --------- --------- --------- -----------'
endif

mlostall_array=fltarr(nmass)
yield_array=fltarr(nmass,nel2follow)
mlost_array=fltarr(nmass,nel2follow)
mzero_array=fltarr(nmass,nel2follow)
for i=0L,nmass-1L do begin
   ind=where(mzas eq mzas_array(i))
   mlostall_array(i)=total(mlost(ind))
   for j=0L,nel2follow-1L do begin
      ind=where(mzas eq mzas_array(i) and element eq list_elements(j),ncheck)
      i0=ind(0)
      yield_array(i,j)=total(yield(ind))
      mlost_array(i,j)=total(mlost(ind))
      mzero_array(i,j)=total(mzero(ind))
      print,mzas(i0),zzas(i0),mfinal(i0),element(i0),total(yield(ind)),total(mlost(ind)),total(mzero(ind)),mlostall_array(i),format='(f5.2,f7.4,f6.3,A5,4e10.2)'
      if keyword_set(fileout)then printf,1,mzas(i0),zzas(i0),mfinal(i0),element(i0),total(yield(ind)),total(mlost(ind)),total(mzero(ind)),mlostall_array(i),format='(f5.2,f7.4,f6.3,A5,4e10.2)'
   endfor
endfor
if keyword_set(fileout)then close,1

if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='toto.eps'
   device,filename=fileeps,/encaps,xs=15,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif
plot,mzas_array,alog10(mlostall_array),psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAS!n (M!dsun!n)',ytitle='log M!dlost!n/M!dsun!n',xr=[0.9,8.0],/xs,yr=[-5.,1.3],/ys
xpos=max(mzas_array)+0.1
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
   oplot,mzas_array,alog10(mlost_array(*,j)),psym=-4,th=th,color=mycolor
   oplot,mzas_array,alog10(mzero_array(*,j)),th=th,color=mycolor,lines=2
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas_array)+0.1
   endif else begin
      xpos=max(mzas_array)+0.4
   endelse
   xyouts,xpos,alog10(mlost_array(nmass-1L,j)),reformated_list_elements(j),color=mycolor
endfor



if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif


if keyword_set(printps)then begin
   device,/close
   device,filename='yield_'+fileeps,/encaps,xs=15,ys=15,/color
endif
plot,mzas_array,yield_array(*,0)/mzas_array,psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAS!n (M!dsun!n)',ytitle='Yield',xr=[0.9,8.0],/xs,yr=[-1.,+1.],/ys,lines=2
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   last_mass(j)=yield_array(nmass-1L,j)
endfor
indsort=reverse(sort(last_mass))
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   mycolor=j+2
   oplot,mzas_array,yield_array(*,0)/mzas_array,psym=-4,th=th,color=mycolor
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas_array)+0.1
   endif else begin
      xpos=max(mzas_array)+0.4
   endelse
   xyouts,xpos,alog10(mlost_array(nmass-1L,j)),reformated_list_elements(j),color=mycolor
endfor

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif

end
