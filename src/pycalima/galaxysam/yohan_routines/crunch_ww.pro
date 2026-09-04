pro crunch_ww,filename=filename,fileout=fileout,printps=printps,fileeps=fileeps

if not keyword_set(filename)then filename='ww95fromMolla15.txt'
readcol,filename,dummy,zzas,mzas,yH,yD,yHe3,yHe4,yC12,yO16,yNe20,yMg24,ySi28,yS32,yCa40,yFe56,mremnant,yC13,yN14,format='(a,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f)',skip=66

list_elements=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
nel2follow=n_elements(list_elements)
; Need to take Anders & Grevesse 1989 abundances 
lA=[12.00d0,10.99d0,8.56d0,8.05d0,8.93d0,8.09d0,7.58d0,7.55d0,7.21d0,7.67d0]
AA=[1d0,4d0,12d0,14d0,16d0,20d0,24d0,28d0,32d0,56d0]
;; laH =12.00d0
;; laHe=10.99d0
;; laC = 8.56d0
;; laN = 8.05d0
;; la0 = 8.93d0
;; laNe= 8.09d0
;; laMg= 7.58d0
;; laSi= 7.55d0
;; laS = 7.21d0
;; laFe= 7.67d0
XH=0.7068d0
XA=10d0^(lA-lA(0))*AA*XH
print,XA
nlines=n_elements(mzas)
mlost_array=fltarr(nlines,nel2follow)
mlost_array(0L:nlines-1L,0)=mzas*(yH   )+(mzas-mremnant)*XA(0)
mlost_array(0L:nlines-1L,1)=mzas*(yHe4 )+(mzas-mremnant)*XA(1)
mlost_array(0L:nlines-1L,2)=mzas*(yC12 )+(mzas-mremnant)*XA(2)
mlost_array(0L:nlines-1L,3)=mzas*(yN14 )+(mzas-mremnant)*XA(3)
mlost_array(0L:nlines-1L,4)=mzas*(yO16 )+(mzas-mremnant)*XA(4)
mlost_array(0L:nlines-1L,5)=mzas*(yNe20)+(mzas-mremnant)*XA(5)
mlost_array(0L:nlines-1L,6)=mzas*(yMg24)+(mzas-mremnant)*XA(6)
mlost_array(0L:nlines-1L,7)=mzas*(ySi28)+(mzas-mremnant)*XA(7)
mlost_array(0L:nlines-1L,8)=mzas*(yS32 )+(mzas-mremnant)*XA(8)
mlost_array(0L:nlines-1L,9)=mzas*(yFe56)+(mzas-mremnant)*XA(9)

ind=where(zzas eq 0.02)
print,mzas(ind),mlost_array(ind,2)

if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='toto.eps'
   device,filename=fileeps,/encaps,xs=15,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif
if not keyword_set(zselect)then zselect=0.02
ind=where(zzas eq zselect,nmass)
mlost=mzas(ind)-mremnant(ind)
plot,mzas(ind),alog10(mlost),psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAS!n (M!dsun!n)',ytitle='log M!dlost!n/M!dsun!n',xr=[8.0,110.0],/xs,yr=[-5.,2.3],/ys
xpos=max(mzas(ind))+0.1
xyouts,xpos,alog10(mlost(nmass-1L)),'all'
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   ind=where(zzas eq zselect)
   last_mass(j)=mlost_array(ind(nmass-1L),j)
endfor
indsort=reverse(sort(last_mass))
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   mycolor=j+2
   oplot,mzas(ind),alog10(mlost_array(ind,j)),psym=-4,th=th,color=mycolor
   ;; oplot,mzas(ind),alog10(mzero_array(*,j)),th=th,color=mycolor,lines=2
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas)+0.1
   endif else begin
      xpos=max(mzas)+0.4
   endelse
   xyouts,xpos,alog10(mlost_array(ind(nmass-1L),j)),list_elements(j),color=mycolor
endfor



if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif


stop
ind_m=where(element eq 'p',nmass)

print,'# of lines in file=',nlines
print,'# of different masses in file=',nmass

mzas_array=mzas(ind_m)
print,'list of masses:',mzas_array

reformated_list_elements=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
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

end
