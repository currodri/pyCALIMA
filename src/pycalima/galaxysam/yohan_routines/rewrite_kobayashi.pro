pro rewrite_kobayashi,zmet=zmet

if not keyword_set(zmet)then zmet=0.0
if(zmet eq 0.   )then zmets='0'
if(zmet eq 0.001)then zmets='0.001'
if(zmet eq 0.004)then zmets='0.004'
if(zmet eq 0.008)then zmets='0.008'
if(zmet eq 0.02 )then zmets='0.02'
if(zmet eq 0.05 )then zmets='0.05'

mass_separatrix=8.0
list_elements=['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
list_elements2=['H','He','C','N','O','F','Ne','Mg','Si','S','Fe']
;; list_elements=['p','He','C','N','O','Ne','Mg','Si','S','Fe']
;; list_elements2=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
nel2follow=n_elements(list_elements)
if(zmet ne 0.)then begin
   mzas=[0.9,1.0,1.2,1.5,1.8,1.9,2.0,2.2,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,10.0,13.0,15.0,18.0,20.0,25.0,30.0,40.0]
   mzas_hn=[20.0,25.0,30.0,40.0]
   mzas_i=[0.9,1.0,1.2,1.5,1.8,1.9,2.0,2.2,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]
   mzas_m=[8.0,10.0,13.0,15.0,18.0,20.0,25.0,30.0,40.0]
   Mrem=[0.473,0.564,0.574,0.600,0.615,0.630,0.640,0.660,0.663,0.682,0.718,0.792,0.852,0.879,0.900,0.929,0.963,1.010,1.120,1.150,1.600,1.500,1.580,1.550,1.804,2.100,2.210] ;; Z=0.02
endif else begin
   mzas=[0.9,1.0,1.2,1.5,1.8,1.9,2.0,2.2,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,8.0,10.0,11.0,13.0,15.0,18.0,20.0,25.0,30.0,40.0,100.0,140.0,140.0,150.0,170.0,200.0,270.0,300.0]
   mzas_hn=[20.0,25.0,30.0,40.0,100.,140.]
   mzas_i=[0.9,1.0,1.2,1.5,1.8,1.9,2.0,2.2,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]
   mzas_m=[8.0,10.0,11.0,13.0,15.0,18.0,20.0,25.0,30.0,40.0,100.0,140.0,140.0,150.0,170.0,200.0,270.0,300.0]
   Mrem=[0.8,0.8,0.9,1.0,1.0,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.5,1.6,1.5,1.6,1.7,1.9,2.1,2.9,10.4,13.2,0.0,0.0,0.0,0.0,0.0,0.0]
endelse
nmassread_i=n_elements(mzas_i)&yield_i=dblarr(nmassread_i,nel2follow)&mlos_i=dblarr(nmassread_i,nel2follow)&mla_i=dblarr(nmassread_i)
nmassread_m=n_elements(mzas_m)&mlos_m=dblarr(nmassread_m,nel2follow)&mla_m=dblarr(nmassread_m)
nmassread=n_elements(mzas)
nmassread_hn=n_elements(mzas_hn)&mlos_hn=dblarr(nmassread_hn,nel2follow)&mla_hn=dblarr(nmassread_hn)

myformat='(A,i'
for i=0L,nmassread-1L do begin
   myformat=myformat+',f'
endfor
myformat=myformat+')'
print,myformat,nmassread,n_elements(Mrem)
fileintermediate='./yield_ck13_z'+zmets+'.txt'
if(zmet eq 0.)then begin
   readcol,fileintermediate,el,At,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32,y33,y34,y35,y36,format='(A,)'
   nlines=n_elements(el)
   yield_tmp=dblarr(nlines,36)
endif else begin
   readcol,fileintermediate,el,At,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,format='(A,)'
   nlines=n_elements(el)
   yield_tmp=dblarr(nlines,27)
endelse
yield_tmp(*,0)=y1
yield_tmp(*,1)=y2
yield_tmp(*,2)=y3
yield_tmp(*,3)=y4
yield_tmp(*,4)=y5
yield_tmp(*,5)=y6
yield_tmp(*,6)=y7
yield_tmp(*,7)=y8
yield_tmp(*,8)=y9
yield_tmp(*,9)=y10
yield_tmp(*,10)=y11
yield_tmp(*,11)=y12
yield_tmp(*,12)=y13
yield_tmp(*,13)=y14
yield_tmp(*,14)=y15
yield_tmp(*,15)=y16
yield_tmp(*,16)=y17
yield_tmp(*,17)=y18
yield_tmp(*,18)=y19
yield_tmp(*,19)=y20
yield_tmp(*,20)=y21
yield_tmp(*,21)=y22
yield_tmp(*,22)=y23
yield_tmp(*,23)=y24
yield_tmp(*,24)=y25
yield_tmp(*,25)=y26
yield_tmp(*,26)=y27
if(zmet eq 0.)then begin
yield_tmp(*,27)=y28
yield_tmp(*,28)=y29
yield_tmp(*,29)=y30
yield_tmp(*,30)=y31
yield_tmp(*,31)=y32
yield_tmp(*,32)=y33
yield_tmp(*,33)=y34
yield_tmp(*,34)=y35
yield_tmp(*,35)=y36
endif
for j=0L,nel2follow-1L do begin
   indel=where(el eq list_elements2(j),nel)
   for i=0L,nmassread-1L do begin
      if(mzas(i) lt mass_separatrix)then begin
         yield_i(i,j)=total(yield_tmp(indel,i))
         if(mzas_i(i) eq 1.0)then begin
            print,yield_tmp(indel,i),'_'+el(indel)
         endif
      endif else begin
         mlos_m(i-nmassread_i,j)=total(yield_tmp(indel,i))
      endelse
   endfor
endfor


;; Read HNe file
myformat='(A,i'
for i=0L,nmassread_hn-1L do begin
   myformat=myformat+',f'
endfor
myformat=myformat+')'
print,myformat,nmassread,n_elements(Mrem)
filehn='./yield_hn_ck13_z'+zmets+'.txt'
if(zmet eq 0.)then begin
   readcol,filehn,el,At,y1,y2,y3,y4,y5,y6,format='(A,)'
   nlines=n_elements(el)
   yield_tmp=dblarr(nlines,6)
endif else begin
   readcol,filehn,el,At,y1,y2,y3,y4,format='(A,)'
   nlines=n_elements(el)
   yield_tmp=dblarr(nlines,4)
endelse
yield_tmp(*,0)=y1
yield_tmp(*,1)=y2
yield_tmp(*,2)=y3
yield_tmp(*,3)=y4
if(zmet eq 0.)then begin
yield_tmp(*,4)=y5
yield_tmp(*,5)=y6
endif
for j=0L,nel2follow-1L do begin
   indel=where(el eq list_elements2(j),nel)
   for i=0L,nmassread_hn-1L do begin
      mlos_hn(i,j)=total(yield_tmp(indel,i))
   endfor
endfor
mla_hn(0)=total(y1)
mla_hn(1)=total(y2)
mla_hn(2)=total(y3)
mla_hn(3)=total(y4)

;; Anders & Grevesse
XHAnders=0.70683d0
lAO =8.93d0
lAF =4.56d0
lAFe=7.67d0
lAC =8.56d0
lAN =8.05d0
lAMg=7.58d0
lASi=7.55d0
lAS =7.21d0
lANe=8.09d0
lAHe=10.98695d0 ; This is the value necessary to get the H, He, Z mass fraction of Anders & Grevesse
lAH =12.0d0
logH =lAH                  +alog10(XHAnders)
logHe=lAHe+alog10(3.9993d0)+alog10(XHAnders)
logF =lAF +alog10(18.998d0)+alog10(XHAnders)
logO =lAO +alog10(15.999d0)+alog10(XHAnders)
logFe=lAFe+alog10(55.845d0)+alog10(XHAnders)
logC =lAC +alog10(12.011d0)+alog10(XHAnders)
logN =lAN +alog10(14.007d0)+alog10(XHAnders)
logMg=lAMg+alog10(24.365d0)+alog10(XHAnders)
logSi=lASi+alog10(28.086d0)+alog10(XHAnders)
logS =lAS +alog10(32.065d0)+alog10(XHAnders)
logNe=lANe+alog10(20.1797d0)+alog10(XHAnders)
lAlist=[lAH ,lAHe ,lAC ,lAN ,lAO ,lAF ,lANe ,lAMg ,lASi ,lAS ,lAFe ]
llist =[logH,logHe,logC,logN,logO,logF,logNe,logMg,logSi,logS,logFe]
nllist=n_elements(llist)
nlist=n_elements(llist)
nfrac=dblarr(nlist)
mfrac=dblarr(nlist)
for i=0L,nlist-1L do begin
   nfrac(i)=10d0^(lAlist(i)-12.0)
   mfrac(i)=10d0^(llist(i)-lAH)
   print,10d0^(lAlist(i)-12.0),10d0^(llist(i)-lAH)
endfor
print,'zsun',1d0-(10d0^(llist(1)-lAH)+10d0^(llist(0)-lAH))

for j=0L,nel2follow-1L do begin
   indel=where(el eq list_elements2(j),nel)
   for i=0L,nmassread_i-1L do begin
      mlos_i(i,j)=mfrac(j)*(mzas_i(i)-mrem(i))+yield_i(i,j)
   endfor
endfor
for i=0L,nmassread-1L do begin
   if(mzas(i) lt mass_separatrix)then begin
      mla_i(i)=total(mlos_i(i,*))
   endif else begin
      mla_m(i-nmassread_i)=total(mlos_m(i-nmassread_i,*))
      print,mzas(i),mla_m(i-nmassread_i)
   endelse
endfor

openw,1,'kobayashi13agb_z'+zmets+'_simplified.txt'
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
for i=0L,nmassread_i-1L do begin
   if(mzas_i(i) gt 0.9 and mzas_i(i) lt 6.5)then begin
      for j=0L,nel2follow-1L do begin
         printf,1,mzas_i(i),zmet,mzas_i(i)-mla_i(i),list_elements(j),yield_i(i,j),mlos_i(i,j),0.,mla_i(i),format='(f5.2,f7.4,f6.3,A5,4e10.2)'
      endfor
   endif
endfor
close,1
openw,2,'kobayashi13snii_z'+zmets+'_simplified.txt'
printf,2,'#---Details of Columns:'
printf,2,'#    M0 (solMass)          (F6.2)  [1/6.5] Initial mass [ucd=phys.mass]'
printf,2,'#    Z/Zsun                (F7.2)  Initial metallicity [ucd=phys.abund.Z]'
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
if(zmet eq 0.)then zmet=1d-10*0.01345
for i=0L,nmassread_m-1L do begin
   if(mzas_m(i) gt 10.0)then begin
      for j=0L,nel2follow-1L do begin
         printf,2,mzas_m(i),alog10(zmet/0.01345),0.,mzas_m(i)-mla_m(i),list_elements(j),mlos_m(i,j),mla_m(i),format='(f6.2,3f7.2,A5,2e10.2)'
      endfor
   endif
endfor
for j=0L,nel2follow-1L do begin
   printf,2,50.0,alog10(zmet/0.01345),0.0,50.0,list_elements(j),1d-10,1d-10,format='(f6.2,3f7.2,A5,2e10.2)'
endfor
close,2

openw,3,'kobayashi13hn_z'+zmets+'_simplified.txt'
printf,3,'#---Details of Columns:'
printf,3,'#    M0 (solMass)          (F6.2)  [1/6.5] Initial mass [ucd=phys.mass]'
printf,3,'#    Z/Zsun                (F7.2)  Initial metallicity [ucd=phys.abund.Z]'
printf,3,'#    vel (km/s)            (F7.2)  Initial rotation [ucd=phys.mass]'
printf,3,'#    M1 (solMass)          (F7.2)  Final mass (1) [ucd=phys.mass]'
printf,3,'#    El                    (a4)    Species i (2) [ucd=phys.atmol.element]'
printf,3,'#    M(i)lost (solMass)    (D9.2)  Mass of species i lost in the wind [ucd=phys.mass]'
printf,3,'#    M(i)lostall (solMass) (D9.2)  Total mass lost in the wind [ucd=phys.mass]'
printf,3,'#----- ------ ------ ------ ---- --------- -----------'
printf,3,'#M0                                                                 '
printf,3,'#(sol         vel    M1 (so      M(i)lost  M(i)lostall'
printf,3,'#Mass) Z0     (km/s) lMass) El   (solMass) (solMass)  '
printf,3,'#----- ------ ------ ------ ---- --------- -----------'
for i=0L,nmassread_hn-1L do begin
   if(mzas_hn(i) gt 10.0)then begin
      for j=0L,nel2follow-1L do begin
         printf,3,mzas_hn(i),alog10(zmet/0.01345),0.,mzas_hn(i)-mla_hn(i),list_elements(j),mlos_hn(i,j),mla_hn(i),format='(f6.2,3f7.2,A5,2e10.2)'
      endfor
   endif
endfor
for j=0L,nel2follow-1L do begin
   printf,3,50.0,alog10(zmet/0.01345),0.0,50.0,list_elements(j),1d-10,1d-10,format='(f6.2,3f7.2,A5,2e10.2)'
endfor
close,3

stop

end
