pro plot_yields,intermediate=intermediate,massive=massive,printps=printps,fileeps=fileeps,ilabel=ilabel,mlabel=mlabel,kobayashi=kobayashi,stop=stop

;; Z=0.02 Kobayashi+060
mkobayashi   =[13.   ,15.     ,18.    ,20.    ,25.    ,30.    ,40.    ]
mg24kobayashi=[2.52d-2,3.79d-2,1.03d-1,7.16d-2,2.18d-1,1.88d-1,3.10d-1]
mg25kobayashi=[2.56d-3,1.47d-3,7.08d-3,1.44d-2,3.13d-2,3.12d-2,7.28d-2]
mg26kobayashi=[2.18d-3,1.73d-3,5.90d-3,8.87d-3,2.73d-2,2.80d-2,7.34d-2]
mgkobayashi=mg24kobayashi+mg25kobayashi+mg26kobayashi

if not keyword_set(intermediate)then intermediate='karakas_z0.02_simplified.txt'
if not keyword_set(massive     )then massive     ='limongichieffi_z0_vel0_simplified.txt'

readcol,intermediate,mzas_i,zzas_i,mfin_i,el_i,y_i,mlos_i,mzer_i,mla_i,format='(f,f,f,a,f,f,f,f)',/silent
readcol,massive     ,mzas_m,zzas_m,vel_m,mfin_m,el_m,mlos_m,mla_m,format='(f,f,f,f,a,f,f)',/silent

nintermediate=n_elements(mzas_i)
nmassive     =n_elements(mzas_m)
ntotal=nintermediate+nmassive

list_elements=['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
nel2follow=n_elements(list_elements)

mass_trans=8.0
epsilon=0.01

if keyword_set(printps)then begin
   set_plot,'ps'
   if not keyword_set(fileeps)then fileeps='toto.eps'
   device,filename=fileeps,/encaps,xs=30,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
endif
ind_i=where(el_i eq list_elements(0),n_i)
ind_m=where(el_m eq list_elements(0),n_m)
plot_oi,mzas_i(ind_i),alog10(mla_i(ind_i)),psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAMS!n (M!dsun!n)',ytitle='log M!dlost!n/M!dsun!n',xr=[0.1,300.0],/xs,yr=[-5.,2.3],/ys
oplot,mzas_m(ind_m),alog10(mla_m(ind_m)),psym=-4,th=th
; Extrapolate missing data
mzas_ex=fltarr(2)
mlos_ex=fltarr(2)
ii=ind_i(0L)
mzas_ex(0)=mzas_i(ii)
mzas_ex(1)=0.1
mlos_ex(0)=mla_i(ii)
mlos_ex(1)=mla_i(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(n_i-1L)
mzas_ex(0)=mzas_i(ii)
mzas_ex(1)=mass_trans-epsilon
mlos_ex(0)=mla_i(ii)
mlos_ex(1)=mla_i(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(0L)
mzas_ex(0)=mzas_m(ii)
mzas_ex(1)=mass_trans+epsilon
mlos_ex(0)=mla_m(ii)
mlos_ex(1)=mla_m(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(n_m-1L)
mzas_ex(0)=mzas_m(ii)
mzas_ex(1)=1000.0
mlos_ex(0)=mla_m(ii)
mlos_ex(1)=mla_m(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor

xpos=max(mzas_m)+20.
xyouts,xpos,alog10(mla_m(ind_m(n_m-1L))),'all'
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   ind_i=where(el_i eq list_elements(j),n_i)
   last_mass(j)=mlos_i(ind_i(n_i-1L))
endfor
indsort=reverse(sort(last_mass))
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   mycolor=j+2
   ind_i=where(el_i eq list_elements(j),n_i)
   ind_m=where(el_m eq list_elements(j),n_m)
   oplot,mzas_i(ind_i),alog10(mlos_i(ind_i)),psym=-4,th=th,color=mycolor
   oplot,mzas_m(ind_m),alog10(mlos_m(ind_m)),psym=-4,th=th,color=mycolor
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas_m(ind_m))+5.
   endif else begin
      xpos=max(mzas_m(ind_m))+35.
   endelse
   xyouts,xpos,alog10(mlos_m(ind_m(n_m-1L))),list_elements(j),color=mycolor

   if(list_elements(j) eq 'Mg' and keyword_set(kobayashi))then begin
      oplot,mkobayashi,alog10(mgkobayashi),psym=-4,th=th,color=mycolor,lines=3
   endif      

   ii=ind_i(0L)
   mzas_ex(0)=mzas_i(ii)
   mzas_ex(1)=0.1
   mlos_ex(0)=mlos_i(ii)
   mlos_ex(1)=mlos_i(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(n_i-1L)
   mzas_ex(0)=mzas_i(ii)
   mzas_ex(1)=mass_trans-epsilon
   mlos_ex(0)=mlos_i(ii)
   mlos_ex(1)=mlos_i(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(0L)
   mzas_ex(0)=mzas_m(ii)
   mzas_ex(1)=mass_trans+epsilon
   mlos_ex(0)=mlos_m(ii)
   mlos_ex(1)=mlos_m(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(n_m-1L)
   mzas_ex(0)=mzas_m(ii)
   mzas_ex(1)=1000.0
   mlos_ex(0)=mlos_m(ii)
   mlos_ex(1)=mlos_m(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex),lines=1,sym=-4,th=th,color=mycolor

endfor

if keyword_set(ilabel)then begin
   xyouts,0.2,1.5,ilabel
endif
if keyword_set(mlabel)then begin
   xyouts,30.0,-4.0,mlabel
endif

plots,[mass_trans,mass_trans],[-5.0,2.3]

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif

if keyword_set(stop)then stop

end
