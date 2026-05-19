pro plot_yields_ratio,intermediate=intermediate,massive=massive,printps=printps,fileeps=fileeps,ilabel=ilabel,mlabel=mlabel,noxtick=noxtick,ymin=ymin,ymax=ymax

if not keyword_set(intermediate)then intermediate='karakas_z0.02_simplified.txt'
if not keyword_set(massive     )then massive     ='limongichieffi_z0_vel0_simplified.txt'

readcol,intermediate,mzas_i,zzas_i,mfin_i,el_i,y_i,mlos_i,mzer_i,mla_i,format='(f,f,f,a,f,f,f,f)',/silent
readcol,massive     ,mzas_m,zzas_m,vel_m,mfin_m,el_m,mlos_m,mla_m,format='(f,f,f,f,a,f,f)',/silent

nintermediate=n_elements(mzas_i)
nmassive     =n_elements(mzas_m)
ntotal=nintermediate+nmassive

;; list_elements =['p','He','C','N','O','Ne','Mg','Si','S','Fe']
;; list_elements2=['H','He','C','N','O','Ne','Mg','Si','S','Fe']
list_elements =['p','He','C','N','O','F','Ne','Mg','Si','S','Fe']
list_elements2=['H','He','C','N','O','F','Ne','Mg','Si','S','Fe']
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
   !p.font=0
endif
ind_i=where(el_i eq list_elements(0),n_i)
ind_m=where(el_m eq list_elements(0),n_m)
xmin=0.5
if not keyword_set(ymin)then ymin=-8.
if not keyword_set(ymax)then ymax=0.1
if keyword_set(noxtick)then begin
   plot_oi,mzas_i(ind_i),alog10(mla_i(ind_i)/mzas_i(ind_i)),psym=-4,xth=th,yth=th,th=th,ytitle='log M!dlost!n/M!dZAMS!n',xr=[xmin,300.0],/xs,yr=[ymin,ymax],/ys,xtickformat='(A1)'
endif else begin
   plot_oi,mzas_i(ind_i),alog10(mla_i(ind_i)/mzas_i(ind_i)),psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAMS!n (M!dsun!n)',ytitle='log M!dlost!n/M!dZAMS!n',xr=[xmin,300.0],/xs,yr=[ymin,ymax],/ys
endelse
oplot,mzas_m(ind_m),alog10(mla_m(ind_m)/mzas_m(ind_m)),psym=-4,th=th
; Extrapolate missing data
mzas_ex=fltarr(2)
mlos_ex=fltarr(2)
ii=ind_i(0L)
mzas_ex(0)=mzas_i(ii)
mzas_ex(1)=xmin
mlos_ex(0)=mla_i(ii)
mlos_ex(1)=mla_i(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(n_i-1L)
mzas_ex(0)=mzas_i(ii)
mzas_ex(1)=mass_trans-epsilon
mlos_ex(0)=mla_i(ii)
mlos_ex(1)=mla_i(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(0L)
mzas_ex(0)=mzas_m(ii)
mzas_ex(1)=mass_trans+epsilon
mlos_ex(0)=mla_m(ii)
mlos_ex(1)=mla_m(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor
ii=ind_i(n_m-1L)
mzas_ex(0)=mzas_m(ii)
mzas_ex(1)=1000.0
mlos_ex(0)=mla_m(ii)
mlos_ex(1)=mla_m(ii)*mzas_ex(1)/mzas_ex(0)
oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor

xpos=max(mzas_m)+20.
xyouts,xpos,alog10(mla_m(ind_m(n_m-1L))/mzas_m(ind_m(n_m-1L))),'all'
tek_color
last_mass=fltarr(nel2follow)
for j=0L,nel2follow-1L do begin
   ind_i=where(el_i eq list_elements(j),n_i)
   last_mass(j)=mlos_i(ind_i(n_i-1L))/mzas_i(ind_i(n_i-1L))
endfor
indsort=reverse(sort(last_mass))
loadct,33,/silent
for jj=0L,nel2follow-1L do begin
   j=indsort(jj)
   mycolor=10+(j-1L)*250./double(nel2follow)
   ;; mycolor=j+2
   ;; if(mycolor eq 7)then mycolor=12 ;; do not use yellow!
   ind_i=where(el_i eq list_elements(j),n_i)
   ind_m=where(el_m eq list_elements(j),n_m)
   oplot,mzas_i(ind_i),alog10(mlos_i(ind_i)/mzas_i(ind_i)),psym=-4,th=th,color=mycolor
   oplot,mzas_m(ind_m),alog10(mlos_m(ind_m)/mzas_m(ind_m)),psym=-4,th=th,color=mycolor
   if((jj+2) mod 2 eq 0)then begin
      xpos=max(mzas_m(ind_m))+5.
   endif else begin
      xpos=max(mzas_m(ind_m))+35.
   endelse
   xyouts,xpos,alog10(mlos_m(ind_m(n_m-1L))/mzas_m(ind_m(n_m-1L))),list_elements2(j),color=mycolor

   ii=ind_i(0L)
   mzas_ex(0)=mzas_i(ii)
   mzas_ex(1)=xmin
   mlos_ex(0)=mlos_i(ii)
   mlos_ex(1)=mlos_i(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(n_i-1L)
   mzas_ex(0)=mzas_i(ii)
   mzas_ex(1)=mass_trans-epsilon
   mlos_ex(0)=mlos_i(ii)
   mlos_ex(1)=mlos_i(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(0L)
   mzas_ex(0)=mzas_m(ii)
   mzas_ex(1)=mass_trans+epsilon
   mlos_ex(0)=mlos_m(ii)
   mlos_ex(1)=mlos_m(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor

   ii=ind_i(n_m-1L)
   mzas_ex(0)=mzas_m(ii)
   mzas_ex(1)=1000.0
   mlos_ex(0)=mlos_m(ii)
   mlos_ex(1)=mlos_m(ii)*mzas_ex(1)/mzas_ex(0)
   oplot,mzas_ex,alog10(mlos_ex/mzas_ex),lines=1,sym=-4,th=th,color=mycolor

endfor

if keyword_set(ilabel)then begin
   xyouts,0.2,1.5,ilabel
endif
if keyword_set(mlabel)then begin
   xyouts,30.0,-4.0,mlabel
endif

loadct,0,/silent
plots,[mass_trans,mass_trans],[ymin,ymax]

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif







end
