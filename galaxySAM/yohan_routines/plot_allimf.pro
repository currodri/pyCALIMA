pro plot_allimf,printps=printps

mmin=0.1d0
mmax=100d0
lmmin=alog10(mmin)
lmmax=alog10(mmax)
nbin=uint(mmax/mmin)
;; msampling=10d0^(lmmin+findgen(nbin)/double(nbin)*(lmmax-lmmin))
msampling=mmin+findgen(nbin)/double(nbin)*(mmax-mmin)

renorm=msampling(nbin-1L)-msampling(0)

xi_chabrier=findgen(nbin)
prefactor=1.0d0
ind=where(msampling lt 1.0d0,nind)
lmm=alog10(msampling(ind))
xi_chabrier(ind)=0.158d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0))
mi=0.5d0*msampling(ind(nind-1L))
ind=where(msampling ge 1.0d0)
mi=mi+0.5d0*msampling(ind(0L))
lmm=alog10(mi)
xii=0.158d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0)) 
xi_chabrier(ind)=msampling(ind)^(-2.3d0)*(xii/mi^(-2.3d0))
xi_chabrier=xi_chabrier/int_tabulated(msampling,msampling*xi_chabrier)

xi_chabrierrev=findgen(nbin)
prefactor=1.0d0
ind=where(msampling lt 1.0d0,nind)
lmm=alog10(msampling(ind))
xi_chabrierrev(ind)=0.093d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.2d0))^2d0/(2d0*0.55d0^2d0))
mi=0.5d0*msampling(ind(nind-1L))
ind=where(msampling ge 1.0d0)
mi=mi+0.5d0*msampling(ind(0L))
lmm=alog10(mi)
xii=0.093d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.2d0))^2d0/(2d0*0.55d0^2d0)) 
xi_chabrierrev(ind)=msampling(ind)^(-2.3d0)*(xii/mi^(-2.3d0))
xi_chabrierrev=xi_chabrierrev/int_tabulated(msampling,msampling*xi_chabrierrev)

alpha_slope=[-2.3d0]
bound_slope=[mmin,mmax]
nbound_slope=n_elements(bound_slope)
xi_salpeter=findgen(nbin)
prefactor=1.0d0
for j=0L,nbound_slope-2L do begin
   ind=where(msampling ge bound_slope(j) and msampling lt bound_slope(j+1),nind)
   xi_salpeter(ind)=msampling(ind)^alpha_slope(j)
   if(j gt 0L)then begin
      prefactor=last_xi/mm^alpha_slope(j)
      xi_salpeter(ind)=prefactor*xi_salpeter(ind)
   endif
   if(j lt nbound_slope-2L)then begin
      ii=ind(nind-1L)
      mm=0.5d0*(msampling(ii)+msampling(ii+1L))
      last_xi=prefactor*mm^alpha_slope(j)
   endif
   if(j eq nbound_slope-2L)then begin
      xi_salpeter(nbin-1L)=prefactor*msampling(nbin-1L)^alpha_slope(j)
   endif
endfor
xi_salpeter=xi_salpeter/int_tabulated(msampling,msampling*xi_salpeter)

alpha_slope=[-1.3,-2.3d0]
bound_slope=[mmin,0.5d0,mmax]
nbound_slope=n_elements(bound_slope)
xi_kroupa=findgen(nbin)
prefactor=1.0d0
for j=0L,nbound_slope-2L do begin
   ind=where(msampling ge bound_slope(j) and msampling lt bound_slope(j+1),nind)
   xi_kroupa(ind)=msampling(ind)^alpha_slope(j)
   if(j gt 0L)then begin
      prefactor=last_xi/mm^alpha_slope(j)
      xi_kroupa(ind)=prefactor*xi_kroupa(ind)
   endif
   if(j lt nbound_slope-2L)then begin
      ii=ind(nind-1L)
      mm=0.5d0*(msampling(ii)+msampling(ii+1L))
      last_xi=prefactor*mm^alpha_slope(j)
   endif
   if(j eq nbound_slope-2L)then begin
      xi_kroupa(nbin-1L)=prefactor*msampling(nbin-1L)^alpha_slope(j)
   endif
endfor
xi_kroupa=xi_kroupa/int_tabulated(msampling,msampling*xi_kroupa)


zmet=[-0.6,-0.3,0.0,+0.3]
nmet=n_elements(zmet)
xi_varying=findgen(nmet,nbin)
aa0=1.0d0
for i=0L,nmet-1L do begin
   aa=2.73d0+3.1d0*zmet(i)
   if(aa lt aa0)then aa=aa0
   alpha_slope=[-aa0,-aa]
   print,zmet(i),aa
   bound_slope=[mmin,0.6d0,mmax]
   nbound_slope=n_elements(bound_slope)
   prefactor=1.0d0
   for j=0L,nbound_slope-2L do begin
      ind=where(msampling ge bound_slope(j) and msampling lt bound_slope(j+1),nind)
      xi_varying(i,ind)=msampling(ind)^alpha_slope(j)
      if(j gt 0L)then begin
         prefactor=last_xi/mm^alpha_slope(j)
         xi_varying(i,ind)=prefactor*xi_varying(i,ind)
      endif
      if(j lt nbound_slope-2L)then begin
         ii=ind(nind-1L)
         mm=0.5d0*(msampling(ii)+msampling(ii+1L))
         last_xi=prefactor*mm^alpha_slope(j)
      endif
      if(j eq nbound_slope-2L)then begin
         xi_varying(i,nbin-1L)=prefactor*msampling(nbin-1L)^alpha_slope(j)
      endif
   endfor
;; xi_varying(i,*)=xi_varying(i,*)*msampling(*)
   xx=xi_varying(i,*)
   xi_varying(i,*)=xi_varying(i,*)/int_tabulated(msampling,xx) ;/renorm
endfor

;; ==========================================================
;; Plot the IMF
;; ==========================================================
if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename='xiimf.eps',/encaps,xs=20,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
   !p.font=0
endif else begin
   iw=0
   window,iw,xs=700,ys=700
   th=1
endelse
lms=alog10(msampling)
plot,lms,alog10(xi_chabrier),/ys,xtitle='log M!dZAMS!n (M!dsun!n)',ytitle='log !9x!X',xth=th,yth=th,th=th,xr=[alog10(mmin),alog10(mmax)],yr=[-5.5,1.5]
oplot,lms,alog10(xi_chabrierrev),color=100
oplot,lms,alog10(xi_salpeter),lines=1,th=th
oplot,lms,alog10(xi_kroupa),lines=2,th=th
x0=1.4
x1=0.9
x2=1.3
dy=0.4
ymax=1.2
eps=0.05
xyouts,x0,ymax-1d0*dy,'Chabrier'
xyouts,x0,ymax-2d0*dy,'Salpeter'
xyouts,x0,ymax-3d0*dy,'Kroupa'
plots,[x1,x2],[ymax-1d0*dy+eps,ymax-1d0*dy+eps],th=th
plots,[x1,x2],[ymax-2d0*dy+eps,ymax-2d0*dy+eps],th=th,lines=1
plots,[x1,x2],[ymax-3d0*dy+eps,ymax-3d0*dy+eps],th=th,lines=2
dlct=250/nmet
print,'dlct=',dlct
mycolor=[20,50,200,250]
for i=0L,nmet-1L do begin   
   ctload,25,/br,/rev,/silent
   ;; mycolor=1+i*dlct
   oplot,lms,alog10(xi_varying(i,*)),color=mycolor(i),th=th
   plots,[x1,x2],[ymax-(4d0+i)*dy+eps,ymax-(4d0+i)*dy+eps],th=th,color=mycolor(i)
   loadct,0,/silent
   xyouts,x0,ymax-(4d0+i)*dy,'10!u'+string(zmet(i),format='(f4.1)')+'!n Z!dsun!n'
endfor
loadct,0,/silent

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charthick=1
endif

if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename='phiimf.eps',/encaps,xs=20,ys=15,/color
   th=5
   !p.charsize=1.5
   !p.charthick=5
   !p.font=0
endif else begin
   iw=iw+1
   window,iw,xs=700,ys=700
   th=1
endelse
lms=alog10(msampling)
plot,lms,alog10(xi_chabrier*msampling),/ys,xtitle='log M!dZAMS!n (M!dsun!n)',ytitle='log !9F!X',xth=th,yth=th,th=th,xr=[alog10(mmin),alog10(mmax)],yr=[-5.5,1.]
oplot,lms,alog10(xi_salpeter*msampling),lines=1,th=th
oplot,lms,alog10(xi_kroupa*msampling),lines=2,th=th
x0=-0.8
x1=-0.2
x2=0.2
dy=0.4
ymax=-2.0
eps=0.05
xyouts,x0,ymax-1d0*dy,'Chabrier'
xyouts,x0,ymax-2d0*dy,'Salpeter'
xyouts,x0,ymax-3d0*dy,'Kroupa'
plots,[x1,x2],[ymax-1d0*dy+eps,ymax-1d0*dy+eps],th=th
plots,[x1,x2],[ymax-2d0*dy+eps,ymax-2d0*dy+eps],th=th,lines=1
plots,[x1,x2],[ymax-3d0*dy+eps,ymax-3d0*dy+eps],th=th,lines=2
dlct=250/nmet
print,'dlct=',dlct
mycolor=[20,50,200,250]
for i=0L,nmet-1L do begin   
   ctload,25,/br,/rev,/silent
   ;; mycolor=1+i*dlct
   oplot,lms,alog10(xi_varying(i,*)*msampling),color=mycolor(i),th=th
   plots,[x1,x2],[ymax-(4d0+i)*dy+eps,ymax-(4d0+i)*dy+eps],th=th,color=mycolor(i)
   loadct,0,/silent
   xyouts,x0,ymax-(4d0+i)*dy,'10!u'+string(zmet(i),format='(f4.1)')+'!n Z!dsun!n'
endfor
loadct,0,/silent

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charthick=1
endif

stop
end
