pro plot_wiersma,vel=vel,printps=printps,fileeps=fileeps,files=files,zmet=zmet,fileout=fileout

dir='/home/dubois/StellarYields/ResultingYields/'
if not keyword_set(files)then begin
print,'Using vel:',vel
if(vel eq 0.)then begin
   files=[ 'yields_evol_z-3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-2_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt']
endif
if(vel eq 150.)then begin
   files=[ 'yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-2_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-1_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z0_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt']
endif
if(vel eq 300.)then begin
   files=[ 'yields_evol_z-3_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-2_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z-1_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt' $
           ,'yields_evol_z0_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-03_dustcpopping17.txt']
endif
endif
nfiles=n_elements(files)
;; Sanity check: verify that all files have the same number of time steps
for i=0L,nfiles-1L do begin
   filename=dir+files(i)
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
   nsteps=n_elements(time)
   if(i gt 0L)then begin
      if(nsteps ne nsteps0)then begin
         print,'fail in # of time steps ', filename,nsteps,nsteps0,i
         stop
      endif else begin
         nsteps0=nsteps
      endelse
   endif else begin
      nsteps0=nsteps
   endelse
endfor

zsun=0.01345
if not keyword_set(zmet)then begin
   zmet=[0.0001345,0.0001345,0.001345,0.01345]
endif

mmet =dblarr(nfiles,nsteps,12L)
mdust=dblarr(nfiles,nsteps,12L)
for j=0L,nfiles-1L do begin
   filename=dir+files(j)
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,hd,hed,cd,nd,od,fluord,neond,mgd,sid,sd,fed,/silent
   ;; Compute the integrated mass return from the sum of all elements
   mmet(j,0L:nsteps-1L,2)=h (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,3)=he(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,4)=c (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,5)=n (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,6)=o (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,7)=mg(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,8)=si(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,9)=s (0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,10)=fe(0L:nsteps-1L)
   mmet(j,0L:nsteps-1L,11)=neon(0L:nsteps-1L)

   mdust(j,0L:nsteps-1L,2)=hd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,3)=hed(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,4)=cd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,5)=nd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,6)=od (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,7)=mgd(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,8)=sid(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,9)=sd (0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,10)=fed(0L:nsteps-1L)
   mdust(j,0L:nsteps-1L,11)=neond(0L:nsteps-1L)

   for i=0L,nsteps-1L do begin
      mmet(j,i,1)=total(mmet(j,i,4L:11L))   ; Total metal mass
      mmet(j,i,0)=total(mmet(j,i,2L:11L))   ; Total gas mass (metal+H+He)
      mdust(j,i,1)=total(mdust(j,i,2L:10L)) ; Total dust mass
   endfor
endfor

magb =dblarr(nfiles,nsteps,12L)
for j=0L,nfiles-1L do begin
   filename=dir+files(j)+'.agb'
   readcol,filename,time,all,h,he,c,n,o,fluor,neon,mg,si,s,fe,/silent
   ;; Compute the integrated mass return from the sum of all elements
   magb(j,0L:nsteps-1L,2)=h (0L:nsteps-1L)
   magb(j,0L:nsteps-1L,3)=he(0L:nsteps-1L)
   magb(j,0L:nsteps-1L,4)=c (0L:nsteps-1L)
   magb(j,0L:nsteps-1L,5)=n (0L:nsteps-1L)
   magb(j,0L:nsteps-1L,6)=o (0L:nsteps-1L)
   magb(j,0L:nsteps-1L,7)=mg(0L:nsteps-1L)
   magb(j,0L:nsteps-1L,8)=si(0L:nsteps-1L)
   magb(j,0L:nsteps-1L,9)=s (0L:nsteps-1L)
   magb(j,0L:nsteps-1L,10)=fe(0L:nsteps-1L)
   magb(j,0L:nsteps-1L,11)=neon(0L:nsteps-1L)

   for i=0L,nsteps-1L do begin
      magb(j,i,1)=total(magb(j,i,4L:11L))
      magb(j,i,0)=total(magb(j,i,2L:11L))
   endfor
endfor

;; Asplun et al 2009
XHAsplund=0.7381d0
;; XHAsplund=0.7065d0

lAO =8.69d0
lAFe=7.50d0
lAC =8.43d0
lAN =7.83d0
lAMg=7.60d0
lASi=7.51d0
lAS =7.12d0
lANe=7.93d0
lAHe=10.925227d0 ; This is the value necessary to get the H, He, Z mass fraction of Asplund+09
lAH =12.0d0
logH =lAH                  +alog10(XHAsplund)
logHe=lAHe+alog10(3.9993d0)+alog10(XHAsplund)
logO =lAO +alog10(15.999d0)+alog10(XHAsplund)
logFe=lAFe+alog10(55.845d0)+alog10(XHAsplund)
logC =lAC +alog10(12.011d0)+alog10(XHAsplund)
logN =lAN +alog10(14.007d0)+alog10(XHAsplund)
logMg=lAMg+alog10(24.305d0)+alog10(XHAsplund)
logSi=lASi+alog10(28.086d0)+alog10(XHAsplund)
logS =lAS +alog10(32.065d0)+alog10(XHAsplund)
logNe=lANe+alog10(20.1797d0)+alog10(XHAsplund)
;; logHe=lAHe+alog10(4d0)+alog10(XHAsplund)
;; logO =lAO +alog10(16d0)+alog10(XHAsplund)
;; logFe=lAFe+alog10(56d0)+alog10(XHAsplund)
;; logC =lAC +alog10(12d0)+alog10(XHAsplund)
;; logN =lAN +alog10(14d0)+alog10(XHAsplund)
;; logMg=lAMg+alog10(24d0)+alog10(XHAsplund)
;; logSi=lASi+alog10(28d0)+alog10(XHAsplund)
;; logS =lAS +alog10(32d0)+alog10(XHAsplund)
;; logNe=lANe+alog10(20d0)+alog10(XHAsplund)

lAlist=[lAH,lAHe,lAO,lAC,lAFe,lAN,lAMg,lASi,lAS,laNe]
llist=[logH,logHe,logO,logC,logFe,logN,logMg,logSi,logS,logNe]
nllist=n_elements(llist)
nfrac=dblarr(nllist)
mfrac=dblarr(nllist)
for i=0L,nllist-1L do begin
   nfrac(i)=10d0^(lAlist(i)-12.0)
   mfrac(i)=10d0^(llist(i)-lAH)
endfor

yhe=alog10(mmet(*,nsteps-1L,3)/mmet(*,nsteps-1L,2))-(logHe-logH)
yc =alog10(mmet(*,nsteps-1L,4)/mmet(*,nsteps-1L,2))-(logC -logH)
yn =alog10(mmet(*,nsteps-1L,5)/mmet(*,nsteps-1L,2))-(logN -logH)
yo =alog10(mmet(*,nsteps-1L,6)/mmet(*,nsteps-1L,2))-(logO -logH)
ymg=alog10(mmet(*,nsteps-1L,7)/mmet(*,nsteps-1L,2))-(logMg-logH)
ysi=alog10(mmet(*,nsteps-1L,8)/mmet(*,nsteps-1L,2))-(logSi-logH)
ys =alog10(mmet(*,nsteps-1L,9)/mmet(*,nsteps-1L,2))-(logS -logH)
yfe=alog10(mmet(*,nsteps-1L,10)/mmet(*,nsteps-1L,2))-(logFe-logH)
yne=alog10(mmet(*,nsteps-1L,11)/mmet(*,nsteps-1L,2))-(logNe-logH)

yheagb=alog10(magb(*,nsteps-1L,3)/magb(*,nsteps-1L,2))-(logHe-logH)
ycagb =alog10(magb(*,nsteps-1L,4)/magb(*,nsteps-1L,2))-(logC -logH)
ynagb =alog10(magb(*,nsteps-1L,5)/magb(*,nsteps-1L,2))-(logN -logH)
yoagb =alog10(magb(*,nsteps-1L,6)/magb(*,nsteps-1L,2))-(logO -logH)
ymgagb=alog10(magb(*,nsteps-1L,7)/magb(*,nsteps-1L,2))-(logMg-logH)
ysiagb=alog10(magb(*,nsteps-1L,8)/magb(*,nsteps-1L,2))-(logSi-logH)
ysagb =alog10(magb(*,nsteps-1L,9)/magb(*,nsteps-1L,2))-(logS -logH)
yfeagb=alog10(magb(*,nsteps-1L,10)/magb(*,nsteps-1L,2))-(logFe-logH)
yneagb=alog10(magb(*,nsteps-1L,11)/magb(*,nsteps-1L,2))-(logNe-logH)

yheII=alog10( (mmet(*,nsteps-1L,3)-magb(*,nsteps-1L,3))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logHe-logH)
ycII =alog10( (mmet(*,nsteps-1L,4)-magb(*,nsteps-1L,4))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logC-logH)
ynII =alog10( (mmet(*,nsteps-1L,5)-magb(*,nsteps-1L,5))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logN-logH)
yoII =alog10( (mmet(*,nsteps-1L,6)-magb(*,nsteps-1L,6))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logO-logH)
ymgII=alog10( (mmet(*,nsteps-1L,7)-magb(*,nsteps-1L,7))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logMg-logH)
ysiII=alog10( (mmet(*,nsteps-1L,8)-magb(*,nsteps-1L,8))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logSi-logH)
ysII =alog10( (mmet(*,nsteps-1L,9)-magb(*,nsteps-1L,9))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logS-logH)
yfeII=alog10( (mmet(*,nsteps-1L,10)-magb(*,nsteps-1L,10))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logFe-logH)
yneII=alog10( (mmet(*,nsteps-1L,11)-magb(*,nsteps-1L,11))/(mmet(*,nsteps-1L,2)-magb(*,nsteps-1L,2)) )-(logNe-logH)


if keyword_set(printps)then begin
   set_plot,'ps'
   device,filename=fileeps,/encaps,xs=30,ys=20
   th=5
   !p.font=0
   !p.charsize=2
   !p.charthick=th
endif else begin
   window,0,xs=1200,ys=800
endelse
!p.multi=[0,3,2]
plot_oi,zmet,yheAGB,xtitle='Z!dZAMS!n',ytitle='[X/H]!dAGB!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,ycAGB,th=th,lines=2
oplot,zmet,ynAGB,th=th,lines=4
x0=1.5d-5
x1=8.0d-5
x2=1e-4
dy=0.35
y0=1.65
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'He'
xyouts,x2,y0-1.0d0*dy,'C'
xyouts,x2,y0-2.0d0*dy,'N'

plot_oi,zmet,yoAGB,xtitle='Z!dZAMS!n',ytitle='[X/H]!dAGB!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,yneAGB,th=th,lines=2
oplot,zmet,ymgAGB,th=th,lines=4
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'O'
xyouts,x2,y0-1.0d0*dy,'Ne'
xyouts,x2,y0-2.0d0*dy,'Mg'

plot_oi,zmet,ysiAGB,xtitle='Z!dZAMS!n',ytitle='[X/H]!dAGB!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,ysAGB,th=th,lines=2
oplot,zmet,yfeAGB,th=th,lines=4
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'Si'
xyouts,x2,y0-1.0d0*dy,'S'
xyouts,x2,y0-2.0d0*dy,'Fe'

plot_oi,zmet,yheII,xtitle='Z!dZAMS!n',ytitle='[X/H]!dII!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,ycII,th=th,lines=2
oplot,zmet,ynII,th=th,lines=4
y0=-1.1
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'He'
xyouts,x2,y0-1.0d0*dy,'C'
xyouts,x2,y0-2.0d0*dy,'N'

plot_oi,zmet,yoII,xtitle='Z!dZAMS!n',ytitle='[X/H]!dII!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,yneII,th=th,lines=2
oplot,zmet,ymgII,th=th,lines=4
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'O'
xyouts,x2,y0-1.0d0*dy,'Ne'
xyouts,x2,y0-2.0d0*dy,'Mg'

plot_oi,zmet,ysiII,xtitle='Z!dZAMS!n',ytitle='[X/H]!dII!n',xth=th,yth=th,th=th,yr=[-2.,2.]
oplot,zmet,ysII,th=th,lines=2
oplot,zmet,yfeII,th=th,lines=4
plots,[x0,x1],[y0-0.0d0*dy,y0-0.0d0*dy],th=th
plots,[x0,x1],[y0-1.0d0*dy,y0-1.0d0*dy],th=th,lines=2
plots,[x0,x1],[y0-2.0d0*dy,y0-2.0d0*dy],th=th,lines=4
xyouts,x2,y0-0.0d0*dy,'Si'
xyouts,x2,y0-1.0d0*dy,'S'
xyouts,x2,y0-2.0d0*dy,'Fe'

!p.multi=0
if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
endif

end
