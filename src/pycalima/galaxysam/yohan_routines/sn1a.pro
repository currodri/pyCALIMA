function myfunc,X,P
  return, P[0]+P[1]*sqrt(P[2]+P[3]*X)
end

function tau_m_pado, m
  res = fltarr(n_elements(m))
  
  ;; Padovani & Matteucci (1993)
  ;; mcrit=6.6
  
  mcrit = 10.^(7.764-1.79/0.2232)*1.1
  fd = where(m ge mcrit, nfd)
  if nfd gt 0 then begin
     a0 = (1.338d0 - sqrt(1.79d0-0.2232d0*(7.764d0-alog10(m(fd)))))/0.1116d0
     res(fd) = 10d0^a0 
  endif
  
  fd = where(m lt mcrit, nfd)
  if nfd gt 0 then begin
     res(fd) = 10.^12
  endif

  fd = where(m ge 6.6, nfd)
  if nfd gt 0 then begin
     res(fd) = (1.2*(m(fd))^(-1.85d0) + 0.003)*1e9
  endif
  res=res*1.3d0 ;; rescaling for a better fit to recent Padova stellar tracks with half (Asplund) solar metallicity

  return, res
end
;#####################################################################
;#####################################################################
;#####################################################################
function tau_m, m
  res = fltarr(n_elements(m))
  fd  = where(m gt 8, nfd)
  if nfd gt 0 then begin
     res(fd) = 1.2*m(fd)^(-1.85) + 0.003
  endif
  
  fd = where( m le 80, nfd)
  if nfd gt 0 then begin
     res(fd) = 5*m(fd)^(-2.7) + 0.012
  endif
  
  return, res*1e9
end
;#####################################################################
;#####################################################################
;#####################################################################
function tau_m_rood,m
;used in Greggio & Renzini 1983
;derived from Rood (1972) and Becker (1979)
	
  res = fltarr(n_elements(m))
  fd  = where(m le 8, nfd)
  res[fd]=10-4.319*alog10(m[fd])+1.543*alog10(m[fd])^2d0
  
  fd  = where(m gt 8, nfd)
  if nfd gt 0 then begin
     res[fd]=10-4.319*alog10(8)+1.543*alog10(8)^2d0
  endif
  return, 10d0^res
end
;#####################################################################
;#####################################################################
;#####################################################################
function inverse, s, rmu, rml, trynum, iseed=iseed
; f=x^s
  if (rmu lt rml) then begin
     tmp=rmu
     rmu=rml
     rml=tmp
  endif
  ;; inverse method
  b = rmu^(1d0+s)/abs(1d0+s)
  c = rml^(1d0+s)/abs(1d0+s)
  a = abs(c-b)
  
  x = a*randomu(iseed,trynum) + min([b,c])
  randist = (x*abs(1d0+s))^(1d0/(1d0+s))

  return, randist
end
;#####################################################################
;#####################################################################
;#####################################################################
pro sn1a,chabrier=chabrier,revisedchabrier=revisedchabrier,bound_slope=bound_slope,alpha_slope=alpha_slope,stop=stop,nbin=nbin,ASNIa=ASNIa,M_Bmi=M_Bmi,M_BMa=M_BMa,gamma=gamma

  if not keyword_set(bound_slope)then begin
     bound_slope=[0.1,100.0]
     print,'Slope boundaries not specified'
     print,'-> Use: [',bound_slope(0),',',bound_slope(1),'] Msun by default',format='(A,f8.2,A,f8.2,A)'
  endif
  if not keyword_set(alpha_slope)then begin
     alpha_slope=[-2.3d0]
     print,'Slope not specified'
     print,'-> Use: ',alpha_slope(0),' by default',format='(A,f8.2,A)'
  endif
  if n_elements(bound_slope) ne n_elements(alpha_slope)+1L then begin
     print,'bound_slope must have as many elements as alpha_slope +1'
     stop
  endif
  nbslope=n_elements(bound_slope)
  mu   = bound_slope(nbslope-1L)
  ml   = bound_slope(0L)
  if not keyword_set(nbin)then nbin = 1000L
  
  ;; #####################
  if not keyword_set(M_BMa)then M_BMa  = 16
  if not keyword_set(M_Bmi)then M_Bmi  = 3
  A_Ia   = 1
  ;; #####################

  mb   = findgen(nbin+1)*(alog10(M_BMa)-alog10(M_Bmi))/float(nbin) + alog10(M_Bmi)
  mb   = 10d0^mb
  
  if keyword_set(chabrier)then begin
     ;; Chabrier 2003
     msampling=ml+findgen(nbin)/double(nbin)*(mu-ml)
     xi_chabrier=dblarr(nbin)
     ind=where(msampling lt 1.0d0,nind)
     lmm=alog10(msampling(ind))
     xi_chabrier(ind)=0.158d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0))
     mi=0.5d0*msampling(ind(nind-1L))
     ind=where(msampling ge 1.0d0)
     mi=mi+0.5d0*msampling(ind(0L))
     lmm=alog10(mi)
     xii=0.158d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0)) 
     xi_chabrier(ind)=msampling(ind)^(alpha_slope(0))*(xii/mi^(alpha_slope(0)))
     renorm=int_tabulated(msampling,msampling*xi_chabrier)
     phi=dblarr(nbin)
     ind=where(mb lt 1.0d0,nind)
     if(nind ge 1L)then begin
        lmm2=alog10(mb(ind))
        phi(ind)=0.158d0/(alog(10d0)*mb(ind))*exp(-(lmm2-alog10(0.079d0))^2d0/(2d0*0.69d0^2d0))
        mi=0.5d0*mb(ind(nind-1L))
     endif
     ind=where(mb ge 1.0d0)
     phi(ind)=mb(ind)^(alpha_slope(0))*(xii/mi^(alpha_slope(0)))
     phi=phi/renorm
  endif else if keyword_set(revisedchabrier)then begin
     ;; Chabrier 2005
     msampling=ml+findgen(nbin)/double(nbin)*(mu-ml)
     xi_chabrier=dblarr(nbin)
     ind=where(msampling lt 1.0d0,nind)
     lmm=alog10(msampling(ind))
     xi_chabrier(ind)=0.093d0/(alog(10d0)*msampling(ind))*exp(-(lmm-alog10(0.200d0))^2d0/(2d0*0.55d0^2d0))
     mi=0.5d0*msampling(ind(nind-1L))
     ind=where(msampling ge 1.0d0)
     mi=mi+0.5d0*msampling(ind(0L))
     lmm=alog10(mi)
     xii=0.093d0/(alog(10d0)*mi)*exp(-(lmm-alog10(0.200d0))^2d0/(2d0*0.55d0^2d0)) 
     xi_chabrier(ind)=msampling(ind)^(alpha_slope(0))*(xii/mi^(alpha_slope(0)))
     renorm=int_tabulated(msampling,msampling*xi_chabrier)
     phi=dblarr(nbin)
     ind=where(mb lt 1.0d0,nind)
     if(nind ge 1L)then begin
        lmm2=alog10(mb(ind))
        phi(ind)=0.093d0/(alog(10d0)*mb(ind))*exp(-(lmm2-alog10(0.200d0))^2d0/(2d0*0.55d0^2d0))
        mi=0.5d0*mb(ind(nind-1L))
     endif
     ind=where(mb ge 1.0d0)
     phi(ind)=mb(ind)^(alpha_slope(0))*(xii/mi^(alpha_slope(0)))
     phi=phi/renorm
  endif else if (nbslope gt 2L) then begin
     ;; Broken power law
     msampling=ml+findgen(nbin)/double(nbin)*(mu-ml)
     xi=dblarr(nbin)
     prefactor=dblarr(nbin)
     prefactor(0L:nbin-1L)=1.0d0
     for j=0L,nbslope-2L do begin
        ind=where(msampling ge bound_slope(j) and msampling lt bound_slope(j+1),nind)
        xi(ind)=msampling(ind)^alpha_slope(j)
        if(j gt 0L)then begin
           prefactor(j)=last_xi/mm^alpha_slope(j)
           xi(ind)=prefactor(j)*xi(ind)
        endif
        if(j lt nbslope-2L)then begin
           ii=ind(nind-1L)
           mm=0.5d0*(msampling(ii)+msampling(ii+1L))
           last_xi=prefactor(j)*mm^alpha_slope(j)
        endif
        if(j eq nbslope-2L)then begin
           xi(nbin-1L)=prefactor(j)*msampling(nbin-1L)^alpha_slope(j)
        endif
     endfor
     renorm=int_tabulated(msampling,msampling*xi)
     phi=dblarr(nbin)
     for j=0L,nbslope-2L do begin
        ind=where(mb ge bound_slope(j) and mb lt bound_slope(j+1),nind)
        if(nind ge 1)then begin
           phi(ind)=prefactor(j)*mb(ind)^alpha_slope(j)
        endif
     endfor
     phi=phi/renorm
  endif else begin
     ;; Salpeter
     print,'Salpeter'
     ssal = alpha_slope(0)
     Asal = (2+ssal)/(mu^(ssal+2)-ml^(ssal+2))
     phi  = Asal*mb^ssal
  endelse

  trynum = 10000L

  nbint = 100L
  mint  = 7d0
  maxt  = 10.6d0
  logt  = findgen(nbint+1)/nbint*(maxt-mint)+mint 
  tsize = (maxt-mint)/nbint
  RIa   = dblarr(nbint+1)

  mprev = M_Bmi
  if not keyword_set(gamma)then gamma=2.0d0 ;; gamma from Greggio & Renzini 83
  for i=0L,nbin-1L do begin
     m2    = inverse(gamma,0.,mb[i]/2d0,trynum)
     dm    = mb[i]-mprev
     logtb = alog10(tau_m_pado(m2))
     hi    = histogram(logtb,min=mint,max=maxt,binsize=tsize)
     RIa  += double(hi)*phi[i]/float(trynum)*dm
     mprev = mb[i]
  endfor
  
  
  t1 = 0d0
  for i=0L,nbint-1L do begin
     t2 = 10d0^logt[i]
     RIa[i]=RIa[i]/(t2-t1)
     t1 = t2
  endfor

  RIa = (RIa+1d-49)*1d9         ; per gigayear
	
;	window,xs=600,ys=600
;	plot, logt, alog10(RIa/max(RIa)),yr=[-4,0.2],/ys,xr=[mint,maxt],/xs
;	oplot, [10,10],[-4,1],linestyle=1	
;	oplot, [7,11],[-2,-2],linestyle=1	
		




;###### for the same time spacing with stellar wind
;	readcol, '~/Population/starburst99/swind_lookup_cmloss.dat', time, format='f'
  time =[10000.0,13183.0,17378.0,19953.0,22909.0,26303.0,30200.0,34674.0,39811.0,$
	       45709.0,52481.0,60257.0,69184.0,79434.0,91202.0,104710.,120230.,138040.,$
	       158490.,181970.,208930.,239890.,275430.,316240.,363090.,416880.,478640.,$
	       549560.,630980.,724460.,831790.,955020.,1.09650e+06,1.25900e+06,1.44550e+06,$
	       1.65960e+06,1.90550e+06,2.18780e+06,2.51200e+06,2.88410e+06,3.31140e+06,$
	       3.80210e+06,4.36530e+06,5.01210e+06,5.75470e+06,6.60720e+06,7.58610e+06,$
	       8.71000e+06,1.00000e+07,1.14820e+07,1.31830e+07,1.51360e+07,1.73790e+07,$
	       1.99540e+07,2.29100e+07,2.63040e+07,3.02010e+07,3.46760e+07,3.98130e+07,$
	       4.57120e+07,5.24840e+07,6.02600e+07,6.91870e+07,7.94380e+07,9.12070e+07,$
	       1.04720e+08,1.20230e+08,1.38040e+08,1.58490e+08,1.81970e+08,2.08930e+08,$
	       2.39880e+08,2.75420e+08,3.16220e+08,3.63070e+08,4.16850e+08,4.78610e+08,$
	       5.49510e+08,6.30910e+08,7.24380e+08,8.31690e+08,9.54900e+08,1.09640e+09,$
	       1.25880e+09,1.44530e+09,1.65940e+09,1.90520e+09,2.18740e+09,2.51150e+09,$
	       2.88350e+09,3.31070e+09,3.80110e+09,4.36420e+09,5.01080e+09,5.75310e+09,$
	       6.60530e+09,7.58390e+09,8.70740e+09,9.99730e+09,1.4e+10]
  logtref = alog10(time)

  RIa_new = interpol(alog10(RIa), logt, logtref)
  RIa_new = 10d0^RIa_new
  fd      = where(RIa_new lt 0, nfd)
  if nfd gt 0 then RIa_new(fd) = 0
  
  openw, 1, 'RIa_ssp.dat'
  nn    = n_elements(RIa_new)
  tprev = 0d0
  Rprev = 0d0
  Rcum  = 0d0
  cRIa_new = dblarr(nn)
  for i=0L, n_elements(RIa_new)-1L do begin
     if RIa_new[i] le 1d-30 then RIa_new[i]=0d0
     Rcum += (time(i)-tprev)/1d9*(RIa_new(i)+Rprev)/2d0
     if i ne 1 and i ne 3 then printf, 1, time(i), RIa_new(i), Rcum
     tprev = time(i)
     Rprev = RIa_new(i)
     cRIa_new(i) = Rcum
  endfor
  close,1
  
  window,0,xs=512,ys=512
  plot, time, cRIa_new,yr=[1d-5,1d-1],/ys,/xlog,/xs,/ylog,xr=[1d7,2d10],xtitle='time (yr)',ytitle='SNIa cumulative rate'
;	oplot, [10,10],[-4,1],linestyle=1	
;	oplot, [7,11],[-2,-2],linestyle=1	

  ok  = where(time ge 4d7)
  x   =  alog10(time[ok])
  y   =  alog10(cRIa_new[ok])
  ndeg =  3
  Afit = poly_fit(x,y,ndeg)
;
;	err = dblarr(n_elements(y))+1	
;	initial_guess = [-4.,1,1,1]
;	Afit = mpfitfun('myfunc',x,y,err,initial_guess,/quite)
  x   =  alog10(time)
  y = dblarr(n_elements(x))
  for ideg=0,ndeg do begin
     y[*] += Afit[ideg]*x^double(ideg)
  endfor
  oplot, 10d0^x,10d0^y,lines=2 ;;,color=getcolor('red',2)


  ;; Madau & Dickinson 2014
  nbinz=100L
  omega_m        = 0.3d0
  omega_l        = 0.7d0
  omega_k        = 0.0d0
  H0             = 70.0d0
  unsurH0        = 977.8/H0     ; in Gyr
  t_c            = dblarr(nbinz)
  t_c(0)         = -1d-10
  t_c(nbinz-1)   = -10.0        ; maximum (absolute value) of conformal star formation time of a star particle
  delta_tc       = (t_c(0)-t_c(nbinz-1)) / double(nbinz-1)
  for i=1,nbinz-2 do begin
     t_c(i) = t_c(0) - double(i)*delta_tc
  endfor
  print,'min & max conformal times: ',min(t_c),max(t_c)
;convert conformal time to real time with precision prec: 
  prec   = 1d-3
  tforz  = dblarr(nbinz)
  zarr   = dblarr(nbinz)
  friedman_2,omega_m,omega_l,omega_k,prec,t_c,tforz,zarr,nbinz
  tforz      = tforz*unsurH0    ; to get look back time in Gyr 
  tforz=max(tforz)-tforz
  redshift=dblarr(nbinz)
  time2   =dblarr(nbinz)
  for i=0L,nbinz-1L do begin
     redshift(i)=zarr (nbinz-1L-i) ;; from z>>1 to z=0
     time2   (i)=tforz(nbinz-1L-i)
  endfor
  sfr_fit_md14=0.015d0*(1d0+redshift)^2.7d0/(1d0+((1d0+redshift)/2.9d0)^5.6d0) ;; for a Salpeter IMF
  sfr_fit_md14=sfr_fit_md14*0.63                                       ;; convert to a Chabrier IMF
  window,1,xs=512,ys=512
  plot_io,redshift,sfr_fit_md14,th=th,xtitle='redshift',ytitle='SFR (Msun/yr/Mpc!u3!n)',yr=[1d-3,1d0],xr=[0.,4.0]
  oplot,redshift,sfr_fit_md14/0.63,th=th,lines=2

  sniarate=dblarr(nbinz)
  alt=alog10(time/1d9)
  for i=0L,nbinz-1L do begin
     for j=1L,i-1L do begin
        logdt=alog10(time2(i)-time2(j))
        if(logdt lt max(alt))then begin
           if(logdt gt min(alt))then begin
              cc=interpol(RIa_new,alt,logdt)
           endif else begin
              cc=0.0d0
           endelse
        endif else begin
           cc=RIa_new(nn-1L)
        endelse
        dm=sfr_fit_md14(j)*(time2(j)-time2(j-1L))*1d9
        sniarate(i)=sniarate(i)+dm*(cc/1d9)
     endfor
  endfor

  window,2,xs=512,ys=512
  if not keyword_set(ASNIa)then ASNIa=3.5d-2
  print,'ASNIa=',ASNIa,' used to plot the cosmic rate of SNIa (for print in files ASNIa=1)'
  ;; plot_io,alog10(1d0+redshift),sniarate/1d-4*ASNIa,yr=[1d-2,1d1],xtitle='log(1+z)',ytitle='SNIa rate (1e-4/yr/Mpc!u3!n)'
  plot_io,redshift,sniarate/1d-4*ASNIa,yr=[1d-2,1d1],xtitle='redshift',ytitle='SNIa rate (1e-4/yr/Mpc!u3!n)',xr=[0.,4.0]

  ;; Cappellaro+15
  xx=[0.05,0.25,0.45,0.65,0.84,1.16,1.64]
  yy=[0.25,0.29,0.44,0.58,0.64,0.87,0.63] ;; 1e-4/yr/Mpc^3
  er=[0.05,0.07,0.11,0.14,0.20,0.22,0.22]
  oplot,xx,yy,psym=4,th=th
  errplot,xx,yy-er,yy+er,th=th

  ;; Rodney+14
  xx =[0.255,0.75,1.25,1.75,2.25]
  yy =[0.36,0.51,0.64,0.72,0.49]
  er1m=[0.26,0.19,0.22,0.30,0.38]
  er2m=[0.35,0.19,0.23,0.28,0.24]
  er1p=[0.60,0.27,0.31,0.45,0.95]
  er2p=[0.12,0.23,0.34,0.50,0.45]
  erm=sqrt(er1m^2d0+er2m^2d0)
  erp=sqrt(er1p^2d0+er2p^2d0)
  oplot,xx,yy,psym=1,th=th
  errplot,xx,yy-erm,yy+erp,th=th

  ;; Dilday+10
  xx=[0.025+0.05,0.075+0.125,0.125+0.175,0.175+0.225,0.225+0.275,0.275+0.325]*0.5d0
  yy  =[2.78,2.59,3.07,3.48,3.65,4.34]/10.0d0
  er1m=[0.83,0.44,0.34,0.30,0.28,0.34]/10.0d0
  er2m=[0.00,0.01,0.05,0.07,0.12,0.16]/10.0d0
  er1p=[1.12,0.52,0.38,0.32,0.31,0.37]/10.0d0
  er2p=[0.15,0.18,0.35,0.82,1.82,3.96]/10.0d0
  erm=sqrt(er1m^2d0+er2m^2d0)  
  erp=sqrt(er1p^2d0+er2p^2d0)
  oplot,xx,yy,psym=2,th=th
  errplot,xx,yy-erm,yy+erp,th=th

  ;; Perret+12
  xx  =[0.16,0.26,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05]
  yy  =[0.14,0.28,0.36,0.36,0.48,0.48,0.58,0.57,0.77,0.74]
  er1m=[0.09,0.07,0.06,0.06,0.06,0.05,0.06,0.05,0.08,0.12]
  er2m=[0.12,0.07,0.06,0.05,0.05,0.06,0.07,0.07,0.12,0.13]
  er1p=[0.09,0.07,0.06,0.06,0.06,0.05,0.06,0.05,0.08,0.12]
  er2p=[0.06,0.06,0.05,0.04,0.04,0.04,0.05,0.06,0.10,0.10]
  erm=sqrt(er1m^2d0+er2m^2d0)  
  erp=sqrt(er1p^2d0+er2p^2d0)
  oplot,xx,yy,psym=5,th=th
  errplot,xx,yy-erm,yy+erp,th=th

  ;; Barbary+12
  xx  =[0.807,1.187,1.535]
  yy  =[1.18,1.33,0.77]
  er1m=[0.45,0.49,0.54]
  er2m=[0.28,0.26,0.77]
  er1p=[0.60,0.65,1.07]
  er2p=[0.44,0.69,0.44]
  erm=sqrt(er1m^2d0+er2m^2d0)  
  erp=sqrt(er1p^2d0+er2p^2d0)
  oplot,xx,yy,psym=6,th=th
  errplot,xx,yy-erm,yy+erp,th=th

  if keyword_set(stop)then stop

end
