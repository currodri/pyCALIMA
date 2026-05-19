pro crunch_lc18esn,zmet=zmet,vel=vel,fileout=fileout

if not keyword_set(zmet)then zmet=0.0
if not keyword_set(vel )then vel=0.0

readcol,'limongichieffi18_esn.txt',velzas,zzas,mzas,esn,format='(f,f,f,f)',skip=43

ind   =where(velzas eq vel and zzas eq zmet,nmass)
esn_array=esn(ind)
mass=mzas(ind)

if keyword_set(fileout)then begin
   openw,1,fileout
   printf,1,'#---Details of Columns:'
   printf,1,'#    Vel (km/s)     (F7.2)    [0/300] Velocity [ucd=phys.veloc;meta.modelled]'
   printf,1,'#    [Fe/H] ([Sun]) (F7.2)    [-3/0] Metallicity [ucd=phys.abund.Z]'
   printf,1,'#    Mass (Msun)    (F7.2)    [13/120] Initial mass [ucd=phys.mass]'
   printf,1,'#    Ebind (10+44J) (E10.2)   [0.6/47.6] Binding energy; 1e+51erg (2) [ucd=phys.energy]'
   printf,1,'#------ ------ ------ ---------'
   printf,1,'#Mass   [Fe/H] Vel    Ebind    '
   printf,1,'#(Msun) (Sun)  (km/s) (1e51erg)'
   printf,1,'#------ ------ ------ ---------'
   for i=0L,nmass-1L do begin
      printf,1,mass(i),zmet,vel,esn_array(i),format='(3f7.2,e10.2)'
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
plot_io,mass,esn_array,psym=-4,xth=th,yth=th,th=th,xtitle='M!dZAS!n (M!dsun!n)',ytitle='E!dSN!n (10!u51!nerg)',xr=[10.0,150.0],/xs,yr=[0.1,50.0],/ys

if keyword_set(printps)then begin
   device,/close
   set_plot,'x'
   !p.charsize=1.5
   !p.charthick=1
endif

end
