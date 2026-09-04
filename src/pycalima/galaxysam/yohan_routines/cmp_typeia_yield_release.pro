pro cmp_typeia_yield_release,fileinput=fileinput,filetimemapping=filetimemapping,ASNIa=ASNIa,stop=stop

  if not keyword_set(ASNIa)then begin
     ASNIa=5d-2
     print,'Using ASNIa:',ASNIa,' by default'
  endif
  if not keyword_set(fileinput)then fileinput='RIa_ssp.dat'
  print,'Reading file:',fileinput,' for the normalized cumulative rate of SNIa'
  readcol,fileinput,t_wind,rate0,cr_Ia,/silent
  if not keyword_set(filetimemapping)then begin
     print,'I need an input file to map the time binning of the SNIa yield evolution onto the SNII+AGN yield evolution'
     filetimemapping='./ResultingYields/yields_evol_z0_v50_chabrier_imf0.1-100msun_asnia0.05_mfailed50msun_dustcpopping17.txt'
     print,'I will read filetimemapping=',filetimemapping
  endif

;;!  Nomoto et al. (1997,1984) W7 (carbon-deflagration model)
;;!            (better fit with Tycho observation)
;;!----------------------------------------------------------------------------
;;!       12C ,     13C ,   14N ,    15N ,    16O ,    17O ,    18O ,    19F ,
;;!       20Ne,     21Ne,   22Ne,    23Na,    24Mg,    25Mg,    26Mg,    27Al,
;;!       28Si,     29Si,   30Si,    31P ,    32S ,    33S ,    34S ,    36S ,
;;!       35Cl,     37Cl,   36Ar,    38Ar,    40Ar,    39K ,    41K ,    40Ca,
;;!       42Ca,     43Ca,   44Ca,    46Ca,    48Ca,    45Sc,    46Ti,    47Ti,
;;!       48Ti,     49Ti,   50Ti,    50V ,    51V ,    50Cr,    52Cr,    53Cr,
;;!       54Cr,     55Mn,   54Fe,    56Fe,    57Fe,    58Fe,    59Co,    58Ni,
;;!       60Ni,     61Ni,   62Ni,    64Ni,    63Cu,    65Cu,    64Zn,    66Zn,
;;!       67Zn,     68Zn                                                   
;;!----------------------------------------------------------------------------

    yield_snIa = [ $  ;; ! Msun per SN
    4.83E-02,1.40E-06,1.16E-06,1.32E-09,1.43E-01,3.54E-08,8.25E-10,5.67E-10,$
    2.02E-03,8.46E-06,2.49E-03,6.32E-05,8.50E-03,4.05E-05,3.18E-05,9.86E-04,$
    1.50E-01,8.61E-04,1.74E-03,4.18E-04,8.41E-02,4.50E-04,1.90E-03,3.15E-07,$
    1.34E-04,3.98E-05,1.49E-02,1.06E-03,1.26E-08,8.52E-05,7.44E-06,1.23E-02,$
    3.52E-05,1.03E-07,8.86E-06,1.99E-09,7.10E-12,2.47E-07,1.71E-05,6.04E-07,$
    2.03E-04,1.69E-05,1.26E-05,8.28E-09,5.15E-05,2.71E-04,5.15E-03,7.85E-04,$
    1.90E-04,8.23E-03,1.04E-01,6.13E-01,2.55E-02,9.63E-04,1.02E-03,1.28E-01,$
    1.05E-02,2.51E-04,2.66E-03,1.31E-06,1.79E-06,6.83E-07,1.22E-05,2.12E-05,$
    1.34E-08,1.02E-08]
   chem_list=['C','N','O','F','Ne','Mg','Si','S','Fe']
   nchem=n_elements(chem_list)
   mz_Ia=dblarr(nchem+1)
   mz_Ia(0)=total(yield_snIa)
   for i=0L,nchem-1L do begin
      if(chem_list(i) eq 'C' )then mz_Ia(i+1)=total(yield_snIa(0:1))
      if(chem_list(i) eq 'N' )then mz_Ia(i+1)=total(yield_snIa(2:3))
      if(chem_list(i) eq 'O' )then mz_Ia(i+1)=total(yield_snIa(4:6))
      if(chem_list(i) eq 'Ne')then mz_Ia(i+1)=total(yield_snIa(8:10))
      if(chem_list(i) eq 'Mg')then mz_Ia(i+1)=total(yield_snIa(12:14))
      if(chem_list(i) eq 'Si')then mz_Ia(i+1)=total(yield_snIa(16:18))
      if(chem_list(i) eq 'S' )then mz_Ia(i+1)=total(yield_snIa(20:23))
      if(chem_list(i) eq 'Fe')then mz_Ia(i+1)=total(yield_snIa(50:53))
      if(chem_list(i) eq 'F' )then mz_Ia(i+1)=yield_snIa(7)
   endfor
   ;; print,yield_snIa(50:53),total(yield_snIa(50:53))


   ;; print,chem_list,format='(8A10)'
   ;; print,mz_Ia(1:nchem),format='(8e10.3)'

   ntia=n_elements(t_wind)
   cMwind=dblarr(ntia,nchem)
   for ichem=0L,nchem-1L do begin
      cMwind(0L:ntia-1L,ichem)=ASNIa*cr_Ia(0L:ntia-1L)*mz_Ia(ichem+1)
   endfor
   ;; help,cMwind

       ;; cMwind_Z = dlog10(10d0**cMwind_Z + ASNIa*cR_Ia*mZ_Ia(1) ) ! per Msun
       ;; cMwind   = dlog10(10d0**cMwind   + ASNIa*cR_Ia*mZ_Ia(1) ) ! per Msun
       ;; cEwind   = dlog10(10d0**cEwind   + ASNIa*cR_Ia*1d51)
       ;; do ichem=1,nchem
       ;;    cMwind_chem(:,:,ichem)=dlog10( 10d0**cMwind_chem(:,:,ichem) &
       ;;                                 & +ASNIa*cR_Ia*mZ_Ia(ichem+1) ) ! per Msun
       ;; enddo


   readcol,filetimemapping,time2,all,h,he,c,n,o,neon,mg,si,s,fe,hd,hed,cd,nd,od,neond,mgd,sid,sd,fed,/silent
   
   ntime2=n_elements(time2)
   newcr_Ia=interpol(cr_Ia,t_wind,time2)
    
   file1='cr_Ia.txt'
   openw,1,file1
   print,'Writing the normalised cumulative rate of SNIa in with maped time steps in ',file1
   for i=0L,ntime2-1L do begin
      printf,1,time2(i),newcr_Ia(i)
   endfor
   close,/all
   
   file1='cr_Ia.ramses'
   openw,1,file1
   print,'Writing the normalised cumulative rate of SNIa in the Ramses HAGN-ready format in ',file1
   print,'(do not forget to have ASNIa<<1 in the Ramses namelist since this is the normalised value)'
   counter=0L
   printf,1,'cr_Ia = (/ &'
   for i=0L,ntime2-1L do begin
      counter=counter+1L
      if(i eq ntime2-1L)then begin
         printf,1,string(newcr_Ia(i),format='(e11.5)')+' /)',format='(A)'
         counter=0L
      endif
      if(counter eq 1L)then begin
         printf,1,' & '+string(newcr_Ia(i),format='(e11.5)')+',',format='(A,$)'
      endif
      if(counter gt 1L and counter le 4L)then begin
         printf,1,string(newcr_Ia(i),format='(e11.5)')+',',format='(A,$)'
      endif
      if(i ne ntime2-1L and counter eq 5L)then begin
         printf,1,string(newcr_Ia(i),format='(e11.5)')+', &',format='(A)'
         counter=0L
      endif
   endfor
   close,/all
   
   file1='cr_yields_Ia.txt'
   openw,1,file1
   print,'Writing the normalised yield evolution by SNIa as a function of time in ',file1
   mm=dblarr(nchem)
   myformat='('+strcompress(1+nchem,/remove_all)+'e10.3)'
   printf,1,'#--------- --------- --------- --------- --------- --------- --------- --------- --------- ---------'
   printf,1,'# time      C total   N total   O total  F  total  Ne total  Mg total  Si total   S total  Fe total'
   printf,1,'#--------- --------- --------- --------- --------- --------- --------- --------- --------- ---------'
   for i=0L,ntime2-1L do begin
      for ichem=0L,nchem-1L do begin
         xx=cMwind(0L:ntia-1L,ichem)/ASNIa
         mm(ichem)=interpol(xx,t_wind,time2(i))
      endfor
      printf,1,time2(i),mm(0L:nchem-1L),format=myformat
   endfor
   close,/all
   
   ASNIastring=string(ASNIa,format='(e8.2)')
   file1='cr_yields_Ia_ASNIa'+ASNIastring+'.txt'
   openw,1,file1
   print,'Writing the yield evolution by SNIa as a function of time in ',file1,' for ASNIa='+ASNIastring
   print,'(values from this file can be compared directly to the yield evolution produced by SNII+AGB from cmp_yield_release.pro)'
   mm=dblarr(nchem)
   myformat='('+strcompress(1+nchem,/remove_all)+'e10.3)'
   printf,1,'#--------- --------- --------- --------- --------- --------- --------- --------- --------- ---------'
   printf,1,'# time      C total   N total   O total  F  total  Ne total  Mg total  Si total   S total  Fe total'
   printf,1,'#--------- --------- --------- --------- --------- --------- --------- --------- --------- ---------'
   for i=0L,ntime2-1L do begin
      for ichem=0L,nchem-1L do begin
         xx=cMwind(0L:ntia-1L,ichem)
         mm(ichem)=interpol(xx,t_wind,time2(i))
      endfor
      printf,1,time2(i),mm(0L:nchem-1L),format=myformat
   endfor
   close,/all
   
   if keyword_set(stop)then stop

end
