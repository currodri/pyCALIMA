pro convert_typeia

   t_wind = [ $
      1.00000E+04,1.31830E+04,1.73780E+04,1.99530E+04,2.29090E+04, $
      2.63030E+04,3.02000E+04,3.46740E+04,3.98110E+04,4.57090E+04, $
      5.24810E+04,6.02570E+04,6.91840E+04,7.94340E+04,9.12020E+04, $
      1.04710E+05,1.20230E+05,1.38040E+05,1.58490E+05,1.81970E+05, $
      2.08930E+05,2.39890E+05,2.75430E+05,3.16240E+05,3.63090E+05, $
      4.16880E+05,4.78640E+05,5.49560E+05,6.30980E+05,7.24460E+05, $
      8.31790E+05,9.55020E+05,1.09650E+06,1.25900E+06,1.44550E+06, $
      1.65960E+06,1.90550E+06,2.18780E+06,2.51200E+06,2.88410E+06, $
      3.31140E+06,3.80210E+06,4.36530E+06,5.01210E+06,5.75470E+06, $
      6.60720E+06,7.58610E+06,8.71000E+06,1.00000E+07,1.14820E+07, $
      1.31830E+07,1.51360E+07,1.73790E+07,1.99540E+07,2.29100E+07, $
      2.63040E+07,3.02010E+07,3.46760E+07,3.98130E+07,4.57120E+07, $
      5.24840E+07,6.02600E+07,6.91870E+07,7.94380E+07,9.12070E+07, $
      1.04720E+08,1.20230E+08,1.38040E+08,1.58490E+08,1.81970E+08, $
      2.08930E+08,2.39880E+08,2.75420E+08,3.16220E+08,3.63070E+08, $
      4.16850E+08,4.78610E+08,5.49510E+08,6.30910E+08,7.24380E+08, $
      8.31690E+08,9.54900E+08,1.09640E+09,1.25880E+09,1.44530E+09, $
      1.65940E+09,1.90520E+09,2.18740E+09,2.51150E+09,2.88350E+09, $
      3.31070E+09,3.80110E+09,4.36420E+09,5.01080E+09,5.75310E+09, $
      6.60530E+09,7.58390E+09,8.70740E+09,9.99730E+09,1.40000E+10 ]

;;! cR_Ia: cumulative frequency of SN Ia per SSP with 1Msun (total M = 1.44 * cR_Ia)
   cR_Ia = [ $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, $
       2.32624E-18, 2.85011E-05, 1.13975E-04, 2.47986E-04, 4.25334E-04, $
       6.43002E-04, 8.98007E-04, 1.18817E-03, 1.51311E-03, 1.87290E-03, $
       2.26651E-03, 2.69440E-03, 3.15704E-03, 3.65339E-03, 4.18569E-03, $
       4.75529E-03, 5.36299E-03, 6.00842E-03, 6.69117E-03, 7.41420E-03, $
       8.17805E-03, 8.98529E-03, 9.83505E-03, 1.07275E-02, 1.16635E-02, $
       1.26436E-02, 1.36693E-02, 1.47387E-02, 1.58524E-02, 1.70087E-02, $
       1.81536E-02, 1.91933E-02, 2.00919E-02, 2.08679E-02, 2.15411E-02, $
       2.21275E-02, 2.26378E-02, 2.30830E-02, 2.34719E-02, 2.38126E-02, $
       2.41126E-02, 2.43766E-02, 2.46086E-02, 2.48133E-02, 2.52377E-02 ]

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
   chem_list=['C','N','O','Ne','Mg','Si','S','Fe']
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
   endfor

   print,chem_list,format='(8A10)'
   print,mz_Ia    ,format='(8e10.3)'

   if not keyword_set(A_snIa)then A_snIa=3.5d-2
   cMwind=A_snIa*cr_Ia*mz_Ia(0)
   help,cMwind

       ;; cMwind_Z = dlog10(10d0**cMwind_Z + A_snIa*cR_Ia*mZ_Ia(1) ) ! per Msun
       ;; cMwind   = dlog10(10d0**cMwind   + A_snIa*cR_Ia*mZ_Ia(1) ) ! per Msun
       ;; cEwind   = dlog10(10d0**cEwind   + A_snIa*cR_Ia*1d51)
       ;; do ichem=1,nchem
       ;;    cMwind_chem(:,:,ichem)=dlog10( 10d0**cMwind_chem(:,:,ichem) &
       ;;                                 & +A_snIa*cR_Ia*mZ_Ia(ichem+1) ) ! per Msun
       ;; enddo


newmet=[ 0.0001345,0.0001345,0.001345,0.01345 ]

nmet=n_elements(newmet)
print,nmet

filename='yields_evol_z-3_v0_chabrier_imf0.01-100msun_dustcpopping17.txt'
readcol,filename,time2,all,h,he,c,n,o,neon,mg,si,s,fe,hd,hed,cd,nd,od,neond,mgd,sid,sd,fed,/silent

ntime2=n_elements(time2)
help,cr_Ia,t_wind,time2
newcr_Ia=interpol(cr_Ia,t_wind,time2)

openw,1,'cr_Ia.ramses'
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

openw,1,'cr_Ia.txt'
for i=0L,ntime2-1L do begin
   printf,1,time2(i),newcr_Ia(i)
endfor
close,/all

stop
end
