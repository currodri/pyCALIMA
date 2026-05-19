pro run_all_interpolate_lc18,modelm=modelm

;; vel=[0,150,300]
;; vstring=['0','150','300']
vel    =[ 0 , 25 , 50 , 100 , 150 , 200 , 250 , 300 ]
vstring=['0','25','50','100','150','200','250','300']
zmet   =[1.345d-5,1.345d-4,1.345d-3,3.3625d-3,6.725d-3,1.345d-2,2.690d-2]
zstring=['z-3','z-2','z-1','z-0.6','z-0.3','z0','z0.3']
nz=n_elements(zmet)
nvel=n_elements(vel)

if keyword_set(modelm)then begin
   rootfile='limongichieffi_modelm'
endif else begin
   rootfile='limongichieffi'
endelse
for i=0,nvel-1L do begin
   for j=0,nz-1L do begin
      fileout=+rootfile+'_'+zstring(j)+'_vel'+vstring(i)+'_simplified.txt'
      print,fileout
      interpolate_lc18,zmet=zmet(j),vel=vel(i),fileout=fileout,modelm=modelm
      ;; if(zmet(j) lt 3.236d-5)then begin
      ;;    print,'cp '+rootfile+'_fe-3_vel'+vstring(i)+'_simplified.txt '+fileout
      ;;    spawn,'cp '+rootfile+'_fe-3_vel'+vstring(i)+'_simplified.txt '+fileout
      ;; endif else if(zmet(j) ge 1.345d-2)then begin
      ;;    print,'cp '+rootfile+'_fe0_vel'+vstring(i)+'_simplified.txt '+fileout
      ;;    spawn,'cp '+rootfile+'_fe0_vel'+vstring(i)+'_simplified.txt '+fileout
      ;; endif else begin
      ;;    interpolate_lc18,zmet=zmet(j),vel=vel(i),fileout=fileout,modelm=modelm
      ;; endelse
   endfor
endfor


end
