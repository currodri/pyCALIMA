pro run_all_crunch_lc18

vel=[0,150,300]
vstring=['0','150','300']
;; zmet   =[-3.,-2.,-1., 0.]
zmet   =[-3,-2,-1, 0]
zstring=['fe-3','fe-2','fe-1','fe0']
nz=n_elements(zmet)
nvel=n_elements(vel)

for i=0,nvel-1L do begin
   for j=0,nz-1L do begin
      fileout='limongichieffi_'+zstring(j)+'_vel'+vstring(i)+'_simplified.txt'
      print,fileout
      crunch_lc18,zmet=zmet(j),vel=vel(i),fileout=fileout
   endfor
endfor

;; model m
for i=0,nvel-1L do begin
   for j=0,nz-1L do begin
      fileout='limongichieffi_modelm_'+zstring(j)+'_vel'+vstring(i)+'_simplified.txt'
      print,fileout
      rewrite_lc18_modelm,vel=vel(i),feoverh=zmet(j),fileout=fileout
   endfor
endfor

for i=0,nvel-1L do begin
   for j=0,nz-1L do begin
      fileout='limongichieffi_esn_'+zstring(j)+'_vel'+vstring(i)+'_simplified.txt'
      print,fileout
      crunch_lc18esn,zmet=zmet(j),vel=vel(i),fileout=fileout
   endfor
endfor


end
