pro run_all_crunch_karakas

zstring=['z0.0001','z0.004','z0.008','z0.02']
nz=n_elements(zstring)
for j=0,nz-1L do begin
   filename='karakas_'+zstring(j)+'.txt'
   fileout ='karakas_'+zstring(j)+'_simplified.txt'
   print,filename
   print,fileout
   crunch_karakas,filename=filename,fileout=fileout
endfor

zmet=[0.00001345,0.0001345,0.001345,0.003362,0.006725,0.01345]
zstring=['z0.00001345','z0.0001345','z0.001345','z0.003362','z0.006725','z0.01345']
nz=n_elements(zstring)
for j=0,nz-1L do begin
   fileout ='karakas_'+zstring(j)+'_simplified.txt'
   print,fileout
   interpolate_metallicity_karakas,zmet=zmet(j),fileout=fileout
endfor


end
