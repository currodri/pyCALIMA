pro plot_chen

readcol,'Chen2000a.txt',a,feh1,feh2,ofe,nafe,mgfe,alfe,sife,kfe,cafe,tife,vfe,crfe,nife,bafe,skip=60,/silent
readcol,'Chen2000b.txt',a,feh,mass,age,skip=42,/silent,format='(a,f,f,f)'

print,n_elements(feh1),n_elements(feh)


openw,1,'Chen2000_tmp.txt'
printf,1,'#---------- ---- ----- ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------'
printf,1,'#                                                                                                        '
printf,1,'#          Mass  Age   [Fe/H  [O/F   [Na/F  [Mg/F  [Al/F  [Si/F  [K/F   [Ca/F  [Ti/F  [V/F   [Cr/F  [Ni/F  [Ba/F'
printf,1,'#                      ]([     e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] (['
printf,1,'#HD                    Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])'
printf,1,'#---------- ---- ----- ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------'
for i=0L,n_elements(a)-1L do begin
   printf,1,a(i),mass(i),age(i),feh(i),ofe(i),nafe(i),mgfe(i),alfe(i),sife(i),kfe(i),cafe(i),tife(i),vfe(i),crfe(i),nife(i),bafe(i),format='(A10,2f5.1,13f7.2)'
endfor
close,1

;; Should try to convert these into Asplund reference
;; openw,2,'Chen2000_tmp.txt'
;; printf,2,'#---------- ---- ----- ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------'
;; printf,2,'#                                                                                                        '
;; printf,2,'#          Mass  Age   [Fe/H  [O/F   [Na/F  [Mg/F  [Al/F  [Si/F  [K/F   [Ca/F  [Ti/F  [V/F   [Cr/F  [Ni/F  [Ba/F'
;; printf,2,'#                      ]([     e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] ([  e] (['
;; printf,2,'#HD                    Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])  Sun])'
;; printf,2,'#---------- ---- ----- ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------ ------'
;; for i=0L,n_elements(a)-1L do begin
;;    printf,2,a(i),mass(i),age(i),feh(i),ofe(i),nafe(i),mgfe(i),alfe(i),sife(i),kfe(i),cafe(i),tife(i),vfe(i),crfe(i),nife(i),bafe(i),format='(A10,2f5.1,13f7.2)'
;; endfor
;; close,2
stop

end
