command_idl_gce,/wind,min=1.5,tscale_inf=[1.5,0.5],tscale_sfr=6.5,prefact=5.3d10 
command_idl_gce,/wind,min=1.5,tscale_inf=[2.,0.5],tscale_sfr=5d4,prefact=3d10,alpha=1.4
command_idl_gce,alpha=1.4,tscale_inf=0.005,tscale_sfr=3d3,pref=3d14,nbint=2000
command_idl_gce,alpha=1.,tscale_sfr=0.25,pref=1.0d12,nbint=2000,/stop,acc=3.
command_idl_gce,/wind,tscale_inf=10.,tscale_sfr=3d5,prefact=2.6d10,mgrenorm=2.5,asnia=3d-2,alpha=1.5,accmodel=1,/rec,/stop,/var
command_idl_gce,/wind,min=.1,tscale_inf=[2.,0.2],tscale_sfr=2.,prefact=4d9 ,max=0.1
