pro command_idl_gce,printps=printps,windmodel=windmodel,suffix=suffix,asnia=asnia,prefactor=prefactor,tscale_inf=tscale_inf,tscale_sfr=tscale_sfr,minmload=minmload,maxmload=maxmload,nbint=nbint,alpha=alpha,cosmic=cosmic,recycling=recycling,stop=stop,varying=varying,accmodel=accmodel,Mgrenorm=Mgrenorm,cosmo=cosmo

  ;; Romano 2020 model with varying velocities
  fileyieldevol=['yields_evol_z-3_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt' ]


  fileyieldevol=['yields_evol_z-3_v150_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_varying_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_salpeter_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_salpeter_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_salpeter_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v150_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v150_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v150_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v300_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v300_chabrier_imf0.1-100msun_mfailed100msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v300_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v150_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt' ]


  ;; galactic_chemical_evolution,/readyield,fileyieldevol=fileyieldevol,tscale_inf=[2.,0.5],tscale_sfr=3.,accmodel=2.,alpha=1.,/windm,nbint=1000,prefactor=2d10

  ;; fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-2_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v100_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v100_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v100_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v25_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v150_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v150_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  fileyieldevol=['yields_evol_z-3_v0_chabrier_imf0.1-100msun_mfailed60msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-2_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed30msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                 'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  ;; Prantzos+18
  ;; fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep1.0e+01msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  if keyword_set(varying)then begin
     fileyieldevol=['yields_evol_z-3_v150_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    'yields_evol_z-2_v100_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    'yields_evol_z-1_v50_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    'yields_evol_z-0.6_v50_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    'yields_evol_z-0.3_v50_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    ;; 'yields_evol_z0_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    ;; 'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]
                    'yields_evol_z0_v50_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
                    'yields_evol_z0.3_v50_varying_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]
  endif

  ;; fileyieldevol=['yields_evol_z-3_v150_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed0.25_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-2_v100_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed0.50_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-1_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed0.50_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.6_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed0.75_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed0.90_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0.3_v50_chabrier_imf0.1-100msun_msep8.0e+00msun_mfailed30msun_ffailed1.00_asnia3.5e-02_dustcpopping17.txt' ]

  ;; fileyieldevol=['yields_evol_z-3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-2_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-1_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.6_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z-0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_z0.3_v0_chabrier_imf0.1-100msun_mfailed25msun_asnia5.0e-02_dustcpopping17.txt' ]

  ;; Kobayashi yields without HNe
  ;; fileyieldevol=['yields_evol_kobayashi_z-3_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_kobayashi_z-1.13_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_kobayashi_z-0.53_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_kobayashi_z-0.23_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_kobayashi_z0.17_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt', $
  ;;                'yields_evol_kobayashi_z0.57_chabrier_imf0.1-100msun_msep1.0e+01msun_asnia5.0e-02_dustcpopping17.txt']
  ;; zyield=alog10([ 0.00001345, 0.001, 0.004, 0.008, 0.02, 0.05 ]/0.01345)


  if not keyword_set(nbint)then nbint=300
  if keyword_set(windmodel)then begin
     if not keyword_set(prefactor )then prefactor=2d10
     if not keyword_set(tscale_inf)then tscale_inf=[2.,0.5]
     if not keyword_set(tscale_sfr)then tscale_sfr=3
     if not keyword_set(alpha     )then alpha=1.
     if not keyword_set(accmodel  )then accmodel=2.
     galactic_chemical_evolution,/readyield,fileyieldevol=fileyieldevol,tscale_inf=tscale_inf,tscale_sfr=tscale_sfr,accmodel=accmodel,windmodel=windmodel,nbint=nbint,prefactor=prefactor,printps=printps,suffix=suffix,asnia=asnia,minmload=minmload,maxmload=maxmload,zyield=zyield,alpha=alpha,cosmic=cosmic,recycling=recycling,stop=stop,Mgrenorm=Mgrenorm
  endif else begin
     if not keyword_set(alpha     )then alpha=1.5
     if not keyword_set(tscale_inf)then tscale_inf=10.
     if not keyword_set(tscale_sfr)then tscale_sfr=3d5
     if not keyword_set(prefactor )then prefactor=1d10
     if not keyword_set(accmodel  )then accmodel=1.
     galactic_chemical_evolution,/readyield,fileyieldevol=fileyieldevol,tscale_inf=tscale_inf,tscale_sfr=tscale_sfr,accmodel=accmodel,nbint=nbint,prefactor=prefactor,printps=printps,suffix=suffix,asnia=asnia,zyield=zyield,alpha=alpha,cosmic=cosmic,recycling=recycling,stop=stop,Mgrenorm=Mgrenorm
  endelse


end
