Search.setIndex({docnames:["api","api/pyemgpipeline.plots","api/pyemgpipeline.processors","api/pyemgpipeline.wrappers","examples","index","installation","notebooks/ex0_input_data_description","notebooks/ex1_EMGMeasurement","notebooks/ex2_EMGMeasurementCollection","notebooks/ex3_DataProcessingManager","notebooks/ex4_DataProcessingManager","quickstart"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:3,sphinx:56},filenames:["api.rst","api\\pyemgpipeline.plots.rst","api\\pyemgpipeline.processors.rst","api\\pyemgpipeline.wrappers.rst","examples.rst","index.rst","installation.rst","notebooks\\ex0_input_data_description.ipynb","notebooks\\ex1_EMGMeasurement.ipynb","notebooks\\ex2_EMGMeasurementCollection.ipynb","notebooks\\ex3_DataProcessingManager.ipynb","notebooks\\ex4_DataProcessingManager.ipynb","quickstart.rst"],objects:{"pyemgpipeline.plots":[[1,0,1,"","EMGPlotParams"],[1,1,1,"","plot_emg"]],"pyemgpipeline.processors":[[2,0,1,"","AmplitudeNormalizer"],[2,0,1,"","BandpassFilter"],[2,0,1,"","BaseProcessor"],[2,0,1,"","DCOffsetRemover"],[2,0,1,"","EndFrameCutter"],[2,0,1,"","FullWaveRectifier"],[2,0,1,"","LinearEnvelope"],[2,0,1,"","Segmenter"]],"pyemgpipeline.processors.AmplitudeNormalizer":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.BandpassFilter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.BaseProcessor":[[2,2,1,"","apply"],[2,2,1,"","assert_input"],[2,2,1,"","export_csv"],[2,2,1,"","get_indices_from_timestamp"],[2,2,1,"","get_param_values_in_str"],[2,2,1,"","get_timestamp"]],"pyemgpipeline.processors.DCOffsetRemover":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.EndFrameCutter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.FullWaveRectifier":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.LinearEnvelope":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.Segmenter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.wrappers":[[3,0,1,"","DataProcessingManager"],[3,0,1,"","EMGMeasurement"],[3,0,1,"","EMGMeasurementCollection"]],"pyemgpipeline.wrappers.DataProcessingManager":[[3,2,1,"","process_all"],[3,2,1,"","set_amplitude_normalizer"],[3,2,1,"","set_bandpass_filter"],[3,2,1,"","set_data_and_params"],[3,2,1,"","set_dc_offset_remover"],[3,2,1,"","set_end_frame_cutter"],[3,2,1,"","set_full_wave_rectifier"],[3,2,1,"","set_linear_envelope"],[3,2,1,"","set_segmenter"],[3,2,1,"","show_current_processes_and_related_params"]],"pyemgpipeline.wrappers.EMGMeasurement":[[3,2,1,"","apply_amplitude_normalizer"],[3,2,1,"","apply_bandpass_filter"],[3,2,1,"","apply_dc_offset_remover"],[3,2,1,"","apply_end_frame_cutter"],[3,2,1,"","apply_full_wave_rectifier"],[3,2,1,"","apply_linear_envelope"],[3,2,1,"","apply_segmenter"],[3,2,1,"","export_csv"],[3,2,1,"","plot"]],"pyemgpipeline.wrappers.EMGMeasurementCollection":[[3,2,1,"","__getitem__"],[3,2,1,"","apply_amplitude_normalizer"],[3,2,1,"","apply_bandpass_filter"],[3,2,1,"","apply_dc_offset_remover"],[3,2,1,"","apply_end_frame_cutter"],[3,2,1,"","apply_full_wave_rectifier"],[3,2,1,"","apply_linear_envelope"],[3,2,1,"","apply_segmenter"],[3,2,1,"","export_csv"],[3,2,1,"","find_max_amplitude_of_each_channel_across_trials"],[3,2,1,"","plot"]]},objnames:{"0":["py","class","Python class"],"1":["py","function","Python function"],"2":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:function","2":"py:method"},terms:{"0":[2,3,4,5,8,9,10,11,12],"00":10,"000":7,"0000e":10,"00022803":11,"00030303":11,"00033644":11,"00036013":11,"00036325":11,"00037394":11,"00037888":11,"00038228":11,"00039053":11,"00039166":11,"00039269":11,"00048853":11,"00049598":11,"00052072":11,"00052723":11,"0005491":11,"00058635":11,"00058947":11,"0007":[8,9],"0008":[8,9],"00081575":11,"00085325":11,"00086887":11,"00095102":11,"00096039":11,"00097289":11,"001":7,"00112769":11,"00113706":11,"00114644":11,"00136287":11,"00138162":11,"00138787":11,"0015":[8,9],"00152569":11,"00156944":11,"00158556":11,"00159493":11,"00161681":11,"00166319":11,"002":7,"0022":9,"0023":9,"00273551":8,"00273592":8,"00273646":8,"00278":[8,9],"003":[3,7,8,9],"0037":9,"0038":9,"004":7,"00403501":8,"00405828":8,"00408111":8,"0045":9,"0046":7,"00481":10,"005":7,"0053":[7,9],"006":[7,9],"0067":7,"0068":9,"007":7,"00735341":9,"0073656":9,"00737811":9,"0075":9,"008":[3,7],"0082":7,"009":7,"00961191":9,"00986346":9,"00988673":9,"00993258":9,"00998159":9,"01":[3,7,10],"010":7,"01010689":9,"0105":9,"011":7,"012":[7,9],"01218871":9,"01220219":9,"01221804":9,"0123954":8,"0123964":8,"01239804":8,"0125976":9,"01262609":9,"01263688":9,"01265461":9,"01266372":9,"01269028":9,"013":7,"01334102":9,"01343851":8,"01344915":8,"01346193":8,"01389486":9,"014":7,"0142":9,"0143":7,"01444859":9,"015":[7,12],"016":7,"0165":9,"017":7,"0173":9,"01757233":9,"01761076":9,"01764664":9,"01773494":8,"01775758":8,"01778186":8,"018":[3,7],"01807356":9,"0180855":9,"0181":9,"01810079":9,"0189266e":11,"019":7,"0195":7,"01953295":9,"01955036":8,"01955799":8,"01956411":8,"020":7,"02025041":9,"02094466":9,"021":7,"02186312":8,"02193963":8,"022":[3,7],"02201485":8,"0225":7,"02298162":9,"023":7,"02309847":9,"02321987":9,"024":7,"025":7,"02519068":8,"02520053":8,"02521111":8,"02522924":9,"02527315":9,"02532435":9,"02684806":9,"0269777":9,"0271074":9,"02820274":9,"02831608":9,"02843199":9,"02957739":9,"02970596":9,"02983576":9,"03":[9,10,11],"031":9,"032":9,"03330972":9,"03371283":9,"03382395":9,"03392962":9,"03402932":9,"03412713":9,"035":[3,7],"0368":7,"04":[3,11],"041":[3,7],"043":8,"047":[3,7],"05":[10,11],"0501765e":11,"054":[3,7],"05676127":9,"05704184":9,"05715868":9,"05733228":9,"05755993":9,"05763172":9,"0622":9,"06336577":9,"06437713":9,"0645":9,"06536599":9,"0675":7,"068":[3,7,8],"069":8,"073":3,"07679376":9,"07700917":9,"07723283":9,"07740053":9,"07772384":9,"07803646":9,"08272218":9,"087":[3,7],"08_21":10,"0908389e":11,"09118":9,"09163069":9,"0920687":9,"1":[1,2,3,4,5,7,9,10,11,12],"10":[1,2,3,7,8,9,10,11],"1000":[2,3,7,8,9,10,12],"10000":11,"10100":11,"1016":[2,3],"102":[3,7],"1033389e":11,"104":3,"1050568e":11,"1052228e":11,"1059278e":11,"106":[3,7],"108":[2,3],"1095889e":11,"10_21":10,"11":[3,7,9,10,11],"112":[2,3],"113":[3,7],"11531935":9,"12":[3,7,9,10,11],"120":8,"125":[3,7],"13":[3,7,10,11],"13096556":9,"13151363":9,"132":[3,7],"13206459":9,"13300577":9,"13480":9,"135":7,"1365":7,"1372":7,"139":[3,7],"14":[3,7,9,10,11],"141":[3,7],"1466444e":11,"14mb":[8,9],"15":[3,7,10,11],"1500":2,"15260":9,"153":[3,7],"16":[2,3,7,9,10,11],"17":[3,7,10],"172":[3,7],"175":2,"17mb":10,"18":[2,3,7,10],"187":2,"19":[3,7,10],"191":3,"1999":[2,3],"1_raw_data_11":10,"1d":[7,10,12],"2":[1,2,3,4,5,7,8,10,11,12],"20":[2,7,8,12],"2000":[3,11],"2006":2,"2018":[2,3],"204":[3,7],"20556472":9,"21":[3,7,12],"22":[3,7],"221":3,"223":[3,7],"2275972e":11,"23":[3,7],"2309276e":11,"2369723e":11,"24":[7,12],"242":[3,7],"2463472e":11,"2469321e":11,"256":3,"26":[3,7,12],"268":[3,7],"27":[3,7],"2719320e":11,"275":[3,7],"279":[3,7],"28":[3,7,10],"289":[3,7],"29":[7,8,12],"296":[3,7],"2969321e":11,"298":3,"2_raw_data_11":10,"2d":[7,8,9,10,11,12],"3":[1,2,3,4,5,7,8,9,11,12],"30":[2,3,8,9,10,11],"30348":[2,3],"308":[3,7],"313":[3,7],"3177502e":11,"3246781e":11,"327":3,"3314267e":11,"341":3,"3440082e":11,"346":[3,7],"3490001e":11,"35":[3,7],"350":2,"36":3,"364":8,"37":[3,7,12],"373":[3,7],"375":[3,7],"3752581e":11,"379mb":11,"38":3,"385":3,"388":[3,7],"38971":8,"39":[7,11,12],"392":[3,7],"3954e":10,"3955e":10,"3956e":10,"3asen":8,"4":[2,3,4,5,7,8,9,10,12],"40":3,"400":2,"409":3,"41":[3,7,12],"411":3,"4138819e":11,"42":[2,3,7,12],"420":3,"425":3,"4278939e":11,"43":[3,7],"4360669e":11,"43665":7,"44":[7,12],"444":[3,7],"45":[3,7,12],"450":[2,3,8,9,10,11],"4554e":10,"4555e":10,"4557e":10,"458":3,"463":[3,7],"470":10,"475":3,"477":[3,7],"48":[7,12],"4802227e":11,"49":3,"496":3,"498":[8,9],"4985673e":11,"499":[8,9],"5":[2,3,7,8,9,10,11,12],"50":[8,9],"500":2,"501":8,"502":8,"51":[3,7,10],"5239861":11,"529":[3,7],"53":[3,7,9,12],"531":9,"532":9,"534":3,"54":3,"5427226e":11,"55":3,"554":3,"56":[7,12],"561":[3,7],"57":3,"573":[3,7],"58":3,"584":3,"594":[3,7],"597":3,"5nmar":9,"5npie":9,"5nsen":9,"6":[2,3,7,8,9,10,11,12],"60":3,"601":9,"602":9,"61":3,"613":[3,7],"61448":10,"61641":10,"62":3,"625":3,"63":[7,12],"632":[3,7],"6411":[2,3],"646":[3,7],"6563":9,"665":[3,7],"67":3,"678":3,"684":[3,7],"6877582e":11,"69":3,"694":[3,7],"6951320e":11,"696":3,"698":[3,7],"7":[3,7,8,9,10,11],"703":3,"717":3,"7173170e":11,"73":[3,7],"734":3,"749":[3,7],"75":11,"756":[3,7],"763":[3,7],"78":[3,7],"782":[3,7],"7925558e":11,"799":3,"8":[3,7,10,11,12],"801":[3,7],"815":[3,7],"823":[3,7],"834":[3,7],"8341444e":11,"839":[3,7],"84":3,"85":3,"8513823e":11,"859":[3,7],"86":[3,7],"864":[3,7],"866":[3,7],"878":[3,7],"884":3,"897":3,"899":[3,7],"9":[2,3,7,9,10,11,12],"901":9,"902":9,"918":[3,7],"924":[3,7],"937":[3,7],"951":[3,7],"9574438e":11,"97":[3,7],"9739996e":11,"984":[3,7],"989":3,"992":3,"995":[3,7],"998":9,"999":[3,9,10],"abstract":2,"case":[3,10,11],"class":[5,7,8,9,10,11,12],"default":[1,2,3,7,10,11],"do":7,"export":[2,3,8,9],"final":[3,12],"float":[2,3,8,9],"function":[3,7],"import":[3,6,7,8,9,10,11,12],"int":[1,2,3,11],"return":[1,2,3,8,9],"static":[2,10],"true":[3,9,10,11],"while":[8,9,10],A:3,For:[2,3,7,8,9,10],If:[1,2,3,7,12],In:[2,3,8,9,10,11],It:5,Its:7,No:[10,11],One:[2,3],The:[1,2,3,5,6,7,8,9,10,11,12],Then:9,With:[3,10,11],__getitem__:3,_as_gen:[1,3],_data:[8,9],_filepath:[8,9],a_txt:8,abov:[10,11],accept:[3,5,10,11],access:[8,9,11],accord:7,across:[3,9,10,11],action:[2,3],actual:[2,3,7],add:10,adher:5,adjust:3,after:[2,3,6,7,8,9,10,11],again:10,all:[1,2,3,8,9,10,11],all_beg_t:[3,9,10],all_csv_path:[3,9],all_data:[3,7,9,10,11],all_end_t:[3,9,10],all_main_titl:10,all_timestamp:[3,7,9,10,11],all_trial_nam:10,alreadi:8,also:[2,3,7,8,9,10,11,12],amplitud:[2,3,8,9,10,11],amplitude_norm:3,amplitudenorm:[0,3,10,11],an:[2,3,5,8,9,10,11],ani:2,annot:11,anterior:11,api:[1,3,5],append:[9,10,11],appli:[2,3,8,9,10,11,12],apply_amplitude_norm:[3,8,9,12],apply_bandpass_filt:[3,8,9,12],apply_dc_offset_remov:[3,8,9,12],apply_end_frame_cutt:[3,8,9,12],apply_full_wave_rectifi:[3,8,9,12],apply_linear_envelop:[3,8,9,12],apply_segment:[3,8,9,12],ar:[1,3,5,7,8,9,10,11],archiv:[8,9,10,11],argument:2,around:2,arrai:[3,8,9,10,11,12],assert_input:2,assess:[2,3],assum:[8,9,10,12],author:3,automat:12,avail:5,ax:3,axes_pos_adjust:[3,11],b:[7,9],bandpass:[2,3,8,9,10,11],bandpass_filt:3,bandpassfilt:[0,3,10,11],bandwidth:2,baseprocessor:0,basic:[8,9,10,11],bbox_to_anchor:[3,11],befor:[8,9,11],beg_idx:2,beg_t:[2,3,12],begin:[2,3,7],being:[10,11],below:7,besid:[8,9],between:3,bf_cutoff_fq_hi:[2,3,8,9,10,11],bf_cutoff_fq_lo:[2,3,8,9,10,11],bf_order:[2,3,8,9,10,11],bicep:[7,8,9,11],biom:[2,3],bk:11,bool:3,both:[1,2,3,5],bottom:3,brown:11,butterworth:[2,3],c:[3,9,10,11],c_raw:3,calcul:[9,10,11],callaghan:2,can:[3,6,7,8,9,10,11,12],certain:1,ch1:3,ch2:3,ch:11,chang:3,channel1:10,channel2:10,channel3:10,channel4:10,channel5:10,channel6:10,channel7:10,channel8:10,channel:[1,2,3,8,9,10,11,12],channel_nam:[1,2,3,8,9,10,11],check:2,circumst:2,code:[3,5],collect:[8,9,10,11],collect_valu:[8,9],color:[1,3,8,9],column:[1,8,9,10,11,12],come:7,command:[8,9,10,11],commun:[2,3],complet:[8,9],compress:[8,9,10,11],concaten:[8,9],concert:[2,3],condit:[2,3],configur:[9,10,11],consid:1,constrained_layout:1,contain:[3,8,9,10,11],contamin:2,continu:[8,9],contract:[8,9],control:[1,3],convent:[3,5,10,11],convert:7,copi:3,correspond:[1,2,3],creat:[1,10,11,12],csv:[2,3,7,8,9],csv_path:3,current:[2,3,10,11],cut:[2,3,8,9],cutoff:[2,3],cutter:[2,3,10,11],cycled_color:[3,11],d:[2,3,10],data:[1,2,3,4,5],data_filenam:[8,9,10,11],data_fold:[8,9,10,11],data_in_list:11,databas:[8,9,10],dataend:11,dataprocessingmanag:[0,5,7,10,11],dataset:[8,9,11],datastart:11,dc:[2,3,8,9,10,11],dc_offset_remov:3,dcoffsetremov:[0,3,10,11],deduc:2,def:[8,9],definit:[4,5],deg:7,delimit:10,demo:5,demonstr:[8,9,10],depend:6,deriv:[8,9],describ:7,descript:[4,5,8,9],design:5,destin:3,detail:1,di:2,dict:[1,3,11],differ:[2,7,10],dim:[2,3],dimens:[2,3],direct:[2,3],discuss:7,displai:[2,3,10,11],divisor:[2,3],doi:[2,3],download:[8,9,10,11],dpi:[1,8],drake:2,dtype:11,e:[2,3,7,8,9,10],each:[3,7,8,9,10,11,12],edgecolor:1,editor:[2,3],edu:[8,9,10],effect:[2,3],eight:10,either:7,electrocardiogram:2,electromyogram:2,electromyographi:[2,3,5,11],element:[1,3,7,9,10,11],elimin:2,emg:[1,2,3,4,5,7,10,11],emg_data_for_gestur:10,emg_plot:[1,3],emg_plot_param:[1,3,8,9,10,11],emgmeasur:[0,7,8,9,12],emgmeasurementcollect:[0,2,7,9,10,11],emgplotparam:[0,3,8,9,10,11],encourag:5,end:[2,3,8,9,10,11],end_frame_cutt:3,end_idx:[2,11],end_t:[2,3,12],endframecutt:[0,3,10,11],engin:7,ensur:5,envelop:[2,3,8,9,10,11],equal:[1,2,3,7],etc:[1,3,8,9,10,11],european:[2,3],evalu:2,ex1_process:8,ex2_gait:9,ex2_sit:9,ex2_stand:9,exactli:1,examin:[10,11],exampl:[5,12],exce:3,execut:3,experi:8,export_csv:[2,3,8,9,10,11],extens:7,extract:[3,7,8,9,10,11],f:[2,3,7,9],facecolor:1,fals:3,feed:7,femor:[7,11],femori:[8,9],few:[8,9],fig_kwarg:[1,3,8,9,10,11],figsiz:[1,3,8,9,10,11],figur:[1,3,8,9,10,11],figure_api:1,file:[2,8,9,10,11],filepath:[2,8,9],fillstyl:1,filter:[2,3,7,8,9,10,11],filtfilt:2,find:3,find_max_amplitude_of_each_channel_across_tri:[3,9],fine:3,finish:[8,9,10,11],first:[2,3,8,9,10,11],fit:1,five:[10,11],flexo:7,float32:11,fn:[10,11],folder:[9,10,11],follow:[7,10,11],fontsiz:[3,11],forearm:10,format:[5,8,9],found:3,four:[8,9],fp:[8,9],frame:[2,3,8,9,10,11],frameon:1,frequenc:[2,3,7],fresh:3,from:[2,3,6,8,9,10,11,12],full:[2,3,8,9,10,11],full_wave_rectifi:3,fullwaverectifi:[0,3,10,11],furtuer:[8,9,10,11],gait:9,gastrocnemio:11,gener:[7,12],genfromtxt:[7,10],gestur:[4,5],get:[1,2,3],get_indices_from_timestamp:2,get_param_values_in_str:2,get_timestamp:2,github:5,give:7,given:7,greater:2,green:11,guid:[3,4,5],h:[2,3],ha:[3,7],half:2,hand:10,handl:7,have:[1,2,3,8,9],header:[2,7],heart:2,height:3,here:[7,8,9],hermen:[2,3],hertz:[2,3],high:[2,3,4,5],higher:[2,3],highest:3,hire:[3,12],hspace:[8,9,10,11],hstack:11,html:[1,3],http:[1,3,8,9,10,11],hydrotherapi:11,hz:[2,3,7,8,9,10,11,12],i:[2,3,7,9,10,11],ic:[8,9,10],ii:[2,3],iii:2,implement:5,includ:[1,3,6,8,9,10,11],inclus:2,increment:[7,12],index:[2,3,7,9,10,11],indic:[2,10],individu:7,inform:[8,9,10,11],initi:[3,8,9,10,11],input:[2,4,5],instal:[5,8,9,10,11],instanc:[3,8,9,10,11],integ:[2,3],interest:[2,3,8,9,10],interfac:[3,5,10,11],intermedi:[3,10,11],internation:5,internu:[8,9],intramuscular:[2,5],introduc:7,inul:[8,9,11],invas:[2,3],io:[7,11],irvin:[8,9,10],is_overlapping_tri:[3,11],is_plot_processing_chain:[3,10,11],isinst:[9,10,11],item:[8,9],its:[1,2,3,7],iv:2,j:[2,3],join:[8,9,10,11],journal:[2,3],k:[3,7,9],k_for_plot:[3,10],kb:[8,9,10],keyword:2,kinesiolog:[2,3],knee:[4,5],knee_oa_therapi:11,known:8,kwarg:2,largest:2,last:[8,9,10],later:11,launch:6,le_cutoff_fq:[2,3,8,9,10,11],le_ord:[2,3,8,9,10,11],learn:[8,9,10],least:[2,3],left:3,legend:3,legend_kwarg:[3,11],len:[3,7,8,9,11],length:[1,2,3],less:[1,2],level:[3,4,5],limb:[4,5,11],line2d:1,line2d_kwarg:[1,8,9],line:[1,7,8,9],linear:[2,3,8,9,10,11],linear_envelop:3,linearenvelop:[0,3,10,11],linestyl:1,linewidth:1,list:[1,2,3,7,9,10,11,12],load:7,load_uci_lower_limb_txt:[8,9],loadmat:[7,11],loc:[3,11],log:7,low:[2,3],lower:[2,4,5,11],lowpass:[2,3],m:[2,3,8,12],machin:[8,9,10],mai:2,main:[1,3],main_titl:[1,3],manag:[10,11],marcha_2:7,marker:1,markers:1,master:10,mat:[7,11],matlab:7,matplotlib:[1,3,6,8,9,10,11],max:3,max_amplitud:[3,8,9,12],maximum:[8,9,10,11],mb:[8,9,10],measur:[4,5],medial:[7,11],meet:3,merletti:2,method:[2,3,10,11],mgr:[3,10,11],might:[8,9],minimum:3,ml:[8,9,10],modul:7,more:[1,2,3],multipl:[2,3,4,5,10,11,12],muscl:[2,3,8,9],must:3,mv:7,mvc:[8,9],n_block:11,n_channel:[1,2,3,7,8,9,10,11,12],n_col:[1,3,9,10,11],n_end_fram:[2,3,8,9,10,11,12],n_row:[1,8,9,10,11],n_sampl:[1,2,3,7,8,9,10,11,12],n_txt:9,name:[1,2,3,7,8,9,10],ncol:3,ndarrai:[1,2,3,7,8,9,10,11,12],need:[7,8,9,10,11],needl:2,non:[2,3,8,9],none:[1,2,3,10,11],normal:[2,3,8,9,10,11],note:[2,3,8,9,10,11],notebook:7,now:[8,9,10],np:[3,7,8,9,10,11,12],number:[1,2,3,7],numpi:[3,6,8,9,10,11,12],nyquist:3,obtain:8,off:[2,3],offset:[2,3,8,9,10,11],one:[1,2,3,8,9,10,11,12],onli:[3,8,9,10,11,12],open:[7,8,9,11],oper:[9,10,11],order:[2,3,5],org:[1,3,11],organ:[7,9,10,11,12],origin:[10,11],os:[8,9,10,11],osteoarthr:[4,5],other:[3,8,9,10,11],otherwis:[1,2,3],overlai:3,overlap:3,p:[2,8,9,10,11],packag:[5,6,7],pad:2,padlen:2,paramet:[1,2,3,5,8,9,10,11],params_in_str:2,part:[7,10],particular:[8,9,10,11],pass:2,path:[3,8,9,10,11],patient:11,pdf:[3,12],pep:[3,6,8,9,10,11,12],perform:10,physiotherapi:11,pip:[6,8,9,10,11],pipelin:5,placement:2,plot:[0,3,5,8,9,10,11,12],plot_emg:0,png:[3,12],posit:[2,3],possess:[3,10,11],postur:8,predefin:[10,11],prefer:3,prepar:3,present:12,prevent:3,previou:7,print:[7,8,9,10,11],process:[1,2,3,4,5,7,12],process_al:[3,10,11],processor:[0,3,5,10,11],program:[2,3],project:[2,3],prop:3,proport:3,provid:[5,10,12],purpos:[2,8,9,10],pyemgpipelin:[0,6,8,9,10,11,12],pypi:6,python:7,q:[8,9,10,11],quick:5,r:2,rang:[7,8,9,10,11],rar:[8,9,11],rate:[2,3,8,9,10,11,12],raw:[3,8,9,10,11],readlin:[8,9],receiv:11,recommend:[10,11],record:[2,11],rectif:[8,9],rectifi:[2,3,10,11],recto:7,rectu:[8,9],red:8,reduc:2,ref:[2,3],refer:[2,3],rehabilit:11,rel:3,relat:[3,10,11],remov:[2,3,8,9,10,11],repeatedli:3,report:[2,3],repositori:[5,6,8,9,10,11],repres:[3,8,9,10,11,12],requir:[2,7],reset:[10,11],reshap:[8,9,11],respect:7,respons:[8,9,10,11],result:[2,3,10,11,12],rf:[8,9,10,11],right:11,rm:[8,9,10,11],routin:11,row:[1,8,9],run:[2,3,10],s1050:[2,3],s1_t1_rf_a:11,s1_t1_rf_b:11,s:[7,8,9,10],same:[2,3,9,10,11],sampl:[2,3,7,8,9,10,11,12],sample_r:[8,9,10,11],save:[3,7,8,9,10,11],scalar:[2,3],scenario:2,scipi:[2,6,7,11],section:7,see:[1,2,3],segment:[0,3,8,9,10,11],segmenter_all_beg_t:3,segmenter_all_end_t:3,semg_db1:[8,9],semitendinoso:7,semitendinosu:[8,9],seniam:[2,3],sequenc:[10,11],seri:10,set:[1,3,8,9,10,11],set_amplitude_norm:[3,10,11],set_bandpass_filt:[3,10],set_data_and_param:[3,10,11],set_dc_offset_remov:3,set_end_frame_cutt:3,set_full_wave_rectifi:3,set_linear_envelop:3,set_segment:[3,10],seven:[3,8,9,12],sever:7,shape:[1,2,3,7,8,9,10,11,12],shift:3,should:[1,2,3,7,8,9,10,11,12],show:[3,7,8,9,10,11],show_current_processes_and_related_param:[3,10,11],shown:[1,2,3,7,10],signal:[1,2,3,5,8,9,10,11,12],simpli:12,sinc:10,singl:[3,4,5,9,10,11],sit:[8,9],six:11,skip_head:10,small:11,smallest:[1,2],soleo:11,some:2,sourc:[3,5,12],space:7,specif:2,split:[8,9],stabl:[1,3],stand:[2,9],standard:[2,3,8,9],start:[2,5,7,8,9],start_idx:11,stegeman:[2,3],step:[2,3,5,8,9,10,11,12],store:[7,8,9,10,11,12],str:[1,2,3],strictli:2,subfold:[8,9,10,11],subject:[8,9,10,11],subplotpar:[1,3,8,9,10,11],subplotparam:[3,8,9,10,11],suggest:[2,3],suitabl:5,summari:0,surfac:[2,3,5,11],t:[3,8,9,10],techniqu:2,term:5,text:7,than:[1,2],thei:[1,3,7,10,11],them:[10,11],theorem:3,therapi:[4,5],thi:[1,2,3,5,6,7,8,9,10,11],those:5,three:[1,9],tibial:11,tight_layout:1,time:[1,2,3,8,9,10],timestamp:[1,2,3,8,9,10,11,12],titl:[1,3,11],togeth:1,top:[3,9,10,11],torino:2,trial:[3,8,9,10,11,12],trial_nam:[3,8,9,11],trunk:2,tupl:[2,3],twice:3,two:[1,2,3,10,11],txt:[8,9,10],type:[1,2,3,7,9,10,11],uc:[8,9,10],uci:[8,9,10],uci_gestur:10,uci_lower_limb:[8,9],uncompress:[8,9,10,11],under:9,undergo:11,unit:7,unrar:[8,9,11],unzip:10,up:[8,9,10,11],updat:10,us:[1,2,3,4,5,7,12],usag:[8,9,10,11],user:[5,7],valid:7,valu:[1,2,3,5,7,9],vasto:[7,11],vastu:[8,9],vector:2,via:[8,9],visual:[8,9],voluntari:[8,9],wa:[8,9,10,11],wai:[3,10,11],wave:[2,3,8,9,10,11],we:[8,9,10,11],wget:[8,9,10,11],when:[1,2,3,7,8,9,10,11],whenev:[8,9],where:[2,3,7,8,9,10,11,12],whether:3,which:[2,3,7,8,9,10,11],whole:[7,8,9,10,11],width:3,wire:3,within:5,won:3,wrapper:[0,5,8,9,10,11,12],wspace:[8,9,10,11],x:[1,2,3,8,9,11],x_process:2,x_shape:2,yet:[10,11],zenodo:11,zip:10},titles:["API","pyemgpipeline.plots","pyemgpipeline.processors","pyemgpipeline.wrappers","Examples","pyemgpipeline Documentation","Installation","Example 0 - Input data description","Example 1 - Single EMG measurement definition and processing (using lower limb data)","Example 2 - Multiple EMG measurement definition and processing (using lower limb data)","Example 3 - High-level, guided processing (using gesture data)","Example 4 - High-level, guided processing (using knee osteoarthritis therapy data)","Quick Start"],titleterms:{"0":7,"1":8,"2":9,"3":10,"4":11,"class":[1,2,3],"function":1,amplitudenorm:2,api:0,arrai:7,bandpassfilt:2,base:2,baseprocessor:2,channel:7,code:12,content:5,covert:7,data:[7,8,9,10,11,12],dataprocessingmanag:3,dcoffsetremov:2,definit:[8,9],demo:12,descript:7,document:5,emg:[8,9],emgmeasur:3,emgmeasurementcollect:3,emgplotparam:1,endframecutt:2,exampl:[3,4,7,8,9,10,11],file:7,format:[7,12],free:7,from:7,fullwaverectifi:2,gestur:10,guid:[10,11],high:[10,11],input:[7,12],instal:6,knee:11,level:[10,11],limb:[8,9],linearenvelop:2,lower:[8,9],measur:[8,9],more:7,multipl:[7,9],numpi:7,one:7,osteoarthr:11,packag:[8,9,10,11],plot:1,plot_emg:1,prepar:[8,9,10,11],process:[8,9,10,11],processor:2,pyemgpipelin:[1,2,3,5],quick:12,raw:7,segment:2,signal:7,singl:[7,8],start:12,structur:7,summari:[1,2,3],therapi:11,timestamp:7,trial:7,us:[8,9,10,11],wrapper:3}})