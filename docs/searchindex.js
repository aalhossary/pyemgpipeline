Search.setIndex({docnames:["api","api/pyemgpipeline.plots","api/pyemgpipeline.processors","api/pyemgpipeline.wrappers","examples","index","installation","notebooks/ex1_EMGMeasurement","notebooks/ex2_EMGMeasurementCollection","notebooks/ex3_DataProcessingManager","quickstart"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:3,sphinx:56},filenames:["api.rst","api\\pyemgpipeline.plots.rst","api\\pyemgpipeline.processors.rst","api\\pyemgpipeline.wrappers.rst","examples.rst","index.rst","installation.rst","notebooks\\ex1_EMGMeasurement.ipynb","notebooks\\ex2_EMGMeasurementCollection.ipynb","notebooks\\ex3_DataProcessingManager.ipynb","quickstart.rst"],objects:{"pyemgpipeline.plots":[[1,0,1,"","EMGPlotParams"],[1,1,1,"","plot_emg"]],"pyemgpipeline.processors":[[2,0,1,"","AmplitudeNormalizer"],[2,0,1,"","BandpassFilter"],[2,0,1,"","BaseProcessor"],[2,0,1,"","DCOffsetRemover"],[2,0,1,"","EndFrameCutter"],[2,0,1,"","FullWaveRectifier"],[2,0,1,"","LinearEnvelope"],[2,0,1,"","Segmenter"]],"pyemgpipeline.processors.AmplitudeNormalizer":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.BandpassFilter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.BaseProcessor":[[2,2,1,"","apply"],[2,2,1,"","assert_input"],[2,2,1,"","export_csv"],[2,2,1,"","get_indices_from_timestamp"],[2,2,1,"","get_param_values_in_str"],[2,2,1,"","get_timestamp"]],"pyemgpipeline.processors.DCOffsetRemover":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.EndFrameCutter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.FullWaveRectifier":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.LinearEnvelope":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.processors.Segmenter":[[2,2,1,"","apply"],[2,2,1,"","get_param_values_in_str"]],"pyemgpipeline.wrappers":[[3,0,1,"","DataProcessingManager"],[3,0,1,"","EMGMeasurement"],[3,0,1,"","EMGMeasurementCollection"]],"pyemgpipeline.wrappers.DataProcessingManager":[[3,2,1,"","process_all"],[3,2,1,"","set_amplitude_normalizer"],[3,2,1,"","set_bandpass_filter"],[3,2,1,"","set_data_and_params"],[3,2,1,"","set_dc_offset_remover"],[3,2,1,"","set_end_frame_cutter"],[3,2,1,"","set_full_wave_rectifier"],[3,2,1,"","set_linear_envelope"],[3,2,1,"","set_segmenter"],[3,2,1,"","show_current_processes_and_related_params"]],"pyemgpipeline.wrappers.EMGMeasurement":[[3,2,1,"","apply_amplitude_normalizer"],[3,2,1,"","apply_bandpass_filter"],[3,2,1,"","apply_dc_offset_remover"],[3,2,1,"","apply_end_frame_cutter"],[3,2,1,"","apply_full_wave_rectifier"],[3,2,1,"","apply_linear_envelope"],[3,2,1,"","apply_segmenter"],[3,2,1,"","export_csv"],[3,2,1,"","plot"]],"pyemgpipeline.wrappers.EMGMeasurementCollection":[[3,2,1,"","__getitem__"],[3,2,1,"","apply_amplitude_normalizer"],[3,2,1,"","apply_bandpass_filter"],[3,2,1,"","apply_dc_offset_remover"],[3,2,1,"","apply_end_frame_cutter"],[3,2,1,"","apply_full_wave_rectifier"],[3,2,1,"","apply_linear_envelope"],[3,2,1,"","apply_segmenter"],[3,2,1,"","export_csv"],[3,2,1,"","find_max_amplitude_of_each_channel_across_trials"],[3,2,1,"","plot"]]},objnames:{"0":["py","class","Python class"],"1":["py","function","Python function"],"2":["py","method","Python method"]},objtypes:{"0":"py:class","1":"py:function","2":"py:method"},terms:{"0":[2,3,7,8,9,10],"00":9,"0000e":9,"0007":[7,8],"0008":[7,8],"0015":[7,8],"0022":8,"0023":8,"00273551":7,"00273592":7,"00273646":7,"003":[3,7,8],"0037":8,"0038":8,"00403501":7,"00405828":7,"00408111":7,"0045":8,"0053":8,"006":8,"0068":8,"00735341":8,"0073656":8,"00737811":8,"0075":8,"008":3,"00961191":8,"00986346":8,"00988673":8,"00993258":8,"00998159":8,"01":[3,9],"01010689":8,"0105":8,"012":8,"01218871":8,"01220219":8,"01221804":8,"0123954":7,"0123964":7,"01239804":7,"0125976":8,"01262609":8,"01263688":8,"01265461":8,"01266372":8,"01269028":8,"01334102":8,"01343851":7,"01344915":7,"01346193":7,"01389486":8,"0142":8,"01444859":8,"015":10,"0165":8,"0173":8,"01757233":8,"01761076":8,"01764664":8,"01773494":7,"01775758":7,"01778186":7,"018":3,"01807356":8,"0180855":8,"0181":8,"01810079":8,"01953295":8,"01955036":7,"01955799":7,"01956411":7,"02025041":8,"02094466":8,"02186312":7,"02193963":7,"022":3,"02201485":7,"02298162":8,"02309847":8,"02321987":8,"02519068":7,"02520053":7,"02521111":7,"02522924":8,"02527315":8,"02532435":8,"02684806":8,"0269777":8,"0271074":8,"02820274":8,"02831608":8,"02843199":8,"02957739":8,"02970596":8,"02983576":8,"03":[8,9],"031":8,"032":8,"03330972":8,"03371283":8,"03382395":8,"03392962":8,"03402932":8,"03412713":8,"035":3,"04":3,"041":3,"043":7,"047":3,"05":9,"054":3,"05676127":8,"05704184":8,"05715868":8,"05733228":8,"05755993":8,"05763172":8,"0622":8,"06336577":8,"06437713":8,"0645":8,"06536599":8,"068":[3,7],"069":7,"073":3,"07679376":8,"07700917":8,"07723283":8,"07740053":8,"07772384":8,"07803646":8,"08272218":8,"087":3,"08_21":9,"09118":8,"09163069":8,"0920687":8,"1":[1,2,3,5,7,8,9,10],"10":[1,2,3,7,8,9],"1000":[2,3,7,8,9,10],"1016":[2,3],"102":3,"104":3,"106":3,"10_21":9,"11":[3,7,8,9],"113":3,"11531935":8,"12":[3,7,8,9],"125":3,"13":[3,7,8,9],"13096556":8,"13151363":8,"132":3,"13206459":8,"13300577":8,"13480":8,"139":3,"14":[3,7,8,9],"141":3,"15":[3,7,8,9],"1500":2,"15260":8,"153":3,"16":[2,3,7,8,9],"17":[3,7,8],"172":3,"175":2,"18":[2,3,8],"187":2,"19":[3,8],"191":3,"1996":[2,3],"1999":[2,3],"1_raw_data_11":9,"1d":[9,10],"2":[1,2,3,5,7,8,9,10],"20":[2,7,8,10],"2000":3,"2006":2,"2018":[2,3],"204":3,"20556472":8,"21":[3,10],"22":3,"221":3,"223":3,"23":3,"24":10,"242":3,"256":3,"26":[3,10],"268":3,"27":3,"275":3,"279":3,"28":[3,9],"289":3,"29":[7,10],"296":3,"298":3,"2_raw_data_11":9,"2d":[7,8,9,10],"3":[1,2,3,5,7,8,9,10],"30":[2,3,7,8,9],"30348":[2,3],"308":3,"313":3,"327":3,"341":3,"346":3,"35":3,"350":2,"36":3,"364":7,"37":[3,10],"373":3,"375":3,"38":3,"385":3,"388":3,"38971":7,"39":[8,10],"392":3,"3954e":9,"3955e":9,"3956e":9,"3asen":7,"4":[2,3,7,8,9,10],"40":3,"400":2,"409":3,"41":[3,10],"411":3,"42":[2,3,10],"420":3,"425":3,"43":3,"44":10,"444":3,"45":[3,10],"450":[2,3,7,8,9],"4554e":9,"4555e":9,"4557e":9,"458":3,"463":3,"470":9,"475":3,"477":3,"48":10,"49":3,"496":3,"498":[7,8],"499":[7,8],"5":[2,3,7,8,9,10],"500":2,"501":7,"502":7,"51":3,"529":3,"53":[3,8,10],"531":8,"532":8,"534":3,"54":3,"55":3,"554":3,"56":10,"561":3,"57":3,"573":3,"58":3,"584":3,"594":3,"597":3,"5nmar":8,"5npie":8,"5nsen":8,"6":[2,3,7,8,9,10],"60":3,"601":8,"602":8,"61":3,"613":3,"61448":9,"61641":9,"62":3,"625":3,"63":10,"632":3,"6411":[2,3],"646":3,"6563":8,"665":3,"67":3,"678":3,"684":3,"69":3,"694":3,"696":3,"698":3,"7":[3,7,8,9],"703":3,"717":3,"73":3,"734":3,"749":3,"756":3,"763":3,"78":3,"782":3,"799":3,"8":[3,7,8,9,10],"801":3,"815":3,"823":3,"834":3,"839":3,"84":3,"85":3,"859":3,"86":3,"864":3,"866":3,"878":3,"884":3,"897":3,"899":3,"9":[2,3,7,8,9,10],"901":8,"902":8,"918":3,"924":3,"937":3,"951":3,"97":3,"984":3,"989":3,"992":3,"995":3,"998":8,"999":[3,8,9],"abstract":2,"case":9,"class":[5,7,8,9,10],"default":[1,2,3,9],"export":[2,3,7,8],"final":[3,10],"float":[2,3,7,8],"function":3,"import":[3,6,7,8,9,10],"int":[1,2,3],"return":[1,2,3,7,8],"static":[2,9],"true":[3,8,9],"while":[7,8,9],A:3,For:[2,3,7,8,9],If:[1,2,3,10],In:[2,3,7,8,9],It:5,No:9,One:[2,3],The:[1,2,3,5,6,7,8,9,10],Then:8,With:[3,9],__getitem__:3,_data:[7,8],_filepath:[7,8],abov:9,accept:[3,5,9],access:[7,8],across:[3,8,9],action:[2,3],actual:[2,3],add:9,adher:5,after:[2,3,6,7,8,9],again:9,all:[1,2,3,8,9],all_beg_t:[3,8,9],all_csv_path:[3,8],all_data:[3,8,9],all_end_t:[3,8,9],all_main_titl:[3,9],all_timestamp:[3,8,9],all_trial_nam:[8,9],alreadi:7,also:[2,3,7,8,9,10],amplitud:[2,3,7,8,9],amplitude_norm:3,amplitudenorm:[0,3,9],an:[2,3,5,7,8,9],ani:2,api:[1,5],append:9,appli:[2,3,7,8,9,10],apply_amplitude_norm:[3,7,8,10],apply_bandpass_filt:[3,7,8,10],apply_dc_offset_remov:[3,7,8,10],apply_end_frame_cutt:[3,7,8,10],apply_full_wave_rectifi:[3,7,8,10],apply_linear_envelop:[3,7,8,10],apply_segment:[3,7,8,10],ar:[1,3,5,7,8,9],argument:2,around:2,arrai:[3,7,8,9,10],assert_input:2,assess:[2,3],assum:[7,8,9,10],author:3,automat:10,avail:5,bandpass:[2,3,7,8,9],bandpass_filt:3,bandpassfilt:[0,3,9],bandwidth:2,baseprocessor:0,basic:[7,8,9],befor:[7,8],beg_idx:2,beg_t:[2,3,10],begin:[2,3],being:9,besid:[7,8],between:3,bf_cutoff_fq_hi:[2,3,7,8,9],bf_cutoff_fq_lo:[2,3,7,8,9],bf_order:[2,3,7,8,9],bicep:[7,8],biom:[2,3],bool:3,both:[1,2,3,5],butterworth:[2,3],c:[3,8,9],c_raw:3,calcul:[8,9],callaghan:2,can:[3,6,7,8,9,10],ch1:3,ch2:3,chang:3,channel1:9,channel2:9,channel3:9,channel4:9,channel5:9,channel6:9,channel7:9,channel8:9,channel:[1,2,3,7,8,9,10],channel_nam:[1,2,3,7,8,9],check:2,circumst:2,code:[3,5],collect:[7,8,9],collect_valu:[7,8],column:[1,7,8,9,10],commun:[2,3],complet:[7,8],concaten:[7,8],concert:[2,3],condit:[2,3],configur:[8,9],consid:1,constrained_layout:1,contain:[3,7,8,9],contamin:2,continu:[7,8],contract:[7,8],control:[1,3],convent:[3,5,9],copi:3,correspond:[1,2,3],creat:[1,9,10],csv:[2,3,7,8],csv_path:3,current:[2,3,9],cut:[2,3,7,8],cutoff:[2,3],cutter:[2,3,9],cwd:[7,8,9],d:[2,3],data:[1,2,3,5],data_filenam:[7,8,9],data_fold:[7,8,9],dataprocessingmanag:[0,5,9],dataset:[7,8],dc:[2,3,7,8,9],dc_offset_remov:3,dcoffsetremov:[0,3,9],deduc:2,def:[7,8],definit:[4,5],delimit:9,demo:5,demonstr:[7,8,9],depend:6,deriv:[7,8],descript:[7,8],design:5,destin:3,detail:1,di:2,dict:[1,3,8,9,10],differ:[2,9],dim:[2,3],dimens:[2,3],direct:[2,3],displai:[2,9],divisor:[2,3],doi:[2,3],dpi:1,drake:2,e:[2,3,7,8,9],each:[3,7,8,9,10],edgecolor:1,editor:[2,3],effect:[2,3],eight:9,electrocardiogram:2,electromyogram:2,electromyographi:[2,3,5],element:[1,3,9],elimin:2,emg:[1,2,3,4,5,9],emg_plot:[1,3],emg_plot_param:[1,3,7,8,9],emgmeasur:[0,7,8,10],emgmeasurementcollect:[0,2,8,9],emgplotparam:[0,3,7,8,9],encourag:5,end:[2,3,7,8,9],end_frame_cutt:3,end_idx:2,end_t:[2,3,10],endframecutt:[0,3,9],ensur:5,envelop:[2,3,7,8,9],equal:[1,2,3],etc:[1,3,7,8,9],european:[2,3],evalu:2,ex1_process:7,ex2_gait:8,ex2_sit:8,ex2_stand:8,exactli:1,examin:9,exampl:[5,7,8,9,10],exce:3,execut:3,experi:7,export_csv:[2,3,7,8,9],extract:[3,7,8,9],f:[2,3,8],facecolor:1,fals:3,femori:[7,8],few:[7,8],fig_kwarg:[1,3,7,8,9],figsiz:[1,3,7,8,9],figur:[1,3,7,8,9],figure_api:1,file:[2,7,8,9],filepath:[2,7,8],filter:[2,3,7,8,9],filtfilt:2,find:3,find_max_amplitude_of_each_channel_across_tri:[3,8],fine:3,first:[2,3,7,8,9],fit:1,five:9,fn:9,follow:9,forearm:9,format:[5,7,8],found:3,four:[7,8],fp:[7,8],frame:[2,3,7,8,9],frameon:1,frequenc:[2,3],fresh:3,from:[2,3,6,7,8,9,10],full:[2,3,7,8,9],full_wave_rectifi:3,fullwaverectifi:[0,3,9],furtuer:[7,8,9],gait:8,gener:10,genfromtxt:9,gestur:9,get:[1,2,3],get_indices_from_timestamp:2,get_param_values_in_str:2,get_timestamp:2,getcwd:[7,8,9],github:5,greater:2,guid:[3,4,5],h:[2,3],ha:3,half:2,hand:9,have:[1,2,3,7,8],header:2,heart:2,here:[7,8],hermen:[2,3],hertz:[2,3],high:[2,3,4,5],higher:[2,3],highest:3,hire:[3,10],hspace:[7,8,9],html:1,http:1,hz:[2,3,7,8,9,10],i:[2,3,8,9],ident:3,ii:[2,3],iii:2,implement:5,includ:[1,6,7,8,9],inclus:2,increment:10,index:[2,3,8,9],indic:[2,9],inform:[7,8,9],initi:[3,7,8,9],inlin:[7,8,9],input:[2,5],instal:5,instanc:[3,7,8,9],integ:[2,3],interest:[2,3,7,8,9],interfac:[3,5,9],intermedi:[3,9],internation:5,internu:[7,8],intramuscular:[2,5],invas:[2,3],irvin:[7,8,9],is_plot_processing_chain:[3,9],isinst:[8,9],item:[7,8],its:[1,2,3],iv:2,j:[2,3],join:[7,8,9],journal:[2,3],k:[3,8],k_for_plot:[3,9],kei:[3,8],keyword:2,kinesiolog:[2,3],known:7,kwarg:2,largest:2,last:[7,8,9],launch:6,le_cutoff_fq:[2,3,7,8,9],le_ord:[2,3,7,8,9],learn:[7,8,9],least:[2,3],len:[3,7,8],length:[1,2,3],less:[1,2],level:[3,4,5],limb:[7,8],line:[7,8],linear:[2,3,7,8,9],linear_envelop:3,linearenvelop:[0,3,9],linewidth:1,list:[1,2,3,8,9,10],load_uci_lower_limb_txt:[7,8],low:[2,3],lower:[2,7,8],lowpass:[2,3],m:[2,3,7,10],machin:[7,8,9],mai:2,main:[1,3],main_titl:[1,3,7],manag:9,matplotlib:[1,3,6,7,8,9],max:3,max_amplitud:[3,7,8,10],maximum:[7,8,9],measur:[4,5],meet:3,merletti:2,method:[2,3,9],mgr:[3,9],might:[7,8],minimum:3,more:[1,2,3],multipl:[2,3,4,5,9,10],muscl:[2,3,7,8],must:3,mvc:[7,8],n_channel:[1,2,3,7,8,9,10],n_col:[1,3,8,9],n_end_fram:[2,3,7,8,9,10],n_row:[1,7,8,9],n_sampl:[1,2,3,7,8,9,10],name:[1,2,3,7,8,9],ndarrai:[1,2,3,7,8,9,10],need:[7,8],needl:2,non:[2,3,7,8],none:[1,2,3,9],normal:[2,3,7,8,9],note:[2,3,7,8,9],now:[7,8,9],np:[3,7,8,9,10],number:[1,2,3],numpi:[3,6,7,8,9,10],nyquist:3,obtain:7,off:[2,3],offset:[2,3,7,8,9],one:[1,2,3,7,8,9,10],onli:[3,9,10],open:[7,8],oper:[8,9],order:[2,3,5],org:1,organ:[8,9,10],origin:9,os:[7,8,9],other:[7,8,9],otherwis:[1,2,3],p:2,packag:[5,6,9],pad:2,padlen:2,paramet:[1,2,3,5,7,8,9],params_in_str:2,parent:[7,8,9],part:9,particular:[7,8,9],pass:2,path:[3,7,8,9],pathlib:[7,8,9],pdf:[3,10],pep:[3,6,7,8,9,10],perform:9,pip:6,pipelin:5,placement:2,plot:[0,3,5,7,8,9,10],plot_emg:0,png:[3,10],posit:[2,3],possess:[3,9],postur:7,predefin:9,prefer:3,prepar:3,present:10,print:[7,8,9],process:[1,2,3,4,5,10],process_al:[3,9],processor:[0,3,5,9],program:[2,3],project:[2,3],provid:[5,9,10],purpos:[2,7,8,9],pyemgpipelin:[0,6,7,8,9,10],pypi:6,quick:5,r:2,rang:[7,8,9],rate:[2,3,7,8,9,10],raw:[3,7,8,9],readlin:[7,8],recommend:9,record:2,rectif:[7,8],rectifi:[2,3,9],rectu:[7,8],reduc:2,ref:[2,3],refer:[2,3],relat:[3,9],remov:[2,3,7,8,9],repeatedli:3,repo_fold:[7,8,9],report:[2,3],repositori:[5,6,7,8,9],repres:[7,8,9,10],requir:2,reset:9,reshap:[7,8],respons:[7,8,9],result:[2,3,9,10],row:[1,7,8],run:[2,3,9],s1050:[2,3],s:8,same:[2,3,8,9],sampl:[2,3,7,8,9,10],sample_r:[7,8,9],satisfi:3,save:[3,9],scalar:[2,3],scenario:2,scipi:[2,6],see:[1,2,3],segment:[0,3,7,8,9],segmenter_all_beg_t:3,segmenter_all_end_t:3,semitendinosu:[7,8],seniam:[2,3],sequenc:9,seri:9,set:[1,3,7,8,9],set_amplitude_norm:[3,9],set_bandpass_filt:[3,9],set_data_and_param:[3,9],set_dc_offset_remov:3,set_end_frame_cutt:3,set_full_wave_rectifi:3,set_linear_envelop:3,set_segment:[3,9],seven:[3,7,8,10],shape:[1,2,3,7,8,9,10],should:[1,2,3,7,8,9,10],show:[3,7,8,9],show_current_processes_and_related_param:[3,9],shown:[1,2,3,9],signal:[1,2,3,5,7,8,9,10],simpli:10,sinc:9,singl:[3,4,5,8,9],sit:[7,8],skip_head:9,smallest:[1,2],some:2,sourc:[3,5,10],specif:2,split:[7,8],stabl:1,stand:[2,8],standard:[2,3,7,8],start:[2,5,7,8],stegeman:[2,3],step:[2,3,5,7,8,9,10],store:[7,8,9,10],str:[1,2,3],strictli:2,subject:[7,8,9],subplotpar:[1,3,7,8,9],subplotparam:[3,7,8,9],suggest:[2,3],suitabl:5,summari:0,surfac:[2,3,5],t:[3,7,8,9],techniqu:2,term:5,than:[1,2],thei:[1,9],them:9,theorem:3,thi:[1,2,3,5,6,7,8,9],those:[3,5],three:[1,8],tight_layout:1,time:[1,2,3,7,8,9],timestamp:[1,2,3,7,8,9,10],titl:[1,3],togeth:1,top:[3,8,9],torino:2,trial:[3,7,8,9,10],trial_nam:7,trunk:2,tupl:2,twice:3,two:[1,2,3,9],txt:[7,8,9],type:[1,2,3,8,9],uc:[7,8,9],uci_gestur:9,uci_lower_limb:[7,8],up:[7,8,9],updat:9,us:[1,2,3,5,7,8,9,10],usag:[7,8,9],user:5,valid:3,valu:[1,2,3,5,8],vastu:[7,8],vector:2,via:[7,8],visual:[7,8],voluntari:[7,8],wa:[7,8,9],wai:[3,9],wave:[2,3,7,8,9],we:[7,8,9],when:[1,2,3,7,9],whenev:[7,8],where:[2,3,7,8,9,10],whether:3,which:[2,3,7,8,9],whole:[7,8,9],wire:3,within:5,won:3,wrapper:[0,5,7,8,9,10],wspace:[7,8,9],x:[1,2,3],x_process:2,x_shape:2,yet:9},titles:["API","pyemgpipeline.plots","pyemgpipeline.processors","pyemgpipeline.wrappers","Examples","pyemgpipeline Documentation","Installation","Single EMG measurement definition and processing","Multiple EMG measurement definition and processing","High-level, guided processing","Quick Start"],titleterms:{"class":[1,2,3],"function":1,amplitudenorm:2,api:0,bandpassfilt:2,base:2,baseprocessor:2,code:10,content:5,data:[7,8,9,10],dataprocessingmanag:3,dcoffsetremov:2,definit:[7,8],demo:10,document:5,emg:[7,8],emgmeasur:3,emgmeasurementcollect:3,emgplotparam:1,endframecutt:2,exampl:[3,4],format:10,fullwaverectifi:2,guid:9,high:9,input:10,instal:6,level:9,linearenvelop:2,measur:[7,8],multipl:8,plot:1,plot_emg:1,prepar:[7,8,9],process:[7,8,9],processor:2,pyemgpipelin:[1,2,3,5],quick:10,segment:2,singl:7,start:10,summari:[1,2,3],wrapper:3}})