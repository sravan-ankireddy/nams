clear all;
close all;

set_env();

randn('state',1234);
rand('state',12345);

%--------- parameters ---------
OFDM_params = set_OFDM_params();
sys_params = set_system_params();

%---------- OFDM 1 ---------------
[dt_bsb1 df_bsb1] = generate_OFDM_sym(OFDM_params);
[rf1 rf1_cpx]= bsb_to_rf(dt_bsb1,sys_params.intp_filter,sys_params.OSR,sys_params.fc(1),sys_params.fs);
 
%---------- OFDM 2 ---------------
[dt_bsb2 df_bsb2] = generate_OFDM_sym(OFDM_params);
[rf2 rf2_cpx]= bsb_to_rf(dt_bsb2,sys_params.intp_filter,sys_params.OSR,sys_params.fc(2),sys_params.fs);

rf1 = normalize(rf1)*0.8;
rf2 = normalize(rf2)*0.8;
rf1_cpx = normalize(rf1_cpx)*0.8;
rf2_cpx = normalize(rf2_cpx)*0.8;


rf = rf1 + rf2;
h = [1 ];
rf = conv(h,rf);


%-------- echo channel estimation ------
ch_num = 1;
isbsb = 0;
[hf1 ht1]= chest_wrapper(dt_bsb1,rf,OFDM_params,sys_params,ch_num,isbsb);


ch_num = 2;
isbsb = 0;
[hf2 ht2]= chest_wrapper(dt_bsb2,rf,OFDM_params,sys_params,ch_num,isbsb);


show_data(to_pow_dB(hf1));
show_data(to_pow_dB(ht1));

%-------  convert to real baseband signal ------
hf = [hf1, hf2];
hf_bsb = [hf hf(end) fliplr(conj(hf(2:end)))]; 
%hf_bsb = hf;

%-------- simulate optical channel -----
d_opt = 0.9*ifft(hf_bsb);
% len = length(d_opt);
% d_opt = conv(d_opt,[0.8 0.1]);
% d_opt = d_opt(1:len);
%---------------------------------------

echos_f = cell(1,1);
echos_t = cell(1,1);

echos_f{1} = hf_bsb;   
echos_t{1} = ifft(echos_f{1});  

figure; plot(abs(echos_t{1})); legend('echo channel in time')


%--------- detect coarse delay ---------
params.peak_th = 1;
params.idx_neighbors_th = 100;

idx_peaks = detect_coarse_delay(echos_t{1},params)

%--------- estimate coefficient ---------
echos_delay = 0;
params.weights  = 1;%[1 1 1 1 1 1];% weights for different cable taps
params.NFFT     = length(echos_f{1});
params.fs       = 819.2e6;
params.peaks    = idx_peaks;
params.N_taps   = 4;
params.tap_res  = 1;
params.tap_name = {'4 taps cable plant,'};%{'29dBw','29dB','26dB','20dB','14dB','10dB'};
params.N_ite    = 1;%4 % number of iterations for attenuation optimation
params.flag_show_cancel = 1;
params.flag_show_cancel_total = 1;

%[w att  total_mse] = estimate_FIR_coe_sim(d_opt,echos_t,echos_f,echos_delay,params);


%---------- for test only ----------
if 0
params.filename = 'data\tx\OFDM_192M_QPSK_ch1_bsb.h';
params.convert_to_int = 1;
params.QI = 1;
params.QF = 15; 
params.scale = 1;
write_to_h_file(params,dt_bsb1);

params.filename = 'data\tx\OFDM_192M_QPSK_ch2_bsb.h';
params.convert_to_int = 1;
params.QI = 1;
params.QF = 15; 
params.scale = 1;
write_to_h_file(params,dt_bsb2);

params.filename = 'data\rx\OFDM_192M_QPSK_rf_sim.h';
params.convert_to_int = 1;
params.QI = 1;
params.QF = 15; 
params.scale = 1;
write_to_h_file(params,rf);

params.filename = 'data\rx\OFDM_192M_QPSK_opt_sim.h';
params.convert_to_int = 1;
params.QI = 1;
params.QF = 15; 
params.scale = 1;
write_to_h_file(params,hf_bsb);

params.filename = 'data\tx\OFDM_192M_QPSK_ch1_bsb.h';
params.convert_to_frac = 1;
params.QF = 15; 
params.iscomplex = 1;
dr = read_from_h_file(params);

end