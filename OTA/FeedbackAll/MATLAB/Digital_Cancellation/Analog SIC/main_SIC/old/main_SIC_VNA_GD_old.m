% windowing of individual branch response and optimize each branch coefficient individually  
% integrated version of wideband and any number of channels 
% program coefficient -> measure residual -> program delta coefficient -> measure residual

clear all;
close all;
clc;

set_env();


f_start = 108e6;
f_stop  = 684e6;
delta_f = 50e3;
NFFT    = 16384;
[f_window_bsb, f_window_pass] = cal_f_index(f_start,f_stop,delta_f,NFFT);

%--------- parameters ---------
N_ite = 1;
N_channel = 1;

if N_channel == 1
    % 1 wideband channel
    OFDM_params = set_OFDM_params_819M();
    sys_params  = set_system_params_819M();
else
    % 2 narrow band channel
    OFDM_params = set_OFDM_params();
    sys_params = set_system_params();
end


flag_load_data            = 1; % 0 to use tcp
flag_program_pic          = 0; % program PIC 
flag_load_tap_vna         = 0; % load vna tap response
flag_delay_taps_digitally = 1; % delay taps digitally


N_branches     = 2;
N_taps         = 8;
tap_spacing    = 0.5e-9;    
flag_keep_opt_delay = 0;
idx_taps_ref        = 3; % index for reference taps

DAC_map = cell(1,2);
DAC_map{1} = [5 2 4 9 0 6 7 1];
DAC_map{2} = [15 10 14 12 13 5 8];

%--------- load tx signal --------------
dt_bsb = cell(1,N_channel);
if N_channel == 1 % 1 wideband channel
    params.filename = 'data\tx\OFDM_819.2M_QPSK_cpx.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
else % 2 narrow band channel
    params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch1_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
    params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch2_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{2} = read_from_h_file(params);
end

%--------- load optical taps response ----------
code_max    = 2^16;
N_codes     = 128;
code_table  = [1:floor(code_max/N_codes):code_max];

if flag_load_tap_vna ~= 1 % load measured taps 
    if flag_delay_taps_digitally == 1 % load one tap and delay digitally
        idx_code = 1;% code index for reference taps
   
        params.N_branches   = N_branches;
        params.N_taps       = N_taps ;
        params.tap_spacing  = tap_spacing;
        params.idx_taps_ref = idx_taps_ref;
        params.NFFT         = 4096*8; %length of tap response
        params.filename     = sprintf('data\\ch_wide_v2\\optical_ch_tap%d_code%d.mat',idx_taps_ref,int32(code_table(idx_code)));
        params.f_window     = OFDM_params.f_window;
        params.R            = 32; % interpolation factor
        params.phase        = 1; % phase of taps
        
        [opt_taps_t, real_spacing]= load_and_delay_taps(params,sys_params);
        
    else % load all measured taps
        params.idx_ref_code = 1;% reference code for coe 1
        params.taps_table   = [1 2 3 4 5 6 7 8]; % taps index table
        params.N_branches   = N_branches;
        params.NFFT         = 4096*8;
        params.code_table   = code_table;
        params.f_window     = ones(1,params.NFFT);
        
        N_taps = length(params.taps_table);
        
        opt_taps_t = load_all_taps(params,sys_params);
    end
else  % load VNA measured taps, need to normailze amp to the same as measured taps
    
%     N_branches = 1;
%     NFFT  = 32768;
%     
%     [taps_f,taps_t,f] = read_VNA_complex_all_taps(f_window_bsb);
%         
%     opt_taps_t = zeros(N_branches,N_taps,NFFT);
%     for idx_branches = 1:N_branches
%         opt_taps_t(idx_branches,:,:) = taps_t;
%     end
   
    % -------- internal --------
    flag_int = 1;
    [taps_f,taps_t,f] = read_VNA_complex_all_taps(f_window_bsb,flag_int);
    
    N_taps     = size(taps_t,1);
    opt_taps_t = zeros(N_branches,N_taps,size(taps_t,2));
    
    idx_branches = 1;
    for idx_taps = 1:N_taps
        opt_taps_t(idx_branches,idx_taps,:) = taps_t(idx_taps,:);
    end
    
    %--------- external ----------
    flag_int = 0;
    [taps_f,taps_t,f] = read_VNA_complex_all_taps(f_window_bsb,flag_int);
    
    N_taps     = size(taps_t,1);
    
    idx_branches = 2;
    for idx_taps = 1:N_taps
        opt_taps_t(idx_branches,idx_taps,:) = taps_t(idx_taps,:);
    end
    
    % normalization to the ADC measured power
    params.idx_ref_code = 1;% reference code for coe 1
    params.taps_table   = [1 2 3 4 5 6 7 8]; % taps index table
    params.N_branches   = N_branches;
    params.NFFT         = 4096*8;
    params.code_table   = code_table;
    params.f_window     = ones(1,params.NFFT);
    
    opt_taps_t = normalize_taps(opt_taps_t,params,sys_params);
end

for idx_ite = 1:N_ite

%--------- load rx signal --------------
if flag_load_data == 1   
    if N_channel == 1 % 1 wideband channel
        params.filename = 'data\rx\OFDM_819M_QPSK_rf_rx_1p6G.h';
        params.convert_to_frac = 0;%1;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    else % 2 narrow band channel
        params.filename = 'data\rx\OFDM_2ch_192M_QPSK_rf_rx_1p6G_tilt.h';
        params.convert_to_frac = 1;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    end
else           
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.180',7);
    %--- PIC command -------
    channel = 0:7;
    pic_code = 2^16-1;
    reset_all_pic(tcp_obj,tcp_const,channel,pic_code);
    pause(0.5)
    %--- capture command ------    
    ADC_channel = 0;        
    N_ADC_samples = 150e3;
    send_ADC_capture_cmd(tcp_obj,tcp_const,N_ADC_samples,0);
    
    rcv_data = fread(tcp_obj, N_ADC_samples*2 + header_size,'int16');
    dr_rf = rcv_data(header_size/2:end).';
    dr_rf = dr_rf./2^15;
end

NFFT = 16384;
fs   = 1.6384e9;
show_freq_time(dr_rf,fs,NFFT);

%-------- echo channel estimation ------
isbsb = 0;
if N_channel == 1
  OFDM_params.th = 5e5;
else
  OFDM_params.th = 5e4;    
end
[ht_bsb,hf_bsb] = multi_chest_wrapper(dt_bsb,dr_rf,OFDM_params,sys_params,isbsb,N_channel,f_window_bsb);


%------- find coarse delay --------------
% echo coarse delay 
params.peak_th = 0.02;%8e-3;
params.idx_neighbors_th = 80;%30;
desired_peaks = detect_coarse_delay(ht_bsb,params)

if length(desired_peaks) ~= N_branches % error checking
    disp('Warning! The number of echo peaks does not equal to the number of branches !'); 
end

% branches delay 
params.peak_th = 0.1;
params.idx_neighbors_th = 80;%30;
branch_peaks = zeros(1,N_branches);
for idx_branches = 1:N_branches
      if N_taps == 1
        idx_taps = 1; 
      else
        idx_taps = floor(N_taps/2); % use middle taps as branches delay 
      end
      branch_peaks(idx_branches) = detect_coarse_delay(opt_taps_t(idx_branches,idx_taps,:),params);
end


% ------ output required delay value for TTD -----------  
delay             = desired_peaks - branch_peaks;
%delay = 0;
fs = 819.2e6;
delay_ns = delay/fs*1e9;
str = sprintf('required branch delay = %f ns,',delay_ns);
disp(str);
%-------------------------------------------------------    


w_size  = 128%256; % size of windowing
NFFT    = w_size;
echos_f = cell(1,1); 
echos_t = cell(1,1);


temp = load('data\code_to_coe\vna_code_to_coe_N128','table_norm');
code_to_coe_table = temp.table_norm;
coe     = cell(1,N_branches);
code_max    = 2^16;
N_codes     = 128;
code_table  = [1:floor(code_max/N_codes):code_max];


N_taps = [7 8];
for idx_branches = 1:N_branches
  coe{idx_branches} = zeros(1,N_taps(idx_branches));
end

for idx_branches = 1:N_branches  
    
    % -------- windowing --------
    % echo windowing    
    w_start = desired_peaks(idx_branches) - w_size/2; 
    if w_start < 1
        w_start = 1;
    end
    
    ht_bsb_w = ht_bsb(w_start:w_start + w_size-1 );
    hf_bsb_w = fft(ht_bsb_w,NFFT);
    desired_peaks(idx_branches) = desired_peaks(idx_branches) - w_start + 1;
    
    echos_t{1} = ht_bsb_w;
    echos_f{1} = hf_bsb_w;
    
    %optical windowing    
    opt_taps_win_t = zeros(1,N_taps,w_size);
    for idx = 1:N_taps
      opt_taps_win_t(1,idx,:) = opt_taps_t(idx_branches,idx,w_start:w_start + w_size-1 );
    end  
    branch_peaks(idx_branches) = branch_peaks(idx_branches) - w_start + 1;
    
    %---------- delay all branches according to coarse delay requirement ---------
    params.N_taps     = N_taps;
    params.N_branches = 1;
    params.NFFT       = w_size;    
    if flag_keep_opt_delay == 1
        delay             = 0;
    else
        delay             = desired_peaks(idx_branches) - branch_peaks(idx_branches);
    end    
    
    [taps_t, taps_f]  = delay_taps(params,delay,opt_taps_win_t);
    
    %----- show all taps ------
    taps_cell = cell(1,size(taps_t,1));
    for idx = 1:size(taps_t,1)
        taps_cell{idx} = real(taps_t(idx,:));
        %taps_cell{idx} = to_pow_dB(taps_t(idx,:));
        
    end    
    show_data_para(taps_cell,{'1','2','3','5','6','7','8','9'});    
    %------------------------
    
    % delay in ns
    fs = 819.2e6;
    delay_ns = delay/fs;
    
    %--------- estimate coefficient ---------
%     echos_delay = 0;
%     params.NFFT     = length(echos_f{1});
%     params.f_idx    = 1:NFFT; % frequency of interest    
%     params.weights  = 1;%[1 1 1 1 1 1];% weights for different cable taps
%     params.fs       = 819.2e6;
%     params.peaks    = 1; % not used now, coarse delay is hardwired
%     params.N_taps   = N_taps;
%     params.tap_res  = 1; % not used now, taps response is measured directly
%     params.tap_name = {['branches = ' num2str(idx_branches) ] };%{'29dBw','29dB','26dB','20dB','14dB','10dB'};
%     params.N_ite    = 1;%4 % number of iterations for attenuation optimation
%     params.flag_show_cancel = 1;
%     params.flag_show_cancel_total = 0;
%     
%     [coe{idx_branches}, att,  total_mse] = estimate_FIR_coe(taps_t,taps_f,echos_t,echos_f,echos_delay,params);     
   
    %---- gradient --------    
    temp =  echos_f{1};
    gradient = taps_f.'*temp(:,1);
    
    step = 0.01;
    coe(idx_branches) = coe{idx_branches} + step*gradient;
    %----------------------
    
    code = find_coe_to_code(real(abs(coe{idx_branches})),code_table,code_to_coe_table);
    coe{idx_branches}
    uint32(code)
    code_v = 5/2^16*code
    
    if flag_program_pic == 1
      program_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code)
    end
  
end

end% iteration

