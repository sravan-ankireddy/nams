% @read echo and optical data from VNA and estimate coefficient 
% @simplified version to reduce the memeory and computational complexity requirement 
% @further simplified version for pcb


function [taps_t,taps_f,echo,desired_peaks] = load_echo_taps()

%!!!!!!!!!! remove this for normal operation !!!!!!!!!!!!
flag_set_peak_locations = 0;
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

flag_filter_type       = 1;% 0:tree filter, 1:cascade filter   
flag_filter_response   = 1;
flag_program_pic       = 1; 
flag_constrained       = 1; 
flag_load_measure      = 1;
flag_load_ADC_capture  = 0;

flag_predict_TTD       = 0;
flag_filter_only       = 0;

small_w_size = 128;%64;      % final number of samples in the frequency domain
w_size  = small_w_size*8; % initial time domain window length   , should not be too large such that it also contain previous peaks
coe_max = 1;    

% time gating window, only cancel external taps 
% idx_gate_start = 480;
% idx_gate_stop = 510;
idx_gate_start = 1;
idx_gate_stop = 32768;



%---------------------------------------------------------
%      down-sampled frequency domain window function 
%----------------------------------------------------------
BW_total   = 819.2e6;

%------- base function BW ------
% BW_base = [190e6 190e6 190e6];
% f_start_base = [108e6 300e6  492e6];

BW_base      = 96e6*6;
f_start_base = [108e6];
delta_f_base = 50e3;

NFFT_base = 16384;   % half-band fft
[f_window_base_bsb, f_window_base_pass,f_idx_base_bsb,f_idx_base_pass] = cal_f_index_v2(f_start_base,BW_base,delta_f_base,NFFT_base);

%---------- signal BW ---------
BW      = 96e6*6;
%BW      = 84e6;
f_start = [108e6];
delta_f = BW_total/(w_size/2);

NFFT = w_size/2;     % half-band fft
[f_window_sig_bsb, f_window_sig_pass,f_idx_sig_bsb,f_idx_sig_pass] = cal_f_index_v2(f_start,BW,delta_f,NFFT);

%----------- determine the down sampling rate --------
BW_signal               = (f_start(end)+ BW(end) - f_start(1)); % the total bandwidth of passband signal

if BW_signal <= 102.4e6
    OSR = 2^3;
elseif BW_signal <= 204.8e6
    OSR = 2^2;
elseif BW_signal <= 409.6e6
    OSR = 2^1;
else
    OSR = 2^0;
end

%------- setup tcp ---------
if flag_program_pic == 1
    % setup pcb board 
    [tcp_obj,tcp_const] = setup_pic_tcp();
    pic_pcb_setup(tcp_obj,tcp_const);    
end

DAC_map = cell(1,1);
DAC_map{1} = [1 2 3 4 5 6 7];

%-------------------------------------------------
%                 load echo 
%-------------------------------------------------
if flag_load_ADC_capture == 1
    filename = '..\data\20180410_ADC_diff\echo.mat';            
    
    temp = load(filename);
    hf_bsb = temp.df;  
else    
    filename = '..\data\20180618_cc3\echo.mat';     
    
    temp = load(filename);
    ht_temp = temp.dt; 
    hf_temp = temp.df;
end

% time gating
ht_bsb = zeros(1,length(ht_temp));
ht_bsb(idx_gate_start:idx_gate_stop) = ht_temp(idx_gate_start:idx_gate_stop);
hf_bsb = fft(ht_bsb);

figure; plot(to_pow_dB(ht_bsb)); legend('echo in t');

%--------------------------------------------------
%              detect peak location 
%--------------------------------------------------
% echo coarse delay 
if flag_load_ADC_capture == 1
   params.peak_th = 0.01;
else
   params.peak_th = 0.04;
end

params.idx_neighbors_th = 10;%30; 
params.small_width_th = 0;% not used now
desired_peaks = detect_coarse_delay(ht_bsb,params)

if flag_set_peak_locations == 1
    %desired_peaks = [96 490] % 819.2 MHz    
    desired_peaks = [51] % 819.2 MHz        
end

%--------------------------------------------------
%          window and filtering the echo
%--------------------------------------------------
idx_peak_small_w = 0; % any random number for echo

params.peaks            = desired_peaks;
params.w_size           = w_size;
params.flag_filter_only = flag_filter_only;
params.small_w_size     = small_w_size;
params.flag_is_echo     = 1; 
params.idx_peak_small_w = idx_peak_small_w;
params.f_window_sig     = f_window_sig_bsb;
params.f_window_base    = f_window_base_bsb;
params.flag_skip_prepare = 0;

%[ht_bsb_w,idx_peak_small_w] = window_filter_down(ht_bsb,params);
[ht_bsb_w,idx_peak_small_w] = window_filter_down_v3(hf_bsb,params);

%-------------------------------------------------
%                load taps response 
%-------------------------------------------------
if flag_load_ADC_capture == 1 % load ADC measured data
    
    N_branches = 2;
    taps_table = cell(1,N_branches);
    taps_table{1} = [1,2,3,4,5,6,7,8];
    taps_table{2} = [10,11,12,13,15];
    
    N_taps = zeros(1,N_branches);
    N_taps(1) = length(taps_table{1});
    N_taps(2) = length(taps_table{2});
    
    params.taps_table           = taps_table; % taps index table
    params.N_branches           = N_branches;
    params.NFFT                 = 32768;
    params.f_window_bsb         = f_window_bsb;
    params.flag_filter_response = flag_filter_response;
    params.flag_predict_TTD     = flag_predict_TTD;
    
    opt_taps_t = read_ADC_all_taps(params);
    
else% load VNA measured data
    
    N_branches = 1;
    taps_table = cell(1,N_branches);
    taps_table{1} = DAC_map{1};
    %taps_table{2} = [10,11,12,13,15];
    
    N_taps = zeros(1,N_branches);
    N_taps(1) = length(taps_table{1});
    %N_taps(2) = length(taps_table{2});
    
    params.taps_table       = taps_table; % taps index table
    params.N_branches       = N_branches;
    params.NFFT             = w_size;
    %params.OSR              = OSR;
    %params.fc               = f_start(1) - 3*delta_f;    % bring down the first start frequency to zero, not center frequency to zero, make sure the lowest frequency is above DC
    %params.fs               = 1.6384e9;      % original sampling rate
    
    params.flag_predict_TTD = flag_predict_TTD;
    params.peaks            = desired_peaks; % peaks before down-sampling
    params.w_size           = w_size;          % need 16 samples for 96 MHz and downsample by 8
    params.flag_filter_only = flag_filter_only;
    params.small_w_size     = small_w_size;
    params.idx_peak_small_w = idx_peak_small_w;
    params.f_window_sig     = f_window_sig_bsb;
    params.f_window_base    = f_window_base_bsb;
    
    [opt_taps_t] = read_VNA_response_windowing(params);
    
end

for idx_branches = 1:N_branches  
    NFFT = small_w_size;
    taps_t = zeros(NFFT,N_taps(idx_branches));
    taps_f = zeros(NFFT,N_taps(idx_branches));
    for idx_taps = 1:N_taps(idx_branches)
        temp  = opt_taps_t(idx_branches,idx_taps,:);
        temp = reshape(temp,size(temp,3),1);
        taps_t(:,idx_taps) = temp;
        taps_f(:,idx_taps) = fft(taps_t(:,idx_taps));
    end
end
echo = ht_bsb_w;
