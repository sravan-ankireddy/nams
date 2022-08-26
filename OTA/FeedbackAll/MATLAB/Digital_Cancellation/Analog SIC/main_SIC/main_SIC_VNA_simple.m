% read echo and optical data from VNA and estimate coefficient 

% simplified version to reduce the memeory and computational complexity
% requirement 

clear all;
close all;
clc;

set_env();

%!!!!!!!!!! remove this for normal operation !!!!!!!!!!!!
flag_set_peak_locations = 1;

flag_normalize_coe_table = 0;
ref_code = 77; % max = 2.594

if flag_normalize_coe_table == 1
    coe_max = 2.5;
else
    coe_max = 1;    
end

flag_swap_tap3_tap8 = 1;
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
flag_filter_type       = 0;% 0:tree filter, 1:cascade filter   

flag_filter_response   = 1;
flag_keep_opt_delay    = 1; % should be 1 for pic programming 
flag_program_pic       = 0; 
flag_constrained       = 1; 
flag_load_measure      = 1;
flag_load_ADC_capture  = 0;
flag_table_down_sample = 0;
flag_swap_tap_table    = 0;

flag_predict_TTD       = 0;
flag_filter_only       = 0;%

small_w_size = 64;      % final number of samples in the frequency domain
w_size  = small_w_size*8; % initial time domain window length   , should not be too large such that it also contain previous peaks

% time gating window, only cancel external taps 
% idx_gate_start = 480;
% idx_gate_stop = 510;
idx_gate_start = 1;
idx_gate_stop = 32768;



%------- for digital delay ----------
OFDM_params_819M = set_OFDM_params_819M();
sys_params_819M  = set_system_params_819M();

OFDM_params = set_OFDM_params_819M();
sys_params  = set_system_params_819M();
%-------------------------------------

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
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.157',7);
    
    DAC_map = cell(1,2);
    %DAC_map{1} = [5 2 4 3 0 6 7 1];
    %             1 2 3 4 5 6 7 8    
    
    if flag_swap_tap_table == 1
        DAC_map{2} = [5 2 1 3 0 6 7 4];
        %             1 2 3 4 5 6 7 8        
        DAC_map{1} = [10 14 12 13 8];        
    else
        % new mapping 20180426  , don't forget to change the coe to vol table
        DAC_map{1} = [5 2 1 3 0 6 7 4];
        %             1 2 3 4 5 6 7 8        
        DAC_map{2} = [10 14 12 13 8];
    end
end

%-------------------------------------------------
%                 load echo 
%-------------------------------------------------
if flag_load_ADC_capture == 1
    filename = '..\data\20180410_ADC_diff\echo.mat';            
    
    temp = load(filename);
    hf_bsb = temp.df;  
else    
    %filename = '..\VNA_CM\data\20180516\echo.mat';            
    %filename = '..\VNA_RS\data\20180313_echo_taps\echo.mat';            
    filename = '..\VNA_RS\data\20180503_external\echo.mat';     
    
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
   params.peak_th = 0.06;
end

params.idx_neighbors_th = 10;%30; 
params.small_width_th = 0;% not used now
desired_peaks = detect_coarse_delay(ht_bsb,params)

if flag_set_peak_locations == 1
    %desired_peaks = [102 497] % 96 MHz
    %desired_peaks = [13 62] % 96 MHz    
    desired_peaks = [96 490] % 819.2 MHz    
end

%--------------------------------------------------
%          window and filtering the echo
%--------------------------------------------------
fc  = f_start(1) - 3*delta_f;    % bring down the first start frequency to zero, not center frequency to zero, make sure the lowest frequency is above DC
fs  = 1.6384e9;                  % original sampling rate
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
if flag_load_measure == 1
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
        
        N_branches = 2;
        taps_table = cell(1,N_branches);
        if flag_swap_tap_table == 1
            taps_table{2} = [1,2,3,4,5,6,7,8];
            taps_table{1} = [10,11,12,13,15];        
        else
            taps_table{1} = [1,2,3,4,5,6,7,8];
            taps_table{2} = [10,11,12,13,15];
        end
        
        N_taps = zeros(1,N_branches);
        N_taps(1) = length(taps_table{1});
        N_taps(2) = length(taps_table{2});
        
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
else
    N_branches = 1;
    N_taps = 16;

    idx_taps_ref = 1;
    idx_code     = 1;% code index for reference taps
    tap_spacing  = 0.5e-9;    
    
    params.N_branches   = N_branches;
    params.N_taps       = N_taps ;
    params.tap_spacing  = tap_spacing;
    params.idx_taps_ref = idx_taps_ref;
    params.NFFT         = 4096*8; %length of tap response
    params.filename     = sprintf('..\\VNA_RS\\data\\taps\\vna_RS_ch_%d_code_1.mat',idx_taps_ref);
    params.f_window     = OFDM_params.f_window;
    params.R            = 32; % interpolation factor
    params.phase        = 1; % phase of taps
    
    [opt_taps_t, real_spacing]= load_and_delay_taps(params,sys_params_819M);    
end


%-------------------------------------------------
%          load coe to voltage table
%-------------------------------------------------
coe     = cell(1,N_branches);
code_max    = 2^16;
N_codes     = 128;
code_table  = [1:floor(code_max/N_codes):code_max];

temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128','table_norm');
if flag_normalize_coe_table == 1
   temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
end

%------------- temp --------------
if flag_swap_tap3_tap8 == 1
   temp2 = temp.table_norm{3};
   temp.table_norm{3} = temp.table_norm{8};
   temp.table_norm{8} = temp2;   
end
%----------------------------------

if flag_table_down_sample == 1 
    % ----------------------------------
    %           down sample
    % ----------------------------------
    R = 8; %8 is the maximum downsampling rate without compromising the performance
    table_down = cell(1,size(temp.table_norm,2));
    for idx = 1:size(temp.table_norm,2)
        temp_down = temp.table_norm{idx};
        table_down{idx} = temp_down(1:R:end);
    end
    code_to_coe_table{1} = table_down;    
    code_table = code_table(1:R:end);
else
    if flag_swap_tap_table == 1
      code_to_coe_table{2} = temp.table_norm;    
    else
      code_to_coe_table{1} = temp.table_norm;
    end
end

temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128_external','table_norm');
% if flag_normalize_coe_table == 1
%    temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
% end
if flag_table_down_sample == 1
    % ----------------------------------
    %           down sample
    % ----------------------------------
    table_down = cell(1,size(temp.table_norm,2));
    for idx = 1:size(temp.table_norm,2)
        temp_down = temp.table_norm{idx};
        table_down{idx} = temp_down(1:R:end);
    end
    code_to_coe_table{2} = table_down;       
else
    if flag_swap_tap_table == 1
      code_to_coe_table{1} = temp.table_norm;    
    else
      code_to_coe_table{2} = temp.table_norm;
    end
end

%-------------------------------------------------
%                  optimization
%-------------------------------------------------
if flag_filter_only == 1
  NFFT    = w_size;    
else
  NFFT    = small_w_size;
end

echos_f = cell(1,1);
echos_t = cell(1,1);
for idx_branches = 1:N_branches  
    idx_branches
    
    echos_t{1} = ht_bsb_w(idx_branches,:);
    echos_f{1} = fft(ht_bsb_w(idx_branches,:));
    
    taps_t = zeros(N_taps(idx_branches),NFFT);
    taps_f = zeros(NFFT,N_taps(idx_branches));
    for idx_taps = 1:N_taps(idx_branches)
        taps_t(idx_taps,:) = opt_taps_t(idx_branches,idx_taps,:);    
        taps_f(:,idx_taps) = fft(taps_t(idx_taps,:)).';

        %---------- test only ----------
        %taps_f(:,idx_taps) = taps_t(idx_taps,:).';        
        %--------------------------------        
    end        
    
    
    %--------- time gating of taps -----------
    %[taps_t, taps_f] = time_gating(taps_t,N_gate);
    
    %----- show all taps ------
    tapst_cell = cell(1,size(taps_t,1));
    taps_amp_cell = cell(1,size(taps_t,1));
    taps_phase_cell = cell(1,size(taps_t,1));
    
    for idx = 1:size(taps_t,1)
        tapst_cell{idx} = (taps_t(idx,:));
        taps_amp_cell{idx} = to_pow_dB(fft(taps_t(idx,:)));        
        taps_phase_cell{idx} = phase(fft(taps_t(idx,:)));        
    end    
    show_data_para(tapst_cell,{'1','2','3','4','5','6','7','8'});    
    %show_data_para(taps_amp_cell,{'1','2','3','4','5','6','7','8'});    
    %show_data_para(taps_phase_cell,{'1','2','3','4','5','6','7','8'});        
    %------------------------
        
    %--------- estimate coefficient ---------
    echos_delay = 0;
    
    if flag_constrained == 0 %unconstrained 
        params.weights  = 1;%[1 1 1 1 1 1];% weights for different cable taps
        params.NFFT     = length(echos_f{1});
        params.f_idx    = 1:params.NFFT;
        params.fs       = 1638.4e6;
        params.peaks    = 1; % not used now, coarse delay is hardwired
        params.N_taps   = N_taps(idx_branches);
        params.tap_res  = 1; % not used now, taps response is measured directly
        params.tap_name = {['branches = ' num2str(idx_branches) ] };%{'29dBw','29dB','26dB','20dB','14dB','10dB'};
        params.N_ite    = 1;%4 % number of iterations for attenuation optimation
        params.flag_show_cancel = 1;
        params.flag_show_cancel_total = 0;
        [coe{idx_branches} att  total_mse] = estimate_FIR_coe(taps_t,taps_f,echos_t,echos_f,echos_delay,params);        
    else % constrained
        if flag_filter_type == 0 % tree structure                                     
            A = [];
            b = [];
            Aeq = [];
            Beq = [];
            
            % negative means no sign change required
            lb = -coe_max*ones(1,N_taps(idx_branches));
            ub = zeros(1,N_taps(idx_branches));
            %ub = ones(1,N_taps(idx_branches));
            
            %----- original ,frequency domain------
            coe{idx_branches} = lsqlin(taps_f,echos_f{1},A,b,Aeq,Beq,lb,ub);
            
            %-------- time domain -----------
            %coe{idx_branches} = lsqlin(taps_t.',echos_t{1}.',A,b,Aeq,Beq,lb,ub);
            
        else % cascade structure           
            [B,FitInfo] = lasso(taps_t.',echos_t{1}.');
            %[B,FitInfo] = lasso(taps_t.',echos_t{1}.','Lambda',1e-5);
            
            %lassoPlot(B);
            
            coe{idx_branches} = B(:,1).';            
        end
    end
    %------ reconstruct echo ----------
    d_tx_hat = reconstruct_t(taps_t,coe{idx_branches});
    
    %------ cancelation ----------
    e = echos_t{1} - d_tx_hat;  
    fs = 1.6384e9/OSR;
    show_cancellation(d_tx_hat,echos_t{1},e,fs,taps_t,coe{idx_branches});
    
    code = find_coe_to_code(real(abs(coe{idx_branches})),code_table,code_to_coe_table{idx_branches});

    coe{idx_branches}
    uint32(code)
    code_v = 5/2^16*code    
    
    if flag_program_pic == 1
        
        %------- if you skip some taps ---------
%         code_new = 65535*ones(1,8);
%         code_new([1 3 5 8]) = code;
%         program_coe(tcp_obj,tcp_const,DAC_map{1},code_new)
        %----------------------------------------
        
      program_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code)
    end
end
