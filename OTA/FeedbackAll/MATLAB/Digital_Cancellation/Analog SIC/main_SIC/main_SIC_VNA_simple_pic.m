% @read echo and optical data from VNA and estimate coefficient
% @simplified version to reduce the memeory and computational complexity requirement
% @further simplified version for pcb

clear all;
close all;
clc;

set_env();

%================= ADC ===================
%foldername = '..\\data\\20180628_cc3_pcb3_ADC';
% flag_set_peak_locations = 1;
% desired_peaks_preset = 442;
%==========================================

%================= VNA ====================
%foldername = '..\\data\\20180628_cc3_pcb3';
%foldername = '..\\data\\20180627_cc3_pcb3';
foldername = '..\\data\\20180705_cc3_pcb3_IA';
flag_set_peak_locations = 0;
%===========================================


flag_filter_type       = 1;% 0:tree filter, 1:cascade filter
flag_filter_response   = 1;
flag_program_pic       = 1;
flag_constrained       = 0;
flag_load_measure      = 1;
flag_load_ADC_capture  = 0;
flag_measure_results   = 0;

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


%------- for digital delay ----------
% OFDM_params_819M = set_OFDM_params_819M();
% sys_params_819M  = set_system_params_819M();
% 
% OFDM_params = set_OFDM_params_819M();
% sys_params  = set_system_params_819M();
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
% BW      = 96e6*6;
BW      = 96e6;
% BW      = 60e6;
% f_start = [108e6];
f_start = [500e6];
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
filename = sprintf('%s\\echo.mat',foldername);
temp = load(filename);
ht_temp = temp.dt;
hf_temp = temp.df;

% time gating
ht_bsb = zeros(1,length(ht_temp));
ht_bsb(idx_gate_start:idx_gate_stop) = ht_temp(idx_gate_start:idx_gate_stop);
hf_bsb = fft(ht_bsb);

figure; plot(20*log10(abs(ht_bsb))); legend('echo in t');

%--------------------------------------------------
%              detect peak location
%--------------------------------------------------
% echo coarse delay
if flag_load_ADC_capture == 1
    params.peak_th = 0.01;
else
%     params.peak_th = 0.06;%% Cable
    params.peak_th = 0.022;%% wireless
end

params.idx_neighbors_th = 10;%30;
params.small_width_th = 0;% not used now
desired_peaks = detect_coarse_delay(ht_bsb,params)

if flag_set_peak_locations == 1
    desired_peaks = desired_peaks_preset;
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
        params.foldername       = foldername;
        
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
% load code table
[code_table,code_to_coe_table] = load_code_and_coe_table(N_branches);


% load power ratio
filename = sprintf('%s\\pow_ratio.mat',foldername);
temp = load(filename);
pow_max = temp.pow_max;
pow_min = temp.pow_min;

%---- test only --------
% pow_max = 1*ones(1,N_taps(1));
% pow_min = 1e-3*ones(1,N_taps(1));
% pow_min(1) = pow_max(1).*10^(-28.63/10);
% pow_min(2) = pow_max(1).*10^(-31.17/10);
% pow_min(3) = pow_max(1).*10^(-34.59/10);
% pow_min(4) = pow_max(1).*10^(-31.93/10);
% pow_min(5) = pow_max(1).*10^(-32.23/10);
% pow_min(6) = pow_max(1).*10^(-33.22/10);
% pow_min(7) = pow_max(1).*10^(-32.55/10);
pow_min = zeros(1,N_taps(1));
%pow_max = 0.8*ones(1,N_taps(1));
%-----------------------

% load gamma
% filename = sprintf('%s\\gamma.mat',foldername);
% temp = load(filename);
% gamma = temp.gamma;
%
%----- test only -------
% pow_max = gamma;
%pow_min = zeros(1,N_taps(1));
%---------------------


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
    
    %----- show all taps and echo ------
    tapst_cell = cell(1,size(taps_t,1)+1);
    taps_amp_cell = cell(1,size(taps_t,1));
    taps_phase_cell = cell(1,size(taps_t,1));
    
    for idx = 1:size(taps_t,1)
        tapst_cell{idx} = (taps_t(idx,:));
        taps_amp_cell{idx} = 20*log10(abs(fft(taps_t(idx,:))));
        %taps_phase_cell{idx} = phase(fft(taps_t(idx,:)));
    end
    tapst_cell{end} = echos_t{1};
%     show_data_para(tapst_cell,{'1','2','3','4','5','6','7','echo'});
    %------------------------
    
    %--------- estimate coefficient ---------
    echos_delay = 0;
    
    
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
        %[coe_cas,FitInfo] = lasso(taps_t.',echos_t{1}.','Lambda',1e-5);
        %[coe_cas,FitInfo] = lasso(taps_t.',echos_t{1}.','Lambda',3e-4);
        
        % my lasso
        %params.lambda = 1e-5;%4.5e-5; % penality for L1 norm
        %params.p      = 1e-7;%1e-7; % penality for z-u
        %[B2,N_ite] = lasso_ADMM(taps_t.',echos_t{1}.',params);
        
        % ridge
        %             lambda = 1.5e-4;
        %            [coe_lin,N_ite] = ridge_regression_ite(taps_t.',echos_t{1}.',lambda);
        %           %[coe_cas,N_ite] = ridge_regression_ite(taps_f,echos_f{1},lambda);
        
        
        % negative means no sign change required
        lb = -2*ones(N_taps(idx_branches),1);
        if flag_constrained == 0
            ub = 2*ones(N_taps(idx_branches),1);
        else
            ub = zeros(N_taps(idx_branches),1);
        end
        coe_lin = lsqlin(taps_t.',echos_t{1}.',[],[],[],[],lb,ub);
        
        % barrier
        %             step_size = 0.1;
        %             N_ite_ext = 10;
        %             N_ite_int = 100;
        %             [coe_lin,e_b] = constrained_LS_barrier(taps_t.',echos_t{1}.',ub,lb,step_size,N_ite_ext,N_ite_int);
        %
        % ADMM
        %N_ite = 100;
        %[coe_lin, e_all] = constrained_LS_ADMM(taps_t.',echos_t{1}.',ub,lb,N_ite);
        
        
        coe_lin
        %coe{idx_branches} = coe_to_cascade_coe(coe_lin,gamma);
        coe{idx_branches} = coe_to_cascade_coe(coe_lin,pow_min,pow_max);
    end
    
    %------ reconstruct echo ----------
    if flag_filter_type == 0 % tree
        d_tx_hat = reconstruct_t(taps_t,coe{idx_branches});
    else                     % cascade
        %d_tx_hat = cascade_filter_analog(coe{idx_branches},sign(coe_lin),taps_t,gamma);
        d_tx_hat = cascade_filter_analog(coe{idx_branches},sign(coe_lin),taps_t,pow_min,pow_max);
    end
    
    
    %------ cancelation ----------
    e = echos_t{1} - d_tx_hat;
    fs = 1.6384e9/OSR;
    %cancel_amount_sim = 
    show_cancellation(d_tx_hat,echos_t{1},e,fs);
    
    [code,code_error] = find_coe_to_code_v2(real(abs(coe{idx_branches})),code_table,code_to_coe_table{idx_branches});
    coe{idx_branches}
    uint32(code)
    
    
    if flag_program_pic == 1
        program_pic_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code);
        switch_sic(tcp_obj,tcp_const,1);  % sic on
        switch_echo(tcp_obj,tcp_const,1); % echo on
    end
    
    filename = sprintf('%s\\filter_coe_branch_%d.mat',foldername,idx_branches);
    save(filename,'code');
end

%% measure the results from vna
if flag_measure_results == 1
    params.f_start  = 100e3;
    params.f_stop   = 819.2e6;
    params.N_points = 16383;
    delta_f = 50e3;
    NFFT = 16384;  % half FFT size
    app = vna_setup_cm(params);
    
    % measure residual
    df = vna_measure_cm(app);
    [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);
    residual_vna = (df);
    
    % measure echo
    switch_sic(tcp_obj,tcp_const,0);  % sic off
    switch_echo(tcp_obj,tcp_const,1); % echo on
    df = vna_measure_cm(app);
    [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);
    echo_vna = (df);
    
    cancel_amount = 20*log10(abs(echo_vna)) - 20*log10(abs(residual_vna));
    mean(cancel_amount(floor(108e6/delta_f):floor(684e6/delta_f)))
    
    cancel_amount = fftshift(cancel_amount);
    echo_vna      = fftshift(echo_vna);
    residual_vna  = fftshift(residual_vna);
    
    NFFT = 32768;
    fs = 1.6384e9;
    f = (-NFFT/2:NFFT/2-1)*fs/NFFT*1e6;
    x = NFFT/2+1:NFFT;
    figure;
    plot(f(x),20*log10(abs(residual_vna(x))),f(x),20*log10(abs(echo_vna(x))));
    legend('residual','echo')
    figure; plot(f(x),cancel_amount(x));
    axis([f(x(1)) f(x(end)) 0 50 ])
    legend('cancel amount');
    
    % compared to the simulation
    %     cancel_amount_down = cancel_amount(1:8:end);
    %     x2 = 128/2+1:128;
    %     figure; plot(x2,cancel_amount_down(x2),x2,cancel_amount_sim(x2));
    %     legend('measurement','simulation')
end


