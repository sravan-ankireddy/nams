% read echo and optical data from VNA and estimate coefficient 
% original version without any simplification 

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
flag_filter_type       = 1;% 0:tree filter, 1:cascade filter   

flag_filter_response   = 1; % filter to the desired band
flag_program_pic       = 0; 
flag_constrained       = 1; 
flag_load_measure      = 1;
flag_load_ADC_capture  = 0;
flag_predict_TTD       = 0;
flag_table_down_sample = 0;
flag_keep_opt_delay    = 1; % should be 1 for pic programming 


w_size  = 16*8; % size of windowing
%N_gate = 120;%120 % for internal time gating
%N_gate = 515; % for external time gating, 812
%N_gate = 492; % for external time gating, 812x6 
%N_gate = 501; %499 for external time gating 20180307
%N_gate = 101; %100 for internal time gating 20180313
%N_gate = 497; %501 for external time gating 20180307

%N_gate = 98; % internal
%N_gate = 562;% external for ADC measurement
N_gate = 32768;


%------- for digital delay ----------
OFDM_params_819M = set_OFDM_params_819M();
sys_params_819M  = set_system_params_819M();

OFDM_params = set_OFDM_params_819M();
sys_params  = set_system_params_819M();
%-------------------------------------

% --------- freq index for tap -----------
% BW = [190e6 190e6 190e6];
% f_start = [108e6 300e6 492e6];
% weight = [1 1 1];

BW = [96e6*6];
f_start = [108e6];
weight = [1];

f_stop = zeros(1,length(f_start));
for idx = 1:length(BW)
 f_stop(idx)  = f_start(idx) + BW(idx);
end
delta_f = 50e3;
NFFT    = 16384;
%[f_window_bsb,f_window_pass,f_idx_bsb,f_idx_pass] = cal_f_index(f_start,f_stop,delta_f,NFFT);
[f_window_bsb,f_window_pass,f_idx_bsb,f_idx_pass] = cal_f_index_weight(f_start,f_stop,delta_f,NFFT,weight);

%------------- test only -------------------------------
%----------- freq index for echo --------
% BW = [96e6*1];
% f_start = [108e6];
% weight = [1];
% 
% f_stop = zeros(1,length(f_start));
% for idx = 1:length(BW)
%  f_stop(idx)  = f_start(idx) + BW(idx);
% end
% delta_f = 50e3;
% NFFT    = 16384;
% [f_window_bsb_echo1,f_window_pass,f_idx_bsb_1,f_idx_pass] = cal_f_index_weight(f_start,f_stop,delta_f,NFFT,weight);
% 
% %------- freq index for echo --------
% BW = [96e6*1];
% f_start = [204e6];
% weight = [1];
% 
% f_stop = zeros(1,length(f_start));
% for idx = 1:length(BW)
%  f_stop(idx)  = f_start(idx) + BW(idx);
% end
% delta_f = 50e3;
% NFFT    = 16384;
% [f_window_bsb_echo2,f_window_pass,f_idx_bsb_2,f_idx_pass] = cal_f_index_weight(f_start,f_stop,delta_f,NFFT,weight);

%--------------------------------------------------

%------- setup tcp ---------
if flag_program_pic == 1
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.157',7);
    
    DAC_map = cell(1,2);
    %DAC_map{1} = [5 2 4 3 0 6 7 1];
    %             1 2 3 4 5 6 7 8    
    
    % new mapping 20180426  , don't forget to change the coe to vol table
    DAC_map{1} = [5 2 1 3 0 6 7 4];
    %             1 2 3 4 5 6 7 8    
    
    DAC_map{2} = [10 14 12 13 8];            
end

%-------------------------------------------------
%                 load echo 
%-------------------------------------------------



if flag_load_ADC_capture == 1
    filename = '..\data\20180410_ADC_diff\echo.mat';            
    
    temp = load(filename);
    hf_bsb = temp.df; 
    
    if flag_filter_response == 1
        hf_bsb = hf_bsb.*f_window_bsb;                
        ht_bsb = ifft(hf_bsb);
    end
 
else    
    % filename = 'data\VNA2\vna2_echo.mat';
    % echos_f = load(filename);
    % echos_f = echos_f.data.';
    %
    % temp = echos_f.';
    % temp_pass  = [temp(1) temp(1) temp(1:end-1)];      % 0,50k,100k
    % temp_bsb   = [temp_pass 0 fliplr(conj(temp_pass(2:end)))]; % convert to baseband
    % temp_bsb   = temp_bsb.*f_window_bsb;
    %
    % ht_bsb = ifft(temp_bsb);
    % figure; plot(to_pow_dB(ht_bsb)); legend('echo in t')
    
    %-----------original--------------- 
    %filename = 'VNA_RS\data\20180313_echo_taps\echo.mat';    
    %filename = 'VNA_RS\data\20180314_echo_taps\echo_96M.mat';        
    %filename = 'VNA_RS\data\20180314_echo_taps\echo.mat';        
    %filename = 'VNA_RS\data\20180327_TTD_v2\echo.mat';            
    %filename = 'VNA_RS\data\20180314_echo_taps\echo_96M_direct.mat';        
    %filename = 'VNA_RS\data\20180315_echo_taps\echo.mat';            
    filename = '..\VNA_RS\data\20180503_external\echo.mat';            
    
    temp = load(filename);
    ht_bsb = temp.dt; 
    
    if flag_filter_response == 1       
        hf_bsb = fft(ht_bsb);
        hf_bsb = hf_bsb.*f_window_bsb;
        
        %------- test only -----------
%         hf_bsb_temp = zeros(1,length(hf_bsb));
%         hf_bsb_temp(f_idx_bsb_1) = hf_bsb(f_idx_bsb_1);
%         hf_bsb_temp(f_idx_bsb_2) = hf_bsb(f_idx_bsb_1);
%         hf_bsb = hf_bsb_temp;
        %------------------------------
               
        ht_bsb = ifft(hf_bsb);
    end
   %-------------------------------------   

    %-------- test with high sampling rate ------------ 
%     filename = 'VNA_RS\data\echo_x6_v2.mat';    
%     temp = load(filename);
%     ht_bsb = temp.dt;
%     ht_bsb = ht_bsb(1:6:end);
   %-------------------------------------   
end

% time gating 
ht_bsb(N_gate:end) = 0;
ht_bsb = double(ht_bsb);
ht_bsb = real(ht_bsb); % ignore very small imaginary part
figure; plot(to_pow_dB(ht_bsb)); legend('echo in t')

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
        N_taps = [8 5];
        
        params.N_branches           = N_branches;        
        params.N_taps               = N_taps;
        params.f_window             = f_window_bsb;
        params.flag_filter_response = flag_filter_response;
        params.flag_predict_TTD     = flag_predict_TTD;       
               
        [opt_taps_t] = read_VNA_all_taps_wrapper(params);
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

%-----------------------------------------------------------
%                    find coarse delay 
%-----------------------------------------------------------

% echo coarse delay 
if flag_load_ADC_capture == 1
   params.peak_th = 0.01;
else
   params.peak_th = 0.1;
end

params.idx_neighbors_th = 80;%30; 
params.small_width_th = 0;% not used now
desired_peaks = detect_coarse_delay(ht_bsb,params)

if flag_set_peak_locations == 1
    %desired_peaks = [102 497] % 96 MHz
    %desired_peaks = [13 62] % 96 MHz    
    desired_peaks = [96 490]; % 819.2 MHz    
end

if length(desired_peaks) ~= N_branches % error checking
    disp('Error! The number of echo peaks does not equal to the number of branches !'); 
end

% branches delay 
if flag_load_ADC_capture == 1
    params.peak_th = 0.3;%0.02; % 600 MHz
else
    params.peak_th = 0.1; % 600 MHz
end
branch_peaks = zeros(1,N_branches);
for idx_branches = 1:N_branches
      if N_taps(idx_branches) == 1
        idx_taps = 1; 
      else
        idx_taps = floor(N_taps(idx_branches)/2); % use middle taps as branches delay 
      end
      branch_peaks(idx_branches) = detect_coarse_delay(opt_taps_t(idx_branches,idx_taps,:),params)
end


NFFT    = w_size;
echos_f = cell(1,1);
echos_t = cell(1,1);


coe     = cell(1,N_branches);
% code_max    = 2^16;
% N_codes     = 128;
% code_table  = [1:floor(code_max/N_codes):code_max];
% 
% 
% temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128','table_norm');
% if flag_normalize_coe_table == 1
%    temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
% end
% 
% %------------- temp --------------
% if flag_swap_tap3_tap8 == 1
%    temp2 = temp.table_norm{3};
%    temp.table_norm{3} = temp.table_norm{8};
%    temp.table_norm{8} = temp2;   
% end
% %----------------------------------
% 
% if flag_table_down_sample == 1
%     %===================================
%     %           down sample
%     %===================================
%     R = 8; %8 is the maximum downsampling rate without compromising the performance
%     table_down = cell(1,size(temp.table_norm,2));
%     for idx = 1:size(temp.table_norm,2)
%         temp_down = temp.table_norm{idx};
%         table_down{idx} = temp_down(1:R:end);
%     end
%     code_to_coe_table{1} = table_down;    
%     code_table = code_table(1:R:end);
% else
%     code_to_coe_table{1} = temp.table_norm;
% end 
% 
% temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128_external','table_norm');
% % if flag_normalize_coe_table == 1
% %    temp.table_norm = normalize_code_to_coe_table(temp.table_norm,ref_code);
% % end
% if flag_table_down_sample == 1
%     %===================================
%     %           down sample
%     %===================================
%     table_down = cell(1,size(temp.table_norm,2));
%     for idx = 1:size(temp.table_norm,2)
%         temp_down = temp.table_norm{idx};
%         table_down{idx} = temp_down(1:R:end);
%     end
%     code_to_coe_table{2} = table_down;       
% else
%     code_to_coe_table{2} = temp.table_norm;
% end

% params.N_branches               = N_branches ;
% params.flag_normalize_coe_table = flag_normalize_coe_table;
% params.flag_swap_tap3_tap8      = flag_swap_tap3_tap8;
% params.flag_table_down_sample   = flag_table_down_sample;
% [code_table,code_to_coe_table]= load_code_table(params);

[code_table,code_to_coe_table] = load_code_and_coe_table(N_branches);

echo_compare = cell(1,2);

for idx_branches = 1:N_branches  
    idx_branches
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
   
    echo_compare{idx_branches} = to_pow_dB(ht_bsb_w/max(abs(ht_bsb_w)));
        
    opt_taps_win_t = zeros(1,N_taps(idx_branches),w_size);
    for idx = 1:N_taps(idx_branches)
      opt_taps_win_t(1,idx,:) = opt_taps_t(idx_branches,idx,w_start:w_start + w_size-1 );
    end  
    branch_peaks(idx_branches) = branch_peaks(idx_branches) - w_start + 1;
    
    %---------- delay all branches according to coarse delay requirement ---------
    params.N_taps     = N_taps(idx_branches);
    params.N_branches = 1;
    params.NFFT       = w_size;
    if flag_keep_opt_delay == 1
        delay             = 0;
    else
        delay             = desired_peaks(idx_branches) - branch_peaks(idx_branches);
    end
    [taps_t, taps_f]  = delay_taps(params,delay,opt_taps_win_t);
    
    %--------- timing gating of taps -----------
    [taps_t, taps_f] = time_gating(taps_t,N_gate- w_start + 1);
    
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
%     show_data_para(taps_amp_cell,{'1','2','3','4','5','6','7','8'});    
%     show_data_para(taps_phase_cell,{'1','2','3','4','5','6','7','8'});        
    %------------------------
    
    
    %--------- estimate coefficient ---------
    
    if flag_constrained == 0 %unconstrained 
        echos_delay = 0;
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
        lb = [];    % negative means no sign change required
        for idx_bnb = 1:N_taps(idx_branches)
            %lb = [lb -Inf];
            lb = [lb -coe_max];
        end
        ub = zeros(1,N_taps(idx_branches));
        %ub = ones(1,N_taps(idx_branches));
        
        if flag_filter_type == 0
            %------ frequency ---------
            coe{idx_branches} = lsqlin(taps_f,echos_f{1},[],[],[],[],lb,ub) ;
            %coe{idx_branches} = lsqlin(taps_f,echos_f{1},[],[],[],[],[],[]) ;
            
            %------ time ---------
            %coe{idx_branches} = lsqlin(taps_t.',echos_t{1}.',[],[],[],[],lb,ub) ;
            %coe{idx_branches} = lsqlin(taps_t.',echos_t{1}.',[],[],[],[],[],[]) ;
            
            %-------- ridge --------
            lambda = 1.5e-4;
            %B = ridge_regression(taps_t.',echos_t{1}.',lambda);      
            %B = ridge_regression(taps_f,echos_f{1}.',lambda);       
            [B,N_ite] = ridge_regression_ite(taps_t.',echos_t{1}.',lambda);
            
            N_ite = 1e4;
            step_size = 0.6;
            lambda  = 1e-4;
            B2 = LS_GD(taps_t.',echos_t{1}.',lambda,N_ite,step_size);
            
            
            show_data_para({coe{idx_branches},B,B2},{'constraint','ridge','GD'});                        
            coe{idx_branches} = B;
            %---------------------------
            
        else            
            % matlab lasso
            %[B,FitInfo] = lasso(taps_t.',echos_t{1}.','Lambda',4.5e-5);
            
            % my lasso 
            %params.lambda = 1e-5;%4.5e-5; % penality for L1 norm
            %params.p      = 1e-7;%1e-7; % penality for z-u     
            %[B2,N_ite] = lasso_ADMM(taps_t.',echos_t{1}.',params);
            
            % ridge
            lambda = 1.5e-4;
            [B3,N_ite] = ridge_regression_ite(taps_t.',echos_t{1}.',lambda);
            
            %show_data_para({B,B2,B3},{'lasso matlab','lasso cc','ridge'});
            
            alpha = coe_to_cascade_coe(B3.');            
            coe{idx_branches} = alpha;
        end
    end    
    %------ reconstruct echo ----------
    if flag_filter_type == 0 % tree 
        d_tx_hat = reconstruct_t(taps_t,coe{idx_branches});
    else                     % cascade        
        d_tx_hat = cascade_filter_analog(coe{idx_branches},sign(B.'),taps_t);        
    end
    %------ cancelation ----------
    e = echos_t{1} - d_tx_hat;  
    fs = 1.6384e9;
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
        
      %program_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code)
       program_pic_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code)
    end
end
%show_data_para(echo_compare,{'internal','external'})
