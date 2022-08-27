% @read echo and optical data from VNA and estimate coefficient 
% @simplified version to reduce the memeory and computational complexity requirement 
% @further simplified version for pcb
% wrapper for coefficient optimization 

function [coe_new,coe_delta,code,desired_peaks] = sic_wrapper(foldername,flag_set_peak_locations,desired_peaks_pre,coe_pre,step,flag_ite_mode,freq_DSR)
                       
flag_filter_type       = 1;% 0:tree filter, 1:cascade filter   
flag_filter_response   = 1;
flag_program_pic       = 1; 
flag_constrained       = 1; 
flag_load_ADC_capture  = 0;
flag_show_plot         = 0;


flag_predict_TTD       = 0;
flag_filter_only       = 0;
flag_use_coe_pre_for_sim = 1; % use coe_pre for simulation of cancellation amount

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
delta_f_base = 50e3*freq_DSR;

NFFT_base = 16384/freq_DSR;   % half-band fft
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
filename = sprintf('%s\\echo.mat',foldername);
temp = load(filename);
ht_temp = temp.dt;

% time gating
ht_bsb = zeros(1,length(ht_temp));
ht_bsb(idx_gate_start:idx_gate_stop) = ht_temp(idx_gate_start:idx_gate_stop);
hf_bsb = fft(ht_bsb);

% downsampling in the frequency domain 
hf_bsb = hf_bsb(1:freq_DSR:end);
ht_bsb = ifft(hf_bsb);

if flag_show_plot == 1
figure; plot(to_pow_dB(ht_bsb)); legend('echo in t');
end

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
desired_peaks = detect_coarse_delay(ht_bsb,params);
desired_peaks = round(desired_peaks);

if flag_set_peak_locations == 1
    desired_peaks = desired_peaks_pre;   
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
    params.freq_DSR         = freq_DSR;
    
    [opt_taps_t] = read_VNA_response_windowing(params);    
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
    
    echos_t{1} = ht_bsb_w(idx_branches,:).';
    echos_f{1} = fft(ht_bsb_w(idx_branches,:));
    
    taps_t = zeros(NFFT,N_taps(idx_branches));
    taps_f = zeros(NFFT,N_taps(idx_branches));
    for idx_taps = 1:N_taps(idx_branches)        
        temp = opt_taps_t(idx_branches,idx_taps,:);
        taps_t(:,idx_taps) = temp(:);        
        taps_f(:,idx_taps) = fft(taps_t(idx_taps,:)).';
    end        
    
    
    %--------- time gating of taps -----------
    %[taps_t, taps_f] = time_gating(taps_t,N_gate);
    
    %----- show all taps and echo ------
    %tapst_cell = cell(1,size(taps_t,1)+1);
    %taps_amp_cell = cell(1,size(taps_t,1));
    %taps_phase_cell = cell(1,size(taps_t,1));
    
    %for idx = 1:size(taps_t,1)
    %    tapst_cell{idx} = (taps_t(idx,:));
    %    taps_amp_cell{idx} = to_pow_dB(fft(taps_t(idx,:)));        
    %    %taps_phase_cell{idx} = phase(fft(taps_t(idx,:)));        
    %end    
    %tapst_cell{end} = echos_t{1};
    %show_data_para(tapst_cell,{'1','2','3','4','5','6','7','echo'});    
    %------------------------
        
    %--------- estimate coefficient ---------
    
    if flag_constrained == 0 %unconstrained 
     
    else % constrained
        if flag_filter_type == 0 % tree structure                                     
            A = [];
            b = [];
            Aeq = [];
            Beq = [];
            
            % negative means no sign change required
            lb = -coe_max*ones(1,N_taps(idx_branches));
            if flag_constrained == 0
              ub = ones(1,N_taps(idx_branches));
            else
              ub = zeros(1,N_taps(idx_branches));
            end
            
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
%            [coe_cas,N_ite] = ridge_regression_ite(taps_t.',echos_t{1}.',lambda);
%           %[coe_cas,N_ite] = ridge_regression_ite(taps_f,echos_f{1},lambda);
            
           
            % negative means no sign cha nge required
            lb = -2*ones(1,N_taps(idx_branches));
            if flag_constrained == 0
              ub = 2*ones(1,N_taps(idx_branches));
            else
              ub = zeros(1,N_taps(idx_branches));
            end            
            
            
            switch flag_ite_mode 
                case 1 % residual update
                    coe_delta = lsqlin(taps_t,echos_t{1},[],[],[],[],lb,ub);
                    coe_new = coe_pre + step*coe_delta;
                case 0 % moving average 
                    coe_delta = lsqlin(taps_t,echos_t{1},[],[],[],[],lb,ub);
                    coe_new = (1-step)*coe_pre + step*coe_delta;
                case 2 % gradient descent, echo should be residual                     
                    coe_new = coe_pre + step*taps_t.'*echos_t{1}; % residual = b - Ax
                otherwise
                    disp('iteration mode error !')
            end
            
            %coe{idx_branches} = coe_to_cascade_coe(coe_lin,gamma);
            coe{idx_branches} = coe_to_cascade_coe(coe_new,pow_min,pow_max);
			
% 			if flag_use_coe_pre_for_sim == 1
% 			   coe_sim = coe_pre;
%             else 
%                coe_sim = coe_lin;
% 			end
        end
    end
    %------ reconstruct echo ----------
   if flag_filter_type == 0 % tree
       d_tx_hat = reconstruct_t(taps_t,coe{idx_branches});
   else                     % cascade       
       %d_tx_hat = cascade_filter_analog(coe{idx_branches},sign(coe_lin),taps_t,gamma);       
       d_tx_hat = cascade_filter_analog(coe{idx_branches},sign(coe_new),taps_t,pow_min,pow_max);
   end
       
    %------ cancelation ----------
    e = echos_t{1} - d_tx_hat;  
    fs = 1.6384e9/OSR;
    if flag_show_plot  == 1
      cancel_amount_sim = show_cancellation(d_tx_hat,echos_t{1},e,fs);
    end
    
    [code,code_error] = find_coe_to_code_v2(real(abs(coe{idx_branches})),code_table,code_to_coe_table{idx_branches});
    
    
    if flag_program_pic == 1                   
       program_pic_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code);       
       switch_sic(tcp_obj,tcp_const,1);  % sic on
       switch_echo(tcp_obj,tcp_const,1); % echo on      
    end
    
    filename = sprintf('%s\\filter_coe_branch_%d.mat',foldername,idx_branches);
    save(filename,'code');
end




