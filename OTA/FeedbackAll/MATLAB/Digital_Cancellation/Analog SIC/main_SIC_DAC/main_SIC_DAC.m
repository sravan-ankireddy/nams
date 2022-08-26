% read DAC tap response and echo response to optimize the coefficient 
% group tap coe estimation to one single function

% -------------- proceduce -----------
% DAC tx echo: fixed tx file
% DAC sic    : 1. variable 2 symbol coefficient in the beginning for echo/DAC measurement 
%              2. filtered tx signal for SIC 

% input:  1. combined echo and tap response
% output: 1. filtered tx signal for SIC 

clear all;
close all;
clc;

set_env();

%=================================================
%                  parameters 
%=================================================

% simulated data
flag_sim_data     = 0;% 1: simulate data, 0:capture or load data 

% captured data
flag_capture_data = 1;% 1: capture data from ethernet, 0:load data from files

% whether data is baseband or not 
flag_is_baseband  = 1;% 1: baseband, 0:RF 

% fixed staring point of OFDM symbol
fix_timing = 0;  % 1: fixed timing, 0: restart synch 
%idx_start = [1,1,1]; % starting index for 3 different channel
idx_start = [4577,6703,16510]-4096-55; % starting index for 3 different channel

OFDM_params = set_OFDM_params();

intp_filter = half_band_filter_design();

%-------- coefficient for measuring echo and taps -------
% r(1) = echo + coe(1)*tap
% r(2) = echo + coe(2)*tap
% r(3) = echo + coe(3)*tap


coe = [0,1,1]; % equivalent to measure echo first 
%coe = [1,-1,1]; 
N_sym_total = 3;
N_sym_est = 2; % number of symbol for estimation 
fc_1638M = [204e6,396e6,588e6];
fc_819M  = [-192e6,0,192e6];
fs  = 819.2e6;
OSR = 2;
N_channel = 3;

OFDM_params.N_sym = N_sym_total;
NOFDM = OFDM_params.NOFDM;

%=================================================
%               ADC data capture 
%=================================================
if flag_sim_data == 1 % simulation data 
    params.coe         = coe;
    params.N_sym_total = N_sym_total;
    params.fc_1638M    = fc_1638M;
    params.fc_819M     = fc_819M;
    params.fs          = fs;
    params.OSR         = OSR;
    params.N_channel   = N_channel;
     
    [rx_combine_rf,rx_echo_rf,rx_combine_bsb,rx_echo_bsb,dt_bsb,tap_ch] = simulate_echo_tap(params,OFDM_params);
else  % captured data 
    if flag_capture_data == 1% capture data from ethernet 
        [tcp_obj, tcp_const,header_size]= setup_tcp_v2('192.168.0.46',7);
           
        cnt_value = 5120*3/2-1;% frame period in samples/2-1
        set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,0);
        set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,1);
        set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,2);

        NFFT = 4096;
        fs = 204.8e6; % Sampling rate
        N_ADC_samples = 5120*10;
        
        N_channel = 3;
        rx_combine_bsb = cell(1,N_channel);
        for idx_ch = 1:N_channel
            rx_combine_bsb{idx_ch} = capture_ADC_sample(tcp_obj,tcp_const,N_ADC_samples,idx_ch-1,header_size);
            rx_combine_bsb{idx_ch} = double(rx_combine_bsb{idx_ch});

            %show_freq_time(data{idx_ch},fs,NFFT,sprintf('ch = %d',idx_ch-1));            
            params.filename = sprintf('..\\data\\rx\\OFDM_3ch_192M_QPSK_ch%d_bsb_rx_combine.h',idx_ch);
            params.convert_to_int = 1;
            params.QI = 1;
            params.QF = 15;
            params.scale = 1;
            write_to_h_file(params,rx_combine_bsb{idx_ch});            
        end                        
        
    else % load data from file
        % combined data for echo/data channel estimation 
        filename = cell(1,N_channel);
        for idx_ch = 1:N_channel
            filename{idx_ch} = sprintf('..\\data\\rx\\OFDM_3ch_192M_QPSK_ch%d_bsb_rx_combine.h',idx_ch);    
        end
        rx_combine_bsb = load_signal_from_file(filename);       
    end    
    flag_is_baseband  = 1;% captured data is always baseband
    
    h = show_3ch_freq_time(rx_combine_bsb);
    keyboard;
end

%=================================================
%         measure echo and DAC tap channel 
%=================================================
%-------------------------------------------------
%              load tx data 
%-------------------------------------------------
filename = cell(1,N_channel);
for idx_ch = 1:N_channel
   filename{idx_ch} = sprintf('..\\data\\tx\\OFDM_3ch_192M_QPSK_ch%d_bsb.h',idx_ch);  
end   
dt_bsb_read = load_signal_from_file(filename);
%dt_bsb_read = dt_bsb;

if flag_sim_data == 1 
    % scale back the signal
    for idx_ch  = 1:N_channel
        scale = real(dt_bsb{idx_ch}(100))/real(dt_bsb_read{idx_ch}(100));
        dt_bsb_read{idx_ch} = scale*dt_bsb_read{idx_ch};
        %compare(dt_bsb_read{idx_ch},dt_bsb{idx_ch},'read','write');
    end
end

% seperate each symbol out 
dt_bsb_sym = cell(N_sym_total,N_channel);
for idx_ch = 1:N_channel
    for idx_sym = 1:N_sym_total
        dt_bsb_sym{idx_sym,idx_ch} = dt_bsb_read{idx_ch}(1+(idx_sym-1)*NOFDM: NOFDM+(idx_sym-1)*NOFDM );
    end
end

%-------------------------------------------------
%              measure channel 
%-------------------------------------------------
sys_params.fs  = fs; % sampling rate 
sys_params.OSR = OSR;       % over-sampling rate 
sys_params.DSR = OSR;       % down-sampling rate 
sys_params.fc  = fc_819M;
sys_params.intp_filter = intp_filter;
OFDM_params.th = 2000;%1e4;
isbsb = flag_is_baseband;

ht_rf  = cell(1,N_sym_est);
hf_rf  = cell(1,N_sym_est);
hf_bsb = cell(1,N_sym_est);
ht_bsb = cell(1,N_sym_est);

%---------- test only, direct measurement -------------
ht_rf_echo  = cell(1,N_sym_est);
hf_rf_echo  = cell(1,N_sym_est);
hf_bsb_echo = cell(1,N_sym_est);
ht_bsb_echo = cell(1,N_sym_est);
%--------------------------------------------------

if flag_is_baseband == 1
    rx_combine = rx_combine_bsb;
    %rx_echo    = rx_echo_bsb;
else
    rx_combine = rx_combine_rf;
    %rx_echo    = rx_echo_rf;
end

%            symbol 1, symbol 2
% channel 1     12        21
% channel 2     12        22
% channel 3     13        23


for idx_sym = 1:N_sym_est
    idx_sym
    dt_bsb_sub = cell(1,N_channel);
    for idx_ch = 1:N_channel
      dt_bsb_sub{idx_ch} = dt_bsb_sym{idx_sym,idx_ch};
    end
    
    %params.fix_timing = fix_timing; 
    %params.idx_start  = idx_start(idx_sym);
    params.isbsb      = isbsb;
    params.N_channel  = N_channel;

    params.fix_timing = fix_timing; % only the first symbol need to do synchronization    
    params.idx_start  = idx_start + (idx_sym-1)*OFDM_params.NOFDM;
    
    [ht_rf{idx_sym},hf_rf{idx_sym},hf_bsb{idx_sym},ht_bsb{idx_sym},flag_recapture] = multi_chest_wrapper(dt_bsb_sub,rx_combine,OFDM_params,sys_params,params);

    % ------------- test only, direct measurement ----------
    %[ht_rf_echo{idx_sym},hf_rf_echo{idx_sym},hf_bsb_echo{idx_sym},ht_bsb_echo{idx_sym},flag_recapture] = multi_chest_wrapper(dt_bsb_sub,rx_echo,OFDM_params,sys_params,params);
    %show_data(to_pow_dB(ht_bsb_echo{idx_sym}{idx_ch}),'echo');    
    %------------------------------------
    keyboard;
end

%-------- observe only ----------
% for idx_sym = 1:N_sym_est   
%     for idx_ch = 1:N_channel
%         show_data(to_pow_dB(ht_bsb{idx_sym}{idx_ch}),sprintf('symbol=%d,channel= %d',idx_sym,idx_ch));
%     end
% end    
%------------------------------

%-------------------------------------------------
%              measure echo,taps 
%-------------------------------------------------
echo = cell(1,N_channel);
tap  = cell(1,N_channel);
for idx_ch = 1:N_channel    
    % time domain
%     r = [ht_bsb{1}{idx_ch}; ht_bsb{2}{idx_ch}];    
%     [echo{idx_ch},tap{idx_ch}] = measure_echo_tap(r,coe);
    
    % frequency domain
    r = [hf_bsb{1}{idx_ch}; hf_bsb{2}{idx_ch}];
    [echo_temp,tap_temp] = measure_echo_tap(r,coe);
    echo{idx_ch} = ifft(echo_temp);
    tap{idx_ch}  = ifft(tap_temp);
    
    show_data(to_pow_dB(echo{idx_ch}),sprintf('echo,channel =%d',idx_ch));
    show_data(to_pow_dB(tap{idx_ch}),sprintf('tap,channel =%d',idx_ch));
end

%-------- test only ,compare measurement------------
% The reference symbol boundary should be different 
% for idx_sym = 1:N_sym_est   
%     for idx_ch = 1:N_channel
%         compare(to_pow_dB(ht_bsb_echo{idx_sym}{idx_ch}),to_pow_dB(echo{idx_ch}),sprintf('direct measured echo channel,symbol=%d,channel= %d',idx_sym,idx_ch),...
%                                                                                                 sprintf('in-direct echo channel,symbol=%d,channel= %d',idx_sym,idx_ch));
%     end
% end    
%-------------------------------

%------------ test only----------------
echo_f = cell(1,N_channel);
for idx_ch = 1:N_channel
  echo_f{idx_ch} = fftshift(fft(echo{idx_ch}));
end

f_start = [108e6,300e6,492e6];
delta_f = 50e3;
hf_combine = combine_channel(echo_f,f_start,delta_f);

show_data(to_pow_dB(hf_combine));
%show_data(phase(hf_combine));

%------------------------------------



%=================================================
%          DAC coefficient estimator  
%=================================================
coe_fir = cell(1,N_channel);
for idx_ch = 1:N_channel
    coe_fir{idx_ch} = estimate_tap_coe(echo{idx_ch},tap{idx_ch});
    coe_fir{idx_ch} = -coe_fir{idx_ch}; % because the combiner is in positive sign
end

if flag_sim_data == 1 % if the data come from simulation , need to include additional tap response
    for idx_ch = 1:N_channel
        coe_fir{idx_ch} = conv(coe_fir{idx_ch},tap_ch{idx_ch});
    end   
end

%=================================================
%          generate filtered tx signal   
%=================================================

% echo signal 
params.filter_coe    = coe_fir;              % baseband filter coefficient for echo emulation 
params.sym_coe       = ones(1,N_sym_total);  % coefficient for each symbol for echo and tap measurement
params.N_sym_total   = N_sym_total;
params.fs            = fs;
params.OSR           = OSR;
params.N_channel     = N_channel;
params.fc_1638M      = fc_1638M;
params.fc_819M       = fc_819M;
params.intp_filter   = intp_filter;
params.rand_seed     = 123;
[echo_hat_rf,echo_hat_bsb,dt_bsb]    = generate_multi_ch_OFDM_sym_cir(OFDM_params,params);

%-----------------------------
%        write tx file  
%-----------------------------
backoff_dB = 10;
params.filename = '..\data\tx\OFDM_3ch_192M_QPSK_rf_cpx_sic.h';
params.convert_to_int = 1;
params.QI = 1;
params.QF = 15; 
params.scale = 1-10^(-backoff_dB/20);
write_to_h_file(params,echo_hat_rf);


%=================================================
%      ADC capture again to show cancellation   
%=================================================

%--------------------------------------
%        simulate data cancellation  
%--------------------------------------
if flag_sim_data == 1
    fs  = 819.2e6;
    e_bsb = cell(1,N_channel);
    for idx = 1:N_channel
        e_bsb{idx} = rx_echo_bsb{idx} + echo_hat_bsb{idx};
        show_cancellation(echo_hat_bsb{idx},rx_echo_bsb{idx},e_bsb{idx},fs);
    end
        
    e_rf = rx_echo_rf + echo_hat_rf;
    show_cancellation(echo_hat_rf,rx_echo_rf,e_rf,fs);
end



