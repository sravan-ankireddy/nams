% simulation of echo and DAC response measure 
clear all;
close all;

set_env();

intp_filter = half_band_filter_design();

randn('state',1234);
rand('state',12345);
%-------------------------------------------------
%             generate transmit signal
%-------------------------------------------------
% coe = [0.9,-0.9,1]; % coefficient to measure echo and tap simultaneously
% 
% N_sym_total = 3;  % number of symbol should not be less than 2 to pass timing match checking block
% N_sym_est = 2; % number of symbol for estimation 
% fs  = 819.2e6;
% OSR = 2;
% OFDM_params = set_OFDM_params();
% OFDM_params.N_sym  = 1; % only one symbol 
% 
% N_channel = 3;
% dt_bsb = cell(N_sym_total,N_channel);
% 
% fc_1638M = [204e6,396e6,588e6];
% fc_819M  = [-192e6,0,192e6];
% 
% rf_all_echo = []; %concatenated version
% rf_all_tap = [];
% for idx_sym = 1:N_sym_total % symbol 
%     rf_all_echo_sym = 0; % rf signal in one symbol     
%     rf_all_tap_sym = 0;  % rf signal in one symbol     
%     
%     % generate one symbol per channel 
%     for idx_ch = 1:N_channel
%         OFDM_params.fc = fc_1638M(idx_ch); % actual carrier frequency for power loading
%         fc             = fc_819M(idx_ch);  % complex baseband carrier frequency. Because of up conversion in DAC        
% 
%         [dt_bsb{idx_sym,idx_ch}] = generate_OFDM_sym(OFDM_params); % generate only one symbol           
%         
%         % echo
%         dt_temp = dt_bsb{idx_sym,idx_ch};            
%         [rf ,rf_cpx]= bsb_to_rf(dt_temp,intp_filter,OSR,fc,fs);
%         rf_all_echo_sym = rf_all_echo_sym + rf; 
%         
%         % tap , go through filter 
%         dt_temp = coe(idx_sym)*dt_bsb{idx_sym,idx_ch};            
%         [rf,rf_cpx]= bsb_to_rf(dt_temp,intp_filter,OSR,fc,fs);
%         rf_all_tap_sym = rf_all_tap_sym + rf; 
%         
%     end
%     
%    rf_all_echo = [rf_all_echo rf_all_echo_sym]; % concatenate symbol 
%    rf_all_tap  = [rf_all_tap rf_all_tap_sym];
% end
  

%-------------------------------------------------
%            simulate channel effect 
%-------------------------------------------------
h_echo = [ zeros(1,30) 1 0.9 -1+i zeros(1,200) 0.1 0.3 zeros(1,400) 0.1 i*0.01 ];
h_tap  = [ 0.1 0.9+i -0.3 ];
% h_echo = 1;
% h_tap = 1;

y_echo = conv(h_echo,rf_all_echo);
y_tap = conv(h_tap,rf_all_tap);

y_echo = y_echo(1:length(rf_all_echo));
y_tap  = y_tap(1:length(rf_all_tap));

y_all = y_echo - y_tap;
%-------------------------------------------------
%              measure channel 
%-------------------------------------------------
sys_params.fs  = 819.2e6; % sampling rate 
sys_params.OSR = 2;       % over-sampling rate 
sys_params.DSR = 2;       % down-sampling rate 
sys_params.fc  = fc_819M;
sys_params.intp_filter = intp_filter;
OFDM_params.th = 1e4;
isbsb = 0;

ht_rf  = cell(1,N_sym_est);
hf_rf  = cell(1,N_sym_est);
hf_bsb = cell(1,N_sym_est);
ht_bsb = cell(1,N_sym_est);

%            symbol 1, symbol 2
% channel 1     12        21
% channel 2     12        22
% channel 3     13        23
for idx_sym = 1:N_sym_est
    idx_sym
    dt_bsb_sub = cell(1,N_channel);
    for idx_ch = 1:N_channel
      dt_bsb_sub{idx_ch} = dt_bsb{idx_sym,idx_ch};
    end
    
    [ht_rf{idx_sym},hf_rf{idx_sym},hf_bsb{idx_sym},ht_bsb{idx_sym},flag_recapture] = multi_chest_wrapper(dt_bsb_sub,y_all,OFDM_params,sys_params,isbsb,N_channel);
end

%-------- observe only ----------
for idx_sym = 1:N_sym_est   
    for idx_ch = 1:N_channel
        show_data(ht_bsb{idx_sym}{idx_ch},sprintf('symbol=%d,channel= %d',idx_sym,idx_ch));
    end
end    
%------------------------------

for idx_ch = 1:N_channel
    %r = [hf_bsb{1}{idx_ch}; hf_bsb{2}{idx_ch}];
    r = [ht_bsb{1}{idx_ch}; ht_bsb{2}{idx_ch}];    
    [echo,tap] = measure_echo_tap(r,coe);
    
    show_data(to_pow_dB(echo),sprintf('echo,channel =%d',idx_ch));
    show_data(to_pow_dB(tap),sprintf('tap,channel =%d',idx_ch));
end

%----------------------------------------------------
%       check the frequency domain channel  
%---------------------------------------------------


