% multi channel channel estimation wrapper 

function [ht_rf,hf_rf,hf_bsb,ht_bsb,flag_recapture] = multi_chest_wrapper(dt_bsb,dr_rf,OFDM_params,sys_params,params)

fix_timing = params.fix_timing; 
idx_start  = params.idx_start;
isbsb      = params.isbsb;
N_channel  = params.N_channel;
%f_window   = params.f_window;

ht_rf = 0;

ht_bsb = cell(1,N_channel);
hf_bsb = cell(1,N_channel);
for idx_ch = 1:N_channel
    if isbsb == 1
        dr = dr_rf{idx_ch}; % baseband signal for each channel
    else
        dr = dr_rf; % rf signal is a scalar
    end
    
    [hf_bsb{idx_ch}, ht_bsb{idx_ch}, flag_recapture]= chest_wrapper(dt_bsb{idx_ch},dr,OFDM_params,sys_params,idx_ch,isbsb,fix_timing,idx_start(idx_ch));
    if(isempty(hf_bsb{idx_ch}))
        disp('ERROR! no channel estimation found')
       break;
    end
    % up-convert 
    [ht_rf_sub, cpx]= bsb_to_rf(ht_bsb{idx_ch},sys_params.intp_filter,sys_params.OSR,sys_params.fc(idx_ch),sys_params.fs);
   
    ht_rf = ht_rf + ht_rf_sub ; 
end

hf_rf = fft(ht_rf);

% if nargin == 7
%     % filtering in the frequency domain    
%     hf_rf = hf_rf.*f_window;
%     ht_rf = ifft(hf_rf);
% end
