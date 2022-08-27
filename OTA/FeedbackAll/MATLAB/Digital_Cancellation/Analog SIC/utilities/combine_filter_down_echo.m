function [ht_bsb,ht_down,ht_bsb_dc] = combine_filter_down_echo(params)

flag_filter_response = params.flag_filter_response;
N_gate               = params.N_gate;
delay_ns             = params.delay_ns;  % seperation between two peaks, too large will give tap error if window size is too small
amp                  = params.amp;
filename             = params.filename;

sys_params  = set_system_params_819M();

BW = 96e6;
f_start = 108e6;
f_stop  = f_start + BW;
delta_f = 50e3;
NFFT    = 16384;
[f_window_bsb,f_window_pass,f_idx_bsb,f_idx_pass] = cal_f_index(f_start,f_stop,delta_f,NFFT);
    
%--------------------------------------------------
%                    load echo
%--------------------------------------------------
N_gate = 300; % remove the external taps
[ht_bsb,hf_bsb] = combine_echo(delay_ns,amp,filename,N_gate);

if flag_filter_response == 1
    hf_bsb = fft(ht_bsb);
    hf_bsb = hf_bsb.*f_window_bsb;
    ht_bsb = ifft(hf_bsb);
end

% ----------- time gating --------------
ht_bsb(N_gate:end) = 0;
% --------------------------------------
ht_bsb = double(ht_bsb);

% ----------------------------
%    down converted version
% ---------------------------
if BW <= 102.4e6
    sys_params.DSR = 3;
elseif BW <= 204.8e6
    sys_params.DSR = 2;
elseif BW <= 409.6e6
    sys_params.DSR = 1;
else
    sys_params.DSR = 0;
end

sys_params.fc = (f_stop-f_start)/2 + f_start;
sys_params.fs = 1.6384e9;
ht_down = rf_to_bsb(ht_bsb,sys_params.intp_filter,sys_params.DSR,sys_params.fc,sys_params.fs);



ht_bsb_dc = ht_bsb.*exp(-i*2*pi*sys_params.fc/sys_params.fs*(0:length(ht_bsb)-1));
temp_f = fft(ht_bsb_dc);

BW = 100e6;
f_start = 50e3;
f_stop  = f_start + BW;
delta_f = 50e3;
NFFT    = 16384;
[f_window_bsb,f_window_pass,f_idx_bsb,f_idx_pass] = cal_f_index(f_start,f_stop,delta_f,NFFT);
  
temp_f = temp_f.*f_window_bsb;
ht_bsb_dc = ifft(temp_f);






