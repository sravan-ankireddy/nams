% convert passband VNA measurement data to baseband without interpolation
% pad zeros only on the unscaned band  

function [df, dt] = pass_convert_to_bsb(df_in,f_start,delta_f,NFFT)

N_zeros    = f_start/delta_f; 

temp_pass  = [zeros(1,N_zeros) df_in(1:end-1)];
temp_pass  = [temp_pass zeros(1,NFFT-length(temp_pass))];

df         = [temp_pass 0 fliplr(conj(temp_pass(2:end)))]; % convert to baseband
dt         = ifft(df);  