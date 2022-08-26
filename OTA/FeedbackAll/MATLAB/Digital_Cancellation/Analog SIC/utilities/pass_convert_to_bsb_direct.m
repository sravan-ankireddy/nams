% convert passband VNA measurement data to baseband directly without zero padding 

function [df, dt] = pass_convert_to_bsb_direct(df_in,NFFT)

temp_pass  = [zeros(1,3) df_in];
temp_pass  = [temp_pass zeros(1,NFFT-length(temp_pass))];

df         = [temp_pass 0 fliplr(conj(temp_pass(2:end)))]; % convert to baseband
dt         = ifft(df);  