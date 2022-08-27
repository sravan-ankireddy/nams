% convert 100k to 819.2M VNA measurement data to baseband

function [df, dt] = convert_to_bsb(df_in)

temp       = df_in;
temp_pass  = [temp(1) temp(1) temp(1:end-1)];      % 0,50k,100k
df         = [temp_pass 0 fliplr(conj(temp_pass(2:end)))]; % convert to baseband
dt         = ifft(df);  