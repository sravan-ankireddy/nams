%combine two echo response

function [ht_all,hf_all] = combine_echo(delay_ns,amp,filename,N_gate)

%filename = 'VNA_RS\data\20180314_echo_taps\echo_96M.mat';
%filename = 'VNA_RS\data\20180327_TTD_v2\echo.mat';   

temp = load(filename);
hf = temp.df;

ht = ifft(hf);
ht(N_gate:end)=0;

fs = 1.6384e9;
Ts = 1/fs;


R = 32;
Ts = 1/(fs*R);
ht_over = interp(double(ht),R);
delay = round(delay_ns/Ts);

ht_delay = [zeros(1,delay) ht_over(1:end-delay)];

ht_all = (amp(1)*ht_over + amp(2)*ht_delay)/(amp(1)+amp(2));

ht_all = ht_all(1:R:end);
hf_all = fft(ht_all);