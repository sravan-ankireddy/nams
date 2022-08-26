% capture ADC data and show 3 channel results
clear all;
close all;
clc;
set_env();

[tcp_obj, tcp_const,header_size]= setup_tcp_v2('192.168.0.46',7);

cnt_value = 5120*3/2-1;
set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,0);
set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,1);
set_ADC_counter_max_value(tcp_obj, tcp_const,cnt_value,2);

        
NFFT = 4096;
fs = 204.8e6; % Sampling rate
N_ADC_samples = 51200;

N_channel = 3;
data = cell(1,N_channel);

flag_init = 1;
%while(1)
for idx = 1:1e6   
    for idx_ch = 1:N_channel
        data{idx_ch} = capture_ADC_sample(tcp_obj,tcp_const,N_ADC_samples,idx_ch-1,header_size);
        data{idx_ch} = double(data{idx_ch});
    end
    
    if flag_init == 1
        h = show_3ch_freq_time(data);
        flag_init = 0;
    else
        update_3ch_freq_time(h,data);
    end
end
