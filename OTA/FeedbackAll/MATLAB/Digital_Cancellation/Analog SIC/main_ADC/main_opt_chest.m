% estimate optical channel by scan through attenuation coefficient
% estimate the coe to voltage table 

clear all;
close all;

set_env();

flag_load_data    = 0;
flag_tcp          = 1;
flag_show_channel = 1;

%--------- parameters ---------
OFDM_params = set_OFDM_params();
sys_params  = set_system_params();

ADC_channel = 0;

code_max = 2^16;
N_codes  = 128;
code_table  = [1:floor(code_max/N_codes):code_max];

N_ADC_samples = 150e3;

DAC_channel_num = 8;
%DAC_map = [4 3 7 0 6 5 2 1]; % 0 is not working 
DAC_map = [5 8 3 0 1 2 6 7]; % 0 is not working 

%--------- setup tcp ----------
if flag_tcp == 1
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.180',7);
end

%--------- load tx signal --------------
params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch1_tilt_bsb.h';
params.convert_to_frac = 1;
params.QF = 15; 
params.iscomplex = 1;
dt_bsb1 = read_from_h_file(params);

params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch2_tilt_bsb.h';
params.convert_to_frac = 1;
params.QF = 15; 
params.iscomplex = 1;
dt_bsb2 = read_from_h_file(params);

%-------- reset all PIC ------
channel = 0:6;
pic_code = 2^16-1;
reset_all_pic(tcp_obj,tcp_const,channel,pic_code);

%idx_code = 1;
%while(idx_code <= length(code_table))
idx_code = 1;
while(idx_code <= 1)
    
    idx_code
    str = sprintf('pic code %d',int32(code_table(idx_code)));
    disp(str);

    if flag_tcp == 1        
    %--------- send SPI code -----------    
        pic_code = code_table(idx_code);
        send_SPI_cmd(tcp_obj,tcp_const,DAC_map(DAC_channel_num),pic_code)
        
    %--------- capture data -----------
        send_ADC_capture_cmd(tcp_obj,tcp_const,N_ADC_samples,ADC_channel);
        rcv_data = fread(tcp_obj, N_ADC_samples*2 + header_size,'int16');
        dr_rf = rcv_data(header_size/2:end).'; % header_size is in bytes, but receive data is array of 2 bytes
        dr_rf = dr_rf/2^15;
    end
    
    %--------- load rx signal ---------
    if flag_load_data == 1
        params.filename = sprintf('data\\rx\\opt\\optical_rx_tap%d_code%d.h',DAC_map(DAC_channel_num),int32(code_table(idx_code)));
        params.convert_to_frac = 0;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    end
    
    NFFT = 4096;
    fs   = 1.6384e9;
    %show_freq_time(dr_rf,fs,NFFT);
    
    %-------- channel estimation ------
    ch_num = 1;
    isbsb = 0;
    [hf1, ht1, flag_recapture]= chest_wrapper(dt_bsb1,dr_rf,OFDM_params,sys_params,ch_num,isbsb);
    
    if flag_recapture == 1      
       continue    
    end
    
    ch_num = 2;
    isbsb = 0;
    [hf2, ht2, flag_recapture]= chest_wrapper(dt_bsb2,dr_rf,OFDM_params,sys_params,ch_num,isbsb);
    
    % convert to real baseband signal
    hf = [fftshift(hf1), fftshift(hf2)];
    hf_bsb = [hf hf(end) fliplr(conj(hf(2:end)))];
    ht_bsb = ifft(hf_bsb);
    
    if flag_show_channel == 1
        t = ( 1:length(ht1) )/(204.8e6)*1e9;
        figure(100); plot(t,to_pow_dB(ht1));
        xlabel('ns');
        ylabel('dB');
        axis([0 t(end) -100 10])
        pause(0.1);
        
        %fig = plot(t,to_pow_dB(ht1));
        %set(fig,'XData',x,'YData',y);
        
        %show_data(to_pow_dB(fftshift(hf1)),'hf1');
        %show_data(to_pow_dB(ht1),'ht1');
        
        %show_data(to_pow_dB(fftshift(hf2)),'hf2');
        %show_data(to_pow_dB(ht2),'ht2');
        
        %show_data(to_pow_dB(fftshift(hf_bsb)),'hf combine');
        %show_data(to_pow_dB(ht_bsb),'ht combine');
    end    
    
    %--------- save channel estimate -----------
%     filename = sprintf('data\\ch\\optical_ch_tap%d_code%d.h',DAC_channel_num,int32(code_table(idx_code)));
%     params.filename = filename;
%     params.convert_to_int = 0;
%     write_to_h_file(params,ht_bsb);

    filename = sprintf('data\\ch_v2\\optical_ch_tap%d_code%d.mat',DAC_channel_num,int32(code_table(idx_code)));    
    save(filename,'ht1','ht2','ht_bsb');
    %keyboard;    
%    pause(2);     
%    close all;

    idx_code = idx_code + 1;
end