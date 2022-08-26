% estimate optical channel by scan through attenuation coefficient 
% estimate the coe to voltage table 

clear all;
close all;

set_env();

flag_load_data    = 0;
flag_tcp          = 1;
flag_show_channel = 1;

%DAC_scan = [1 2 3 4 5 6 7 8 10 11 12 13 15];
DAC_scan = [2 3 4 5 6 7 8 10 11 12 13 15];

DAC_map = [5 2 4 3 0 6 7 1 -1 10 14 12 13 -1 8 -1];
          %1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16 

%--------- parameters ---------
OFDM_params = set_OFDM_params_819M();
sys_params  = set_system_params_819M();

ADC_channel = 0;

code_max = 2^16;
N_codes  = 128;
code_table  = [1:floor(code_max/N_codes):code_max];

N_ADC_samples = 150e3;

%--------- setup tcp ----------
if flag_tcp == 1
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.157',7);
end

%--------- load tx signal --------------
params.filename = 'data\tx\OFDM_819.2M_QPSK_cpx.h';
params.convert_to_frac = 1;
params.QF = 15; 
params.iscomplex = 1;
dt_bsb1 = read_from_h_file(params);



for DAC_channel_num = DAC_scan
 DAC_channel_num   
 
max_cap = 10;
for idx_version = 1:10 % capture the same code multiple times
idx_version

recap_cnt = 0;
idx_code = 1;
%while( (idx_code <= length(code_table)) & recap_cnt < max_cap ) 
while(idx_code <= 1)
    
    idx_code
    str = sprintf('pic code %d',int32(code_table(idx_code)));
    disp(str);

    if flag_tcp == 1       
        %-------- reset all PIC ------
        channel = 0:15;
        pic_code = 2^16-1;
        reset_all_pic(tcp_obj,tcp_const,channel,pic_code);
        
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
        %params.filename = sprintf('data\\rx\\opt\\optical_rx_tap%d_code%d.h',DAC_map(DAC_channel_num),int32(code_table(idx_code)));
        params.filename = 'data\rx\OFDM_819.2M_QPSK_sim.h';        
        params.convert_to_frac = 1;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    end
    
    NFFT = 4096;
    fs   = 1.6384e9;
    show_freq_time(dr_rf,fs,NFFT);
    
    %-------- channel estimation ------
    ch_num = 1;
    isbsb = 0;
    OFDM_params.th = 5e4;
    [hf1, ht1, flag_recapture] = chest_wrapper(dt_bsb1,dr_rf,OFDM_params,sys_params,ch_num,isbsb);
    
    if flag_recapture == 1      
         recap_cnt = recap_cnt + 1;
        continue    
    else
         recap_cnt = 0;
    end   
    
    if flag_show_channel == 1
        t = ( 1:length(ht1) )/(204.8e6)*1e9;
        figure(101); plot(t,to_pow_dB(ht1));
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
    filename = sprintf('data\\20180316\\optical_ch_tap%d_code%d_v%d.mat',DAC_channel_num,int32(code_table(idx_code)),idx_version);    
    save(filename,'ht1');

%    keyboard;    
%    pause(2);     
%    close all;

    idx_code = idx_code + 1;
end % code

end% version

end% DAC number