function [ht_bsb,hf_bsb,hf_pass,ht_pass] = execute_chest()

% estimate channel and save the frequency response to files 


flag_load_data = 0;
flag_reset_pic = 0;
flag_no_window = 1;

if flag_no_window == 1
    NFFT    = 16384;
    f_window_pass = ones(1,NFFT);
    f_window_bsb  = ones(1,2*NFFT);
else
    f_start = [108e6 492e6];
    f_stop  = [300e6 684e6];
    NFFT    = 16384;
    delta_f = 50e3;
    [f_window_bsb f_window_pass] = cal_f_index(f_start,f_stop,delta_f,NFFT);
end

%--------- parameters ---------
N_channel      = 1;

if N_channel == 1
    % 1 wideband channel
    OFDM_params_819M = set_OFDM_params_819M();
    sys_params_819M  = set_system_params_819M();
    
    OFDM_params = set_OFDM_params_819M();
    sys_params  = set_system_params_819M();
else
    % 2 narrow band channel
    OFDM_params_819M = set_OFDM_params_819M();
    sys_params_819M  = set_system_params_819M();
    
    OFDM_params = set_OFDM_params();
    sys_params = set_system_params();
end

%--------- setup tcp ----------
if flag_load_data == 0
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.180',7);
end

pic_channel = 0;
ADC_channel = 0;

%--------- load tx signal --------------
dt_bsb = cell(1,N_channel);

if N_channel == 1 % 1 wideband channel
    params.filename = 'data\tx\OFDM_819.2M_QPSK_cpx.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
else % 2 narrow band channel
    params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch1_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
    params.filename = 'data\tx\OFDM_2ch_192M_QPSK_ch2_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{2} = read_from_h_file(params);
end

%--------- capture data -----------
if flag_load_data == 1   
    if N_channel == 1 % 1 wideband channel
        params.filename = 'data\rx\OFDM_819M_QPSK_rf_rx_1p6G.h';
        params.convert_to_frac = 0;%1;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    else % 2 narrow band channel
        params.filename = 'data\rx\OFDM_2ch_192M_QPSK_rf_rx_1p6G_tilt.h';
        params.convert_to_frac = 1;
        params.QF = 15;
        params.iscomplex = 0; % note that rf signal is real baseband signal
        dr_rf = read_from_h_file(params);
    end
else           
    [tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.180',7);
    %--- PIC command -------
    if flag_reset_pic == 1
        channel = 0:7;
        pic_code = 2^16-1;
        reset_all_pic(tcp_obj,tcp_const,channel,pic_code);
        pause(0.5)
    end
    %--- capture command ------    
    ADC_channel = 0;        
    N_ADC_samples = 150e3;
    send_ADC_capture_cmd(tcp_obj,tcp_const,N_ADC_samples,0);
    
    rcv_data = fread(tcp_obj, N_ADC_samples*2 + header_size,'int16');
    dr_rf = rcv_data(header_size/2:end).';
    dr_rf = dr_rf./2^15;
end

NFFT = 4096;
fs   = 1.6384e9;
show_freq_time(dr_rf,fs,NFFT);

%-------- channel estimation ------
isbsb = 0;
if N_channel == 1
  OFDM_params.th = 5e4;
else
  OFDM_params.th = 5e4;    
end
[ht_bsb,hf_bsb,hf_pass,ht_pass] = multi_chest_wrapper(dt_bsb,dr_rf,OFDM_params,sys_params,isbsb,N_channel,f_window_bsb);



show_data(to_pow_dB(hf_bsb),'hf');
show_data(to_pow_dB(ht_bsb),'ht');

