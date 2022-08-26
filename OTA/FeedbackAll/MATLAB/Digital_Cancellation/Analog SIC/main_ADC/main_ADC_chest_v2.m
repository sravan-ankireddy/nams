clear all;
close all;

% ADC measurement of tap and echo response
% estimate channel and save the frequency response to files
set_env();

measure_mode    = 2; 
% 0:measure tap
% 1:measure echo  
% 2:differential measurement for taps
% 3:differential measurement for echos


flag_no_window = 1;
N_channel      = 1;
flag_dont_save = 0; % don't save anything
max_cap        = 10;

% for differential tap measurement
DAC_channel_num_ref =  7;

echo_folder = sprintf('..\\data\\20180410_ADC_diff');
tap_folder = '..\\data\\20180410_ADC_diff';


flag_reset_all_pic = 1;
switch measure_mode
    case 0 % tap measurement
        flag_diff_measure = 0;
        flag_set_one_pic_value = 1;
        flag_set_two_pic_value = 0;
        flag_save_taps = 1;
        flag_save_echo = 0;
        %DAC_scan = [1 2 3 4 5 6 7 8 10 11 12 13 15];
        %DAC_scan = [10 11 12 13 15];
        %DAC_scan = [1 2 3 4 5 6 7 8];
        DAC_scan = DAC_channel_num_ref;        
    case 1 % echo measurement
        flag_diff_measure = 0;
        flag_set_one_pic_value = 0;
        flag_set_two_pic_value = 0;
        flag_save_taps = 0;
        flag_save_echo = 1;
        DAC_scan = 10;
    case 2 % differential measurement taps
        flag_diff_measure = 1;
        flag_set_one_pic_value = 0;
        flag_set_two_pic_value = 1;
        flag_save_taps = 1;
        flag_save_echo = 0;
        DAC_scan = [1 2 3 4 5 6 7 8 10 11 12 13 15];    
        %DAC_scan = 7;
    case 3 % differential measurement echos 
        flag_diff_measure = 1;
        flag_set_one_pic_value = 1;
        flag_set_two_pic_value = 0;
        flag_save_taps = 0;
        flag_save_echo = 1;
        DAC_scan = DAC_channel_num_ref;
end

if flag_dont_save == 1
    flag_save_taps = 0;
    flag_save_echo = 0;
end

DAC_map = [5 2 4 3 0 6 7 1 -1 10 14 12 13 -1 8 -1];
          %1 2 3 4 5 6 7 8  9 10 11 12 13 14 15 16

%----- frequency domain filter ------
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

%----- load parameters ------
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


%---------------------------------------
%           load tx signal
%---------------------------------------
dt_bsb = cell(1,N_channel);

if N_channel == 1 % 1 wideband channel
    params.filename = '..\data\tx\OFDM_819.2M_QPSK_cpx.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
else % 2 narrow band channel
    params.filename = '..\data\tx\OFDM_2ch_192M_QPSK_ch1_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{1} = read_from_h_file(params);
    
    params.filename = '..\data\tx\OFDM_2ch_192M_QPSK_ch2_tilt_bsb.h';
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{2} = read_from_h_file(params);
end


%----------------------------------
%            setup tcp
%----------------------------------
[tcp_obj, tcp_const,header_size]= setup_tcp('192.168.0.157',7);


%----------------------------------
%          capture data
%----------------------------------


for DAC_channel_num = DAC_scan
    DAC_channel_num
    
    %---------- reset all pic ---------
    if flag_reset_all_pic == 1
        pic_channel = 0:16;
        pic_code = 2^16-1;
        reset_all_pic(tcp_obj,tcp_const,pic_channel,pic_code);
    end
    
    
    %----------- set one pic value ------------
    code_max = 2^16;
    N_codes  = 128;
    code_table  = [1:floor(code_max/N_codes):code_max];
    
    if flag_set_one_pic_value == 1
            coe = 1;
            idx_code = 1;
            pic_code = code_table(idx_code);
            
            send_SPI_cmd(tcp_obj,tcp_const,DAC_map(DAC_channel_num),pic_code)
            pause(0.5);
    elseif flag_set_two_pic_value == 1
        
%         if(DAC_channel_num_ref == DAC_channel_num) % reference tap should be programmed to max value 
%             % set the strongest reference tap
%             pic_code = 1;
%             send_SPI_cmd(tcp_obj,tcp_const,DAC_map(DAC_channel_num_ref),pic_code)
%             pause(0.5); 
%             
%             idx_code = round(128*3/5); % for easier filename only, not ture for the real code value
%         else            
            % set the strongest reference tap
            pic_code = 1;
            send_SPI_cmd(tcp_obj,tcp_const,DAC_map(DAC_channel_num_ref),pic_code)
            pause(0.5);
            
            % set the intented tap measurement
            %idx_code = round(128*3/5);
            idx_code = 1;
            pic_code = code_table(idx_code);
            send_SPI_cmd(tcp_obj,tcp_const,DAC_map(DAC_channel_num),pic_code)
            pause(0.5);
%        end
    end
    
    recap_cnt = 0;
    flag_recapture = 1;
    while(flag_recapture == 1 && recap_cnt < max_cap)
        %--- capture command ------
        ADC_channel = 0;
        N_ADC_samples = 150e3;
        send_ADC_capture_cmd(tcp_obj,tcp_const,N_ADC_samples,0);
        
        rcv_data = fread(tcp_obj, N_ADC_samples*2 + header_size,'int16');
        dr_rf = rcv_data(header_size/2:end).';
        dr_rf = dr_rf./2^15;
        
        NFFT = 4096;
        fs   = 1.6384e9;
        show_freq_time(dr_rf,fs,NFFT);
        
        %----------------------------------
        %       channel estimation
        %----------------------------------
        
        isbsb = 0;
        if N_channel == 1
            if flag_diff_measure == 1
              OFDM_params.th = 8e5;          
            else
              OFDM_params.th = 5e4;
            end
        else
            OFDM_params.th = 5e4;
        end
        [ht_bsb,hf_bsb,hf_pass,ht_pass,flag_recapture] = multi_chest_wrapper(dt_bsb,dr_rf,OFDM_params,sys_params,isbsb,N_channel,f_window_bsb);
        df = hf_bsb;
        
        
        if flag_recapture == 1
            disp('recapture');
            recap_cnt = recap_cnt + 1;
            continue
        else
            figure(10);
            plot(to_pow_dB(ht_bsb)); legend('echo channel in time');
            
            figure(11);
            plot(to_pow_dB(hf_bsb)); legend('echo channel in frequency');
            
            figure(12);
            plot(ht_bsb); legend('echo channel in time(linear)');
            
            
            if flag_save_taps == 1
                if flag_diff_measure == 1
                    filename_taps = sprintf('%s\\tap%d_%d_code%d.mat',tap_folder,DAC_channel_num_ref,DAC_channel_num,idx_code);
                else
                    filename_taps = sprintf('%s\\tap%d_code%d.mat',tap_folder,DAC_channel_num,idx_code);
                end
                save(filename_taps,'df');
            end
            
            if flag_save_echo == 1
                if flag_diff_measure == 1
                    filename_echo = sprintf('%s\\echo_tap%d.mat',echo_folder,DAC_channel_num_ref);                    
                else
                    filename_echo = sprintf('%s\\echo.mat',echo_folder);                                        
                end
                save(filename_echo,'df');
            end
            
            recap_cnt = 0;
            
        end
    end
end



