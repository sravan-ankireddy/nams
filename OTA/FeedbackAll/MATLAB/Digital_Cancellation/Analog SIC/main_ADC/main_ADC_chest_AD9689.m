clear all;
close all;

% ADC measurement of tap and echo response
% estimate channel and save the frequency response to files
% For AD9689, downsample to 3 different channel directly in the chips  

set_env();

measure_mode    = 1; 
% 0:measure tap
% 1:measure echo  
flag_dont_save     = 0; % don't save anything
max_cap            = 10;
flag_program_taps  = 0;

echo_folder = sprintf('..\\data\\20180607');
tap_folder = '..\\data\\20180607';

%-------------------------------------------------
%                  parameters 
%-------------------------------------------------
% fixed staring point of OFDM symbol
fix_timing = 0;  % 1: fixed timing, 0: restart synch 
idx_start = [1,1,1]; % starting index for 3 different channel, don't care if restart synch

N_ADC_samples = 51200;
N_channel = 3;
fs = 1638.4e6;
fc_1638M = [204e6,396e6,588e6];

OFDM_params = set_OFDM_params();
intp_filter = half_band_filter_design();
sys_params.fs  = fs;     % sampling rate 
sys_params.OSR = 3;      % over-sampling rate , for up-sampling channel estimation
sys_params.DSR = 3;      % down-sampling rat
sys_params.fc  = fc_1638M; 
sys_params.intp_filter = intp_filter;

% pcb setup 
dac_channels = 1:7;
tap_min_pts = [3487, 3487, 3135, 3324, 3541, 3595, 3460];
tap_max_pts = zeros(length(dac_channels));
pic_channels = [1 1 1 1 0 0 0;   % port sign (p=1/n=0)
                6 2 4 7 4 2 0;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel
          

            
% setup code table           
code_max = 2^16;
N_codes  = 128;
code_table  = [1:floor(code_max/N_codes):code_max];
                        
%------------------------------------------------------
%          flag configuration based on mode 
%------------------------------------------------------
switch measure_mode
    case 0 % tap measurement
        flag_save_taps = 1;
        flag_save_echo = 0;
              DAC_scan = [1 2 3 4 5 6 7 8 10 11 12 13 15];        
    case 1 % echo measurement
        flag_save_taps = 0;
        flag_save_echo = 1;
              DAC_scan = 10;
end
if flag_dont_save == 1
    flag_save_taps = 0;
    flag_save_echo = 0;
end



%---------------------------------------
%           load tx signal
%---------------------------------------
filename = cell(1,N_channel);
for idx_ch = 1:N_channel
   filename{idx_ch} = sprintf('..\\data\\tx\\OFDM_3ch_192M_QPSK_ch%d_bsb.h',idx_ch);  
end   
dt_bsb = load_signal_from_file(filename);


%----------------------------------
%            setup tcp
%----------------------------------
[tcp_obj_fpga, tcp_const_fpga,header_size]= setup_tcp_v2('192.168.0.46',7);

% setup pcb board 
[tcp_obj,tcp_const] = setup_pic_tcp();
pic_pcb_setup(tcp_obj,tcp_const);

cnt_value = 5120*3/2-1; % frame period in samples /2 - 1
set_ADC_counter_max_value(tcp_obj_fpga, tcp_const_fpga,cnt_value,0);
set_ADC_counter_max_value(tcp_obj_fpga, tcp_const_fpga,cnt_value,1);
set_ADC_counter_max_value(tcp_obj_fpga, tcp_const_fpga,cnt_value,2);

%----------------------------------
%          capture data
%----------------------------------

switch_sic(tcp_obj,tcp_const,1);  % sic on
switch_echo(tcp_obj,tcp_const,0); % echo off

for idx_taps = DAC_scan
    idx_taps
    %----------------------- pcb setup ---------------------
    % set all channel gain to min except the current one 
    chgain_level = 0;
    set_all_channel_gains(tcp_obj,tcp_const,chgain_level);
    chgain_level = 4095;
    set_one_channel_gain(tcp_obj,tcp_const,pic_channels(1,idx_taps),pic_channels(2,idx_taps),chgain_level);    
    
    % set all taps to their min    
    for idx = 1:length(dac_channels)        
       set_one_tap_level(tcp_obj,tcp_const,dac_channels(idx),tap_min_pts(idx))      
    end
    % set current taps to max  
    set_one_tap_level(tcp_obj,tcp_const,dac_channels(idx_taps),tap_max_pts(idx_taps));
    %-------------------------------------------------------
    
    recap_cnt = 0;
    flag_recapture = 1;
    while(flag_recapture == 1 && recap_cnt < max_cap)
        
        %--- capture command ------                              
        rx = cell(1,N_channel);
             
        for idx_ch = 1:N_channel
            %----------------------------------
            %       ADC capture command
            %----------------------------------
            rx{idx_ch} = capture_ADC_sample(tcp_obj_fpga,tcp_const_fpga,N_ADC_samples,idx_ch-1,header_size);
            rx{idx_ch} = double(rx{idx_ch});     
            
            NFFT = 4096;
            fs = 204.8e6;
            show_freq_time(rx{idx_ch},fs,NFFT,sprintf('ch = %d',idx_ch-1));
        end
        %----------------------------------
        %       channel estimation
        %----------------------------------
        params.isbsb      = 1;
        params.N_channel  = N_channel;
        params.fix_timing = fix_timing; % only the first symbol need to do synchronization
        params.idx_start  = idx_start;
        params.th = 1e5;
        
        [ht_rf,hf_rf,hf_bsb,ht_bsb,flag_recapture] = multi_chest_wrapper(dt_bsb,rx,OFDM_params,sys_params,params);
        
        h = show_3ch_freq_time(ht_bsb);
        
                
        if flag_recapture == 1 %failed capture
            disp('recapture');
            recap_cnt = recap_cnt + 1;
            continue
        else% successful capture 
            %---------------------------------------------
            %            combine all channels 
            %---------------------------------------------         
            %f_start = [108e6,300e6,492e6];
            f_start = [204e6-102.4e6,396e6-102.4e6,588e6-102.4e6];
            
            delta_f = 50e3;
            df = combine_channel(hf_bsb,f_start,delta_f);
           
            show_data(to_pow_dB(df),'channel estimation in all 3 ch')            
            
            %----------------------------------------------
            %        save combined channel measurement 
            %----------------------------------------------                     
            if flag_save_taps == 1
                filename_taps = sprintf('%s\\tap%d_code%d.mat',tap_folder,idx_taps,idx_code);
                save(filename_taps,'df');
            end
            
            if flag_save_echo == 1
                filename_echo = sprintf('%s\\echo.mat',echo_folder);
                save(filename_echo,'df');
            end
            
            recap_cnt = 0;            
        end
    end
end



