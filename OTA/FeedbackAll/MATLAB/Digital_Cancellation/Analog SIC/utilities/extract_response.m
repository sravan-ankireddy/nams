% extract tap response from the ADC differential measurement 

function extract_response()

DAC_channel_num_ref =  7;
echo_folder = sprintf('data\\20180410_ADC_diff');
tap_folder = 'data\\20180410_ADC_diff';
idx_code = 77; % for filename only
%idx_code = 1; % for filename only

DAC_scan = [1 2 3 4 5 6 7 8 10 11 12 13 15];


% read reference taps 
filename_taps = sprintf('%s\\tap7_code1',tap_folder);
temp = load(filename_taps,'df');
tap_ref = temp.df;

%---------------------
%    read all taps 
%---------------------
for DAC_channel_num = DAC_scan
    filename_taps = sprintf('%s\\tap%d_%d_code%d.mat',tap_folder,DAC_channel_num_ref,DAC_channel_num,idx_code);
    temp = load(filename_taps,'df');
    tap_combine_f = temp.df;
    
    if DAC_channel_num == DAC_channel_num_ref
      tap_f         = tap_combine_f;
    else
      tap_f         = tap_combine_f - tap_ref;       
    end
    
    %---- observe only ------
    tap_t = ifft(tap_f);    
    tap_combine_t = ifft(tap_combine_f);    
    tap_ref_t     = ifft(tap_ref);   
    %show_data_para({tap_combine_t(100:250),tap_ref_t(100:250),tap_t(100:250)},{'combine','reference','intended'})
    show_data_para({tap_combine_t,tap_ref_t,tap_t},{'combine','reference','intended'})    
    %------------------------
    
    filename_taps = sprintf('%s\\tap%d_code%d.mat',tap_folder,DAC_channel_num,idx_code);   
    df = tap_f; 
    save(filename_taps,'df');    
end

%--------------------
%     read echo
%--------------------
filename_echo = sprintf('%s\\echo_tap%d.mat',echo_folder,DAC_channel_num_ref);                    
temp = load(filename_echo);
echo_combine_f = temp.df;
echo_f = echo_combine_f - tap_ref;

          
%---- observe only ------
echo_t = ifft(echo_f);
echo_combine_t = ifft(echo_combine_f);
tap_ref_t     = ifft(tap_ref);
%show_data_para({tap_combine_t(100:250),tap_ref_t(100:250),tap_t(100:250)},{'combine','reference','intended'})
show_data_para({echo_combine_t,tap_ref_t,echo_t},{'echo combine','echo reference','echo intended'})
%------------------------

df = echo_f;
filename_echo = sprintf('%s\\echo.mat',echo_folder);                                     
save(filename_echo,'df');   

