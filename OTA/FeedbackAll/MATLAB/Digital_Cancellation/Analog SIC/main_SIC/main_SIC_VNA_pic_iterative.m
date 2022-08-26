% measure echo and taps , not compelted yet 

close all
clear all

set_env();

%% params
folder_name ='..\\..\\..\\data\\20180618_cc3';


dac_channels = 1:7;
tap_max_pts = zeros(1,length(dac_channels));

% pcb 1
% tap_min_pts = [3539, 3647,3319,3431,3632,3613,3564];
% pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
%                 1 5 7 7 3 5 3;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel

% pcb 3
tap_min_pts = [3750,3669,3383,3552,3750,3662,3506];
pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
                0 6 4 2 4 6 2;]; % 18-037-1   

DAC_map = cell(1,1);
DAC_map{1} = [1 2 3 4 5 6 7];

N_taps = length(DAC_map{1});

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% make sure the last tap is removed 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% load tap response
[taps_t,taps_f,echo,desired_peaks] = load_echo_taps();


%% setup pcb board 
[tcp_obj,tcp_const] = setup_pic_tcp();
pic_pcb_setup(tcp_obj,tcp_const);

switch_sic(tcp_obj,tcp_const,1);  % sic on
switch_echo(tcp_obj,tcp_const,1); % echo on 

% load code table 
N_branches = 1;
[code_table,code_to_coe_table] = load_code_and_coe_table(N_branches);

% program the initial coefficient 
idx_branches = 1;
temp = load('..\data\20180618_cc3\filter_coe.mat');       
coe_init  = temp.coe;
code_init = temp.code;

program_pic_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code);

%% from code search table
% find the largest coefficient 
[coe_sort idx]= sort(coe_init,'descend');

N_near = 10;
N_step = 2;
N_ite  = N_taps^N_near;
code_search_table = zeros(1,N_near);

for idx = 1:N_ite

      code_search =
      code_search(idx,:) =  code_init + 0.1;
end


%% vna measurement parameters 
params.f_start  = 100e3;
params.f_stop   = 819.2e6;
params.N_points = 16383;
delta_f = 50e3;
NFFT = 16384;  % half FFT size
app = vna_setup_cm(params);

%% measurement 

A = taps_f;

N_ite  = 100;
for idx = 1:N_ite
    % measure frequency response from VNA
    df = vna_measure_cm(app);
    [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);
    
    % windowing the residual
    %     idx_peak_small_w = 0; % any random number for echo
    %     params.peaks            = desired_peaks;
    %     params.w_size           = w_size;
    %     params.flag_filter_only = flag_filter_only;
    %     params.small_w_size     = small_w_size;
    %     params.flag_is_echo     = 1;
    %     params.idx_peak_small_w = idx_peak_small_w;
    %     params.f_window_sig     = f_window_sig_bsb;
    %     params.f_window_base    = f_window_base_bsb;
    %     params.flag_skip_prepare = 0;
    %     [ht_bsb_w,idx_peak_small_w] = window_filter_down_v3(df,params);
    
    
    % program the new codes 
    %code = find_coe_to_code_v2(real(abs(coe{idx_branches})),code_table,code_to_coe_table{idx_branches});
    program_pic_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code);       
end
% % save files
% filename = sprintf('%s\\echo.mat',folder_name);
% save(filename,'df','dt');

