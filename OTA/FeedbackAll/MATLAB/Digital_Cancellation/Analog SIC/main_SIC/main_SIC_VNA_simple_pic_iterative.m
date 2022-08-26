% measure residual and compute the delta coefficient , update coe = coe + delta
% use vna for echo/residual measuremnt 

clear all;
close all;

N_taps = 7;
step = 0.9;
step_init = 1;

foldername = '..\\data\\20180627_cc3_pcb3_ite';

flag_residual_ite = 1; % 1: iteration based on residual , 0: iteration based on weighted average

%% setup pcb board 
[tcp_obj,tcp_const] = setup_pic_tcp();
pic_pcb_setup(tcp_obj,tcp_const);

switch_sic(tcp_obj,tcp_const,0);  % sic off
switch_echo(tcp_obj,tcp_const,1); % echo on 

%% vna measurement parameters 
params.f_start  = 100e3;
params.f_stop   = 819.2e6;
params.N_points = 16383;
delta_f = 50e3;
NFFT = 16384;  % half FFT size
app = vna_setup_cm(params);


%% iterative process
% ------------- measure initial echo ---------------------
switch_sic(tcp_obj,tcp_const,0);  % sic off
switch_echo(tcp_obj,tcp_const,1); % echo on

% measure echo/residual from VNA
df = vna_measure_cm(app);
[df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);

filename = sprintf('%s\\echo.mat',foldername);
save(filename,'df','dt');

filename = sprintf('%s\\echo_ite_0.mat',foldername);
save(filename,'df','dt');

flag_set_peak_locations = 0;
coe_pre = zeros(N_taps,1);
desired_peaks_pre = 68; % any random number     


[coe_init,coe_delta,code_init,desired_peaks] = sic_wrapper(foldername,flag_set_peak_locations,desired_peaks_pre,coe_pre,step_init,flag_residual_ite);    
coe_init
coe_delta
%keyboard;

%------------------ measure residual -------------------
flag_set_peak_locations = 1;
desired_peaks_pre = desired_peaks;
coe_pre = coe_init;


coe_delta_all = [];
coe_all       = [];

N_ite = 10;
for idx = 1:N_ite      
    % ------------ measure echo from VNA ----------------
    if flag_residual_ite == 0
        switch_sic(tcp_obj,tcp_const,0);  % sic off
        switch_echo(tcp_obj,tcp_const,1); % echo on       
    end
    
    df = vna_measure_cm(app);
    [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);

    filename = sprintf('%s\\echo.mat',foldername);
    save(filename,'df','dt');

%     filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
%     save(filename,'df','dt');

    [coe,coe_delta,code,desired_peaks] = sic_wrapper(foldername,flag_set_peak_locations,desired_peaks_pre,coe_pre,step,flag_residual_ite);    
     coe_pre = coe;
    idx
    coe       
    coe_delta 
    coe_all = [coe_all; coe.'];
    coe_delta_all = [coe_delta_all; coe_delta.'];
    
    % measure residual, just for observation  
    switch_sic(tcp_obj,tcp_const,1);  % sic off
    switch_echo(tcp_obj,tcp_const,1); % echo on

    df = vna_measure_cm(app);
    [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);
    
    filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
    save(filename,'df','dt');    
end

filename = sprintf('%s\\coe_delta_all.mat',foldername);
save(filename,'coe_delta_all');

filename = sprintf('%s\\coe_all.mat',foldername);
save(filename,'coe_all');

%% read the successive residual 
df = cell(1,N_ite+1);
dt = cell(1,N_ite+1);
str = cell(1,N_ite+1);

idx = 0;
filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
temp = load(filename,'df','dt');
df{idx+1} = to_pow_dB(temp.df);
dt{idx+1} = to_pow_dB(temp.dt);
str{1} = '0';

for idx = 1:N_ite
    filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
    temp = load(filename,'df','dt');
    df{idx+1} = to_pow_dB(temp.df);
    dt{idx+1} = to_pow_dB(temp.dt);    
    str{idx+1}  = num2str(idx);
end

show_data_para(dt,str);

show_data_para(df,str);




