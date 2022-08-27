% measure residual and compute the delta coefficient , update coe = coe + delta
% use vna for echo/residual measuremnt 

clear all;
close all;

N_ite = 10; % number of iterations 
N_taps = 7;
step_size_init = 10;    % step size for iteration 
tap_trial_set = [3 4 5 6];
alpha = 0.9;
DAC_MAP = [1 2 3 4 5 6 7];
code_max = 4095;

foldername = '..\\data\\20180803_cc3_pcb7';

flag_ite_mode = 1; % 1: iteration based on residual , 0: iteration based on weighted average, 3: gradient descent 

% always compute coefficient directly regardless of the mode 
flag_ite_mode_init = 1; % initial mode  
step_init = 1; % initial step size


NFFT = 16384;
fs = 1.6384e9;
f = (1:NFFT)/NFFT*fs;


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
%-------- measure echo from VNA -----------
switch_sic(tcp_obj,tcp_const,0);  % sic off
switch_echo(tcp_obj,tcp_const,1); % echo on

df = vna_measure_cm(app);
[df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);

filename = sprintf('%s\\echo.mat',foldername);
save(filename,'df','dt');

filename = sprintf('%s\\echo_ite_0.mat',foldername);
save(filename,'df','dt');

echo_pow = sum(abs(df).^2);
echo = df;
%----------------------------------------------------

flag_set_peak_locations = 0;
coe_pre = zeros(N_taps,1);
desired_peaks_pre = 68; % any random number     
freq_DSR = 4;

% always compute coefficient directly regardless of the mode 
[coe_init,coe_delta,code_init,desired_peaks] = sic_wrapper(foldername,flag_set_peak_locations,desired_peaks_pre,coe_pre,step_init,flag_ite_mode_init,freq_DSR);    


% ------------ measure residual from VNA ----------------
switch_sic(tcp_obj,tcp_const,1);  % sic on
switch_echo(tcp_obj,tcp_const,1); % echo on

df = vna_measure_cm(app);
[df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);

filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
save(filename,'df','dt');

residual_pow = sum(abs(df).^2);

figure(1);
h = plot(f,echo(1:NFFT),f,df(1:NFFT));
xlabel('MHz')
ylabel('pow(dB)')
legend('echo','residual');
%-------------------------------------------------------
            

%------------------ measure residual -------------------
flag_set_peak_locations = 1;
desired_peaks_pre = desired_peaks;


residual_pow_min_best = residual_pow;
code = code_init;
for idx = 1:N_ite    
    step_size = step_size_init*ones(1,N_taps);
  
    for idx_tap = tap_trial_set
        trial_end = 0;
        while( trial_end == 0 )
            
            % search for the code that minimize the residual power
            residual_pow_min = 1e10;
            for sign_trial = [1,-1]
                code_trial = code;
                code_trial(idx_tap) = code(idx_tap) + round( (sign_trial*step_size(idx_tap)*residual_pow/echo_pow) );

                if abs(code_trial(idx_tap)) == code_max
                   code_trial(idx_tap) = sign(code_trial(idx_tap))*code_max;
                   %trial_end = 1;
                end
                
                % ------------- program the coefficient ------------------
                program_pic_coe(tcp_obj,tcp_const,DAC_MAP,code_trial);           
                
                % ------------ measure residual from VNA ----------------
                switch_sic(tcp_obj,tcp_const,1);  % sic on
                switch_echo(tcp_obj,tcp_const,1); % echo on
                
                df = vna_measure_cm(app);
                [df, dt] = pass_convert_to_bsb(df,params.f_start,delta_f,NFFT);
                                
                figure(1);
                plot(f,echo(1:NFFT),f,df(1:NFFT));
                xlabel('MHz')
                ylabel('pow(dB)')
                legend('echo','residual');
                title( ['number of iteration' num2str(idx)] );
%                 filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
%                 save(filename,'df','dt');
                
                residual_pow = sum(abs(df).^2);
                if residual_pow < residual_pow_min
                    residual_pow_min = residual_pow;
                    code_min         = code_trial;
                end
                %-------------------------------------------------------
            end                    

             % check if the trial is better than current
            if residual_pow_min <= residual_pow_min_best
                code = code_min;
                residual_pow_min_best = residual_pow_min;
                trial_end = 1;
            else   
                step_size(idx_tap) = step_size(idx_tap)*alpha; % reduce step size               
            end            
        end
    end
end

% filename = sprintf('%s\\coe_delta_all.mat',foldername);
% save(filename,'coe_delta_all');
% 
% filename = sprintf('%s\\coe_all.mat',foldername);
% save(filename,'coe_all');

%% read the successive residual 
% df = cell(1,N_ite+1);
% dt = cell(1,N_ite+1);
% str = cell(1,N_ite+1);
% 
% idx = 0;
% filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
% temp = load(filename,'df','dt');
% df{idx+1} = to_pow_dB(temp.df);
% dt{idx+1} = to_pow_dB(temp.dt);
% str{1} = '0';
% 
% for idx = 1:N_ite
%     filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
%     temp = load(filename,'df','dt');
%     df{idx+1} = to_pow_dB(temp.df);
%     dt{idx+1} = to_pow_dB(temp.dt);    
%     str{idx+1}  = num2str(idx);
% end
% 
% show_data_para(dt,str);
% 
% show_data_para(df,str);




