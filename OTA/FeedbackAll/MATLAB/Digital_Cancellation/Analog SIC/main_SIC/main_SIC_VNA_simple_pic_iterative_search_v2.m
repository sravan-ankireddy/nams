% measure residual and compute the delta coefficient , update coe = coe + delta
% use vna for echo/residual measuremnt

clear all;
close all;

N_ite = 100; % number of iterations
N_taps = 7;
step_size_init = 1500;    % step size for iteration
tap_trial_set = [7 6 5 4 3];
alpha = 0.8;
DAC_MAP = [1 2 3 4 5 6 7];
code_max_lim = 4095;
code_min_lim = 0;

foldername = '..\\data\\20180705_cc3_pcb3_IA';

flag_ite_mode = 1; % 1: iteration based on residual , 0: iteration based on weighted average, 3: gradient descent

% always compute coefficient directly regardless of the mode
flag_ite_mode_init = 1; % initial mode
step_init = 1; % initial step size




%% setup pcb board
[tcp_obj,tcp_const] = setup_pic_tcp();
pic_pcb_setup(tcp_obj,tcp_const);

switch_sic(tcp_obj,tcp_const,0);  % sic off
switch_echo(tcp_obj,tcp_const,1); % echo on

%% vna measurement parameters
params.f_start  = 500e6;
params.f_stop   = 560e6;
params.N_points = 32;
params.if_bw = 10e3;

% delta_f = (params.f_stop - params.f_start)/params.N_points;
% f = 500e6:delta_f:560e6;
% f = f(1:params.N_points);
NFFT = 256;  % half FFT size
app = vna_setup_cm(params);


%% iterative process
%-------- measure echo from VNA -----------
switch_sic(tcp_obj,tcp_const,0);  % sic off
switch_echo(tcp_obj,tcp_const,1); % echo on

[df,f] = vna_measure_cm(app);

% filename = sprintf('%s\\echo.mat',foldername);
% save(filename,'df','dt');
%
% filename = sprintf('%s\\echo_ite_0.mat',foldername);
% save(filename,'df','dt');

echo_pow = sum(abs(df).^2);
echo = df;
t = 1:length(f);
%----------------------------------------------------

flag_set_peak_locations = 0;
coe_pre = zeros(N_taps,1);
desired_peaks_pre = 68; % any random number
freq_DSR = 4;

% always compute coefficient directly regardless of the mode
%[coe_init,coe_delta,code_init,desired_peaks] = sic_wrapper(foldername,flag_set_peak_locations,desired_peaks_pre,coe_pre,step_init,flag_ite_mode_init,freq_DSR);
code_init = [3457,3633,4047,3759,3585,3306,3380]


% ------------ measure residual from VNA ----------------
switch_sic(tcp_obj,tcp_const,1);  % sic on
switch_echo(tcp_obj,tcp_const,1); % echo on

[df,f] = vna_measure_cm(app);

% filename = sprintf('%s\\echo_ite_%d.mat',foldername,idx);
% save(filename,'df','dt');

residual_pow = sum(abs(df).^2);
%-------------------------------------------------------


%------------------ measure residual -------------------
residual_pow_min_best = residual_pow;
code = code_init

residual_pow_search = zeros(1,length(tap_trial_set));
code_search = cell(1,length(tap_trial_set));

step_size = step_size_init*ones(1,N_taps);    
for idx = 1:N_ite
    % measure echo
    switch_sic(tcp_obj,tcp_const,0);  % sic off
    switch_echo(tcp_obj,tcp_const,1); % echo on
    [df,f] = vna_measure_cm(app);
    echo = df;
    echo_t = ifft(echo);
    echo_pow = sum(abs(echo).^2);
        
    pow_ratio = residual_pow_min_best/echo_pow;
    
    for idx_tap = 1:length(tap_trial_set)
        str = sprintf('================== tap = %d ==================',tap_trial_set(idx_tap));
        disp(str);
        
        % search for the code that minimize the residual power
        residual_pow_min = 1e10;
        for sign_trial = [1,-1]
            code_trial = code;
            delta =  round( (sign_trial*step_size(tap_trial_set(idx_tap))*pow_ratio) )
            code_trial(tap_trial_set(idx_tap)) = code(tap_trial_set(idx_tap)) + delta;
            
            if code_trial(tap_trial_set(idx_tap)) > code_max_lim
                code_trial(tap_trial_set(idx_tap)) = code_max_lim;
                %trial_end = 1;
            end
            
            if code_trial(tap_trial_set(idx_tap)) < code_min_lim
                code_trial(tap_trial_set(idx_tap)) = code_min_lim;
                %trial_end = 1;
            end
            
            % ------------- program the coefficient ------------------
            program_pic_coe(tcp_obj,tcp_const,DAC_MAP,code_trial);
            
            % ------------ measure residual from VNA ----------------
            switch_sic(tcp_obj,tcp_const,1);  % sic on
            switch_echo(tcp_obj,tcp_const,1); % echo on
            
            [df,f] = vna_measure_cm(app);
             dt = ifft(df);
            
            figure(1);
            subplot(2,2,1);
            plot(f,to_pow_dB(echo),f,to_pow_dB(df));
            xlabel('MHz')
            ylabel('pow(dB)')
            legend('echo','residual');
            title( ['number of iteration' num2str(idx)] );
            
            subplot(2,2,2);
            plot(f,to_pow_dB(echo)-to_pow_dB(df));
            xlabel('MHz')
            ylabel('pow(dB)')
            legend('cancel amount');
            title( ['number of iteration' num2str(idx)] );
                        
            subplot(2,2,3);
            plot(t,to_pow_dB(echo_t),t,to_pow_dB(dt));
            xlabel('time')
            ylabel('pow(dB)')
            legend('echo','residual');
            title( ['number of iteration' num2str(idx)] );                       

            subplot(2,2,4);                       
            plot(t,to_pow_dB(echo_t)-to_pow_dB(dt));
            xlabel('time')
            ylabel('pow(dB)')
            legend('cancel amount');
            title( ['number of iteration' num2str(idx)] );                       

            
            pause(0.1);
            
            
            residual_pow = sum(abs(df).^2);
            if residual_pow < residual_pow_min
                residual_pow_min = residual_pow;
                code_min  = code_trial;
            end
            %-------------------------------------------------------
        end
        
        residual_pow_search(idx_tap) = residual_pow_min;
        code_search{idx_tap} = code_min ;        
    end
    
    [residual_pow_min_best, idx_min] = min(residual_pow_search);
    code = code_search{idx_min}; 
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




