% measure residual and compute the delta coefficient , update coe = coe + delta
% use vna for echo/residual measuremnt

clear all;
close all;
set_env();
N_ite = 1000; % number of iterations
N_taps = 7;
step_size_init = 100;    % step size for iteration
tap_trial_set = [7 6 5 4 3];
tap_trial_set = [ 5 3];
alpha = 0.8;
DAC_MAP = [1 2 3 4 5 6 7];
code_max_lim = 4095;
code_min_lim = 0;
Cancel_Mag = [];
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
params.f_start  = 520e6;
params.f_stop   = 580e6;
params.N_points = 2^7;
params.if_bw = 30e3;

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
% code_init = [3457,3633,4047,3759,3585,3306,3380];
% code_init = [3487,3487,3135,3324,2900,3595,2580];
% code_init = [3441,3480,2650,3500,3290,3777,3809];
code_init = [3409   3550   3600   3290   3260   3480   3350];


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
code = code_init;

residual_pow_search = zeros(1,length(tap_trial_set));
code_search = cell(1,length(tap_trial_set));

step_size = step_size_init*ones(1,N_taps);    
for idx = 1:N_ite
    % measure echo
    switch_sic(tcp_obj,tcp_const,0);  % sic off
    switch_echo(tcp_obj,tcp_const,1); % echo on
    [df,f] = vna_measure_cm(app);
    echo = df;
%     echo_mag_freq = 20*log10(abs(echo));
    echo_t = ifft(echo);
%     echo_mag_time = 20*log10(abs(echo_t));
    echo_pow = sum(abs(echo).^2);
        
    pow_ratio = residual_pow_min_best/echo_pow;
    
    for idx_tap = 1:length(tap_trial_set)
%         str = sprintf('================== tap = %d ==================',tap_trial_set(idx_tap));
%         disp(str);
     
        % search for the code that minimize the residual power
        residual_pow_min = 1e10;
        for sign_trial = [1,-1]
            code_trial = code;
            delta =  round( (sign_trial*step_size(tap_trial_set(idx_tap))*pow_ratio) );
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
            
            
            
            subplot(1,2,1);
            cancel = min(to_pow_dB(echo)-lin2db_f(df));
            Cancel_Mag(idx) = cancel;
            plot(f/1e6,to_pow_dB(echo),f/1e6,lin2db_f(df),'LineWidth',2);
            
            ylim([-70 -5]);
            xlabel('Frequency (MHz)','FontSize',12)
            ylabel('Magnitue(dB)','FontSize',12)
            legend({'Echo','Residual'},'FontSize',12,'Location','southoutside');
            title( 'Frequency Domain' );
            grid on;
            ax=gca;
            ax.GridAlpha = 0.4;
            
                 
           
           
            subplot(1,2,2);
            plot(t,to_pow_dB(echo_t),t,lin2db(dt),'LineWidth',2);
            
         
            xlim([1 8]);ylim([-70 -10]);
            xlabel('Time (ns)','FontSize',12)
            ylabel('Magnitude(dB)','FontSize',12)
            legend({'Echo','Residual'},'FontSize',12,'Location','southoutside');
            title( 'Time Domain');                       
            grid on;
            ax=gca;
            ax.GridAlpha = 0.4;
            
            str = {' Cancelation Amount: ' num2str(cancel)};
           ax1 = axes('Position',[0.432547619047619 0.0578778135048231 0.179357142857143 0.0921757770632368],'Visible','off');
           axes(ax1) % sets ax1 to current axes
           text(.025,0.01,str,'FontSize',14) 
           
           figure(2)

           hist(Cancel_Mag);grid on;
           xlim([20 50]);
       
            
            residual_pow = sum(abs(df).^2);
            if residual_pow < residual_pow_min
                residual_pow_min = residual_pow;
                code_min  = code_trial;
                
            
                
            end
            %-------------------------------------------------------
        end
        
        residual_pow_search(idx_tap) = residual_pow_min;
        code_search{idx_tap} = code_min ; 
       
%        str_max={num2str(max(Cancel_Mag))};
%        text(4,-65,str_max);
    end
     
    [residual_pow_min_best, idx_min] = min(residual_pow_search);
    code = code_search{idx_min}; 
end
    


