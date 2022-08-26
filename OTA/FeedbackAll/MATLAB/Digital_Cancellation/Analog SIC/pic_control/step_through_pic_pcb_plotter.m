clc; clearvars; close all;

%% README
% This script is similar to step_through_pic_pcb.m
%
% One key difference, however, is that it connects to the VNA and produces
% plots in the frequency domain and time domain.
%
% Once run, each tap is measured in FD and TD and plotted (on its own plot 
% and together with all other taps).
%
% Complex VNA data is converted to TD data via IFFT or ICZT.
%
% Plots are saved to: step_through_output/
%
% The maximum attenuation points (tap_max_atten) for the PIC under test 
% must be known and the TIA channels must be set for each tap output 
% (pic_channels).
%
% The echo can also be measured and plotted if toggle_echo is true.
%
% This script uses Mofei's server program to interface with the PCB.
%
% Markers on the max/min points can be enabled if markers is true.
%
% FD y-limits and TD x-limits can be set.

%% Setup
% DAC channels being used
dac_channels = [1:7];
taps = 1:7;

toggle_echo = true;

% DAC values resulting in maximum attenuation for each tap
% tap_max_atten = [3487, 3487, 3135, 3324, 3541, 3595, 3460]; % 18-037-4
% tap_max_atten = [3217, 3272, 3001, 3190, 3569, 3623, 3542]; % 18-037-3
% tap_max_atten = [3390, 3422, 3181, 3293, 3534, 3550, 3406]; % 18-037-2
tap_max_atten = [3679,3743,3406,3566,3807,3743,3711]; % 18-037-1
% tap_max_atten = zeros(1,8); % MC1s

% DAC values resulting in minimum attenuation for each tap
tap_min_atten = zeros(length(dac_channels)); 
% tap_min_atten = [2527,2399,2357,1546,2271,2154,2197,2325]; % MC1s

% PIC channels 
pic_channels = [0 1 0 0 1 0 1;   % port sign (p=1/n=0)
                1 7 7 5 3 3 5;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel

                       
% pic_channels = [0 0 1 0 1 0 1;   % port sign (p=1/n=0)
%                 7 6 2 4 4 2 6;]; % 18-037-3       
%  
%                        
% pic_channels = [0 0 1 0 1 0 1;   % port sign (p=1/n=0)
%                 7 6 2 4 4 2 6;]; % 18-037-2     
            
% pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
%                 1 5 7 7 3 5 3;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel
%             
% pic_channels = [0 0 1 0 1 0 1;   % port sign (p=1/n=0)
%                 7 6 2 4 4 2 6;]; % 18-037-1 
% 
% pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1vA
%                 1 4 7 2 2 4 0 ;];     
%             
% % pic_channels = [1 0 0 1 1 1 0 0;   % port sign (p=1/n=0) PCB 1vB
% %                 6 0 2 1 4 2 7 4;];     
            
% pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
%                 1 4 7 2 2 4 0 ;];    

pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
                2 4 7 2 1 4 0 ;];  
      
pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0) PCB 1 
                0 6 4 2 4 6 2;];  % chechun's setup
            
pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
                7 6 4 2 4 6 2;];  % green box setup (pcb-3)
             
              % E A C G K F B
pic_channels = [0 0 0 1 1 1 0;   % port sign (p=1/n=0)
                1 4 5 1 5 7 2;];  % green box setup (pcb-5)

              % E A C G D F B
pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
                7 6 4 2 0 6 2;];  % green box setup (pcb-3)

              % E A C G K F B
pic_channels = [0 0 0 1 1 1 0;   % port sign (p=1/n=0)
                1 4 5 1 5 7 2;];  % green box setup (pcb-5)

              % E A C G D F B
pic_channels = [0 0 0 1 1 1 0;   % port sign (p=1/n=0)
                1 4 5 1 5 7 2;];  % green box setup (pcb-5)
            
              % D A C G E F B
pic_channels = [1 0 0 1 0 1 0 0;   % port sign (p=1/n=0)
                5 4 5 1 1 7 2 3;];  % green box setup (pcb-5) swap
       
pic_channels = [0 1 1 0 1 0 0 0;   % port sign (p=1/n=0)
                4 5 1 5 7 1 2 3;];  % green box setup (pcb-5) swap polarities

              % F G A C D B E K
pic_channels = [1 0 0 1 1 1 0 0;   % port sign (p=1/n=0)
                7 6 4 2 4 6 2 3;]; % 18-037-1, PCB-3 (green box setup)
            
              % F G D C A E B K
pic_channels = [1 0 1 1 0 0 1 0;   % port sign (p=1/n=0)
                7 6 4 2 4 2 6 3;]; % 18-037-1, PCB-3 (green box setup, swapped)
       
% pic_channels = [1 0 1 0 0 1 1 0;
%                 4 0 6 7 4 2 1 2;]; % PCB-1, MC1-2
            
              % F G D C A E B K
pic_channels = [0 0 1 1 0 0 0 0;   % port sign (p=1/n=0)
                0 6 4 2 4 2 1 3;]; % 18-037-1, PCB-5, TTD (green box setup, swapped)
            
              % F A D C G E B K
pic_channels = [0 1 1 1 0 0 0 0;   % port sign (p=1/n=0)
                0 3 4 2 6 1 2 3;]; % 18-037-1, PCB-5, TTD (green box setup, swapped ch 5, swapped A&G)
            
              % F A D C G E B K
pic_channels = [0 1 1 1 0 0 0 0;   % port sign (p=1/n=0)
                0 3 4 2 6 2 1 3;]; % 18-037-1, PCB-5, TTD (green box setup, fixed)
            
[tcp_obj, tcp_const, tap_min_pts, tap_max_pts, pic_channels, args] = prep_for_cancellation('external0',2);
            
channel_gain_min = 0;
channel_gain_max = 4095;

delay = 0.1; % seconds between transmissions

%% Control Settings
header          = swapbytes(uint32(hex2dec('abcd1020')));
type_req        = swapbytes(uint32(hex2dec('00000010')));
type_res        = swapbytes(uint32(hex2dec('00000020')));

cmd_pic         = swapbytes(uint32(hex2dec('0000001A')));
cmd_chgain      = swapbytes(uint32(hex2dec('00000010')));
cmd_sic         = swapbytes(uint32(hex2dec('00000015')));
cmd_echo        = swapbytes(uint32(hex2dec('00000016')));

% Establish connection to Python server
t = tcpclient('localhost',1555, 'Timeout', 60);

%% Script Params
markers     = false;

window      = 'none'; % 'hamming', 'gaussian', 'kaiser', 'none'
beta        = 0.5; % default is 0.5
alpha       = 2.5; % default is 2.5

nifft           = 256; % num IFFT/ICZT points to take
inv_transform   = 'ifft'; % 'ifft' or 'iczt'

t1          = 20;  % ns, xlim min
t2          = 50; % ns, xlim max

save_plots  = {'png'}; % types to save figs as: 'png', 'eps', 'pdf'
prompting   = false;

% Lims on y-axis of freq plot
F_min = -25; % dB
F_max = 25; % dB

% Plotting
lines_row = {'-b','-r','-k','-g','-m','-c','-y','--b','--r','--k','--c','--g','--m','--y',':b',':r',':k',':c',':g',':m',':y'};
mylinewidth = 2;

if(length(save_plots) > 0)
    output_dir = ['step_through_output/' datestr(now,'mm-dd-yyyy_HH-MM-SS') '/']; % output directory based on time of executionend
    mkdir(output_dir);
end

%% VNA Connection
instrument          = 'S2VNA'; 
use_center_and_span = false;
f1_hz               = 100e3;
f2_hz               = 819.2e6;
num_points          = 256;
power_level_dbm     = -20;
parameter           = 'S21';
IF_BW               = 100;
% format              = 'MLOGarithmic';
% format              = 'MLINear';
% format = 'POLar';

% Setup VNA connection
try
    app = actxserver([instrument,'.application']);
catch ME
    disp('Error establishing COM server connection.');
    disp('Check that the VNA application COM server was registered');
    disp('at the time of software installation.');
    disp('This is described in the VNA programming manual.');
    return
end

ready = 0;
count = 0;

while ~ready
    ready = app.ready;
    if count > 20
        disp('Error, instrument not ready.');
        disp('Check that VNA is powered on and connected to PC.');
        disp('The status Ready should appear in the lower right');
        disp('corner of the VNA application window.');
        return
    end
    pause(1)
    count = count + 1;
end

disp(sprintf(app.name));

if use_center_and_span
    app.scpi.get('sense',1).frequency.set('center',f1_hz);
    app.scpi.get('sense',1).frequency.set('span',f2_hz);
else
    app.scpi.get('sense',1).frequency.set('start',f1_hz);
    app.scpi.get('sense',1).frequency.set('stop',f2_hz);
end

app.scpi.get('sense',1).sweep.set('points',num_points);
app.SCPI.get('SENSE',1).BANDwidth.RESolution = IF_BW;

if(instrument(1) ~= 'R')
    app.SCPI.get('SOURce',1).POWer.LEVel.IMMediate.set('AMPLitude',power_level_dbm);
end

% Configure the measurement type
app.scpi.get('calculate',1).get('parameter',1).set('define',parameter);
% app.scpi.get('calculate',1).selected.set('format',format);
app.scpi.trigger.sequence.set('source','bus');

%% Setup Figs
figure('pos',[40 40 900 600]); % FD individual
figure('pos',[40 40 900 600]); % TD individual
figure('pos',[40 40 900 600]); % FD combined
figure('pos',[40 40 900 600]); % TD combined

%% Work
% Switch Echo Off
disp(['Turning echo path to ' num2str(0)]);
data_echo          = swapbytes(uint32(0));
len_echo           = swapbytes(uint32(4*length(data_echo)));
packet_echo        = uint32([header,type_req,len_echo,cmd_echo,data_echo]);
write(t,packet_echo);
pause(delay);

% Switch SIC On
disp(['Turning SIC path to ' num2str(1)]);
data_sic           = swapbytes(uint32(1));
len_sic            = swapbytes(uint32(4*length(data_sic)));
packet_sic         = uint32([header,type_req,len_sic,cmd_sic,data_sic]);
write(t,packet_sic);
pause(delay);

for idx = 1:length(dac_channels)
    i = idx;
    
    disp(['Tap ' num2str(taps(i)) ' set to minimum attenuation. All other taps set to max.']);
    % Set previous taps to their max attenuation and channels set to zero
    for j = 1:length(dac_channels)
        if j == i
            pause(0.01);
        else
            % Set tap j's DAC channel to maximum attenuation
            disp(['Setting DAC channel ' num2str(dac_channels(j)) '   to ' num2str(tap_max_atten(j))]);
            pic_channel        = swapbytes(uint32(dac_channels(j)));
            pic_code           = swapbytes(uint32(tap_max_atten(j)));
            data_pic           = [pic_channel, pic_code];
            len_pic            = swapbytes(uint32(4*length(data_pic)));
            packet_pic         = uint32([header,type_req,len_pic,cmd_pic,data_pic]);
            write(t,packet_pic);
            pause(delay);
            
            % Set tap j's PIC channel gain to min
            disp(['Setting PIC channel ' num2str(pic_channels(1,j)) '.' num2str(pic_channels(2,j)) ' to ' num2str(channel_gain_min)]);
            p_ch            = swapbytes(uint32(pic_channels(1,j)));
            ch_num          = swapbytes(uint32(pic_channels(2,j)));
            gain_ch         = swapbytes(uint32(channel_gain_min));
            data_chgain     = [p_ch,ch_num,gain_ch];
            len_chgain      = swapbytes(uint32(4*length(data_chgain)));
            packet_chgain   = uint32([header,type_req,len_chgain,cmd_chgain,data_chgain]);
            write(t,packet_chgain);
            pause(delay);
        end
    end
    
    j = i;
    % Set tap i's DAC channel to minimum attenuation
    disp(['Setting DAC channel ' num2str(dac_channels(j)) '   to ' num2str(tap_min_atten(j))]);
    pic_channel        = swapbytes(uint32(dac_channels(j)));
    pic_code           = swapbytes(uint32(tap_min_atten(j)));
    data_pic           = [pic_channel, pic_code];
    len_pic            = swapbytes(uint32(4*length(data_pic)));
    packet_pic         = uint32([header,type_req,len_pic,cmd_pic,data_pic]);
    write(t,packet_pic);
    pause(delay);

    % Set tap i's PIC channel gain to max
    disp(['Setting PIC channel ' num2str(pic_channels(1,j)) '.' num2str(pic_channels(2,j)) ' to ' num2str(channel_gain_max)]);
    p_ch            = swapbytes(uint32(pic_channels(1,j)));
    ch_num          = swapbytes(uint32(pic_channels(2,j)));
    gain_ch         = swapbytes(uint32(channel_gain_max));
    data_chgain     = [p_ch,ch_num,gain_ch];
    len_chgain      = swapbytes(uint32(4*length(data_chgain)));
    packet_chgain   = uint32([header,type_req,len_chgain,cmd_chgain,data_chgain]);
    write(t,packet_chgain);
    pause(delay);
    
    app.scpi.trigger.sequence.invoke('single');
    pause(0.5);
    app.scpi.get('calculate',1).selected.set('format','POLar');
    Y_temp = app.scpi.get('calculate',1).selected.data.fdata;
    Y(idx,:) = Y_temp(1:2:end) + 1j .* Y_temp(2:2:end);
    
    figure(1); plot(abs(Y(idx,:)));
    
    f = (app.scpi.get('sense',1).frequency.data) / 1e6; % freq in MHz
    
    % Windowing
    N = length(Y(idx,:));
    if(strcmp(window,'hamming'))
        w = hamming(N)';
    elseif(strcmp(window,'kaiser'))
        w = kaiser(N,beta)';
    elseif(strcmp(window,'gaussian'))
        w = gausswin(N,alpha)';
    else
        w = ones(1,N);
    end
      
    Z = Y(idx,:) .* w;
    Z = [0 Z fliplr(conj(Z))];
    BW = f2_hz - f1_hz;
    M = nifft;
    
    T = 1 / BW;
    T_f = T * N; % total time
    
    if(strcmp(inv_transform,'ifft')) % IFFT
        y(idx,:) = ifft(Z,nifft);
        ty = (0:nifft-1) / (BW) / (nifft / N);
    else % ICZT
        t_1 = t1 * 1e-9;
        t_2 = t2 * 1e-9;
        W = exp(-1j*2*pi*(t_2-t_1)/((M-1)*(T_f)));
        A = exp(1j*2*pi*t_1/(T_f));
        y(idx,:) = czt(Z',M,W,A)' ./ M; % conjugate of transform of conjugate right?
        ty = t_1 + (0:M-1).*(t_2  - t_1)./(M-1);
    end
        
    f1 = figure(1); hold off;
    p1 = plot(f,20.*log10(abs(Y(idx,:))),char(lines_row(idx)),'LineWidth',mylinewidth);
    grid on;
    xlabel('Frequency (MHz)');
    ylabel('Magnitude (dB)');
    if(F_min && F_max)
        ylim([F_min F_max]);
    end

    [Y_max, index_max] = max(Y(idx,:));

    if(markers && false)
        cursorMode1 = datacursormode(f1);
        data_marker1 = cursorMode1.createDatatip(p1);
        marker1 = [f(index_max) Y(idx,index_max) 0];
        set(data_marker1,'Position',marker1);
        updateDataCursors(cursorMode1);
    end
    
    f3 = figure(3); hold on;
    p3 = plot(f,20.*log10(abs(Y(idx,:))),char(lines_row(idx)),'LineWidth',mylinewidth);
    hold on;
    grid on;
    xlabel('Frequency (MHz)');
    ylabel('Magnitude (dB)');
    ylim([F_min F_max]);
    
    if(markers && false)
        cursorMode3 = datacursormode(f3);
        data_marker3 = cursorMode1.createDatatip(p3);
        set(data_marker3,'Position',marker1); % use previously found marker
        updateDataCursors(cursorMode3);
    end
    
    f2 = figure(2); hold off;
    p2 = plot(ty*1e9,(real(y(idx,:))),char(lines_row(idx)),'LineWidth',mylinewidth);
    grid on;
    xlabel('Time (ns)');
    ylabel('Real Amplitude');
%     xlim([t1 t2]);

    if(pic_channels(1,idx) == 1)
        [Y_max, index_max] = min(real(y(idx,:)));
    else
        [Y_max, index_max] = max(real(y(idx,:)));
    end
    
    if(markers)
        cursorMode2 = datacursormode(f2);
        data_marker2 = cursorMode2.createDatatip(p2);
        marker2 = [ty(index_max)*1e9 y(idx,index_max) 0];
        set(data_marker2,'Position',marker2);
        updateDataCursors(cursorMode2);
    end

    f4 = figure(4); hold on;
    p4 = plot(ty*1e9,(real(y(idx,:))),char(lines_row(idx)),'LineWidth',mylinewidth);
    grid on;
    xlabel('Time (ns)');
    ylabel('Real Amplitude');
%     xlim([t1 t2]);
    
    if(markers)
        cursorMode4 = datacursormode(f4);
        data_marker4 = cursorMode2.createDatatip(p4);
        set(data_marker4,'Position',marker2); % use previously found marker
        updateDataCursors(cursorMode4);
    end
    
    if(prompting)
        leg{idx} = char((input('> Line label: ','s')));
        input('Press enter to continue: ');    
    else
        leg{idx} = ['tap ' num2str(idx), ', ch ' num2str(pic_channels(1,idx)) '.' num2str(pic_channels(2,idx))];
    end
    
    figure(1);
    legend(char(leg{idx}));

    figure(2);
    legend(char(leg{idx}));
    
    figure(3);
    legend(char(leg));

    figure(4);
    legend(char(leg));
    
    pause(1);
    
    if(length(save_plots) > 0)
        disp(['Saving FD plot to: ' output_dir]);
        for ii = 1:length(save_plots)
           if(strcmp(char(save_plots(ii)),'eps'))
             print('-f1',[output_dir 'FD_' num2str(idx)],'-depsc');
           elseif(strcmp(char(save_plots(ii)),'png'))
             print('-f1',[output_dir 'FD_' num2str(idx)],'-dpng');
           elseif(strcmp(char(save_plots(ii)),'pdf'))
             figure(1);
             set(gcf,'PaperPosition',[0 0 10 5]); % position plot at left hand corner with width 5 and height 5.
             set(gcf,'PaperSize',[10 5]); % set the paper to have width 5 and height 5.
             saveas(gcf,[output_dir 'FD_' num2str(idx)],'pdf') 
           end
        end
    end  
    
    if(length(save_plots) > 0)
        disp(['Saving TD plot to: ' output_dir])
        for ii = 1:length(save_plots)
           if(strcmp(char(save_plots(ii)),'eps'))
             print('-f2',[output_dir 'TD_' num2str(idx)],'-depsc');
           elseif(strcmp(char(save_plots(ii)),'png'))
             print('-f2',[output_dir 'TD_' num2str(idx)],'-dpng');
           elseif(strcmp(char(save_plots(ii)),'pdf'))
             figure(2);
             set(gcf,'PaperPosition',[0 0 10 5]); % position plot at left hand corner with width 5 and height 5.
             set(gcf,'PaperSize',[10 5]); % set the paper to have width 5 and height 5.
             saveas(gcf,[output_dir 'TD_' num2str(idx)],'pdf') 
           end
        end
    end
        
    pause(delay);
    
%     input('Press enter to continue: ');    
end
%%
if(toggle_echo)
    % Switch Echo On
    disp(['Turning echo path to ' num2str(1)]);
    data_echo          = swapbytes(uint32(1));
    len_echo           = swapbytes(uint32(4*length(data_echo)));
    packet_echo        = uint32([header,type_req,len_echo,cmd_echo,data_echo]);
    write(t,packet_echo);
    pause(delay);

    % Switch SIC Off
    disp(['Turning SIC path to ' num2str(0)]);
    data_sic           = swapbytes(uint32(0));
    len_sic            = swapbytes(uint32(4*length(data_sic)));
    packet_sic         = uint32([header,type_req,len_sic,cmd_sic,data_sic]);
    write(t,packet_sic);
    pause(delay);
    
    % Get sweep from VNA
    app.scpi.trigger.sequence.invoke('single');
    app.scpi.get('calculate',1).selected.set('format','POLar');
    Y_temp = app.scpi.get('calculate',1).selected.data.fdata;
    E = Y_temp(1:2:end) + 1j .* Y_temp(2:2:end);
       
    Z = E .* w;
    Z = [0 Z fliplr(conj(Z))];
    BW = f2_hz - f1_hz;
    M = nifft;
    
    T = 1 / BW;
    T_f = T * N; % total time
    
    if(strcmp(inv_transform,'ifft')) % IFFT
        e = ifft(Z,nifft);
        ty = (0:nifft-1) / (BW) / (nifft / N);
    else % ICZT
        t_1 = t1 * 1e-9;
        t_2 = t2 * 1e-9;
        W = exp(-1j*2*pi*(t_2-t_1)/((M-1)*(T_f)));
        A = exp(1j*2*pi*t_1/(T_f));
        e = czt(Z',M,W,A)' ./ M; % conjugate of transform of conjugate right?
        ty = t_1 + (0:M-1).*(t_2  - t_1)./(M-1);
    end
    
    f3 = figure(3); hold on;
    p3 = plot(f,20.*log10(abs(E(:))),char(lines_row(idx+1)),'LineWidth',mylinewidth);
    hold on;
    grid on;
    xlabel('Frequency (MHz)');
    ylabel('Magnitude (dB)');
%     ylim([F_min F_max]);
    
    f4 = figure(4); hold on;
    p4 = plot(ty*1e9,(real(e)),char(lines_row(idx+1)),'LineWidth',mylinewidth);
    grid on;
    xlabel('Time (ns)');
    ylabel('Real Amplitude');
%     xlim([t1 t2]);
    
    [Y_max, index_max] = max(real(e));
    
    if(markers)
        cursorMode4 = datacursormode(f4);
        data_marker4 = cursorMode2.createDatatip(p4);
        marker4 = [ty(index_max)*1e9 e(index_max) 0];
        set(data_marker4,'Position',marker4);
        updateDataCursors(cursorMode4);
    end
    
    leg{idx+1} = ['echo'];    
    
    figure(3);
    legend(char(leg));
    
    figure(4);
    legend(char(leg));
end

%% Save combined plots
if(length(save_plots) > 0)
    disp(['Saving combined FD plot to: ' output_dir]);
    for ii = 1:length(save_plots)
       if(strcmp(char(save_plots(ii)),'eps'))
         print('-f3',[output_dir 'FD_combined'],'-depsc');
       elseif(strcmp(char(save_plots(ii)),'png'))
         print('-f3',[output_dir 'FD_combined'],'-dpng');
       elseif(strcmp(char(save_plots(ii)),'pdf'))
         figure(3);
         set(gcf,'PaperPosition',[0 0 10 5]); % position plot at left hand corner with width 5 and height 5.
         set(gcf,'PaperSize',[10 5]); % set the paper to have width 5 and height 5.
         saveas(gcf,[output_dir 'FD_combined'],'pdf') 
       end
    end
end  

if(length(save_plots) > 0)
    disp(['Saving combined TD plot to: ' output_dir])
    for ii = 1:length(save_plots)
       if(strcmp(char(save_plots(ii)),'eps'))
         print('-f4',[output_dir 'TD_combined'],'-depsc');
       elseif(strcmp(char(save_plots(ii)),'png'))
         print('-f4',[output_dir 'TD_combined'],'-dpng');
       elseif(strcmp(char(save_plots(ii)),'pdf'))
         figure(4);
         set(gcf,'PaperPosition',[0 0 10 5]); % position plot at left hand corner with width 5 and height 5.
         set(gcf,'PaperSize',[10 5]); % set the paper to have width 5 and height 5.
         saveas(gcf,[output_dir 'TD_combined'],'pdf') 
       end
    end
end

%% Save important variables in .mat file for loading later
disp(['Saving variables to: ' output_dir 'vars.mat']);
save([output_dir 'vars.mat'],'channel_gain_max','channel_gain_min','dac_channels','tap_max_atten','tap_min_atten','Y','y','f1_hz','f2_hz','BW','alpha','beta','window','inv_transform','nifft','M','N','output_dir','parameter','power_level_dbm','t1','t2','T','T_f');
disp('Complete.');
