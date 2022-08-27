clc; clearvars;

%% README
% This script steps through each tap so that only the tap of interest can
% be viewed on the S2VNA program. 
%
% When executed, this script starts with tap 1 being the tap of interest.
%
% To step to the next tap, hit ENTER in the Command Window.
%
% For the current tap:
% - set its tap DAC value to 0
% - set its TIA channel gain to 4095
%
% For all other taps:
% - set its tap DAC value to its maximum attenuation point
% - set its TIA channel gain to 0
%
% This script does not connect to S2VNA so you must open it independently.
%
% This script connects to the PCB/PIC via Mofei's server program.
%
% step_through_pic_pcb_plotter.m is an alternative that connects to the VNA
% and automatically plots and saves the FD and TD responses.

%% Setup

% DAC channels being used
dac_channels = [1:7];
taps = [1:7];

% DAC values resulting in maximum attenuation for each tap
tap_max_atten = [3470, 3599, 3229, 3309, 3518, 3518, 3390]; % 18-037-4
% tap_max_atten = [3217, 3272, 3001, 3190, 3569, 3623, 3542]; % 18-037-3
% tap_max_atten = [3390, 3422, 3181, 3293, 3534, 3550, 3406]; % 18-037-2
% tap_max_atten = [3679,3743,3406,3566,3807,3743,3711]; % 18-037-1
% tap_max_atten = [3502	3470	3502	3422	3502	3454	3374	3438]; % 18-062-7 MC2
% tap_max_atten = zeros(1,8);

% DAC values resulting in minimum attenuation for each tap
tap_min_atten = zeros(length(dac_channels)); 
% tap_min_atten = [2527,2399,2357,1546,2271,2154,2197,2325];

% PIC channels 
% pic_channels = [1 1 1 1 0 0 0;   % port sign (p=1/n=0)
%                 6 2 4 7 4 2 0;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel
            
% pic_channels = [1 1 1 1 0 0 0;   % port sign (p=1/n=0)
%                 5 1 3 7 5 3 1;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel

% pic_channels = [0 1 1 1 0 0 0;   % port sign (p=1/n=0)
%                 0 2 4 7 4 2 1;]; % port number [0,7]; pic_channels(:,n) is tap n fiber channel

% pic_channels = [1 1 1 1 0 0 0;   % port sign (p=1/n=0)
%                 7 2 6 4 4 2 0;];
            
%  pic_channels = [1 1 1 1 0 0 0;   % port sign (p=1/n=0)
%                  7 2 6 5 4 2 0;];

% pic_channels = [1 1 1 1 1 1 1;   % port sign (p=1/n=0)
%                7 7 7 7 7 7 7;];

% pic_channels = [0 1 0 0 1 0 1;   % port sign (p=1/n=0)
%                 1 7 7 5 3 3 5;]; % 18-037-4 (Che-Chun)
%             
% pic_channels = [0 0 0 0 0 0 0;   % port sign (p=1/n=0)
%                 7 7 7 7 7 7 7;]; % 18-037-3       

% pic_channels = [0 0 1 0 1 0 1;   % port sign (p=1/n=0)
%                 7 6 2 4 4 2 6;]; % 18-037-3  
%             
% pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
%                 1 4 7 2 2 4 0 ;];    

%               % E A C G K F B
% pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
%                 2 4 7 2 1 4 0 ;];    
% 
%               % E A C G D F B
% pic_channels = [1 0 0 1 1 1 0;   % port sign (p=1/n=0)
%                 7 6 4 2 0 6 2;];  % green box setup (pcb-3)
% 
% %               % E A C G K F B
% % pic_channels = [0 0 0 1 1 1 0;   % port sign (p=1/n=0)
% %                 1 4 5 1 5 7 2;];  % green box setup (pcb-5)
% 
%               % E A C G D F B
% pic_channels = [0 0 0 1 1 1 0;   % port sign (p=1/n=0)
%                 1 4 5 1 5 7 2;];  % green box setup (pcb-5)
% 
% pic_channels = [0 1 1 0 1 0 0 0;   % port sign (p=1/n=0)
%                 4 5 1 5 7 1 2 3;];  % green box setup (pcb-5) swap polarities
% 
% pic_channels = [1 0 0 0 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
%                 2 4 7 2 1 4 0 ;]; 
%             
%             
% pic_channels = [1 0 0 1 1 1 0 ;   % port sign (p=1/n=0) PCB 1 
%                 6 0 2 1 4 2 7 ;]; 
%             
% pic_channels = [0 1 0 0 1 1 0 1;
%                 0 6 2 4 4 2 7 1;];
%                         
% pic_channels = [1 0 1 0 0 1 1 0;
%                 4 0 6 7 4 2 1 2;]; % PCB-1, MC1-2
%             
% pic_channels = [0 1 0 0 1 1 0 ;
%                 0 6 2 4 4 2 7 ;]; % PCB -1, CC3

            
% pic_channels = [1 0 1 1 0 0 1 ;
%                 6 0 2 4 4 2 1 ;]; % PCB -1, CC3      
%             
% pic_channels = [1 0 1 1 0 0 1 0;
%                 6 0 2 4 4 2 1 7;]; % PCB -7, MC1 best 
%             
pic_channels = [1 0 1 1 0 0 1 0;
                6 0 2 4 4 2 1 7;];  
            
%             
pic_channels = [1 0 1 1 0 0 1 0;
                6 0 2 4 4 2 1 7;];              
channel_gain_min = 0;
channel_gain_max = 4095;

delay = 0.1; % seconds between transmissions
            
%% Control Settings
header          = swapbytes(uint32(hex2dec('abcd1020')));
type_req        = swapbytes(uint32(hex2dec('00000010')));
type_res        = swapbytes(uint32(hex2dec('00000020')));


cmd_pic         = swapbytes(uint32(hex2dec('0000001A')));
cmd_chgain      = swapbytes(uint32(hex2dec('00000010')));

% Establish connection to Python server
t = tcpclient('localhost',1555, 'Timeout', 60);

%% Step Through Taps
for i = 1:length(dac_channels)
    disp(['Tap ' num2str(taps(i)) ' set to minimum attenuation. All other taps set to max.']);
    % Set previous taps to their max attenuation and channels set to zero
    for j = 1:length(dac_channels)
        if j == i
            pause(0.01);
        else
%             Set tap j's DAC channel to maximum attenuation
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
    
    if(j ~= length(pic_channels(1,:)))
      cont = input('Press enter to continue: ');
    end
end