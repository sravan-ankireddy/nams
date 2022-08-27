clc; clearvars;

%% README
% This script sets various settings of the PCB using Mofei's server.
%
% The following is set when executing this script:
% - tap levels of the PIC via external DAC (dac_level)
% - laser bias level (bias_level)
% - all TIA channel gain levels (chgain_level)
% - SIC power level (spower_level)
% - SIC path enable (sic)
% - echo path enable (echo)
% 
% This script is typically used to initially configure/reset the PCB/PIC as
% desired immediately after turn-on.

%% Parameters
dac_channels    = [1:7];  % DAC channels to program [0,7]
dac_level       = 0;    % value to write to each DAC channel [0,4095]
bias_level      = 140;  % value to set laser bias [0,255]
chgain_level    = 4095; % value to set each channel gain [0,4095]
spower_level    = 0;    % SIC power level [0,63] (0 is min attenuation)
delay           = 0.1;  % seconds to wait between writing packets to server
sic             = 0;    % SIC enable (1=on,0=off)
echo            = 1;    % echo enable (1=on,0=off)

%% Control settings
header          = swapbytes(uint32(hex2dec('abcd1020')));
type_req        = swapbytes(uint32(hex2dec('00000010')));
type_res        = swapbytes(uint32(hex2dec('00000020')));

cmd_pic         = swapbytes(uint32(hex2dec('0000001A')));
cmd_chgain      = swapbytes(uint32(hex2dec('00000010')));
cmd_bias        = swapbytes(uint32(hex2dec('00000011')));
cmd_slope       = swapbytes(uint32(hex2dec('00000012')));
cmd_spower      = swapbytes(uint32(hex2dec('00000013')));
cmd_rxpower     = swapbytes(uint32(hex2dec('00000014')));
cmd_sic         = swapbytes(uint32(hex2dec('00000015')));
cmd_echo        = swapbytes(uint32(hex2dec('00000016')));

% Establish connection to Python server
t = tcpclient('localhost',1555,'Timeout',60);

%% Switch SIC On
% Turns the SIC path on
disp(['Turning SIC path to ' num2str(sic)]);
data_sic           = swapbytes(uint32(sic));
len_sic            = swapbytes(uint32(4*length(data_sic)));
packet_sic         = uint32([header,type_req,len_sic,cmd_sic,data_sic]);
write(t,packet_sic);
pause(delay);

%% Switch Echo Off
disp(['Turning echo path to ' num2str(echo)]);
data_echo          = swapbytes(uint32(echo));
len_echo           = swapbytes(uint32(4*length(data_echo)));
packet_echo        = uint32([header,type_req,len_echo,cmd_echo,data_echo]);
write(t,packet_echo);
pause(delay);

%% Program SIC Power
% Sets SIC power to a desired level
disp(['Setting SIC power level to ' num2str(spower_level)]);
data_spower        = swapbytes(uint32(spower_level));
len_spower         = swapbytes(uint32(4*length(data_spower)));
packet_spower      = uint32([header,type_req,len_spower,cmd_spower,data_spower]);
write(t,packet_spower);
pause(delay);

%% Laser Bias
% Sets laser bias to a desired bias level
disp(['Setting laser bias level to ' num2str(bias_level)]);
data_bias          = swapbytes(uint32(bias_level));
len_bias           = swapbytes(uint32(4*length(data_bias)));
packet_databias    = uint32([header,type_req,len_bias,cmd_bias,data_bias]);
write(t,packet_databias);
pause(delay);

%% DAC Channel Levels
% Sets a desired set of DAC channels to same level
for i = 1:length(dac_channels)
    disp(['Setting DAC channel ' num2str(i) ' to ' num2str(dac_level)]);
    pic_channel        = swapbytes(uint32(dac_channels(i)));
    pic_code           = swapbytes(uint32(dac_level));
    data_pic           = [pic_channel, pic_code];
    len_pic            = swapbytes(uint32(4*length(data_pic)));
    packet_pic         = uint32([header,type_req,len_pic,cmd_pic,data_pic]);
    write(t,packet_pic);
    pause(delay);
end

%% Channel Gains
% Sets each channel gain to the same desired value
for i = 0:1 % pos/neg selector
    for j = 0:7 % PIC lines from each side (pos/neg)
        disp(['Setting PIC channel ' num2str(i) ' ' num2str(j) ' to ' num2str(chgain_level)]);
        p_ch            = swapbytes(uint32(i));
        ch_num          = swapbytes(uint32(j));
        gain_ch         = swapbytes(uint32(chgain_level));
        data_chgain     = [p_ch,ch_num,gain_ch];
        len_chgain      = swapbytes(uint32(4*length(data_chgain)));
        packet_chgain   = uint32([header,type_req,len_chgain,cmd_chgain,data_chgain]);
        write(t,packet_chgain);
        pause(delay);
    end
end

%% Signoff
disp('Setup completed.')