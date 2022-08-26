clear all;
clc;


%% Define header and type

header          = swapbytes(uint32(hex2dec('abcd1020')));
type_req        = swapbytes(uint32(hex2dec('00000010')));
type_res        = swapbytes(uint32(hex2dec('00000020')));


%% Command control

cmd_pic         = swapbytes(uint32(hex2dec('0000001A')));
cmd_chgain      = swapbytes(uint32(hex2dec('00000010')));
cmd_bias        = swapbytes(uint32(hex2dec('00000011')));
cmd_slope       = swapbytes(uint32(hex2dec('00000012')));
cmd_spower      = swapbytes(uint32(hex2dec('00000013')));
cmd_rxpower     = swapbytes(uint32(hex2dec('00000014')));
cmd_sic         = swapbytes(uint32(hex2dec('00000015')));
cmd_echo        = swapbytes(uint32(hex2dec('00000016')));
cmd_sweep       = swapbytes(uint32(hex2dec('0000001C')));

%% Program Channel Gain 

p_ch            = swapbytes(uint32(1));
ch_num          = swapbytes(uint32(5));
gain_ch         = swapbytes(uint32(4095));
data_chgain     = [p_ch,ch_num,gain_ch];


len_chgain      = swapbytes(uint32(4*length(data_chgain)));
packet_chgain   = uint32([header,type_req,len_chgain,cmd_chgain,data_chgain]);



%% Program Laser Bias

data_bias          = swapbytes(uint32(100));
len_bias           = swapbytes(uint32(4*length(data_bias)));
packet_databias    = uint32([header,type_req,len_bias,cmd_bias,data_bias]);


%% Program SIC Power

data_spower        = swapbytes(uint32(0));
len_spower         = swapbytes(uint32(4*length(data_spower)));
packet_spower      = uint32([header,type_req,len_spower,cmd_spower,data_spower]);


%% Switch SIC ON/OFF

data_sic           = swapbytes(uint32(0));
len_sic            = swapbytes(uint32(4*length(data_sic)));
packet_sic         = uint32([header,type_req,len_sic,cmd_sic,data_sic]);



%% Switch Echo ON/OFF

data_echo          = swapbytes(uint32(1));
len_echo           = swapbytes(uint32(4*length(data_echo)));
packet_echo        = uint32([header,type_req,len_echo,cmd_echo,data_echo]);


%% Program PIC

pic_channel        = swapbytes(uint32(1));
pic_code           = swapbytes(uint32(3679));
data_pic           = [pic_channel, pic_code];
len_pic            = swapbytes(uint32(4*length(data_pic)));

packet_pic         = uint32([header,type_req,len_pic,cmd_pic,data_pic]);


%% Sweep  Laser Bias

bias_start      = swapbytes(uint32(70));
bias_incr       = swapbytes(uint32(2));
bias_stop       = swapbytes(uint32(120));
data_sweep      = [bias_start, bias_incr, bias_stop];
len_sweep       = swapbytes(uint32(4*length(data_sweep)));
packet_sweep    = uint32([header,type_req,len_sweep,cmd_sweep,data_sweep]);

%% Program rx Power

data_spower        = swapbytes(uint32(32));
len_spower         = swapbytes(uint32(4*length(data_spower)));
packet_rxpower      = uint32([header,type_req,len_spower,cmd_rxpower,data_spower]);

%% Connect and write 

t = tcpclient('localhost',1555, 'Timeout', 60);
write(t,packet_spower);



