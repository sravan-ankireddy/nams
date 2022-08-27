
function [tcp_obj,tcp_const] = setup_pic_tcp()

%% Control settings
tcp_const.header          = swapbytes(uint32(hex2dec('abcd1020')));
tcp_const.type_req        = swapbytes(uint32(hex2dec('00000010')));
tcp_const.type_res        = swapbytes(uint32(hex2dec('00000020')));

tcp_const.cmd_pic         = swapbytes(uint32(hex2dec('0000001A')));
tcp_const.cmd_chgain      = swapbytes(uint32(hex2dec('00000010')));
tcp_const.cmd_bias        = swapbytes(uint32(hex2dec('00000011')));
tcp_const.cmd_slope       = swapbytes(uint32(hex2dec('00000012')));
tcp_const.cmd_spower      = swapbytes(uint32(hex2dec('00000013')));
tcp_const.cmd_rxpower     = swapbytes(uint32(hex2dec('00000014')));
tcp_const.cmd_sic         = swapbytes(uint32(hex2dec('00000015')));
tcp_const.cmd_echo        = swapbytes(uint32(hex2dec('00000016')));
tcp_const.cmd_rprxpw      = swapbytes(uint32(hex2dec('00000017')));
tcp_const.cmd_rplzpw      = swapbytes(uint32(hex2dec('00000018')));
tcp_const.cmd_rplztp      = swapbytes(uint32(hex2dec('00000019')));
tcp_const.cmd_picctrl     = swapbytes(uint32(hex2dec('0000001A')));
tcp_const.cmd_ping        = swapbytes(uint32(hex2dec('0000001B')));
tcp_const.cmd_swstart     = swapbytes(uint32(hex2dec('0000001C')));

% Establish connection to Python server
tcp_obj = tcpclient('localhost',1555, 'Timeout', 60);