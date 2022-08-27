function switch_echo(tcp_obj,tcp_const,onoff)
% Switch echo On
% Turns the SIC path on
data_sic           = swapbytes(uint32(onoff));
len_sic            = swapbytes(uint32(4*length(data_sic)));
packet_sic         = uint32([tcp_const.header,tcp_const.type_req,len_sic,tcp_const.cmd_echo,data_sic]);
write(tcp_obj,packet_sic);