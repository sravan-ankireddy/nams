function set_one_channel_gain(tcp_obj,tcp_const,pn,ch_num,chgain_level)

% Channel Gains
% Sets each channel gain to the same desired value
p_ch            = swapbytes(uint32(pn));% pos/neg selector
ch_num          = swapbytes(uint32(ch_num));% PIC lines from each side (pos/neg)
gain_ch         = swapbytes(uint32(chgain_level));
data_chgain     = [p_ch,ch_num,gain_ch];
len_chgain      = swapbytes(uint32(4*length(data_chgain)));
packet_chgain   = uint32([tcp_const.header,tcp_const.type_req,len_chgain,tcp_const.cmd_chgain,data_chgain]);
write(tcp_obj,packet_chgain);

pause(0.1)