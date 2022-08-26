
function set_one_tap_level(tcp_obj,tcp_const,dac_channel,dac_level)

% DAC Channel Levels
% Sets a desired set of DAC channels to same level
pic_channel        = swapbytes(uint32(dac_channel));
pic_code           = swapbytes(uint32(dac_level));
data_pic           = [pic_channel, pic_code];
len_pic            = swapbytes(uint32(4*length(data_pic)));
packet_pic         = uint32([tcp_const.header,tcp_const.type_req,len_pic,tcp_const.cmd_pic,data_pic]);
write(tcp_obj,packet_pic);

pause(0.2);