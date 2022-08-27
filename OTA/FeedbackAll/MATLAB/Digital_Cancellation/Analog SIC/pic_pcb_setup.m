function pic_pcb_setup(tcp_obj,tcp_const)

%bias was 100 before 
bias_level      = 120;  % value to set laser bias [0,255], 80~110 is linear 
spower_level    = 1;    % SIC power level [0,63?], 0 is min attenuation
rxpower_level   = 1;    % SIC power level [0,63?], 0 is min attenuation
chgain_level    = 4095; % value to set each channel gain [0,4095] , 0 is min

set_laser_bias(tcp_obj,tcp_const,bias_level);

set_all_channel_gains(tcp_obj,tcp_const,chgain_level);
program_sic_power(tcp_obj,tcp_const,spower_level);
program_rx_power(tcp_obj,tcp_const,rxpower_level)


