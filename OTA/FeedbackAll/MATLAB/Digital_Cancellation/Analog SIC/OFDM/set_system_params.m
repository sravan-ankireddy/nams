function params = set_system_params()

params.fs  = 1.6384e9; % sampling rate 
params.OSR = 3; % over-sampling rate 
params.DSR = 3; % down-sampling rate 
params.fc  = [204e6 588e6];
[params.intp_filter , params.delay ] = half_band_filter_design();