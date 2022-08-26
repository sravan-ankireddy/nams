function params = set_system_params_819M()

params.fs  = 1.6384e9; % sampling rate 
params.OSR = 1; % over sampling rate ,log2
params.DSR = 1; % down sampling rate 
params.fc  = [409.6e6];
[params.intp_filter , params.delay ] = half_band_filter_design();
