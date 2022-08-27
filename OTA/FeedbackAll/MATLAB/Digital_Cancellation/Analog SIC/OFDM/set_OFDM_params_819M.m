function params = set_OFDM_params_819M()

params.NFFT   = 4096*4; % FFT size 
params.NCP    = 1024*4; % cp size 
params.N_data = 3800*4; % number of data subcarriers 
params.NRP    = 256*4;  % number of samples in roll off period

params.N_sym  = 3; % number of symbols 
params.N_rpt  = 1;  % number of repetitions
params.mod_order = 4;

params.NOFDM = params.NFFT + params.NCP;
params.idx_data = [1:params.N_data/2 (params.NFFT-params.N_data/2+1):params.NFFT];

% power loading 
params.en_pow_loading = 0;
params.idx_data_freq = [0:params.N_data/2-1 -params.N_data/2:-1 ]; % index of data in the frequency domain

total_subcarriers = params.NFFT; % (684e6-108e6)/50e3
params.power_loading = 10.^(10*[0:total_subcarriers-1]/(20*total_subcarriers));
params.f_low = 108e6; % lowest frequency edge of the power loading 
params.delta_f = 50e3; % subcarrier spacing 

% receiver match threshold 
params.th = 5e5;
params.N_match = 4096*4; 


% calculate interested frequency range 

%f_start = [108e6 488e6];
%f_stop  = [300e6 684e6];
f_start = [108e6];
f_stop  = [684e6];
fs      = 1.6384e9;

idx = [];
for idx_band = 1:length(f_start)
  idx      = [idx floor(f_start(idx_band)/params.delta_f):floor(f_stop(idx_band)/params.delta_f)];
end

NFFT_total = fs/2/params.delta_f;
f_window = zeros(1,NFFT_total);
f_window(idx) = ones(1,length(idx));
f_window_mirror = fliplr(f_window(2:end));
params.f_window  = [f_window 0 f_window_mirror];
% symmetry as in 0, 1 2 3, 4 ,-3 -2 -1
scale = NFFT_total/sum(params.f_window);
params.f_window = params.f_window*scale; % make sure the total power is still the same 