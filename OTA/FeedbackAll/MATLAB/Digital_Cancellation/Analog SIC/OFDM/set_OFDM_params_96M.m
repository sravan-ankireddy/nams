function params = set_OFDM_params_96M()

params.NFFT   = 4096; % FFT size 
params.NCP    = 1024; % cp size 
params.N_data = 1882; % number of data subcarriers 
params.NRP    = 256;  % number of samples in roll off period

% params.NFFT   = 4096; % FFT size 
% params.NCP    = 1024; % cp size 
% params.N_data = 3800; % number of data subcarriers 
% params.NRP    = 256;  % number of samples in roll off period

% params.NFFT   = 128; % FFT size 
% params.NCP    = 32; % cp size 
% params.N_data = 96; % number of data subcarriers 
% params.NRP    = 8;  % number of samples in roll off period

% params.NFFT   = 256; 
% params.NCP    = params.NFFT/4;
% params.N_data = params.NFFT*3/4;

params.N_sym  = 16; % number of symbols 
params.N_rpt  = 1;  % number of repetitions
params.mod_order = 1024;

params.NOFDM = params.NFFT + params.NCP;
%params.idx_data = [2:params.N_data/2+1 (params.NFFT-params.N_data/2+1):params.NFFT];
params.idx_data = [1:params.N_data/2 (params.NFFT-params.N_data/2+1):params.NFFT];


