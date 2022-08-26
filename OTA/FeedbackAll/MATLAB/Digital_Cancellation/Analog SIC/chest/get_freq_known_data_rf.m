
function df = get_freq_known_data_rf(params,dt)

fc    = params.fc; % carrier frequency 
fs    = params.fs; % sampling rate
DSR   = params.DSR; % downampling rate
intp_filter = params.intp_filter;
NFFT  = params.NFFT;
NCP   = params.NCP;
NOFDM = NFFT+NCP;

% downsample first 
dt_bsb = rf_to_bsb(dt,intp_filter,DSR,fc,fs);

% get frequecy domain data
idx = NCP+1;
df = [];
while( idx+NFFT-1 <= length(dt_bsb) )
    temp = fft(dt_bsb(idx:idx+NFFT-1),NFFT);
    df = [df ; temp];    
    idx = idx + NOFDM;
%    show_data(to_pow_dB(temp));
%     keyboard;
end