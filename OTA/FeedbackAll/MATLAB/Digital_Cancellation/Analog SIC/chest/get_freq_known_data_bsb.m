% get frequency domain known data in baseband

function df = get_freq_known_data_bsb(params,dt)

NFFT  = params.NFFT;
NCP   = params.NCP;
NOFDM = NFFT+NCP;

idx = NCP+1;
df = [];
while( idx+NFFT-1 <= length(dt) )
    temp = fft(dt(idx:idx+NFFT-1),NFFT);
    df = [df ; temp];    
    idx = idx + NOFDM;
%    show_data(to_pow_dB(temp));
%     keyboard;
end