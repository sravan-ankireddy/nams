% averae frequency domain power 
function data_f_mean = ave_pow_f(data,params)

NOFDM =  params.NFFT + params.NCP;
%NOFDM =  params.NFFT;
NFFT = params.NFFT;
NCP = params.NCP;

data_f_mean = zeros(1,NFFT);

N_sym = 0;
idx = NCP;
while( idx + NFFT < length(data))
    data_f = fftshift(fft(data(idx:idx+NFFT-1),NFFT));
    idx = idx + NOFDM;
    
    data_f_mean = data_f_mean +  abs(data_f).^2;
    N_sym = N_sym + 1;
end
data_f_mean = 10*log10(data_f_mean./N_sym);
