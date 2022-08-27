% averae frequency domain power, don't care about CP  
function data_f_mean = ave_pow_f_v2(data,params)

NFFT = params.NFFT;

data_f_mean = zeros(1,NFFT);

N_sym = 0;
idx = 1;
while( idx + NFFT < length(data))
    data_f = fftshift(fft(data(idx:idx+NFFT-1),NFFT));
    idx = idx + NFFT;
    
    data_f_mean = data_f_mean +  abs(data_f).^2;
    N_sym = N_sym + 1;
end
data_f_mean = 10*log10(data_f_mean./N_sym);
