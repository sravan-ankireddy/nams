function [taps_f_out,taps_t_out] = filter_taps(taps_f,f_window)

N_taps = size(taps_f,2);

NFFT = 16384*2;
taps_f_out = zeros(NFFT,N_taps);
taps_t_out = zeros(N_taps,NFFT);
for idx = 1:N_taps    
    temp = taps_f(:,idx).';
    temp = temp.*f_window;
    
    taps_f_out(:,idx) = temp.';
    taps_t_out(idx,:) = ifft(temp,NFFT);
end