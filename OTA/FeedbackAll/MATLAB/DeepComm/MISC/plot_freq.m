function H = plot_freq(Y)

NFFT        = 2048;
% S = 64*upsample;
for i = 1:10000
    a = randi([500,length(Y)-NFFT]);
    
    Data = Y(a+1:a+NFFT);
    H(1:NFFT,i) = fftshift(fft(Data));
    
end
H = H./NFFT;
H = 20*log10(abs(H));

H = mean(H,2);
% delf = (S-1)/NFFT;
% f  = 1:delf:S;
% 
% plot(f(1:NFFT),H);
end