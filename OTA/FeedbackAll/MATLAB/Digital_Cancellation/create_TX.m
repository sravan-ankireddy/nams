clc;
clearvars;
close all;


N        = 10;
NFFT     = 64;

for upsample = 1
    
    [TX, TX_f] = FFT_Symb(N,NFFT,upsample);
    
    %     TX   = TX.*(.8/(max(max(abs(real(TX))),max(abs(imag(TX))))));
    %     TX_f = TX_f.*(.8/(max(max(abs(real(TX_f))),max(abs(imag(TX_f))))));
    
    str_f = strcat("TX_O",num2str(upsample),"_F.bin");
    str   = strcat("TX_O",num2str(upsample),".bin");
    
    write_complex_binary(TX,str);
    write_complex_binary(TX_f,str_f);
end
% figure;
% hold on;
% % plot(plot_freq(TX))
% % plot(plot_freq(TX_f))
% plot(10*log10(abs(fftshift(pwelch(resample(TX_f,4,1),256)))))