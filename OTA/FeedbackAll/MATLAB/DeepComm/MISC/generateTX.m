function TX = generateTX(P, old)

string = strcat("TXData/TX_",num2str(P),".bin");

if old
    TX = read_complex_binary(string);
else
    N        = 5000;
    NFFT     = 64;
    
    for upsample = 1
        
        [~, TX] = FFT_Symb(N,NFFT,upsample);
        
        % Normalization
        TX = TX.*(.8/(max(max(abs(real(TX))),max(abs(imag(TX))))));
        
        % File Names for TX
        str_f = strcat("TX_O",num2str(upsample),"_F.bin");
        str   = strcat("TX_O",num2str(upsample),".bin");
        
        % Convert to bin files
        write_complex_binary(TX,str);
        write_complex_binary(TX_f,str_f);
    end
end
figure;
hold on;
grid on;
plot(10*log10(abs(fftshift(pwelch(resample(TX_f,4,1),256)))))