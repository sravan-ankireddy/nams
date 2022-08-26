function show_freq_time(dt,fs,NFFT,str)

if nargin < 4
  str = '';  
end


f = (-NFFT/2:NFFT/2-1)*fs/NFFT/1e6;
df = fftshift(fft(dt,NFFT));

figure;
if(isreal(dt))
 
    subplot(2,1,1);
    plot(f,to_pow_dB(df));
    xlabel('MHz')
    ylabel('pow(dB)')
    legend([str ' frequency']);
    
    subplot(2,1,2);
    plot((dt)); legend([ str ' time (real)']);
        
else
    
    subplot(3,1,1);
    plot(f,to_pow_dB(df));
    xlabel('MHz')
    ylabel('pow(dB)')
    legend([str ' frequency']);
    
    subplot(3,1,2);
    plot(real(dt)); legend([ str ' time (real)']);        
    
    subplot(3,1,3);
    plot(imag(dt)); legend([ str ' time (imag)']);        
end