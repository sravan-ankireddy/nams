function show_cancel(dt_cancel,dt_before,dt_reconstruct,fs,NFFT,noise)


f = (-NFFT/2:NFFT/2-1)*fs/NFFT/1e6;

df_cancel = zeros(1,NFFT);
df_before = zeros(1,NFFT);
df_noise  = zeros(1,NFFT);

N = length(dt_cancel);

window = NFFT;
idx = 1;
idx_ave = 1;
while( idx + NFFT-1 <= min(length(dt_cancel),length(df_noise)))
    df_cancel = df_cancel + fft(dt_cancel(idx:idx+NFFT-1),NFFT);
    df_before = df_before + fft(dt_before(idx:idx+NFFT-1),NFFT);
    df_noise = df_noise + fft(noise(idx:idx+NFFT-1),NFFT);

    idx = idx + window;
    idx_ave = idx_ave + 1;
end

df_cancel = df_cancel/(idx_ave-1);
df_before = df_before/(idx_ave-1);
df_noise = df_noise/(idx_ave-1);

df_cancel = to_pow_dB(fftshift(df_cancel));
df_before = to_pow_dB(fftshift(df_before));
df_noise = to_pow_dB(fftshift(df_noise));

figure;
subplot(2,1,1);
plot(f,df_before,f,df_cancel,f,df_noise);
xlabel('MHz')
ylabel('pow(dB)')
legend('before cancellation','after cancellation','noise')
grid on;

subplot(2,1,2);
plot(f,df_before-df_cancel,f,df_before - df_noise);
xlabel('MHz')
ylabel('pow(dB)')
legend('cancellation','max possible cancellation')
grid on;
    
    
figure;
t = 1:N;
subplot(2,1,1); 
plot(t,dt_before(1:N),'-+',t,dt_reconstruct(1:N),'-o'); legend('original','reconstruct');
grid on;

N = min(length(dt_cancel),length(noise));
t = 1:N;
subplot(2,1,2);
plot(t,dt_cancel(1:N),'.-',t,noise(1:N)); legend('residual');
grid on;
