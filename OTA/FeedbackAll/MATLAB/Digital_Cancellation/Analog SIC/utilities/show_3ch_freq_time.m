function h = show_3ch_freq_time(dt)

NFFT = 4096;
fs = 204.8e6;

% figure;
% pause(0.00001);
% frame_h = get(handle(gcf),'JavaFrame');
% set(frame_h,'Maximized',1); 

figure('units','normalized','outerposition',[0 0 1 1]);

N_channel = 3;
df = cell(1,N_channel);
for idx = 1:N_channel
    df{idx} = 20*log10(abs(fftshift(fft(dt{idx},NFFT))));
end

f = (-NFFT/2:NFFT/2-1)*fs/NFFT/1e6;

% freq
subplot(3,3,1); 
h{1} = plot(f,df{1}(1:NFFT)); 
xlabel('MHz')
ylabel('pow(dB)')
legend('ch1 frequency');

subplot(3,3,2); 
h{2} = plot(f,df{2}(1:NFFT));
xlabel('MHz')
ylabel('pow(dB)')
legend('ch2 frequency');

subplot(3,3,3); 
h{3} = plot(f,df{3}(1:NFFT));
xlabel('MHz')
ylabel('pow(dB)')
legend('ch3 frequency');

% time real
t = 1:length(dt{1});
subplot(3,3,4); 
h{4} = plot(t,real(dt{1}));
legend('ch1 time(real)');

t = 1:length(dt{2});
subplot(3,3,5); 
h{5} = plot(t,real(dt{2}));
legend('ch2 time(real)');

t = 1:length(dt{3});
subplot(3,3,6); 
h{6} = plot(t,real(dt{3}));
legend('ch3 time(real)');

% time imag
t = 1:length(dt{1});
subplot(3,3,7); 
h{7} = plot(t,imag(dt{1}));
legend('ch1 time(imag)');

t = 1:length(dt{2});
subplot(3,3,8); 
h{8} = plot(t,imag(dt{2}));
legend('ch2 time(imag)');

t = 1:length(dt{3});
subplot(3,3,9); 
h{9} = plot(t,imag(dt{3}));
legend('ch3 time(imag)');

