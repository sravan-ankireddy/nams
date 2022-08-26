clc;
clearvars;
close all;


% Extraction of the received data
upsample = 1;
stringTX = strcat("TX_O",num2str(upsample),"_F.bin");
stringRX = strcat("RX_O",num2str(upsample),"_F.bin");

X = read_complex_binary(stringTX);
Y = read_complex_binary(stringRX);
RX = acorr(X,Y);

% Plot the received data frequency response
% plot(10*log10(abs(fftshift(pwelch(resample(Y(1:1e6),4,1),256)))))

% Channel Estimation
h = Equalization_NLMS(X,RX);
H = fftshift(fft(h));

% Plot the time domain channel response
% plot(abs(h))

% Plot the frequency domain channel response
% plot(abs(H))



% Remove CP from the received samples
RX = reshape(X,80, floor(length(RX)/80));

rx = zeros(64, 5000);
for i = 1:5000
    rx(:,i) = fftshift(fft(RX(17:80,i)));
end

scatter(real(rx(:)), imag(rx(:)))




