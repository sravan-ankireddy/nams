% % clc;
% clearvars;
% close all;

RX = read_complex_binary('RX.bin');

% figure;
% plot(abs(RX.*conj(RX)));
window_size = 16;
corr1 = zeros(1000000,1);
for tt = 1:1000000
corr1(tt) = (sum(RX(tt:tt + window_size - 1) .* conj(RX(tt + 16:tt + 16 + window_size - 1))));
        corr1(tt) = corr1(tt) / (sum(RX(tt:tt + window_size - 1) .* conj(RX(tt:tt + window_size - 1))));
end


figure;
plot(abs(corr1))