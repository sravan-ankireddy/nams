function [y]= channel(x,h,SNR)

y = conv(x,h);

if SNR ~= 'inf'
 y = awgn(y,SNR,'measured');    
end

% sig_pow = var(y);
% sigma = sig_pow/10^(SNR/10);
% noise = sqrt(sigma/2)*( randn(1,length(y)) + i*randn(1,length(y)) ); 
% y = y + noise;

