clc;
clearvars;
close all;

N_captures = 300;

TX = read_complex_binary('TX.bin');
SNR_db = 20;
SNR = 10^(SNR_db/10);

RX = zeros(length(TX)*N_captures,1);
for i = 1:N_captures
    Z = (0.5/sqrt(SNR))*(randn(length(TX),1) + 1j*randn(length(TX),1));
    RX((i-1)*length(TX) + 1: i*length(TX)) = TX + Z;
end
write_complex_binary(RX,'RX.bin');
