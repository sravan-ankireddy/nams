clc;
clearvars;
close all;

% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_ofdm_symbols = 3840;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_symbols = no_of_ofdm_symbols * no_of_subcarriers;
mod_order = 64;
bit_per_symbol = log2(mod_order);
total_no_bits = total_symbols * bit_per_symbol;
enc_type = 'turbo'; %'convolutional'
block_len = 40; % Convolutional Code Parameter
rate = 1/3;
no_of_blocks = floor((rate * total_no_bits)/ (block_len + 12*rate));
encoded_no_bits = block_len * no_of_blocks;
no_encoder_out_bits = (encoded_no_bits / rate) + 12 * no_of_blocks;
extra_bits = total_no_bits - no_encoder_out_bits;
no_preamble_symbols = 4;
preamble_len = no_preamble_symbols * (size_of_FFT + cp_length);
total_no_of_data_samples = no_of_ofdm_symbols * (size_of_FFT + cp_length);
total_no_of_samples = total_no_of_data_samples + preamble_len;
no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];
TX_attenuation = 0; % dB
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Encoding
data_input = randi([0 1], block_len, no_of_blocks);
save('data_input.mat', 'data_input');
encoded_data = Encoder(data_input, enc_type, no_of_blocks, block_len, rate);
save('encoded_data.mat', 'encoded_data');
encoded_data = [encoded_data(:); zeros(extra_bits, 1)];

% Modulation
mod_symbols = qammod(encoded_data, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);
save('mod_symbols.mat', 'mod_symbols');

% Subcarrier Allocation
P = open('Pilot_matrix.mat');
P = P.PILOT;

Pilots = zeros(no_of_pilot_carriers, no_of_ofdm_symbols);

A = zeros(size_of_FFT, no_of_ofdm_symbols);
k = 1;

for i = 1:no_of_ofdm_symbols

    for j = subcarrier_locations

        if any(pilot_carriers(:) == j)
            A(j, i) = pilot_values(j, 1) * P(1 + mod(i - 1, 127));
        else
            A(j, i) = mod_symbols(k);
            k = k + 1;
        end

    end

    Pilots(:, i) = A(pilot_carriers, i);
end

save('Pilots.mat', 'Pilots');

% IFFT to generate tx symbols
for i = 1:no_of_ofdm_symbols
    IFFT_Data = ifft(fftshift(A(1:size_of_FFT, i)), size_of_FFT);
    A(1:size_of_FFT + cp_length, i) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
end

TX = A(:);

% Normalize the modulated data Power
TX = TX .* (.8 / (max(max(abs(real(TX))), max(abs(imag(TX))))));

% Short Preamble Field
STS = open('STS.mat');
STS = STS.STS;
IFFT_Data = ifft(fftshift(STS(1:size_of_FFT, 1)), size_of_FFT);
sts(1:size_of_FFT + cp_length, 1) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
sts(1) = sts(1) * 0.5;
sts(size_of_FFT + cp_length) = sts(size_of_FFT + cp_length) * 0.5;
sts = [sts; sts];

% Long Preamble Field
LTS = open('LTS.mat');
LTS = LTS.LTS;
IFFT_Data = ifft(fftshift(LTS(1:size_of_FFT, 1)), size_of_FFT);
lts(1:size_of_FFT + cp_length, 1) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
lts = [lts; lts];

% Concatenate the lts and sts to the transmit data
TX = [sts; lts; TX];
TX = TX .* (sqrt(10^(-TX_attenuation / 10)));
% Write the data to a bin file to be used by GNURadio
save('TX.mat', 'TX');
write_complex_binary(TX, 'TX.bin');
