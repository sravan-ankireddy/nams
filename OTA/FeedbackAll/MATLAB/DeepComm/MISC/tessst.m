clc;
clearvars;
close all;
% N = 100000;
% data = randi([0,1],N,1);
% enc_data = Encoder(data,'convolutional', N,10);
% mod_data = 2*enc_data - 1;
% dec_data = Decoder(mod_data,'MAP',N,10);
% 
% sum(abs(data - dec_data))/N 
no_of_ofdm_symbols = 800;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_symbols = no_of_ofdm_symbols * no_of_subcarriers;
mod_order = 2;
bit_per_symbol = log2(mod_order);
total_no_bits = total_symbols * bit_per_symbol;
enc_type = 'convolutional'; %'turbo'; %'convolutional'
dec_type = 'MAP';
blk_len = 10;
encoded_no_bits = 0.5 * total_no_bits; %(total_no_bits - 12) / 3;
total_no_of_samples = no_of_ofdm_symbols * (size_of_FFT + cp_length);
no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];

data_to_encode = randi([0 1], encoded_no_bits, 1);
encoded_data = Encoder(data_to_encode, enc_type, encoded_no_bits, blk_len);
demod_data = 2*encoded_data - 1;
    decoded_data = Decoder(demod_data, dec_type, encoded_no_bits, blk_len);

    biterr(decoded_data, data_to_encode)/encoded_no_bits