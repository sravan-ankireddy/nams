clearvars;
clc;
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_frames = 20;
no_of_ofdm_symbols_per_frame = 4500;
no_signal_symbols = 1;
total_ofdm_symbols_per_frame = no_of_ofdm_symbols_per_frame + no_signal_symbols;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_msg_symbols = no_of_ofdm_symbols_per_frame * no_of_subcarriers;
signal_field_symbols = no_signal_symbols * no_of_subcarriers;
mod_order = 2;
bit_per_symbol = log2(mod_order);
total_no_bits = total_msg_symbols * bit_per_symbol;
enc_type = 'turbo'; %'convolutional'
dec_type = 'turbo'; %'convolutional' 'MAP'

% msg_len = 36; % Convolutional Code Parameter
% term_bits = 4; % 0;
% rate = 36/63; %1/2; % 1/3;
% code_len = 63;%(block_len + term_bits) / rate;

% LDPC Params
load('data_files/par_gen_data/G_BCH_63_36.mat','G');
load('data_files/par_gen_data/H_BCH_63_36.mat','H');
G = double(G);
H = double(H);
msg_len = size(G,2);
code_len = size(G,1);
rate = msg_len/code_len;

no_of_blocks = floor(total_no_bits / code_len);%(block_len + term_bits));
encoded_no_bits = msg_len * no_of_blocks;
no_encoder_out_bits = code_len * no_of_blocks;
extra_bits = total_no_bits - no_encoder_out_bits;
no_preamble_symbols = 4;
preamble_len = no_preamble_symbols * (size_of_FFT + cp_length);
total_no_of_data_samples = total_ofdm_symbols_per_frame * (size_of_FFT + cp_length);
total_no_of_samples = total_no_of_data_samples + preamble_len;
no_of_pilot_carriers = 4;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
