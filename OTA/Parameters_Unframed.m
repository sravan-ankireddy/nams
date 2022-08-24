clearvars;
clc;
% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_frames = 1;
total_msg_symbols = 1e6;
mod_order = 2;
bit_per_symbol = log2(mod_order);
total_no_bits = total_msg_symbols * bit_per_symbol;
enc_type = 'turbo'; %'convolutional'
dec_type = 'turbo'; %'convolutional' 'MAP'
block_len = 40; % Convolutional Code Parameter
term_bits = 4;
rate = 1/3;
coded_block_len = (block_len + term_bits) / rate;
no_of_blocks = floor((rate * total_no_bits) / (block_len + term_bits));
encoded_no_bits = block_len * no_of_blocks;
no_encoder_out_bits = coded_block_len * no_of_blocks;
extra_bits = total_no_bits - no_encoder_out_bits;
total_no_of_data_samples = total_msg_symbols;
total_no_of_samples = total_no_of_data_samples;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
