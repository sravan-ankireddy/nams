% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
no_of_frames = 1;
no_of_ofdm_symbols_per_frame = 3600;
no_signal_symbols = 1;
total_ofdm_symbols_per_frame = no_of_ofdm_symbols_per_frame + no_signal_symbols;
size_of_FFT = 64;
cp_length = 16;
no_of_subcarriers = 48;
total_msg_symbols = no_of_ofdm_symbols_per_frame * no_of_subcarriers;
signal_field_symbols = no_signal_symbols * no_of_subcarriers;
mod_order = 2;
mod_type = "NN";
bit_per_symbol = log2(mod_order);
total_no_bits = total_msg_symbols * bit_per_symbol;
block_len = 40; % 40; % Convolutional Code Parameter
term_bits = 0; % 4 % 0;
rate = 1; %1/2; %1/3;
coded_block_len = (block_len + term_bits) / rate;
no_of_blocks = floor((rate * total_no_bits) / (block_len + term_bits));
encoded_no_bits = block_len * no_of_blocks;
no_encoder_out_bits = coded_block_len * no_of_blocks;
extra_bits = total_no_bits - no_encoder_out_bits;
no_preamble_symbols = 4;
preamble_len = no_preamble_symbols * (size_of_FFT + cp_length);
total_no_of_data_samples = total_ofdm_symbols_per_frame * (size_of_FFT + cp_length);
total_no_of_samples = total_no_of_data_samples + preamble_len;
no_of_pilot_carriers = 4;
total_carriers = no_of_pilot_carriers + no_of_subcarriers;
subcarrier_locations = [7:32 34:59];
pilot_carriers = [12 26 40 54];
pilot_values = zeros(size_of_FFT, 1);
pilot_values(pilot_carriers, 1) = [1; 1; 1; -1];

tx_gain = 1;
rx_gain = 1;
sample_offset = 1000000;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
