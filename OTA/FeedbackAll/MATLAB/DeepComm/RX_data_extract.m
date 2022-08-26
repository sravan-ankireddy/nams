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
dec_type = 'turbo'; %'convolutional' 'MAP'
block_len = 40; % Convolutional Code Parameter
term_bits = 4;
rate = 1/3;
coded_block_len = (block_len + term_bits) / rate;
no_of_blocks = floor((rate * total_no_bits) / (block_len + term_bits));
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extraction of the received data
RX = read_complex_binary('RX.bin');

k = 1;
pkt_received = 0;
lt_set = 0;

good_pkt1 = 0;
receive_matrix1 = {};

good_pkt2 = 0;
receive_matrix2 = {};

good_pkt3 = 0;
receive_matrix3 = {};

good_pkt4 = 0;
receive_matrix4 = {};

while k + total_no_of_samples < length(RX)

    if k < 100000
        k = k + 1;
        continue;
    end

    if lt_set == 0

        % STS Packet Detection
        window_size = 16;
        count = 0;
        i = k;

        while count < 12

            corr = (sum(RX(i:i + window_size - 1) .* conj(RX(i + 16:i + 16 + window_size - 1))));
            corr = corr / (sum(RX(i:i + window_size - 1) .* conj(RX(i:i + window_size - 1))));

            if corr > 0.97
                count = count + 1;
            else
                count = 0;
            end

            i = i + 1;
        end

        st_id = i + 16;

        % LTS Symbol Alignment
        L = zeros(200, 1);

        LTS = open('LTS.mat');
        LTS = LTS.LTS;
        lts = ifft(fftshift(LTS(1:size_of_FFT, 1)), size_of_FFT);

        for j = 1:200
            L(j) = sum(RX(st_id + j - 1:st_id + j - 1 +63) .* conj(lts));
        end

        [~, lt_id1] = max(abs(L));
        L(lt_id1) = 0;
        [~, lt_id2] = max(abs(L));
        lt_id = min(lt_id1, lt_id2);

        lt_id = st_id + lt_id - 1 - 16;
    else
        lt_id = lt_id + total_no_of_samples;
    end

    sts_start_id = lt_id - 160;
    sts_end_id = lt_id - 1;

    lts1_start_id = lt_id;
    lts1_end_id = lt_id + 79;

    lts2_start_id = lt_id + 80;
    lts2_end_id = lt_id + 159;
    data_start_id = lt_id + 160;
    data_end_id = data_start_id + total_no_of_data_samples - 1;

    if data_end_id > length(RX)
        break;
    end

    % Packet Extraction
    pkt_received = pkt_received + 1;
    sts = RX(sts_start_id:sts_end_id);
    lts1 = RX(lts1_start_id:lts1_end_id);
    lts2 = RX(lts2_start_id:lts2_end_id);
    y = RX(data_start_id:data_end_id);

    % Coarse Frequency offset
    alpha = (1/16) * angle(sum(conj(sts(1:144)) .* sts(17:160)));

    lts1 = lts1 .* exp(-1j .* (0:79)' * alpha);
    lts2 = lts2 .* exp(-1j .* (80:159)' * alpha);
    y = y .* exp(-1j .* (160:159 + total_no_of_data_samples)' * alpha);

    % Data Arranged
    LTS1 = fftshift(fft(lts1(cp_length + 1:size_of_FFT + cp_length, 1)));
    LTS2 = fftshift(fft(lts2(cp_length + 1:size_of_FFT + cp_length, 1)));

    y = reshape(y, size_of_FFT + cp_length, no_of_ofdm_symbols);
    Y = zeros(size_of_FFT, no_of_ofdm_symbols);

    for i = 1:no_of_ofdm_symbols
        Y(:, i) = fftshift(fft(y(cp_length + 1:size_of_FFT + cp_length, i)));
    end

    % Channel Estimation
    H = zeros(size_of_FFT, 1);

    for j = subcarrier_locations
        H(j) = 0.5 * (LTS1(j) + LTS2(j)) * sign(LTS(j));
    end

    % Channel Equalization and Phase Offset Correction
    Pilots = open('Pilots.mat');
    Pilots = Pilots.Pilots;
    detected_symbols = zeros(total_symbols, 1);
    l = 1;

    for i = 1:no_of_ofdm_symbols
        theta = angle(sum(conj(Y(pilot_carriers, i)) .* Pilots(:, i) .* H(pilot_carriers, 1)));

        for j = subcarrier_locations

            if ~(any(pilot_carriers(:) == j))
                detected_symbols(l, 1) = (Y(j, i) / H(j)) * exp(1j * theta);
                l = l + 1;
            end

        end

    end

    outlier_id = find(abs(detected_symbols) > 10);
    detected_symbols(outlier_id) = detected_symbols(outlier_id) ./ abs(detected_symbols(outlier_id));
    mean_abs_val = mean(abs(detected_symbols));
    detected_symbols = detected_symbols / mean_abs_val;
    % SNR Estimate
    h = abs(ifft(fftshift(H)));
    snr_estimate = 20 * log10(abs(h(1))) - 20 * log10(mean(abs(h(10:end))));

    if snr_estimate > 18
        lt_set = 1;
    else
        lt_set = 0;
    end

    % Constellation View

    % Symbols that were transmitted
    mod_symbols = open('mod_symbols.mat');
    mod_symbols = mod_symbols.mod_symbols;
    data_tx = qamdemod(mod_symbols, mod_order, 'UnitAveragePower', true);

    %     color_map = jet(mod_order);
    %     figure();
    %     hold on;
    %     grid on;
    %
    %     for i = 1:mod_order
    %         scatter(real(detected_symbols(data_tx == i - 1)), imag(detected_symbols(data_tx == i - 1)), [], rand(1, 3))
    %     end

    % Decoder

    demod_data = -qamdemod(detected_symbols, mod_order, 'OutputType', 'llr', 'UnitAveragePower', true);
    demod_data = demod_data(1:end - extra_bits);

    coded_block_length = block_len / rate + 12;
    demod_data = reshape(demod_data, coded_block_length, no_of_blocks);

    Encoded_data = open('encoded_data.mat');
    Encoded_data = Encoded_data.encoded_data;

    decoded_data = Decoder(demod_data, dec_type, no_of_blocks, block_len);
    data_to_encode = open('data_input.mat');
    data_to_encode = data_to_encode.data_input;
    bit_err = biterr(decoded_data, data_to_encode) / encoded_no_bits;
    fprintf("Packet: %d  SNR:  %.2f  BER: %1.4f\n", pkt_received, snr_estimate, biterr(decoded_data, data_to_encode) / encoded_no_bits)
    k = data_end_id;

    if bit_err > .1 && bit_err < 0.4
        good_pkt1 = good_pkt1 + 1;
        DataIn1(good_pkt1, 1:no_of_blocks, 1:block_len) = data_to_encode.';
        DataOut1(good_pkt1, 1:no_of_blocks, 1:coded_block_len) = demod_data.';
        EncOut1(good_pkt1, 1:no_of_blocks, 1:coded_block_len) = Encoded_data.';
    end

    if bit_err > .05 && bit_err < 0.1
        good_pkt2 = good_pkt2 + 1;
        DataIn2(good_pkt2, 1:no_of_blocks, 1:block_len) = data_to_encode.';
        DataOut2(good_pkt2, 1:no_of_blocks, 1:coded_block_len) = demod_data.';
        EncOut2(good_pkt2, 1:no_of_blocks, 1:coded_block_len) = Encoded_data.';
    end

    if bit_err > .01 && bit_err < 0.05
        good_pkt3 = good_pkt3 + 1;
        DataIn3(good_pkt3, 1:no_of_blocks, 1:block_len) = data_to_encode.';
        DataOut3(good_pkt3, 1:no_of_blocks, 1:coded_block_len) = demod_data.';
        EncOut3(good_pkt3, 1:no_of_blocks, 1:coded_block_len) = Encoded_data.';
    end

    if bit_err < .01
        good_pkt4 = good_pkt4 + 1;
        DataIn4(good_pkt4, 1:no_of_blocks, 1:block_len) = data_to_encode.';
        DataOut4(good_pkt4, 1:no_of_blocks, 1:coded_block_len) = demod_data.';
        EncOut4(good_pkt4, 1:no_of_blocks, 1:coded_block_len) = Encoded_data.';
    end

    if pkt_received > 50
        break;
    end

end

save('DataIn1_0.1_0.4.mat', 'DataIn1');
save('DataOut1_0.1_0.4.mat', 'DataOut1');
save('EncOut1_0.1_0.4.mat', 'EncOut1');
save('DataIn2_0.05_0.1.mat', 'DataIn2');
save('DataOut2_0.05_0.1.mat', 'DataOut2');
save('EncOut2_0.05_0.1.mat', 'EncOut2');
save('DataIn3_0.01_0.05.mat', 'DataIn3');
save('DataOut3_0.01_0.05.mat', 'DataOut3');
save('EncOut3_0.01_0.05.mat', 'EncOut3');
save('DataIn4_0.0_0.01.mat', 'DataIn4');
save('DataOut4_0.0_0.01.mat', 'DataOut4');
save('EncOut4_0.0_0.01.mat', 'EncOut4');
