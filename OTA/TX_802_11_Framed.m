function TX_802_11_Framed(seed)
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/utils'));

    run('Parameters.m');
    
    rng(seed);
    %Frame Data
    msg_data = zeros(msg_len, no_of_blocks, no_of_frames);
    enc_data = zeros(code_len, no_of_blocks, no_of_frames);
    mod_data = zeros(total_msg_symbols, no_of_frames);
    tx_data = zeros(total_no_of_samples, no_of_frames);

    % Encoding
    data_start = zeros(no_of_frames, 1);

    for n_frame = 1:no_of_frames

        sts = [];
        lts = [];

        data_input = randi([0 1], msg_len, no_of_blocks);
        if (code == "LDPC" || code == "BCH")
            encoded_data = ldpcEncode(data_input,ldpcEncCfg);
            encoder_data = [encoded_data(:); zeros(extra_bits, 1)];
        elseif (code == "Turbo")
            encoded_data = turbo_encode(data_input, enc_type, no_of_blocks, msg_len, rate);
            encoder_data = [encoded_data(:); zeros(extra_bits, 1)];         
        else
            rel = rs(rs < code_len+1);
            data_pos = sort(rel(1:msg_len));
            encoded_data = zeros(code_len, no_of_blocks);
            for ii = 1:no_of_blocks
                data = data_input(:,ii);
                data_in = zeros(code_len,1);
                data_in(data_pos) = data;
                encoded_data(:,ii) = polar_encode(data_in);
            end
            encoder_data = [encoded_data(:); zeros(extra_bits, 1)];
        end

        % Modulation
        mod_symbols = qammod(encoder_data, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);

        % Signal Frame containing the frame number
        sig_symb = zeros(no_signal_symbols, 1);
        % disp(n_frame);
        F = de2bi(n_frame - 1, 6);

        for i = 1:6
            sig_symb((i - 1) * 8 + 1:i * 8) = F(i) * ones(8, 1);
        end

        sig_symb = qammod(sig_symb, 2, 'InputType', 'bit', 'UnitAveragePower', true);
        % Subcarrier Allocation
        P = open('utils/Pilot_matrix.mat');
        P = P.PILOT;
        Pilots = zeros(no_of_pilot_carriers, total_ofdm_symbols_per_frame);

        A = zeros(size_of_FFT, total_ofdm_symbols_per_frame);
        k = 1;
        l = 1;

        for i = 1:total_ofdm_symbols_per_frame

            for j = subcarrier_locations

                if any(pilot_carriers(:) == j)
                    A(j, i) = pilot_values(j, 1) * P(1 + mod(i - 1, 127));
                else

                    if i <= no_signal_symbols
                        A(j, i) = sig_symb(l);
                        l = l + 1;
                    else
                        A(j, i) = mod_symbols(k);
                        k = k + 1;
                    end

                end

            end

            Pilots(:, i) = A(pilot_carriers, i);
        end

        save('utils/Pilots.mat', 'Pilots');

        % IFFT to generate tx symbols
        for i = 1:total_ofdm_symbols_per_frame
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
        sts(size_of_FFT + cp_length) = sts(size_of_FFT + cp_length);
        sts = [sts(1) * 0.5; sts(2:end); sts; sts(1) * 0.5];

        % Long Preamble Field
        LTS = open('LTS.mat');
        LTS = LTS.LTS;
        IFFT_Data = ifft(fftshift(LTS(1:size_of_FFT, 1)), size_of_FFT);
        lts(1:size_of_FFT + cp_length, 1) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
        lts = [lts(1) * 0.5; lts(2:end); lts; lts(1) * 0.5];

        % Concatenate the lts and sts to the transmit data
        TX = [sts(1:end - 1); sts(end) + lts(1); lts(2:end - 1); lts(end) + TX(1); TX(2:end)];

        % Frame Data
        msg_data(:, :, n_frame) = data_input;
        enc_data(:, :, n_frame) = encoded_data;
        mod_data(:, n_frame) = mod_symbols;
        tx_data(:, n_frame) = TX;
        data_start(n_frame) = (n_frame - 1) * length(TX) + 320 + 1;
    end

    % Save Frame Data
    save('data_files/ota_data/msg_data.mat', 'msg_data');
    save('data_files/ota_data/enc_data.mat', 'enc_data');
    save('data_files/ota_data/mod_data.mat', 'mod_data');
    save('data_files/ota_data/tx_data.mat', 'tx_data');
    save('data_files/ota_data/data_start.mat', 'data_start');
    % Write to Transmitter
    TX = tx_data(:);
    write_complex_binary(TX, 'TX.bin');
end
