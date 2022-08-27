function RX_Feedback_Encoder(num)
    clc;

    if nargin < 1
        num = 3;
    end

    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));

    run('Parameters_feedback.m');

    %Frame Data
    B_Output = zeros(total_msg_symbols, no_of_frames);

    if num == 3
        Bit_Input = open('Feedback_Files/Bit_Input.mat');
        Bit_Input = Bit_Input.Bit_Input;
        BB_Output = zeros(total_msg_symbols, no_of_frames);
    else
        BX_Output = zeros(total_msg_symbols, no_of_frames);
    end

    TX_Out = zeros(total_no_of_samples, no_of_frames);

    YL = open(strcat('Feedback_Files/Y', num2str(num), '_Output.mat'));
    YL = YL.YL;

    for n_frame = 1:no_of_frames

        sts = [];
        lts = [];

        % Modulation
        if strcmp(mod_type, "NN")
            encoder_data = YL(:, n_frame);
            save(strcat('Data_Files/RX_Encoded', num2str(num), '.mat'), 'encoder_data');
            system(strcat('python3 Imp_Functions/RX_NN_Encoder', num2str(num), '.py'));
            mod_symbols = open(strcat('Data_Files/RX_Modulated', num2str(num), '.mat'));
            mod_symbols = double(mod_symbols.output);
        else
            mod_symbols = ActiveDecoder(0, 0, YL(:, n_frame), 0, 0, 1);

            if num == 3
                B_Output(:, n_frame) = mod_symbols;
                mod_symbols = 1 ./ (1 + exp(-mod_symbols));
            end

        end

        if num == 3
            decoded_data = (mod_symbols > 0.5);
            data_to_encode = Bit_Input(:, n_frame);
            bit_err = biterr(decoded_data, data_to_encode) / encoded_no_bits;

            fprintf("Frame: %d  BER: %1.4f\n", n_frame, bit_err);

            BB_Output(:, n_frame) = decoded_data;

        else

            % Signal Frame containing the frame number
            sig_symb = zeros(no_signal_symbols, 1);

            F = de2bi(n_frame - 1, 6);

            for i = 1:6
                sig_symb((i - 1) * 8 + 1:i * 8) = F(i) * ones(8, 1);
            end

            sig_symb = qammod(sig_symb, 2, 'InputType', 'bit', 'UnitAveragePower', true);
            % Subcarrier Allocation
            P = open('Pilot_matrix.mat');
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

            save('Imp_Files/Pilots.mat', 'Pilots');

            % IFFT to generate tx symbols
            for i = 1:total_ofdm_symbols_per_frame
                IFFT_Data = ifft(fftshift(A(1:size_of_FFT, i)), size_of_FFT);
                A(1:size_of_FFT + cp_length, i) = [IFFT_Data(size_of_FFT - cp_length + 1:size_of_FFT); IFFT_Data];
            end

            TX = A(:);

            % Normalize the modulated data Power
            % TX = TX .* (.8 / (max(max(abs(real(TX))), max(abs(imag(TX))))));

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

            tx_scale = .8 / max(max(real(TX), imag(TX)));
            TX = tx_scale .* TX;

            save('Feedback_Files/tx_scale.mat', 'tx_scale');
            % Frame Data

            B_Output(:, n_frame) = mod_symbols;
            BX_Output(:, n_frame) = tx_gain * mod_symbols;
            TX_Out(:, n_frame) = TX;
            Data_start(n_frame) = (n_frame - 1) * length(TX) + 320 + 1;
        end

        % Save Frame Data

        save(strcat('Feedback_Files/B', num2str(num), '_Output.mat'), 'B_Output');

        if num == 3
            save(strcat('Feedback_Files/BB', num2str(num), '_Output.mat'), 'BB_Output');
        else
            save(strcat('Feedback_Files/BX', num2str(num), '_Output.mat'), 'BX_Output');
        end

        % Write to Transmitter
        TX = TX_Out(:);
        write_complex_binary(TX, 'TX.bin');
    end

end
