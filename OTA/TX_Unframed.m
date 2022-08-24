function TX_Unframed()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));

    run('Parameters_Unframed.m');

    %Frame Data
    Data_Input = zeros(block_len, no_of_blocks, no_of_frames);
    Encoder_Output = zeros(coded_block_len, no_of_blocks, no_of_frames);
    Modulator_Output = zeros(total_msg_symbols, no_of_frames);
    TX_Out = zeros(total_no_of_samples, no_of_frames);

    % Encoding
    for n_frame = 1:no_of_frames
        data_input = randi([0 1], block_len, no_of_blocks);
        encoded_data = Encoder(data_input, enc_type, no_of_blocks, block_len, rate);
        encoder_data = [encoded_data(:); zeros(extra_bits, 1)];

        % Modulation
        mod_symbols = qammod(encoder_data, mod_order, 'InputType', 'bit', 'UnitAveragePower', true);

        TX = mod_symbols(:);

        % Normalize the modulated data Power
        TX = TX .* (.8 / (max(max(abs(real(TX))), max(abs(imag(TX))))));

        % Frame Data
        Data_Input(:, :, n_frame) = data_input;
        Encoder_Output(:, :, n_frame) = encoded_data;
        Modulator_Output(:, n_frame) = mod_symbols;
        TX_Out(:, n_frame) = TX;
    end

    % Save Frame Data
    save('Data_Files/Data_Input.mat', 'Data_Input');
    save('Data_Files/Encoded_Output.mat', 'Encoder_Output');
    save('Data_Files/Modulator_Output.mat', 'Modulator_Output');
    save('Data_Files/TX_Out.mat', 'TX_Out');

    % Write to Transmitter
    TX = TX_Out(:);
    write_complex_binary(TX, 'TX.bin');
end
