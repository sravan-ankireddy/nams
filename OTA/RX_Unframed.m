function RX_Unframed()
    currentFolder = pwd;
    addpath(strcat(currentFolder, '/Imp_Files'));
    addpath(strcat(currentFolder, '/Imp_Functions'));
    run('Parameters_Unframed.m');

    % Extraction of the received data
    RX = read_complex_binary('RX.bin');
    Receiver_Output = zeros(total_no_of_data_samples, 1);
    Data_Output = zeros(coded_block_len, no_of_blocks, 1);
    Frame_Error = zeros(1, 2);

    for n_detect = 1:1
        % Decoder
        Receiver_Output(:, 1) = RX;
        demod_data = -qamdemod(RX, mod_order, 'OutputType', 'approxllr', 'UnitAveragePower', true);
        demod_data = demod_data(1:end - extra_bits);

        demod_data = reshape(demod_data, coded_block_len, no_of_blocks);

        decoded_data = Decoder(demod_data, dec_type, no_of_blocks, block_len);
        data_to_encode = open('Data_Files/Data_Input.mat');
        data_to_encode = data_to_encode.Data_Input;
        data_to_encode = data_to_encode(:, :, 1);
        bit_err = biterr(decoded_data, data_to_encode) / encoded_no_bits;
        fprintf("BER: %1.4f\n", bit_err)

        Frame_Error(1, 1) = bit_err;
        Frame_Error(1, 2) = 0;
        Data_Output(:, :, n_detect) = demod_data;
    end

    save('Data_Files/Frame_Error.mat', 'Frame_Error')
    save('Data_Files/Data_Output.mat', 'Data_Output')
    save('Data_Files/Receiver_Output.mat', 'Receiver_Output')
end
