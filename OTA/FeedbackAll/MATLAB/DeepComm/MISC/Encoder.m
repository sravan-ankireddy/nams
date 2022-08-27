function encoded_data = Encoder(data_input, type, no_encoded_bits, block_length)

    if strcmp(type, 'turbo')
        turboEnc = comm.TurboEncoder('InterleaverIndicesSource', 'Input port');
        intrlvrInd = randperm(length(dataIn));
        save('Interleaver.mat', 'intrlvrInd');
        encoded_data = step(turboEnc, dataIn, intrlvrInd);

    elseif strcmp(type, 'convolutional')
        code_rate = 2; % Coding Rate
        total_encoded_bits = code_rate * no_encoded_bits;
        no_of_blocks = no_encoded_bits / block_length;
        coded_block_length = code_rate * block_length;

        encoded_data = zeros(total_encoded_bits, 1);
        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % Convolutional Encoder
        hConEnc = comm.ConvolutionalEncoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated');

        for i = 1:no_of_blocks
            st_id = (i - 1) * block_length;
            data = data_input(st_id + 1:st_id + block_length);
            encoded_data(2 * st_id + 1:2 * st_id + coded_block_length, 1) = hConEnc(data);
        end

    end

end
