function Encoded_data = Encoder(data_input, type, no_of_blocks, block_length, code_rate)

    if strcmp(type, 'turbo')
        turboEnc = comm.TurboEncoder('InterleaverIndicesSource', 'Input port');
        coded_block_length = block_length / code_rate + 12;
        Encoded_data = zeros(coded_block_length, no_of_blocks);

        for i = 1:no_of_blocks
            X = data_input(:, i);
            intrlvrInd = [22, 20, 25, 4, 10, 15, 28, 11, 18, 29, 27, ...
                        35, 37, 2, 39, 30, 34, 16, 36, 8, 13, 5, 17, 14, 33, 7, ...
                        32, 1, 26, 12, 31, 24, 6, 23, 21, 19, 9, 38, 3, 0] + 1;
            encoded_data = step(turboEnc, X, intrlvrInd);
            Encoded_data(:, i) = encoded_data;
        end

    elseif strcmp(type, 'convolutional')
        coded_block_length = block_length / code_rate;
        Encoded_data = zeros(coded_block_length, no_of_blocks);
        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % Convolutional Encoder
        hConEnc = comm.ConvolutionalEncoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated');

        for i = 1:no_of_blocks
            X = data_input(:, i);
            encoded_data = hConEnc(X);
            Encoded_data(:, i) = encoded_data;
        end
    end

end
