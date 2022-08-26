function Decoded_data = Decoder(Demod_data, type, no_of_blocks, block_length)

    if strcmp(type, 'turbo')
        turboDec = comm.TurboDecoder('InterleaverIndicesSource', 'Input port', 'NumIterations', 6);
        Interleaver = open('Interleaver.mat');
        Interleaver = Interleaver.Interleaver;

        Decoded_data = zeros(block_length, no_of_blocks);

        for i = 1:no_of_blocks
            %             intrlvrInd = Interleaver(:, i);
            intrlvrInd = [22, 20, 25, 4, 10, 15, 28, 11, 18, 29, 27, ...
                        35, 37, 2, 39, 30, 34, 16, 36, 8, 13, 5, 17, 14, 33, 7, ...
                        32, 1, 26, 12, 31, 24, 6, 23, 21, 19, 9, 38, 3, 0] + 1;
            demod_data = Demod_data(:, i);
            decoded_data = step(turboDec, demod_data, intrlvrInd);
            Decoded_data(:, i) = decoded_data;
        end

    elseif strcmp(type, 'MAP')

        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % BCJR Decoder
        hAPPDec = comm.APPDecoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated', ...
            'Algorithm', 'True APP', 'CodedBitLLROutputPort', true);

        Decoded_data = zeros(block_length, no_of_blocks);

        for i = 1:no_of_blocks
            demod_data = Demod_data(:, i);
            ll0 = zeros(block_length, 1);
            llr = step(hAPPDec, ll0, demod_data);
            data = (llr > 0); % MAP decoded bits
            decoded_data = data(1:block_length);
            Decoded_data(:, i) = decoded_data;
        end

    end
