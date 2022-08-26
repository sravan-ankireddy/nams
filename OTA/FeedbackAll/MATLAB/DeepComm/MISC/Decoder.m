function decoded_data = Decoder(demod_data, type, encoded_no_bits, block_length)

    if strcmp(type, 'turbo')
        turboDec = comm.TurboDecoder('InterleaverIndicesSource', 'Input port', 'NumIterations', 6);
        intrlvrInd = open('Interleaver.mat');
        intrlvrInd = intrlvrInd.intrlvrInd;
        decoded_data = step(turboDec, demod_data, intrlvrInd);

    elseif strcmp(type, 'MAP')

        code_rate = 2; % Coding Rate
        no_of_blocks = encoded_no_bits / block_length;
        coded_block_length = code_rate * block_length;
        
        constraint_length = 3;
        TRELLIS = poly2trellis(constraint_length, [5 7], 7);

        % BCJR Decoder
        hAPPDec = comm.APPDecoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated', ...
            'Algorithm', 'True APP', 'CodedBitLLROutputPort', true);

        decoded_data = zeros(encoded_no_bits, 1);

        for i = 1:no_of_blocks
            st_id = (i - 1) * block_length;
            ll0 = zeros(block_length, 1);
            llr = step(hAPPDec, ll0, demod_data(2*st_id + 1:2*st_id + coded_block_length));
            data = (llr > 0); % MAP decoded bits
            decoded_data(st_id +1:st_id + block_length) = data(1:block_length);
        end

    end
