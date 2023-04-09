function Encoded_data = turbo_encoder(data_input, type, no_of_blocks, block_length, code_rate)

    if strcmp(type, 'turbo')
        TRELLIS = poly2trellis(4, [13, 15], 13);
        turboEnc = comm.TurboEncoder('TrellisStructure', TRELLIS, 'InterleaverIndicesSource', 'Input port');
        coded_block_length = block_length / code_rate + 12;
        Encoded_data = zeros(coded_block_length, no_of_blocks);
        
        for i = 1:no_of_blocks
            X = data_input(:, i);

            if (length(X) == 40)

                intrlvrInd = [22, 20, 25, 4, 10, 15, 28, 11, 18, 29, 27, ...
                        35, 37, 2, 39, 30, 34, 16, 36, 8, 13, 5, 17, 14, 33, 7, ...
                        32, 1, 26, 12, 31, 24, 6, 23, 21, 19, 9, 38, 3, 0] + 1;
            elseif (length(X) == 100)
                intrlvrInd = [0, 33, 14, 47, 28, 61, 42, 75, 56, 89, 70, 103, 84, 13, 98, 27, 8, 41, 22, 55, 36, 69, 50, 83, 64, 97, 78, 7, 92, 21, ...
                        2, 35, 16, 49, 30, 63, 44, 77, 58, 91, 72, 1, 86, 15, 100, 29, 10, 43, 24, 57, 38, 71, 52, 85, 66, 99, 80, 9, 94, 23, 4, 37, 18, 51, ...
                            32, 65, 46, 79, 60, 93, 74, 3, 88, 17, 102, 31, 12, 45, 26, 59, 40, 73, 54, 87, 68, 101, 82, 11, 96, 25, 6, 39, 20, 53, 34, 67, 48, ...
                            81, 62, 95, 76, 5, 90, 19] +1;
            elseif (length(X) == 200)
                intrlvrInd = [0, 63, 26, 89, 52, 115, 78, 141, 104, 167, 130, 193, 156, 19, 182, 45, 8, 71, 34, 97, 60, 123, 86, 149, 112, 175, 138, 1, 164, 27,...
                    190, 53, 16, 79, 42, 105, 68, 131, 94, 157, 120, 183, 146, 9, 172, 35, 198, 61, 24, 87, 50, 113, 76, 139, 102, 165, 128, 191, 154, 17, 180, 43,...
                    6, 69, 32, 95, 58, 121, 84, 147, 110, 173, 136, 199, 162, 25, 188, 51, 14, 77, 40, 103, 66, 129, 92, 155, 118, 181, 144, 7, 170, 33, 196, 59, ...
                    22, 85, 48, 111, 74, 137, 100, 163, 126, 189, 152, 15, 178, 41, 4, 67, 30, 93, 56, 119, 82, 145, 108, 171, 134, 197, 160, 23, 186, 49, 12, 75,...
                    38, 101, 64, 127, 90, 153, 116, 179, 142, 5, 168, 31, 194, 57, 20, 83, 46, 109, 72, 135, 98, 161, 124, 187, 150, 13, 176, 39, 2, 65, 28, 91, ...
                    54, 117, 80, 143, 106, 169, 132, 195, 158, 21, 184, 47, 10, 73, 36, 99, 62, 125, 88, 151, 114, 177, 140, 3, 166, 29, 192, 55, 18, 81, 44, ...
                    107, 70, 133, 96, 159, 122, 185, 148, 11, 174, 37] + 1;
            end
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
    