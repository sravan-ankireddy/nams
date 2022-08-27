% Author = 'hyejikim'
% Viterbi decoder and MAP decoder for convolutional ([7,5]) code
%
%%

constraint_length = 3;
TRELLIS = poly2trellis(constraint_length, [5 7], 7); % % HERE

close all;

frameLength = 10; % Blocklength
num_test = 10000; %0; %000; % Number of blocks to test

code_rate = 2; % Coding Rate

SNR_points = 1; % Number of points for test SNR
test_SNRs = linspace(6, 6, SNR_points); % Test SNR (and sigma)
test_sigmas = 10.^(-test_SNRs / 20);

tail = false; % Whether to use tail bits

if tail == true
    % Convolutional Encoder
    hConEnc = comm.ConvolutionalEncoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Terminated');

    % BCJR Decoder
    hAPPDec = comm.APPDecoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Terminated', ...
        'Algorithm', 'True APP', 'CodedBitLLROutputPort', true);

    vitdec_tail = 'term';

else

    % Convolutional Encoder
    hConEnc = comm.ConvolutionalEncoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated'); % % HERE

    % BCJR Decoder
    hAPPDec = comm.APPDecoder('TrellisStructure', TRELLIS, 'TerminationMethod', 'Truncated', ...
        'Algorithm', 'True APP', 'CodedBitLLROutputPort', true);

    vitdec_tail = 'trunc';

end

tic
rnd_seed = 0;
rng(rnd_seed)

nb_errors = zeros(SNR_points, num_test);
nb_bl_errors = zeros(SNR_points, 1);

map_nb_errors = zeros(SNR_points, num_test);
map_nb_bl_errors = zeros(SNR_points, 1);

for idx = 1:SNR_points
    idx

    for counter = 1:num_test
        data = randi([0 1], frameLength, 1);

        encodedData = hConEnc(data);

        noisy_encodedData = 2 * encodedData - 1 + test_sigmas(idx) * randn(size(encodedData));

        %        decc = hAPPDec(zeros(frameLength,1),noisy_encodedData);
        %
        %
        %         % Bursty noise
        %         %noisy_encodedData = noisy_encodedData + sigma_b*randn(size(noisy_encodedData)).*(rand(size(noisy_encodedData))> 1-alpha);
        %
        %
        %         if tail == true
        %             noisy_received=reshape(noisy_encodedData,[code_rate,frameLength+constraint_length-1])';
        %         else
        %             noisy_received=reshape(noisy_encodedData,[code_rate,frameLength])';
        %         end
        %
        %
        %
        %         alpha = 0.2;
        %         sigma_b = test_sigmas(idx)*10;
        %         DSNR = 10*log(1/sigma_b^2);
        %
        %         alphabar = 1-alpha;
        %
        %
        %
        %         SNR = -10*log(test_sigmas(idx)^2);
        %         decc = custom_viterbi(noisy_received(:,1),noisy_received(:,2),SNR,alpha,DSNR);
        %        decc = decc';
        %
        %         if tail == true
        %             ll0 = zeros(frameLength+constraint_length-1,1);
        %         else
        %             ll0 = zeros(frameLength,1);
        %         end

        llr = step(hAPPDec, ll0, noisy_encodedData); % % %*2/test_sigmas(idx)^2);
        map_decc = (llr > 0); % MAP decoded bits
        map_decc = map_decc(1:frameLength);

        nb_errors(idx, counter) = sum(abs(decc - data)); % count Viterbi BER

        if sum(abs(decc - data)) > 0
            nb_bl_errors(idx) = nb_bl_errors(idx) + 1; % count Viterbi BLER
        end

        map_nb_errors(idx, counter) = sum(abs(map_decc(1:frameLength) - data)); % count MAP BER

        if sum(abs(map_decc - data)) > 0
            map_nb_bl_errors(idx) = map_nb_bl_errors(idx) + 1; % count MAP BLER
        end

    end

end

% Average out over trials
%vit_BER=mean(nb_errors,2)/frameLength;
%vit_BLER=nb_bl_errors/num_test;

map_BER = mean(map_nb_errors, 2) / frameLength
map_BLER = map_nb_bl_errors / num_test

% figure(1);
%title(strcat('rate ', num2str(code_rate) , vitdec_tail,' BER'));
%semilogy(test_SNRs,vit_BER,'ro-');
%hold on;
semilogy(test_SNRs, map_BER, 'bx-');
legend('Viterbi', 'MAP');
title('BER')
%
% figure(2);
% title(strcat('rate ', num2str(code_rate) , vitdec_tail,' BLER'));
% semilogy(test_SNRs,vit_BLER,'ro-');
% hold on;
% semilogy(test_SNRs, map_BLER,'bx-');
% legend('Viterbi','MAP');
% title('BLER')

test_SNRs

%vit_BER
%vit_BLER
