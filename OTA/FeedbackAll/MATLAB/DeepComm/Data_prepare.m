% clearvars;
% close all;
% clc;
%
% % Input Matrix
% no_of_ofdm_symbols = 3840;
% no_of_subcarriers = 48;
% mod_order = 64;
% total_symbols = no_of_ofdm_symbols * no_of_subcarriers;
% bit_per_symbol = log2(mod_order);
% blk_len = 10;
% code_rate = 2;
% no_of_msgs = 1024;
%
% total_no_bits = total_symbols * bit_per_symbol;
% no_of_captures = total_no_bits / (code_rate * blk_len * no_of_msgs);
% N = 18;
%
% DataIn = zeros(no_of_msgs, N * no_of_captures, blk_len);
% send_matrix = open(strcat('data_input_matrix_',num2str(mod_order),'.mat'));
% send_matrix = send_matrix.data_input_matrix;
% 42 70 134 19
N = 19;
K = 4;
Encoded_data = open('encoded_data.mat');
Encoded_data = Encoded_data.encoded_data;
EncOut4 = zeros(no_of_blocks*N, block_len + 4, 3);
for i = 1:N
    for j = 1:no_of_blocks
        EncOut4((i - 1) * block_len + j, :,1) = Encoded_data(1:3:end,j);
EncOut4((i - 1) * block_len + j, :,2) = Encoded_data(2:3:end,j);
EncOut4((i - 1) * block_len + j, :,3) = Encoded_data(3:3:end,j);

    end
end
% % %
% DataOut1 = zeros(no_of_blocks*N, block_len + 4, 3);
% % %
% receive_matrix = open(strcat('receive', num2str(K), '.mat'));
% receive_matrix = receive_matrix.receive_matrix1;
% 
% for i = 1:N
%     D1 = receive_matrix{i};
% for j = 1: no_of_blocks
%     DataOut1((i - 1) *no_of_blocks +j, : , 1) = D1(1:3:end, j);
%     DataOut1((i - 1) *no_of_blocks +j, : , 2)  = D1(2:3:end, j);
%     DataOut1((i - 1) *no_of_blocks +j, : , 3)  = D1(3:3:end, j);
% end
% 
% end
% save(strcat('EncOut4.mat'), 'EncOut4')
% save(strcat('DataIn4.mat'), 'DataIn4')
% save(strcat('DataOut', num2str(K), '.mat'), strcat('DataOut',num2str(K)))
save(strcat('EncOut', num2str(K), '.mat'), strcat('EncOut',num2str(K)))