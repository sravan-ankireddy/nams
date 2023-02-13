% load parity check matrix

load('../data_files/par_gen_data/H_BCH_63_36.mat');
load('../data_files/par_gen_data/G_BCH_63_36.mat');
H = double(H);
G = double(G);

% creating sparse logical version of H
Hs = sparse(logical(H));

% length of code
N = size(H,2);
K = N - size(H,1);
rate = K/N;

decoder_type = "nms";

% Convert EbN0 to SNR (=EbN0?)
SNR_dB = 1;

% Varinace of noise in linear scale
var_N = 10.^(-SNR_dB/10)/(2*rate);

% no. of simulations
num_frames = 1e4;

% no. of iterations
max_iter = 5;

% error variables
BER_bch = zeros(size(SNR_dB));

% extract and store the incoming messages to check node for correlation
% analysis : storing num_frames x max_iter x size(H) values for each SNR

llr_in_check_node = zeros(length(SNR_dB),num_frames,max_iter,N-K,N);
llr_in_var_node = zeros(length(SNR_dB),num_frames,max_iter,N-K,N);
tic;
for i_SNR = 1:length(SNR_dB)
    % fix seed of randn
    rng(i_SNR,'twister')

    disp(SNR_dB(i_SNR));
    err_bch = 0;

    for i_sim = 1:num_frames
        
        if (mod(i_sim,100) == 0)
            disp(i_sim);
        end

        % generate a random message
        m = randi([0 1], K, 1);
        
        % encode the message
        c_bch = mod(G*m,2);
        
        % BPSK modulation
        x_bch = 2*c_bch-1;

        % Passing through channel
        sig = sqrt(var_N(i_SNR)*(2*rate));
        noise = sig * randn(size(x_bch));
        channel = "AWGN";
        y_bch = apply_channel(x_bch, sig, noise, channel);

        % BPSK demodulation
        llr = 2*y_bch/(sig^2);

        [llr_out_nms, llr_in_check_node(i_SNR,i_sim,:,:,:), llr_in_var_node(i_SNR,i_sim,:,:,:)] = neural_ms(llr,H,max_iter,decoder_type);

        c_bch_hat = (llr_out_nms > 0)';

        % bit error rate  
        err_bch = err_bch + sum(c_bch_hat ~= c_bch);
    end
    BER_bch(i_SNR) = err_bch/(N*num_frames);
end
%%
% for each variable node, compute the mean abs correlation between all edges
mean_abs_corr_nms = zeros(max_iter,N);

% figure(1);
for iter = 1:max_iter
    for v = 1:N
        % extract the non-zero entries
        ind = find(H(:,v)> 0);
        llr_in_v = squeeze(llr_in_var_node(1,:,iter,ind,v));
        if (length(ind) > 1)
            % compute the correlation coefficient matrix
            corr_v = abs(corr(llr_in_v));
            % extract the off diagonal elements
            corr_v_u = triu(corr_v,1);
            mean_abs_corr_nms(iter,v) = mean(corr_v_u(corr_v_u>0));
        else
            mean_abs_corr_nms(iter,v) = 0;
        end
    end
%     plot(mean_abs_corr_nms(iter,:)); hold on;
end
final_corr = mean_abs_corr_nms(end,:);
final_corr(final_corr==0) = NaN;
% plot(final_corr); hold on;

% plot in standard format
f = figure;
plot(final_corr,'-o','MarkerSize',12,'LineWidth',3);

ylabel("Mean correlation coefficient");

xlabel("Variable node index");

leg = legend('Min-Sum');

% leg.FontSize = 32;
% legend('Location','southwest');
% set(leg,'color','none');
% leg.BoxFace.ColorType='truecoloralpha';
% leg.BoxFace.ColorData=uint8(255*[1 1 1 0.5]');

grid on;
% ax = gca;
% fs = 28;
% set(gca,'FontSize',fs);

% figure
f.Position = [1500 1000 1250 750];

% mean_abs_corr_nms = zeros(max_iter,N-K);
% figure(2);
% for iter = 1:max_iter
%     for c = 1:N-K
%         % extract the non-zero entries
%         ind = find(H(c,:)> 0);
%         llr_in_c = squeeze(llr_in_check_node(1,:,iter,c,ind));
%         if (size(llr_in_c,1) > 1)
%             % compute the correlation coefficient matrix
%             corr_c = abs(corr(llr_in_c));
%             % extract the off diagonal elements
%             corr_c_u = triu(corr_c);
%             mean_abs_corr_nms(iter,c) = mean(corr_c_u(corr_c_u>0));
%         else
%             mean_abs_corr_nms(iter,c) = 0;
%         end
%     end
% %     plot(mean_abs_corr_nms(iter,:)); hold on;
% end
% final_corr = mean_abs_corr_nms(end,:);
% plot(final_corr(final_corr>0)); hold on;
