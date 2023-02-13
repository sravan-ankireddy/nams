% load parity check matrix

filename = 'data_files/par_gen_data/BCH_63_36.alist';
% filename = 'data_files/par_gen_data/LDPC_128_64.alist';

H = alist2full(filename);

% creating sparse logical version of H
Hs = sparse(logical(H));

% length of code
N = size(H,2);
K = N - size(H,1);
rate = K/N;

% create comm objects
bchenc = comm.BCHEncoder(N,K);
bchdec = comm.BCHDecoder(N,K);
bpskmod = comm.BPSKModulator;
bpskdemod = comm.BPSKDemodulator;
bpskdemod_soft = comm.BPSKDemodulator('DecisionMethod' ,'Log-likelihood ratio');

ldpcEncCfg = ldpcEncoderConfig(Hs);
ldpcDecCfg = ldpcDecoderConfig(Hs,'norm-min-sum');

% modutalion bits : bpsk 1
mod_bits = 1;

% step size
delta = 1;

channel = "ETU";

% Eb/N0 range of signal in dB
if (channel == "ETU")
    Eb_N0_dB = 1:18;
else
    Eb_N0_dB = 1:8;
end

% Convert EbN0 to SNR (=EbN0?)
SNR_dB = Eb_N0_dB; %+ 10*log10(rate) + 10*log10(mod_bits);

% Varinace of noise in linear scale
var_N = 10.^(-SNR_dB/10)/(2*rate);

% no. of simulations
num_frames = 1e4;

% no. of iterations
max_iter = 5;

% error variables
BER_nms = zeros(size(SNR_dB));

err_stat_nms = zeros(N,1);

% Store cv beliefs for analysis : mu_vc = l_ch + s_vc
% Store the sum of incodming messages to all variable nodes in each
% iteration
s_vc_nms = zeros(num_frames,max_iter,N,length(Eb_N0_dB));
s_cv_nms = zeros(num_frames,max_iter,N-K,N,length(Eb_N0_dB));
llr_in_nms = zeros(num_frames,N,length(Eb_N0_dB));

decoder_type = "minsum";

llr_ref = 0;
enc_ref = 0;

if (channel == "ETU")
    data = load("data_files/lte_data/BCH_63_36_ETU_df_0_data_train_5_22.mat");
    llr_ref = data.llr;
    enc_ref = data.enc;
end

tic;
for i_SNR = 1:length(SNR_dB)
    % fix seed of randn for reproducability
    rng(i_SNR,'twister')

    disp(SNR_dB(i_SNR));
    err_nms = 0;

    parfor i_frame = 1:num_frames
        
        if (mod(i_frame,1000) == 0)
            disp(i_frame);
        end

        if (channel == "AWGN")

            % generate a random message - FIX to 0
            m = 0*randi([0 1], K, 1); 
            
            % encode the message
            c_nms = ldpcEncode(m,ldpcEncCfg);
            
            % BPSK modulation
%             x_nms = bpskmod(c_nms);
            x_nms = 1 - 2*c_nms;
    
            % Passing through channel
            sig = sqrt(var_N(i_SNR));
            noise = sig * randn(size(x_nms));
            
            y_nms = apply_channel(x_nms, sig, noise, channel);
    
            % BPSK demodulation
%             llr = bpskdemod_soft(y_nms);
            llr = 2*y_nms/(sig^2);
            [llr_out_nms, llr_updates, llr_updates_full] = weighted_min_sum(llr,H,max_iter);
    
            s_vc_nms(i_frame,:,:,i_SNR) = llr_updates;
            s_cv_nms(i_frame,:,:,:,i_SNR) = llr_updates_full;
            llr_in_nms(i_frame,:,i_SNR) = llr;
    
            % estimating the codeword from bch decoding
            c_hat_nms = (llr_out_nms < 0)';
        else

            llr = llr_ref(:,i_frame,SNR_dB(i_SNR));
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum(llr,H,max_iter);

            s_vc_nms(i_frame,:,:,i_SNR) = llr_updates;
            s_cv_nms(i_frame,:,:,:,i_SNR) = llr_updates_full;
            llr_in_nms(i_frame,:,i_SNR) = llr;

            c_nms = enc_ref(:,i_frame,SNR_dB(i_SNR));

            c_hat_nms = (llr_out_nms > 0)';
        end

        % bit error rate  
        err_nms = err_nms + sum(c_hat_nms ~= c_nms);

    end
    BER_nms(i_SNR) = err_nms/(N*num_frames);
end

% plots

% semilogy(Eb_N0_dB,BER_nms);
% grid on;
% hold on;

%% Emp distribuition of s_vc : Gaussian approx at variable node

% extract data for node i_n
W_vc_opt = zeros(N,length(Eb_N0_dB));
for i_SNR = 1:length(Eb_N0_dB)
    disp(i_SNR);
    % extract iter 1 data
    s_vc = squeeze(s_vc_nms(:,1,:,i_SNR));
    for i_n = 1:N
        data_int = squeeze(llr_in_nms(:,i_n,i_SNR));
        data_ext = squeeze(s_vc_nms(:,1,i_n,i_SNR));

        m1 = mean(data_int);
        m2 = mean(data_ext);
        
        v1 = var(data_int);
        v2 = var(data_ext);
        
        c1 = m1/m2;
        c2 = v1/v2;
        
        f = @(x)-1*(x+c1)^2/(x^2+c2);
        x0 = 1;
        [xmin,fmin] = fminsearch(f,x0);
        
        W_vc_opt(i_n,i_SNR) = min(xmin,1);
    end
end

% save('W_cv.mat',"W_cv_opt","W_cv_ana");

figure;
for i_s = 1:length(Eb_N0_dB)
    plot(W_vc_opt(:,i_s));
    hold on;
end

% get ber plots
W_cv_opt_mat = zeros(size(H,1),size(H,2),length(Eb_N0_dB));
for i_s = 1:length(Eb_N0_dB)
    W_cv_opt_mat(:,:,i_s) = repmat(W_vc_opt(:,i_s)',size(H,1),1);
end

% % load NN weights and compare with Gaussian weights
if (channel == "ETU")
    load("data_files/weights/nams_BCH_63_36_st_20000_lr_0.005_ETU_df_0_ent_2_nn_eq_1_relu_1_max_iter_5_5_22.mat");
else
    load("data_files/weights/nams_BCH_63_36_st_20000_lr_0.005_AWGN_ent_2_nn_eq_1_relu_1_max_iter_5_1_8.mat");
end
W_vc_nn = W_vc;
figure;
plot(W_vc_opt(:,4));
hold on;
plot(W_vc_nn);
legend("Gaussian weights","Weights from NN");

% compare BER performance
W_nn_cur = repmat(W_vc_nn,size(H,1),1);
simulate_ber_gauss_vs_nn;

%% Emp distribuition of s_cv : Gaussian approx at check node
% For each edge connected to each variable node, find the empirical
% distribuition

% extract data for node i_n
W_cv_opt_mat = zeros(N-K,N,length(Eb_N0_dB));
for i_SNR = 1:length(Eb_N0_dB)
    disp(i_SNR);
    % extract iter 1 data
    s_cv = squeeze(s_cv_nms(:,1,:,i_SNR));

    mean_data = H;
    var_data = H;
    
    for i_h = 1:size(H,1)
        for j_h = 1:size(H,2)
            data = s_cv_nms(:,1,i_h,j_h,i_SNR);
    
            mean_data(i_h,j_h) = mean(data);
            var_data(i_h,j_h) = var(data);
        end
    end

    % find optimal weights for each check node by solving multivariate
    % optimization at each variable node; then take mean
    for i_n = 1:N
        % channel llr
        data_int = squeeze(llr_in_nms(:,i_n,i_SNR));    

       % get mean and var of all edges
        mean_cur = mean_data(:,i_n);
        var_cur = var_data(:,i_n);

        % formulate the optimization problem
        x0 = ones(1,size(H,1));
        LB = zeros(size(x0));
        UB = H(:,i_n);
        % channel llr stats
        m1 = mean(data_int);
        v1 = var(data_int);
        
        f = @(x)-1*(mean_cur(1)*x(1) + mean_cur(2)*x(2) + mean_cur(3)*x(3) + mean_cur(4)*x(4) + mean_cur(5)*x(5) + mean_cur(6)*x(6) + ...
                    mean_cur(7)*x(7) + mean_cur(8)*x(8) + mean_cur(9)*x(9) + mean_cur(10)*x(10) + mean_cur(11)*x(11) + mean_cur(12)*x(12) + ...
                    mean_cur(13)*x(13) + mean_cur(14)*x(14) + mean_cur(15)*x(15) + mean_cur(16)*x(16) + mean_cur(17)*x(17) + mean_cur(18)*x(18) + ...
                    mean_cur(19)*x(19) + mean_cur(20)*x(20) + mean_cur(21)*x(21) + mean_cur(22)*x(22) + mean_cur(23)*x(23) + mean_cur(24)*x(24) + ...
                    mean_cur(25)*x(25) + mean_cur(26)*x(26) + mean_cur(27)*x(27) + m1)^2/ ...
                (var_cur(1)*(x(1)^2) + var_cur(2)*(x(2)^2) + var_cur(3)*(x(3)^2) + var_cur(4)*(x(4)^2) + var_cur(5)*(x(5)^2) + var_cur(6)*(x(6)^2) + ...
                    var_cur(7)*(x(7)^2) + var_cur(8)*(x(8)^2) + var_cur(9)*(x(9)^2) + var_cur(10)*(x(10)^2) + var_cur(11)*(x(11)^2) + var_cur(12)*(x(12)^2) + ...
                    var_cur(13)*(x(13)^2) + var_cur(14)*(x(14)^2) + var_cur(15)*(x(15)^2) + var_cur(16)*(x(16)^2) + var_cur(17)*(x(17)^2) + var_cur(18)*(x(18)^2) + ...
                    var_cur(19)*(x(19)^2) + var_cur(20)*(x(20)^2) + var_cur(21)*(x(21)^2) + var_cur(22)*(x(22)^2) + var_cur(23)*(x(23)^2) + var_cur(24)*(x(24)^2) + ...
                    var_cur(25)*(x(25)^2) + var_cur(26)*(x(26)^2) + var_cur(27)*(x(27)^2) + v1);
        
        [W_cv_opt_mat(:,i_n,i_SNR),fmin] = fminsearchbnd(f,x0,LB,UB);
    end
end

W_cv_opt = zeros(N,length(Eb_N0_dB));

for i_e = 1:length(Eb_N0_dB)
    for i_n = 1:N
        nz_ind = H(:,i_n) > 0;
        W_cv_opt(i_n,i_e) = mean(W_cv_opt_mat(nz_ind,i_n,i_e));
    end
end

% load nn weights : reshape W_cv_nn and compare with Gaussian weights
if (channel == "ETU")
    load("data_files/weights/nams_BCH_63_36_st_20000_lr_0.005_ETU_df_0_ent_1_nn_eq_1_relu_1_max_iter_5_5_22.mat");
else
    load("data_files/weights/nams_BCH_63_36_st_20000_lr_0.005_AWGN_ent_1_nn_eq_1_relu_1_max_iter_5_1_8.mat");
end
W_cv_nn = W_vc;
W_cv_nn_mat = H;
edge_count = 0;
for i_col = 1:size(H,2)
    for i_row = 1:size(H,1)
        if (H(i_row,i_col) > 0)
            edge_count = edge_count + 1;
            W_cv_nn_mat(i_row,i_col) = W_cv_nn(edge_count);
        end
    end
end

W_cv_nn = zeros(N,1);
for i_n = 1:N
    nz_ind = H(:,i_n) > 0;
    W_cv_nn(i_n,1) = mean(W_cv_nn_mat(nz_ind,i_n));
end

f = figure;
plot(W_cv_opt(:,4),'-.*');
hold on;
plot(W_cv_nn,'--o');

ylabel("Weights");

xlabel("Variable node index");

leg = legend('Mean weights from Gaussian Approximation', 'Mean weights from NNMS');

leg.FontSize = 28;
legend('Location','southwest');

grid on;
ax = gca;
fs = 30;
set(gca,'FontSize',fs);

% figure
f.Position = [1500 1000 1250 750];

W_nn_cur = W_cv_nn_mat;
simulate_ber_gauss_vs_nn;
