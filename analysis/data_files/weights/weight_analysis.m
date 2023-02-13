w0 = load("nams_BCH_63_30_st_20000_lr_0.01_ETU_ent_0_nn_eq_1_relu_1_max_iter_5_9_18.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

hhf = histfit(w0.W_cv(1,:),50);
df = fitdist(w0.W_cv(1,:)', 'Normal')

hold on;



w0 = load("nams_BCH_63_36_st_20000_lr_0.01_ETU_ent_0_nn_eq_1_relu_1_max_iter_5_9_18.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

% figure
hhf = histfit(mean(w0.W_cv,1),50);
df = fitdist(mean(w0.W_cv,1)', 'Normal')

hold on;

w0 = load("nams_BCH_63_57_st_20000_lr_0.01_ETU_ent_0_nn_eq_1_relu_1_max_iter_5_9_18.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

hhf = histfit(w0.W_cv(1,:),50);
df = fitdist(w0.W_cv(1,:)', 'Normal')

%% 

w0 = load("nams_BCH_63_30_st_20000_lr_0.01_AWGN_ent_0_nn_eq_1_relu_1_max_iter_5_1_8.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

hhf = histfit(w0.W_cv(1,:),50);
df = fitdist(w0.W_cv(1,:)', 'Normal')

hold on;



w0 = load("nams_BCH_63_36_st_20000_lr_0.01_AWGN_ent_0_nn_eq_1_relu_1_max_iter_5_1_8.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

% figure
hhf = histfit(mean(w0.W_cv,1),50);
df = fitdist(mean(w0.W_cv,1)', 'Normal')

hold on;

w0 = load("nams_BCH_63_57_st_20000_lr_0.01_AWGN_ent_0_nn_eq_1_relu_1_max_iter_5_1_8.mat");

% for i = 1:5
%     histogram(w0.W_cv(i,:),100);
%     hold on;
% end

hhf = histfit(w0.W_cv(1,:),50);
df = fitdist(w0.W_cv(1,:)', 'Normal')

%%
for i = 1:5
    plot(W_cv(i,:)); 
    disp(mean(W_cv(i,:)));
    hold on;
end




for j = 1:27
    start_ind = 1 + (j-1)*18;
    end_ind = start_ind + 18 - 1;

    plot(W_cv(i,start_ind:end_ind)); hold on;
    disp(mean(W_cv(i,start_ind:end_ind)));
end

W_cv_re = reshape(W_cv(1,:),27,[]);

W_cv_re = transpose(W_cv_re);
W_cv_re = W_cv_re(:);

plot(W_cv_re);



%%
edge_count = 0;
W_awgn_mat = zeros(size(H));
W_etu_mat = zeros(size(H));
N = size(H,2);
K = size(H,1);
for i_col = 1:size(H,2)
    for i_row = 1:size(H,1)
        if (H(i_row,i_col) > 0)
            edge_count = edge_count + 1;
            W_awgn_mat(i_row,i_col) = W_awgn(edge_count);
            W_etu_mat(i_row,i_col) = W_etu(edge_count);
        end
    end
end

% mean weight per check node
W_awgn_chk_mean = zeros(K,1);
for i_k = 1:K
    nz_ind = H(i_k,:) > 0;
    W_awgn_chk_mean(i_k,1) = mean(W_awgn_mat(i_k,nz_ind));
end
W_awgn_var_mean = zeros(N,1);
for i_n = 1:N
    nz_ind = H(:,i_n) > 0;
    W_awgn_var_mean(i_n,1) = mean(W_awgn_mat(nz_ind,i_n));
end

W_etu_chk_mean = zeros(K,1);
for i_k = 1:K
    nz_ind = H(i_k,:) > 0;
    W_etu_chk_mean(i_k,1) = mean(W_etu_mat(i_k,nz_ind));
end
W_etu_var_mean = zeros(N,1);
for i_n = 1:N
    nz_ind = H(:,i_n) > 0;
    W_etu_var_mean(i_n,1) = mean(W_etu_mat(nz_ind,i_n));
end

figure;
plot(W_awgn_chk_mean); hold on; plot(W_etu_chk_mean);

figure;
plot(W_awgn_var_mean); hold on; plot(W_etu_var_mean);