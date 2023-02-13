N = 63;
K = 36;
filename = "data_files/par_gen_data/BCH_" + N + "_" + K + ".alist";

% N = 384;
% K = 192;
% filename = "data_files/par_gen_data/LDPC_" + N + "_" + K + ".alist";

H = alist2full(filename);

%% FIX ME
H = H';
m = size(H,1);
n = size(H,2);

disp_count = 10e4;

tic;
num_cyc_len_4 = 0;

% for each check/var node, count the number of 4 cycles present
num_chk_cyc_len_4 = zeros(size(H,1),1);
num_var_cyc_len_4 = zeros(size(H,2),1);
comb_count = 0;
for i = 1:m-1
    for j = i+1:m
        comb_count = comb_count+1;
        if (mod(comb_count,disp_count) == 0)
            percent_done = round(100*comb_count/nchoosek(m,2),2);
            disp("Counting length 4 cycles : " + percent_done + "% done");
        end
        row1 = H(i,:);
        row2 = 2*H(j,:);
        row_sum = row1+row2;

        num_3_vec = row_sum == 3;
        num_3 = sum(num_3_vec);
        cyc_4 = num_3*(num_3-1)/2;
        
        num_cyc_len_4 = num_cyc_len_4 + cyc_4;
        num_chk_cyc_len_4(i) = num_chk_cyc_len_4(i) + cyc_4;
        num_chk_cyc_len_4(j) = num_chk_cyc_len_4(j) + cyc_4;
 
        var_cyc_ind = find(num_3_vec);
        num_var_cyc_len_4(var_cyc_ind) = num_var_cyc_len_4(var_cyc_ind) + 1;
    end
end
t4 = toc;
disp("Length 4 cycles .. done");

%%
% load gw ent 1 awgn weights
load("data_files/weights/weights_for_analysis/nams_BCH_63_36_st_10000_lr_0.005_AWGN_ent_1_nn_eq_1_relu_1_max_iter_5_1_8.mat");
W_gw_mat_awgn = zeros(size(H));
W_gw_mean_awgn = zeros(1,size(H,1));
edge_count = 1;
for i = 1:size(H,1)
    temp = [];
    for j = 1:size(H,2)
        if (H(i,j) == 1)
            temp = [temp W_gw(1,edge_count)];
            W_gw_mat_awgn(i,j) = W_gw(1,edge_count);
            edge_count = edge_count+1;
        end
    end
    W_gw_mean_awgn(i) = mean(temp);
end
clear W_gw;
% load gw ent 1 etu weights
load("data_files/weights/weights_for_analysis/nams_BCH_63_36_st_20000_lr_0.005_ETU_df_0_ent_1_nn_eq_1_relu_1_max_iter_5_5_22.mat");
W_gw_mat_etu = zeros(size(H));
W_gw_mean_etu = zeros(1,size(H,1));
edge_count = 1;
for i = 1:size(H,1)
    temp = [];
    for j = 1:size(H,2)
        if (H(i,j) == 1)
            temp = [temp W_gw(1,edge_count)];
            W_gw_mat_etu(i,j) = W_gw(1,edge_count);
            edge_count = edge_count+1;
        end
    end
    W_gw_mean_etu(i) = mean(temp);
end

f = figure(1);
yyaxis left;
plot(num_chk_cyc_len_4,'-.*','MarkerSize',12,'LineWidth',3);
ylabel("No. of length-4 cycles");

yyaxis right;
plot(W_gw_mean_awgn,'--o','MarkerSize',12,'LineWidth',3);
hold on;
plot(W_gw_mean_etu,'--ms','MarkerSize',12,'LineWidth',3);
ylabel("Mean weight of edges");

xlabel("Variable node index");

leg = legend('No. of length-4 cycles', 'Mean weight of edges -- AWGN', 'Mean weight of edges -- ETU');

leg.FontSize = 32;
legend('Location','northeast');
set(leg,'color','none');
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.5]');

grid on;
ax = gca;
fs = 36;
set(gca,'FontSize',fs);

% figure
f.Position = [1500 1000 1250 750];


% plot(num_chk_cyc_len_4/max(num_chk_cyc_len_4)); hold on; plot(1./(mean_weight_chk_node/max(mean_weight_chk_node)));

