BERs_wms_gauss = zeros(size(BER_nms));
for i_SNR = 1:length(SNR_dB)
    % fix seed of randn for reproducability
    rng(i_SNR,'twister')

    disp(SNR_dB(i_SNR));
    err_nms = 0;

    W_ana_cur = squeeze(W_ana_mat(:,:,i_SNR));
    B_ana_cur = zeros(size(W_ana_cur));

    parfor i_frame = 1:num_frames
        
        if (mod(i_frame,1000) == 0)
            disp(i_frame);
        end

        if (channel == "AWGN")

            % generate a random message - FIX to 0
            m = randi([0 1], K, 1); 
            
            % encode the message
            c_nms = G*m;
            
            % BPSK modulation
            x_nms = 1-2*c_nms;
     
            % Passing through channel   
            sig = sqrt(var_N(i_SNR));
            noise = sig * randn(size(x_nms));
            
            y_nms = apply_channel(x_nms, sig, noise, channel);
    
            % BPSK demodulation
            llr = 2*y_nms/(sig^2);
    
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum_gs(llr,H,max_iter,W_ana_cur,B_ana_cur);
    
            s_vc_nms(i_frame,:,:,i_SNR) = llr_updates;
            s_cv_nms(i_frame,:,:,:,i_SNR) = llr_updates_full;
            llr_in_nms(i_frame,:,i_SNR) = llr;
    
            % estimating the codeword from bch decoding
            c_hat_nms = (llr_out_nms < 0)';
        else

            llr = llr_ch_etu(:,i_frame,i_SNR);
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum_gs(llr,H,max_iter,W_ana_cur,B_ana_cur);

            c_nms = enc_etu(:,i_frame,i_SNR);

            c_hat_nms = (llr_out_nms > 0);
        end

        % bit error rate  
        err_nms = err_nms + sum(c_hat_nms ~= c_nms);

    end
    BERs_wms_gauss(i_SNR) = err_nms/(N*num_frames);
end

BERs_wms_nn = zeros(size(BER_nms));
for i_SNR = 1:length(SNR_dB)
    % fix seed of randn for reproducability
    rng(i_SNR,'twister')

    disp(SNR_dB(i_SNR));
    err_nms = 0;

    for i_frame = 1:num_frames
        
        if (mod(i_frame,1000) == 0)
            disp(i_frame);
        end

        if (channel == "AWGN")

            % generate a random message - FIX to 0
            m = randi([0 1], K, 1); 
            
            % encode the message
            c_nms = G*m;
            
            % BPSK modulation
            x_nms = 1-2*c_nms;
    
            % Passing through channel   
            sig = sqrt(var_N(i_SNR));
            noise = sig * randn(size(x_nms));
            
            y_nms = apply_channel(x_nms, sig, noise, channel);
    
            % BPSK demodulation
%             llr = bpskdemod_soft(y_nms);
            llr = 2*y_nms/(sig^2);
    
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum_gs(llr,H,max_iter,W_nn_cur,B_nn_cur);
    
            s_vc_nms(i_frame,:,:,i_SNR) = llr_updates;
            s_cv_nms(i_frame,:,:,:,i_SNR) = llr_updates_full;
            llr_in_nms(i_frame,:,i_SNR) = llr;
    
            % estimating the codeword from bch decoding
            c_hat_nms = (llr_out_nms < 0)';
        else

            llr = llr_ch_etu(:,i_frame,i_SNR);
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum_gs(llr,H,max_iter,squeeze(W_nn_cur(:,:,i_SNR)),squeeze(B_nn_cur(:,:,i_SNR)));

            c_nms = enc_etu(:,i_frame,i_SNR);

            c_hat_nms = (llr_out_nms > 0);
        end

        % bit error rate  
        err_nms = err_nms + sum(c_hat_nms ~= c_nms);

    end
    BERs_wms_nn(i_SNR) = err_nms/(N*num_frames);
end


%% 
BER_nms_temp = BER_nms%(9:end);
BERs_wms_gauss_temp = BERs_wms_gauss%(9:end);
BERs_wms_nn_temp = BERs_wms_nn%(9:end);
SNR_dB_temp = SNR_dB%13:22;%SNR_dB(9:end);

f = figure;
semilogy(SNR_dB_temp,BER_nms_temp,'-o','MarkerSize',12,'LineWidth',3);
hold on;
semilogy(SNR_dB_temp,BERs_wms_gauss_temp,'-bs','MarkerSize',12,'LineWidth',3);
semilogy(SNR_dB_temp,BERs_wms_nn_temp,'-rd','MarkerSize',12,'LineWidth',3);
ylabel("BER");

xlabel("SNR");

leg = legend('Min-Sum', 'NNMS with weights from Gaussian Approx.', 'NNMS trained on ETU data');

leg.FontSize = 32;
legend('Location','southwest');
set(leg,'color','none');
leg.BoxFace.ColorType='truecoloralpha';
leg.BoxFace.ColorData=uint8(255*[1 1 1 0.5]');

grid on;
ax = gca;
fs = 36;
set(gca,'FontSize',fs);

% figure
f.Position = [1500 1000 1250 750];