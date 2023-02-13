BERs_wms = zeros(size(BER_nms));
for i_SNR = 1:length(SNR_dB)
    % fix seed of randn for reproducability
    rng(i_SNR,'twister')

    disp(SNR_dB(i_SNR));
    err_nms = 0;

    W_cur = squeeze(W_cv_opt_mat(:,:,i_SNR));
    B_cur = zeros(size(H));

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
            x_nms = 1-2*c_nms;
    
            % Passing through channel   
            sig = sqrt(var_N(i_SNR));
            noise = sig * randn(size(x_nms));
            
            y_nms = apply_channel(x_nms, sig, noise, channel);
    
            % BPSK demodulation
%             llr = bpskdemod_soft(y_nms);
            llr = 2*y_nms/(sig^2);
    
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum(llr,H,max_iter,W_cur,B_cur);
    
            s_vc_nms(i_frame,:,:,i_SNR) = llr_updates;
            s_cv_nms(i_frame,:,:,:,i_SNR) = llr_updates_full;
            llr_in_nms(i_frame,:,i_SNR) = llr;
    
            % estimating the codeword from bch decoding
            c_hat_nms = (llr_out_nms < 0)';
        else

            llr = llr_ref(:,i_frame,SNR_dB(i_SNR));
            [llr_out_nms,llr_updates, llr_updates_full] = weighted_min_sum(llr,H,max_iter,W_cur,B_cur);

            c_nms = enc_ref(:,i_frame,SNR_dB(i_SNR));

            c_hat_nms = (llr_out_nms > 0)';
        end

        % bit error rate  
        err_nms = err_nms + sum(c_hat_nms ~= c_nms);

    end
    BERs_wms(i_SNR) = err_nms/(N*num_frames);
end

figure;
semilogy(SNR_dB,BER_nms);
hold on;
semilogy(SNR_dB,BERs_wms);
legend("min-sum","weighted min-sum ch gauss");