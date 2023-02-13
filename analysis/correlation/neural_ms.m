function [llr_out,llr_in_check_node,llr_in_var_node] = neural_ms(llr_in,H,max_iter,decoder_type)
    
    if (decoder_type == "ms" || decoder_type == "bp")
        B_cv = zeros(max_iter,sum(sum(H)));
        W_cv = ones(max_iter,sum(sum(H)));
    elseif (decoder_type == "nms")
        % load pretrained model weights
        load("nms_weights/nams_BCH_63_36_st_20000_lr_0.005_AWGN_ent_0_nn_eq_1_relu_1_max_iter_5_1_8.mat","B_cv","W_cv");
    end

    m = size(H,1); % num. check nodes 
    n = size(H,2); % num. var nodes
    
    llr_in_check_node = zeros(max_iter,m,n);
    llr_in_var_node = zeros(max_iter,m,n);

    llr_in_rep = repmat(transpose(llr_in),m,1).*H;
    L_cv = zeros(size(H));

    for i_iter = 1:max_iter        
        % vc
        % parse all the var nodes
        L_vc = (sum(L_cv,1) - L_cv).*H  ;
        % L_vc is the extrinsic information that passes to check node
        
        L_vc = llr_in_rep + L_vc;
        
        % cv
        % parse all the check nodes
        L_cv = zeros(size(H));
        edge_count = 0;
        for i_row = 1:m
            % extract the current LLRs
            L_cur = L_vc(i_row,:);
            
            % tune the LLRs using weights and bias, 1 row at a time
            W = W_cv(i_iter,edge_count+1:edge_count+sum(H(i_row,:)));
            B = B_cv(i_iter,edge_count+1:edge_count+sum(H(i_row,:)));
                
            % get positions of active edges
            ind_non_zero = find(H(i_row,:));
            L_cur_val = L_cur(ind_non_zero);
            if (decoder_type == "bp")   

                llr_update = zeros(1,length(ind_non_zero));

                % iterate over var nodes
                for i_v = 1:length(ind_non_zero)
                    v = ind_non_zero(i_v);
                    ind_v = ind_non_zero(ind_non_zero~=v);
                    llr_update(i_v) = 2*atanh(prod(tanh(L_cur(ind_v)/2)));
                end
                % update the sign and tune the llrs
                llr_update = W.*(llr_update - B);
            else 
                % calculate exact llr updates for each edge
                % calculate min and 2nd min and use them
                [val, ind] = mink(abs(L_cur_val),2);
    
                sign_vec = sign(L_cur_val);
                prod_sign = prod(sign_vec);
                % first update everyithing with the min abs and correct the update of min pos
                llr_update = val(1)*ones(size(sign_vec));
                llr_update(ind(1)) = val(2);
                % update the sign and tune the llrs
                llr_update = W.*sign_vec.*prod_sign.*(llr_update - B);
            end

            % store incoming updates to check node
            llr_in_check_node(i_iter,i_row,ind_non_zero) = llr_update;

            % assign the update to matrix
            L_cv(i_row,ind_non_zero) = llr_update;

            edge_count = edge_count + sum(H(i_row,:));
        end
        
        % marginalise and calculated updated llr
        llr_in_updated = sum(L_cv,1) + llr_in';

        % store incoming updates to var node
        llr_in_var_node(i_iter,:,:) = L_cv;
       
    end
    llr_out = llr_in_updated;
    
end