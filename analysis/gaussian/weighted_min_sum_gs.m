function [llr_out,llr_updates, llr_updates_full] = weighted_min_sum_gs(llr_in,H,max_iter,W,B)
    
    if (nargin <= 3)
        W = zeros(size(H,2),1);
        B = zeros(size(W));
    end
    W_ref = W;
    B_ref = B;
    if (size(W,2) == 1)
        W = repmat(W_ref,1,size(H,1))';
        B = repmat(B_ref,1,size(H,1))';
    end

    m = size(H,1); % num. check nodes 
    
    llr_in_rep = repmat(transpose(llr_in),m,1).*H;
    L_cv = zeros(size(H));

    % save the incoming messages at the var node : need size iter x 
    llr_updates = zeros(max_iter,size(H,2));
    llr_updates_full = zeros(max_iter,size(H,1),size(H,2));
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
            
            % get positions of active edges
            ind_non_zero = find(H(i_row,:));
            L_cur_val = L_cur(ind_non_zero);
 
            % calculate exact llr updates for each edge
            % calculate min and 2nd min and use them
            [val, ind] = mink(abs(L_cur_val),2);

            sign_vec = sign(L_cur_val);
            prod_sign = prod(sign_vec);
            % first update everyithing with the min abs and correct the update of min pos
            llr_update = val(1)*ones(size(sign_vec));
            llr_update(ind(1)) = val(2);

%             % update the sign and tune the llrs
%             W_cur = W(i_row,ind_non_zero);
%             B_cur = B(i_row,ind_non_zero);
%             llr_update = W_cur.*sign_vec.*prod_sign.*(llr_update - B_cur);

            % update the llrs
            llr_update = sign_vec.*prod_sign.*(llr_update);

            % assign the update to matrix
            L_cv(i_row,ind_non_zero) = llr_update;

            edge_count = edge_count + sum(H(i_row,:));
        end

        % marginalise and calculated updated llr
        llr_in_updated = llr_in + (sum(W.*L_cv - B,1)');


        % store incoming messages at var node
        llr_updates(i_iter,:) = sum(L_cv,1);
        llr_updates_full(i_iter,:,:) = L_cv;
    end
    llr_out = llr_in_updated;

    if (size(llr_out,2) > 1)
        x = 1;
    end
    
end