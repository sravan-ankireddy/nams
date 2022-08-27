function [desired_peaks,echo_width,desired_peaks_bsb,echo_width_bsb,cancel_mean,cancel_mean_f,cancel_max_f,mode,desired_peaks_judge,echo_width_judge,residual_mean_f] = cancel_wrapper(ht_bsb,ht_bsb_dc,ht_down,opt_taps_t,params,sys_params)

    flag_down_sample        = params.flag_down_sample;
    flag_set_peak_locations = params.flag_set_peak_locations;
    flag_set_opt_delay      = params.flag_set_opt_delay;
    flag_keep_opt_delay     = params.flag_keep_opt_delay;
    flag_show_fig           = params.flag_show_fig;
    flag_constrained        = params.flag_constrained;
    flag_program_pic        = params.flag_program_pic;
    
    N_branches              = params.N_branches; 
    w_size                  = params.w_size;
    N_taps                  = params.N_taps;
    echo_width_threshold    = params.echo_width_threshold;
    peak_sep_threshold      = params.peak_sep_threshold; 
    N_gate                  = params.N_gate;    
    tap_spacing             = params.tap_spacing;
    observe_idx             = params.observe_idx;
    f_idx_bsb               = params.f_idx_bsb;  
    
    %--------------------------------------------------
    %               find coarse delay
    %--------------------------------------------------
    % echo coarse delay
      % ----------- passband peak location --------
     params.peak_th          = 0.27;%0.3;%0.35;%0.39;
     params.idx_neighbors_th = 3;%3;
     params.small_width_th   = 2;
     
     [desired_peaks,echo_width,total_pow] = detect_coarse_delay_v2(ht_bsb_dc,params);
     desired_peaks = round(desired_peaks);
     % ----------- baseband peak location --------
     params.peak_th          = 0.2;%0.2;%0.4
     params.idx_neighbors_th = 3;%2%4
     params.small_width_th   = 1;
     
     [desired_peaks_bsb,echo_width_bsb,total_pow_bsb] = detect_coarse_delay_v2(ht_down,params);
     % convert peak value to the domain of original sampling rate
     %total_delay = cal_halfband_filter_delay(sys_params.delay,  sys_params.DSR ,0);
     %desired_peaks = (desired_peaks.*2^sys_params.DSR - total_delay ) - (2^sys_params.DSR)+ 1;
     desired_peaks_bsb = round(desired_peaks_bsb);
     %show_data(abs(ht_down(1:200)),'echo down conversion');

    desired_peaks
    echo_width     
        
    %------ ranking of different peaks -------
    % if it is symmetric , it is signle peak 
    
    % not sure it is single peak           
    %[v idx_peaks] = sort(total_pow,'descend');
    [v idx_peaks] = sort(echo_width,'descend');
    
    
    if length(idx_peaks) >= 2
        desired_peaks_judge = desired_peaks(idx_peaks(1:2)) % chose the largest two peaks
        echo_width_judge    = echo_width(idx_peaks(1:2))   % actually only used for single peak case
        
        if echo_width_judge(1) > echo_width_threshold  % the first large echo is too wide 
            disp('the first large echo is too wide')
            desired_peaks_judge = desired_peaks_judge(1)
            echo_width_judge    = echo_width_judge(1)
        end        
    else
        desired_peaks_judge = desired_peaks
        echo_width_judge    = echo_width             % actually only used for single peak case
    end
    
    % check if it is a single peak
    if length(desired_peaks) == 3
        if (desired_peaks(2) - desired_peaks(1)) == (desired_peaks(3) - desired_peaks(2)) && (abs(echo_width(3) - echo_width(1)) < 2) && (echo_width(2)>echo_width(1)) && (echo_width(2)>echo_width(1))
            disp('It is actually signle narrow peak')
            desired_peaks_judge = desired_peaks(2)
            echo_width_judge    = echo_width(2)            
        end        
    end
    %-----------------------------------------

    %--------------------------------------------------
    %              mode determination 
    %--------------------------------------------------    
    two_far_peaks      = (sum(isnan(desired_peaks_judge)) == 0) && (length(desired_peaks_judge) == 2)  && (abs(desired_peaks_judge(2) - desired_peaks_judge(1)) >= peak_sep_threshold);
    single_narrow_peak = (length(desired_peaks_judge) == 1) && (echo_width_judge <= echo_width_threshold);
    single_wide_peak   = (length(desired_peaks_judge) == 1) && (echo_width_judge > echo_width_threshold);
    
    if single_wide_peak == 1 % check if it is actually two separable peaks 
        disp('check if it is actullay two separable peaks ')
%         w = 128;
%         idx_start = desired_peaks_judge - w/2;
%         idx_end   = idx_start + w -1;
%         ht_temp = zeros(1,length(ht_bsb_dc));
%         ht_temp(idx_start:idx_end) = ht_bsb_dc(idx_start:idx_end);

        params.peak_th          = 0.5;% increase the threshold 
        params.idx_neighbors_th = 3;%3;
        params.small_width_th   = 2;

        [desired_peaks,echo_width,total_pow] = detect_coarse_delay(ht_bsb_dc,params);
        desired_peaks = round(desired_peaks);        
        
              
          %[v idx_peaks] = sort(total_pow,'descend');
          [v idx_peaks] = sort(echo_width,'descend');
          
          
          if length(idx_peaks) >= 2
              desired_peaks_judge = desired_peaks(idx_peaks(1:2)) % chose the largest two peaks
              
%               if echo_width_judge(1) > echo_width_threshold  % the first large echo is too wide
%                   disp('the first large echo is too wide')
%                   desired_peaks_judge = desired_peaks_judge(1)
%                   echo_width_judge    = echo_width_judge(1)
%               end
          else
              desired_peaks_judge = desired_peaks
          end        
        
          if length(desired_peaks_judge) == 2 % it is two close peaks 
              two_far_peaks      = 0;
              single_narrow_peak = 0;
              single_wide_peak   = 0;
          end
    end
    
    
    
    if flag_set_peak_locations == 1
        %desired_peaks = [102 497] % 96 MHz
        %desired_peaks = [13 62] % 96 MHz
        desired_peaks = [96 490]; % 819.2 MHz
    end
    
    if length(desired_peaks) ~= N_branches % error checking
        disp('Error! The number of echo peaks does not equal to the number of branches !');
    end
    
    % branches delay
    params.peak_th = 0.5;
    params.idx_neighbors_th = 10;     
    branch_peaks = zeros(1,N_branches);
    tap_width = zeros(1,N_branches);
    for idx_branches = 1:N_branches
        if N_taps(idx_branches) == 1
            idx_taps = 1;
        else
            idx_taps = floor(N_taps(idx_branches)/2); % use middle taps as branches delay
        end
        %[branch_peaks(idx_branches),tap_width(idx_branches)] = detect_coarse_delay(opt_taps_t(idx_branches,idx_taps,:),params)
        [branch_peaks(idx_branches)] = detect_coarse_delay(opt_taps_t(idx_branches,idx_taps,:),params);       
        branch_peaks(idx_branches) = round(branch_peaks(idx_branches));
    end
    
    %--------------------------------------------------
    %             find optimized coefficient
    %--------------------------------------------------
    
    
    NFFT    = w_size;
    echos_f = cell(1,1);
    echos_t = cell(1,1);
    
    temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128','table_norm');
    code_to_coe_table{1} = temp.table_norm;
    
    temp = load('..\VNA_RS\data\code_to_coe\vna_RS_code_to_coe_N128_external','table_norm');
    code_to_coe_table{2} = temp.table_norm;
    
    coe     = cell(1,N_branches);
    code_max    = 2^16;
    N_codes     = 128;
    code_table  = [1:floor(code_max/N_codes):code_max];
    
    echo_compare = cell(1,2);
    delay_all = zeros(1,N_branches);
    
    %-------------------------------------------------
    %              start cancellation 
    %-------------------------------------------------

       
    if single_narrow_peak == 1
       mode = 0;
    elseif single_wide_peak == 1
       mode = 1;    
    elseif two_far_peaks == 1
       mode = 2;    
    else
       mode = 3;            
    end
    
      if ( two_far_peaks || single_narrow_peak ) % seperate optimization
        
          if( two_far_peaks )
             disp('======================== two far away peaks ===============================');    
          else
             disp('======================== single narrow peak ===============================');                  
          end
        
        
        N_branches = length(desired_peaks_judge);
        
        %-----------------------------------
        %     if two peaks are far away
        %-----------------------------------            
            for idx_branches = 1:N_branches
                idx_branches                               
                
                %---------- delay all branches according to coarse delay requirement ---------
                params.N_taps     = N_taps(idx_branches);
                params.N_branches = 1;
                params.NFFT       = size(opt_taps_t,3);
                if flag_set_opt_delay == 1
                    delay             = delay_set;
                else
                    if flag_keep_opt_delay == 1
                        delay             = 0;
                    else
                        delay             = desired_peaks_judge(idx_branches) - branch_peaks(idx_branches)
                    end
                end
                [temp_taps_t, temp_taps_f]  = delay_taps(params,delay,opt_taps_t);
                
                % -------- windowing --------
                % echo windowing
                w_start = desired_peaks_judge(idx_branches) - w_size/2;
                if w_start < 1
                    w_start = 1;
                end
                ht_bsb_w = ht_bsb(w_start:w_start + w_size-1 );
                hf_bsb_w = fft(ht_bsb_w,NFFT);                
                echos_t{1} = ht_bsb_w;
                echos_f{1} = hf_bsb_w;
                
                
                echo_compare{idx_branches} = to_pow_dB(ht_bsb_w/max(abs(ht_bsb_w)));
                
                % tap windowing 
                taps_t = zeros(N_taps(idx_branches),w_size);
                taps_f = zeros(w_size,N_taps(idx_branches));
                for idx = 1:N_taps(idx_branches)
                   taps_t(idx,:) = temp_taps_t(idx, w_start : w_start + w_size-1 );
                   taps_f(:,idx) = fft(taps_t(idx,:)).';
                end
                                                             
                delay_all(idx_branches) = delay;
                
                %--------- timing gating of taps -----------
                [taps_t, taps_f] = time_gating(taps_t,N_gate- w_start + 1);
                
                %----- show all taps ------
                if flag_show_fig == 1
                    tapst_cell = cell(1,size(taps_t,1));
                    taps_amp_cell = cell(1,size(taps_t,1));
                    taps_phase_cell = cell(1,size(taps_t,1));
                    
                    for idx = 1:size(taps_t,1)
                        tapst_cell{idx} = real(taps_t(idx,:));
                        %taps_amp_cell{idx} = to_pow_dB(fft(taps_t(idx,:)));
                        %taps_phase_cell{idx} = phase(fft(taps_t(idx,:)));
                    end
                    %show_data_para(tapst_cell,{'1','2','3','4','5','6','7','8'});
                    %     show_data_para(taps_amp_cell,{'1','2','3','4','5','6','7','8'});
                    %     show_data_para(taps_phase_cell,{'1','2','3','4','5','6','7','8'});
                end
                %------------------------
                
                
                %--------- estimate coefficient ---------
                echos_delay = 0;
                params.weights  = 1;%[1 1 1 1 1 1];% weights for different cable taps
                params.NFFT     = length(echos_f{1});
                params.f_idx    = 1:params.NFFT;
                params.fs       = 1638.4e6;
                params.peaks    = 1; % not used now, coarse delay is hardwired
                params.N_taps   = N_taps(idx_branches);
                params.tap_res  = 1; % not used now, taps response is measured directly
                params.tap_name = {['branches = ' num2str(idx_branches) ] };%{'29dBw','29dB','26dB','20dB','14dB','10dB'};
                params.N_ite    = 1;%4 % number of iterations for attenuation optimation
                params.flag_show_cancel = 1;
                params.flag_show_cancel_total = 0;
                
                if flag_constrained == 0 %unconstrained
                    [coe{idx_branches}, att, total_mse] = estimate_FIR_coe(taps_t,taps_f,echos_t,echos_f,echos_delay,params);
                else % constrained
                    lb = [];    % negative means no sign change required
                    for idx_bnb = 1:N_taps(idx_branches)
                        %lb = [lb -Inf];
                        lb = [lb -1];
                    end
                    %ub = zeros(1,N_taps(idx_branches));
                    ub = ones(1,N_taps(idx_branches));
                    
                    coe{idx_branches} = lsqlin(taps_f,echos_f{1},[],[],[],[],lb,ub) ;
                end
                %------ reconstruct echo ----------
                d_tx_hat = reconstruct_t(taps_t,coe{idx_branches});
                
                %------ cancellation ----------
                e = echos_t{1} - d_tx_hat;
                cancel_amount = to_pow_dB(echos_t{1}) - to_pow_dB(e);
                
                fs = 1.6384e9;
                if flag_show_fig == 1
                    %show_cancellation(d_tx_hat,echos_t{1},e,fs,taps_t,coe{idx_branches});
                end
                
                %mse = mean(cancel_amount);
                
                coe{idx_branches}
                
                if flag_program_pic == 1
                    code = find_coe_to_code(real(abs(coe{idx_branches})),code_table,code_to_coe_table{idx_branches});
                    
                    uint32(code)
                    code_v = 5/2^16*code
                    
                    %------- if you skip some taps ---------
                    %         code_new = 65535*ones(1,8);
                    %         code_new([1 3 5 8]) = code;
                    %         program_coe(tcp_obj,tcp_const,DAC_map{1},code_new)
                    %----------------------------------------
                    
                    program_coe(tcp_obj,tcp_const,DAC_map{idx_branches},code)
                end
            end
            
            %==================================================
            %         show global cancellation results
            %==================================================
            params.N_branches = N_branches;
            params.NFFT = 32768;
            [taps_t, taps_f]  = delay_taps(params,delay_all,opt_taps_t);
            [taps_t, taps_f] = time_gating(taps_t,N_gate);
            
            coe_all = [];
            for idx_branches = 1:N_branches
                if idx_branches == 1
                    coe_all = coe{idx_branches};
                else
                    coe_all = [coe_all coe{idx_branches}];
                end
            end
            
            d_tx_hat = reconstruct_t(taps_t,coe_all);
            
            e = ht_bsb - d_tx_hat;
            cancel_amount = to_pow_dB(ht_bsb) - to_pow_dB(e);

            cancel_mean = mean(cancel_amount(observe_idx));
            temp = to_pow_dB(fft(ht_bsb)) - to_pow_dB(fft(e));
            cancel_mean_f = mean(temp(f_idx_bsb));
            cancel_max_f  = max(temp(f_idx_bsb));            
            ef = to_pow_dB(fft(e));
            residual_mean_f = mean(ef(f_idx_bsb));
            
            fs = 1.6384e9;
            if flag_show_fig == 1
                show_cancellation(d_tx_hat(observe_idx),ht_bsb(observe_idx),e(observe_idx),fs,taps_t,coe{1});
                %show_all_taps(taps_t);
            end
      else
        % joint optimization
        %-----------------------------------
        %     if two peaks are close
        %-----------------------------------
          if (single_wide_peak == 1) % single peak with wide width
              disp('===================== single peak with wide width ===========================');
 
              echo_peaks = desired_peaks_judge(1);
              
              %----------------------------------
              %          tap delay
              %----------------------------------
              delay_all = zeros(1,N_branches);
              for idx_branches = 1:N_branches
                  params.N_taps     = N_taps(idx_branches);
                  params.N_branches = 2;
                  params.NFFT       = 32768;
                  if flag_set_opt_delay == 1
                      delay_all(idx_branches)      = delay_set;
                  else
                      if flag_keep_opt_delay == 1
                          delay_all(idx_branches)  = 0;
                      else
                          if idx_branches == 1
                            %delay_all(idx_branches)  = desired_peaks_judge(1) - branch_peaks(idx_branches) + ceil(tap_spacing*sys_params.fs*N_taps(idx_branches)/2);
                            delay_all(idx_branches)  = desired_peaks_judge(1) - branch_peaks(idx_branches) + ceil(tap_spacing*sys_params.fs*(N_taps(idx_branches)/2+3));                            
                          else
                            %delay_all(idx_branches)  = desired_peaks_judge(1) - branch_peaks(idx_branches) - ceil(tap_spacing*sys_params.fs*N_taps(idx_branches)/2);              
                            delay_all(idx_branches)  = desired_peaks_judge(1) - branch_peaks(idx_branches) - ceil(tap_spacing*sys_params.fs*(N_taps(idx_branches)/2+3));              
                          end
                      end
                  end
              end
              
          else % double peak 
              disp(['==================== two close peaks with distance:' num2str(abs(desired_peaks_judge(2) - desired_peaks_judge(1))) '================================']);
 
              echo_peaks = ( desired_peaks_judge(1) + desired_peaks_judge(2) )/2;
              w_size = w_size*2;
              
              %----------------------------------
              %          tap delay
              %----------------------------------
              delay_all = zeros(1,N_branches);
              for idx_branches = 1:N_branches
                  params.N_taps     = N_taps(idx_branches);
                  params.N_branches = 2;
                  params.NFFT       = 32768;
                  if flag_set_opt_delay == 1
                      delay_all(idx_branches)      = delay_set;
                  else
                      if flag_keep_opt_delay == 1
                          delay_all(idx_branches)  = 0;
                      else
                          delay_all(idx_branches)  = desired_peaks_judge(idx_branches) - branch_peaks(idx_branches);
                      end
                  end
              end
          end
            [taps_t, taps_f]  = delay_taps(params,delay_all,opt_taps_t);
            
                      
            %--------- time gating of taps -----------
            [taps_t, taps_f] = time_gating(taps_t,N_gate);
            
            %------------------------------
            %        echo windowing
            %------------------------------
            w_start = echo_peaks - w_size/2;
            if w_start < 1
                w_start = 1;
            end
            
            ht_bsb_w = ht_bsb(w_start:w_start + w_size-1 );
            hf_bsb_w = fft(ht_bsb_w);
            
            echos_t{1} = ht_bsb_w;
            echos_f{1} = hf_bsb_w;
                       
            
            %----------------------------------
            %          tap windowing 
            %----------------------------------
            taps_w_t = zeros(size(taps_t,1),w_size);            
            taps_w_f = zeros(w_size,size(taps_t,1));
            
            for idx_taps = 1:size(taps_t,1)
                taps_w_t(idx_taps,:) = taps_t(idx_taps,w_start:w_start + w_size - 1 );
                taps_w_f(:,idx_taps) = fft(taps_w_t(idx_taps,:),w_size).';
            end
           
            if flag_show_fig == 1 % show all tap response
               show_all_taps(taps_w_t);
            end
            %----------------------------------
            %       estimate coefficient
            %----------------------------------
            echos_delay = 0;
            params.weights  = 1;%[1 1 1 1 1 1];% weights for different cable taps
            params.NFFT     = length(echos_f{1});
            params.f_idx    = 1:params.NFFT;
            params.fs       = 1638.4e6;
            params.peaks    = 1; % not used now, coarse delay is hardwired
            params.N_taps   = N_taps(1)+N_taps(2);
            params.tap_res  = 1; % not used now, taps response is measured directly
            params.tap_name = {['branches = ' num2str(idx_branches) ] };%{'29dBw','29dB','26dB','20dB','14dB','10dB'};
            params.N_ite    = 1;%4 % number of iterations for attenuation optimation
            params.flag_show_cancel = 1;
            params.flag_show_cancel_total = 0;
            
            if flag_constrained == 0 %unconstrained
                [coe{1}, att, total_mse] = estimate_FIR_coe(taps_w_t,taps_w_f,echos_t,echos_f,echos_delay,params);
            else % constrained
                lb = [];    % negative means no sign change required
                for idx_bnb = 1:N_taps(1)+N_taps(2)
                    %lb = [lb -Inf];
                    lb = [lb -1];
                end
                %ub = zeros(1,N_taps(idx_branches));
                ub = ones(1,N_taps(1)+N_taps(2));
                
                coe{1} = lsqlin(taps_w_f,echos_f{1},[],[],[],[],lb,ub) ;
            end
            %------ reconstruct echo ----------
            d_tx_hat = reconstruct_t(taps_w_t,coe{1});
            
            %------ cancellation ----------
            e = echos_t{1} - d_tx_hat;
%            cancel_amount = to_pow_dB(echos_t{1}) - to_pow_dB(e);
            
            fs = 1.6384e9;
%             if flag_show_fig == 1
%                 show_cancellation(d_tx_hat,echos_t{1},e,fs,taps_w_t,coe{idx_branches});
%             end  
                                     
            %==================================================
            %         show global cancellation results
            %==================================================
            params.N_branches = N_branches;
            params.NFFT = 32768;
            params.N_taps = N_taps(1);
            [taps_t, taps_f]  = delay_taps(params,delay_all,opt_taps_t);
            [taps_t, taps_f] = time_gating(taps_t,N_gate);
            

            temp = coe{1};
            coe_all = [];
            for idx_branches = 1:N_branches
                if idx_branches == 1
                    coe_all = temp(1:N_taps(1)).';
                else
                    coe_all = [coe_all temp(N_taps(1)+1:end).'];
                end
            end
            
            d_tx_hat = reconstruct_t(taps_t,coe_all);
            
            e = ht_bsb - d_tx_hat;
            
            cancel_amount = to_pow_dB(ht_bsb) - to_pow_dB(e);
            
            cancel_mean = mean(cancel_amount(observe_idx));
            temp = to_pow_dB(fft(ht_bsb)) - to_pow_dB(fft(e));
            cancel_mean_f = mean(temp(f_idx_bsb));
            cancel_max_f  = max(temp(f_idx_bsb));
            ef = to_pow_dB(fft(e));
            residual_mean_f = mean(ef(f_idx_bsb));
            
            
            fs = 1.6384e9;
            if flag_show_fig == 1
                show_cancellation(d_tx_hat(observe_idx),ht_bsb(observe_idx),e(observe_idx),fs,taps_t,coe{1});
                %show_all_taps(taps_t);
            end
            
      end