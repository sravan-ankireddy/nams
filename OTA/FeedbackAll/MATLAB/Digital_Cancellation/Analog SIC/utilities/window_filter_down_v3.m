% change the order of windowing

% 1. frequency filtering to simulate the correct bandwidth
% 2. time domain windowing 
% 3. store frequency domain data
% 4. fitlering in the frequency domain to match the echo signal bandwidth 

% time domain windowing 
% filter 
% downsample
% go back to time domain for frequency domain down sampling


function [dt_d,idx_peak_small_w_out] = window_filter_down_v3(hf,params)

peaks            = params.peaks;
f_window_base    = params.f_window_base; % frequency domain filtering for high BW tap response , high frequency resolution
f_window_sig     = params.f_window_sig;  % frequency domain filtering for echo signal bandwidth, low frequency resolution
w_size           = params.w_size;
flag_filter_only = params.flag_filter_only;
small_w_size     = params.small_w_size;
flag_is_echo     = params.flag_is_echo;
idx_peak_small_w = params.idx_peak_small_w;
flag_skip_prepare = params.flag_skip_prepare;


idx_peak_small_w_out = zeros(1,length(peaks));

if flag_filter_only == 1
    dt_d = zeros(length(peaks),w_size);
else
    dt_d = zeros(length(peaks),small_w_size);
end

for idx = 1:length(peaks)
    %====================================================
    %             preparing the base function 
    %====================================================
    if flag_skip_prepare == 1 % only TTD predicted response don't need to go through the preparation process
        hf_w = hf;
    else
        %-------------------------------------------------
        %       filtering in the original 32k domain
        %-------------------------------------------------
        hf_filter_base = hf.*f_window_base;
        ht_filter_base = ifft(hf_filter_base);
        
        %-------------------------------
        %    time domain windowing
        %------------------------------
        w_start = peaks(idx) - w_size/2;
        if w_start < 1
            w_start = 1;
        end
        
        ht_w = ht_filter_base(w_start:w_start + w_size-1 );
        hf_w = fft(ht_w);       
    end
    
    %====================================================
    %             simplification process  
    %====================================================
    %---------------------------------------------------------------
    %       filtering to match the signal BW in downsampled domain
    %---------------------------------------------------------------
    hf_filter = hf_w.*f_window_sig;
    
       
    if flag_filter_only == 1
        dt_d(idx,:) = ifft(hf_filter);
    else
        %-------------------------------------------------
        %          down-convert and make it real
        %-------------------------------------------------
        %[df_w,dt_w] = down_convert_real(hf_filter,fc,fs);
        
        NFFT = w_size; % fft size before down-sampling
        
        % down conversion
        df_half = hf_filter(1:end/2);
        idx_nz = find(df_half~=0);
        
        df_down_half = zeros(1,length(df_half));
        df_down_half(2:length(idx_nz)+1) = df_half(idx_nz);
        
        %-------------------------------------------------
        %                down-sampling
        %-------------------------------------------------
        % find the largest down-sampling rate
        DSR = 2^floor( log2 ( (NFFT/2)/length(idx_nz) ) ) ;
        NFFT_down = NFFT/DSR;
        
        df_down_dec_half = df_down_half(1:NFFT_down/2);
        
        df_bsb  = [df_down_dec_half 0 fliplr(conj(df_down_dec_half(2:end)))]; % make it real
        dt_bsb = ifft(df_bsb);
        
        %-------------------------------------------------
        %       windowing again in the time domain
        %-------------------------------------------------
        if( length(dt_bsb) > small_w_size)
            if flag_is_echo == 1
                [v, idx_peak] = max(abs(dt_bsb));
                
                %----------- test ------------
                %idx_peak = idx_peak + 1;
                %-----------------------------
                
                idx_peak_small_w_out(idx) = idx_peak ;
            else
                idx_peak = idx_peak_small_w(idx) ;
            end
            
            
            w_start = idx_peak - small_w_size/2;
            if w_start < 1
                w_start = 1;
            end
            
            if ( w_start + small_w_size-1 > length(dt_bsb))% abnormal condition, just ignore this  
              dt_d(idx,:) = dt_bsb(1:small_w_size);         
            else
              dt_d(idx,:) = dt_bsb(w_start:w_start + small_w_size-1 );                        
            end
        else            
            dt_d(idx,:) = dt_bsb;            
        end
    end
end