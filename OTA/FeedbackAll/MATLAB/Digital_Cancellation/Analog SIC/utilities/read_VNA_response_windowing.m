% read taps and windowing

function opt_taps_t = read_VNA_response_windowing(params)

taps_table           = params.taps_table; % taps index table
N_branches           = params.N_branches;
NFFT                 = params.NFFT;
N_taps               = length(taps_table{1});

flag_predict_TTD     = params.flag_predict_TTD;
peaks                = params.peaks;
w_size               = params.w_size;
flag_filter_only     = params.flag_filter_only;
small_w_size         = params.small_w_size;
idx_peak_small_w     = params.idx_peak_small_w;

f_window_sig         = params.f_window_sig;
f_window_base        = params.f_window_base;

flag_is_echo = 0;

if flag_filter_only == 1
    opt_taps_t = zeros(N_branches,N_taps,NFFT);
else
    opt_taps_t = zeros(N_branches,N_taps,small_w_size);
end

%---------- for window_filter_down ---------
params.f_window_bsb     = f_window_sig;
params.w_size           = w_size;
params.flag_filter_only = flag_filter_only;
params.small_w_size     = small_w_size;
params.flag_is_echo     = flag_is_echo;
params.idx_peak_small_w = idx_peak_small_w;
%for v2
params.NFFT             = NFFT;
%for v3
params.f_window_base    = f_window_base; % frequency domain filtering for high BW tap response , high frequency resolution
params.f_window_sig     = f_window_sig;  % frequency domain filtering for echo signal bandwidth, low frequency resolution

for idx_branches = 1:N_branches
    
    if( flag_predict_TTD == 1 && idx_branches == 2 )
        % predict TTD response
        flag_load_ADC = 0;
        %[taps_f, taps_t]= predict_TTD_freq_response_switch_simple(flag_load_ADC,small_w_size);
        
%         params.flag_skip_prepare = 0;        
%         [taps_f, taps_t]= predict_TTD_freq_response_switch_simple_v2(flag_load_ADC,small_w_size);

        params.flag_skip_prepare = 1;
        [taps_f, taps_t]= predict_TTD_freq_response_switch_simple_v3(flag_load_ADC,peaks(idx_branches),w_size);
        
        
        params.peaks            = peaks(idx_branches);
        params.idx_peak_small_w = idx_peak_small_w(2);
        
        for idx_taps = 1:length(taps_table{idx_branches})
            %opt_taps_t(idx_branches,idx_taps,:) = window_filter_down(taps_t(idx_taps,:),peaks(idx_branches),f_window_bsb,w_size,fc,fs,OSR,flag_filter_only,small_w_size,flag_is_echo,idx_peak_small_w(2));
            %opt_taps_t(idx_branches,idx_taps,:) = window_filter_down(taps_t(idx_taps,:),params);
            opt_taps_t(idx_branches,idx_taps,:) = window_filter_down_v3(taps_f(:,idx_taps).',params);                             
        end
    else
        % load directly
        params.peaks            = peaks(idx_branches);
        params.idx_peak_small_w = idx_peak_small_w(idx_branches);
        params.flag_skip_prepare = 0;
        
        for idx_taps = 1:length(taps_table{idx_branches})
            %------------------------------
            %      load taps response
            %------------------------------
            tap_name = taps_table{idx_branches};
%             if idx_branches == 1
%                filename = sprintf('..\\VNA_RS\\data\\20180502_no_external\\tap_%d_code_1',tap_name(idx_taps));            
%             else
%                filename = sprintf('..\\VNA_RS\\data\\20180503_external\\tap_%d_code_1',tap_name(idx_taps));            
%             end             
            filename = sprintf('%s\\tap_%d_code_0',params.foldername,tap_name(idx_taps));
 
            h_struct = load(filename);
            hf = h_struct.df;
            ht = ifft(hf);
            
            %opt_taps_t(idx_branches,idx_taps,:) = window_filter_down(ht,peaks(idx_branches),f_window_bsb,w_size,fc,fs,OSR,flag_filter_only,small_w_size,flag_is_echo,idx_peak_small_w(idx_branches));
            %opt_taps_t(idx_branches,idx_taps,:) = window_filter_down(ht,params);
            opt_taps_t(idx_branches,idx_taps,:) = window_filter_down_v3(hf,params);
        end
    end
end
