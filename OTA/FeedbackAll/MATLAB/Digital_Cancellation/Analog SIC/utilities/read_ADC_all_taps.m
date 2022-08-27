% read all taps from ADC measurement
function opt_taps_t = read_ADC_all_taps(params)

taps_table    = params.taps_table; % taps index table
N_branches    = params.N_branches;
NFFT          = params.NFFT;
N_taps        = length(taps_table{1});

f_window_bsb         = params.f_window_bsb;
flag_filter_response = params.flag_filter_response;
flag_predict_TTD     = params.flag_predict_TTD;

opt_taps_t = zeros(N_branches,N_taps,NFFT);
for idx_branches = 1:N_branches
   
    if( flag_predict_TTD == 1 && idx_branches == 2 ) % predict TTD response
        flag_load_ADC = 1;
        [taps_f, taps_t]= predict_TTD_freq_response_switch(flag_load_ADC);

        for idx_taps = 1:length(taps_table{idx_branches})
           opt_taps_t(idx_branches,idx_taps,:) = taps_t(idx_taps,:);
        end
        
    else % load directly     
        for idx_taps = 1:length(taps_table{idx_branches})
            tap_name = taps_table{idx_branches};
            if idx_branches == 1 % use code 77 for internal
               filename = sprintf('data\\20180410_ADC_diff\\tap%d_code77',tap_name(idx_taps));
               %filename = sprintf('data\\20180410_ADC_diff\\tap%d_code1',tap_name(idx_taps));               
            else % use code 1 for external
               filename = sprintf('data\\20180410_ADC_diff\\tap%d_code1',tap_name(idx_taps));             
            end
            h_struct = load(filename);
            hf = h_struct.df;
            
            if flag_filter_response == 1
                hf = hf.*f_window_bsb;
                ht = ifft(hf,NFFT);
            end
            
            opt_taps_t(idx_branches,idx_taps,:) = ht;
            
            %figure(100); plot(to_pow_dB(real(ht_bsb))); hold on;
        end            
    end
end


