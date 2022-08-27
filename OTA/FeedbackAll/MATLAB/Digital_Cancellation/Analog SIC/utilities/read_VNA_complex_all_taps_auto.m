% read automatic vna measurement

function [taps_f_out,taps_t_out,f] = read_VNA_complex_all_taps_auto(params)

flag_int = params.flag_int;
flag_predict_TTD = params.flag_predict_TTD;

if flag_int == 1 % internal taps
    N_taps = 8;
    taps_f = cell(1,N_taps);

    %filename = 'VNA_RS\data\20180315_echo_taps\tap_1_code_1.mat';
    filename = 'VNA_RS\data\20180411\tap_1_code_1.mat';    
    temp = load(filename);
    taps_f{1} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_2_code_1.mat';
    temp = load(filename);    
    taps_f{2} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_3_code_1.mat';
    temp = load(filename);
    taps_f{3} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_4_code_1.mat';
    temp = load(filename);
    taps_f{4} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_5_code_1.mat';
    temp = load(filename);
    taps_f{5} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_6_code_1.mat';
    temp = load(filename);
    taps_f{6} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_7_code_1.mat';
    temp = load(filename);
    taps_f{7} = temp.df;
    
    filename = 'VNA_RS\data\20180411\tap_8_code_1.mat';
    temp = load(filename);
    taps_f{8} = temp.df;

    
    NFFT = 16384*2;
    taps_f_out = zeros(NFFT,N_taps);
    taps_t_out = zeros(N_taps,NFFT);
    for idx = 1:N_taps
        temp       = taps_f{idx};
        temp_bsb   = temp;
        
        taps_f_out(:,idx) = temp_bsb.';
        taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
    end

else % external taps
    if flag_predict_TTD ~= 1 % load measured external
        N_taps = 5;
        taps_f = cell(1,N_taps);
        
        %filename = 'VNA_RS\data\20180327_TTD_v2\tap_10_code_1.mat';  
        filename = 'VNA_RS\data\20180411\tap_10_code_1.mat';          
        temp = load(filename);
        taps_f{1} = temp.df;

        filename = 'VNA_RS\data\20180411\tap_11_code_1.mat';
        temp = load(filename);
        taps_f{2} = temp.df;
        
        filename = 'VNA_RS\data\20180411\tap_12_code_1.mat';
        temp = load(filename);
        taps_f{3} = temp.df;
        
        filename = 'VNA_RS\data\20180411\tap_13_code_1.mat';
        temp = load(filename);
        taps_f{4} = temp.df;
        
        filename = 'VNA_RS\data\20180411\tap_15_code_1.mat';
        temp = load(filename);
        taps_f{5} = temp.df;
        
        NFFT = 16384*2;
        taps_f_out = zeros(NFFT,N_taps);
        taps_t_out = zeros(N_taps,NFFT);
        for idx = 1:N_taps
            temp       = taps_f{idx};
            temp_bsb   = temp;
            
            taps_f_out(:,idx) = temp_bsb.';
            taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
        end        
    else % load predicted external
        %[taps_f_out, taps_t_out]= predict_TTD_freq_response();       
        %[taps_f_out, taps_t_out]= predict_TTD_freq_response_x6();       
        flag_load_ADC = 0;
        [taps_f_out, taps_t_out]= predict_TTD_freq_response_switch(flag_load_ADC);
        
        %-------- constuct other taps based on the first tap-----------        
%         if 0   % measure the first tap by disconnecting the cable plant 
%             filename = 'VNA_RS\data\20180309_echo_taps\tap_10_code_1.mat';
%             temp = load(filename);
%             base_hf_v = temp.df;
%         else   % measure the first tap without disconnecting cable plant
%             [base_hf_v,base_ht_v] = recover_tap();
%         end
%         [taps_f_out, taps_t_out]= predict_TTD_freq_response_measure_base(base_hf_v);
        %------------------------------------   
    end
end


