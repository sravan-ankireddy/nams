% read automatic vna measurement

function [taps_f_out,taps_t_out,f] = read_VNA_complex_all_taps_auto_96M_direct(flag_int)


if flag_int == 1 % internal taps
    N_taps = 8;
    taps_f = cell(1,N_taps);
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_1_code_1_96M_direct.mat';
    temp = load(filename);
    taps_f{1} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_2_code_1_96M_direct.mat';    
    temp = load(filename);    
    taps_f{2} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_3_code_1_96M_direct.mat';    
    temp = load(filename);
    taps_f{3} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_4_code_1_96M_direct.mat';
    temp = load(filename);
    taps_f{4} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_5_code_1_96M_direct.mat';  
    temp = load(filename);
    taps_f{5} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_6_code_1_96M_direct.mat';    
    temp = load(filename);
    taps_f{6} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_7_code_1_96M_direct.mat';    
    temp = load(filename);
    taps_f{7} = temp.df;
    
    filename = 'VNA_RS\data\20180314_echo_taps\tap_8_code_1_96M_direct.mat';    
    temp = load(filename);
    taps_f{8} = temp.df;

    NFFT = 4096;
    taps_f_out = zeros(NFFT,N_taps);
    taps_t_out = zeros(N_taps,NFFT);
    for idx = 1:N_taps
        temp       = taps_f{idx};
        temp_bsb   = temp;
        
        taps_f_out(:,idx) = temp_bsb.';
        taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
    end
    
else % external taps
    if 1 % load measured external
        N_taps = 5;
        taps_f = cell(1,N_taps);
                
        filename = 'VNA_RS\data\20180314_echo_taps\tap_10_code_1_96M_direct.mat';          
        temp = load(filename);
        taps_f{1} = temp.df;
        
        filename = 'VNA_RS\data\20180314_echo_taps\tap_11_code_1_96M_direct.mat';          
        temp = load(filename);
        taps_f{2} = temp.df;
        
        filename = 'VNA_RS\data\20180314_echo_taps\tap_12_code_1_96M_direct.mat';          
        temp = load(filename);
        taps_f{3} = temp.df;
        
        filename = 'VNA_RS\data\20180314_echo_taps\tap_13_code_1_96M_direct.mat';          
        temp = load(filename);
        taps_f{4} = temp.df;
        
        filename = 'VNA_RS\data\20180314_echo_taps\tap_15_code_1_96M_direct.mat';          
        temp = load(filename);
        taps_f{5} = temp.df;
        
        NFFT = 4096;
        taps_f_out = zeros(NFFT,N_taps);
        taps_t_out = zeros(N_taps,NFFT);
        for idx = 1:N_taps
            temp       = taps_f{idx};
            temp_bsb   = temp;
            
            taps_f_out(:,idx) = temp_bsb.';
            taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
        end
    else % load predicted external
        [taps_f_out, taps_t_out]= predict_TTD_freq_response();       
        %[taps_f_out, taps_t_out]= predict_TTD_freq_response_x6();       
        %keyboard;
    end
end


