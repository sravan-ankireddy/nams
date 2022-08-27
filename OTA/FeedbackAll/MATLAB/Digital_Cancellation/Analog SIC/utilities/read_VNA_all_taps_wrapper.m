% read all VNA taps response

function [opt_taps_t] = read_VNA_all_taps_wrapper(params)


flag_predict_TTD     = params.flag_predict_TTD;
flag_filter_response = params.flag_filter_response;
f_window             = params.f_window;
N_branches           = params.N_branches;
N_taps               = params.N_taps;

if flag_filter_response == 0
    f_window = ones(1,length(f_window)); % should not modify tap response
end

NFFT = 16384*2;
opt_taps_t = zeros(N_branches,N_taps(1),NFFT);

%======================================
%            internal taps
%======================================
taps_f = cell(1,N_taps(1));

%filename = 'VNA_RS\data\20180315_echo_taps\tap_1_code_1.mat';
foldername = '..\\VNA_RS\\data\\20180502_no_external'; 

filename = sprintf('%s\\tap_1_code_1.mat',foldername);
temp = load(filename);
taps_f{1} = temp.df;

filename = sprintf('%s\\tap_2_code_1.mat',foldername);
temp = load(filename);
taps_f{2} = temp.df;

filename = sprintf('%s\\tap_3_code_1.mat',foldername);
temp = load(filename);
taps_f{3} = temp.df;

filename = sprintf('%s\\tap_4_code_1.mat',foldername);
temp = load(filename);
taps_f{4} = temp.df;

filename = sprintf('%s\\tap_5_code_1.mat',foldername);
temp = load(filename);
taps_f{5} = temp.df;

filename = sprintf('%s\\tap_6_code_1.mat',foldername);
temp = load(filename);
taps_f{6} = temp.df;

filename = sprintf('%s\\tap_7_code_1.mat',foldername);
temp = load(filename);
taps_f{7} = temp.df;

filename = sprintf('%s\\tap_8_code_1.mat',foldername);
temp = load(filename);
taps_f{8} = temp.df;



taps_f_out = zeros(NFFT,N_taps(1));
taps_t_out = zeros(N_taps(1),NFFT);
for idx = 1:N_taps(1)
    temp       = taps_f{idx};
    temp_bsb   = temp.*f_window;
    
    taps_f_out(:,idx) = temp_bsb.';
    taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
end

idx_branches = 1;
for idx_taps = 1:N_taps(1)
    opt_taps_t(idx_branches,idx_taps,:) = taps_t_out(idx_taps,:);
end

%======================================
%            external taps
%======================================

if flag_predict_TTD ~= 1 % load measured external
    taps_f = cell(1,N_taps(2));
    foldername = '..\\VNA_RS\\data\\20180503_external';
    
    filename = sprintf('%s\\tap_10_code_1.mat',foldername);
    temp = load(filename);
    taps_f{1} = temp.df;
    
    filename = sprintf('%s\\tap_11_code_1.mat',foldername);
    temp = load(filename);
    taps_f{2} = temp.df;
    
    filename = sprintf('%s\\tap_12_code_1.mat',foldername);
    temp = load(filename);
    taps_f{3} = temp.df;
    
    filename = sprintf('%s\\tap_13_code_1.mat',foldername);
    temp = load(filename);
    taps_f{4} = temp.df;
    
    filename = sprintf('%s\\tap_15_code_1.mat',foldername);
    temp = load(filename);
    taps_f{5} = temp.df;
    
    taps_f_out = zeros(NFFT,N_taps(2));
    taps_t_out = zeros(N_taps(2),NFFT);
    for idx = 1:N_taps(2)
        temp       = taps_f{idx};
        temp_bsb   = temp.*f_window;
        
        taps_f_out(:,idx) = temp_bsb.';
        taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
    end
else % load predicted external
    %[taps_f_out, taps_t_out]= predict_TTD_freq_response();
    %[taps_f_out, taps_t_out]= predict_TTD_freq_response_x6();
    flag_load_ADC = 0;
    [taps_f, taps_t]= predict_TTD_freq_response_switch(flag_load_ADC);
    
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

    [taps_f_out,taps_t_out] = filter_taps(taps_f,f_window);
end

idx_branches = 2;
for idx_taps = 1:N_taps(2)
    opt_taps_t(idx_branches,idx_taps,:) = taps_t_out(idx_taps,:);
end
