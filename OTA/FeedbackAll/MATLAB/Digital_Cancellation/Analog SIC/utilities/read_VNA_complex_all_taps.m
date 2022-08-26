function [taps_f_out,taps_t_out,f] = read_VNA_complex_all_taps(f_window,flag_int)


if flag_int == 1 % internal taps  
    N_taps = 7;
    taps_f = cell(1,N_taps);    

    filename_real = 'data\VNA\20180214\internal\tap2_new_real_imag.dat';
    [taps_f{1}, taps_t{1}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap3_new_real_imag.dat';
    [taps_f{2}, taps_t{2}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap4_new_real_imag.dat';
    [taps_f{3}, taps_t{3}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap5_new_real_imag.dat';
    [taps_f{4}, taps_t{4}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap6_new_real_imag.dat';
    [taps_f{5}, taps_t{5}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap7_new_real_imag.dat';
    [taps_f{6}, taps_t{6}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\internal\tap8_new_real_imag.dat';
    [taps_f{7}, taps_t{7}, x] = read_VNA_compex_v2(filename_real);


else % external taps    
    N_taps = 5;
    taps_f = cell(1,N_taps);
    
%      filename_real = 'data\VNA\20180213\tap1_v6.dat';
%      [taps_f{1}, taps_t{1}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap2_v6.dat';
%     [taps_f{2}, taps_t{2}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap3_v6.dat';
%     [taps_f{3}, taps_t{3}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap4_v6.dat';
%     [taps_f{4}, taps_t{4}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap5_v6.dat';
%     [taps_f{5}, taps_t{5}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap6_v6.dat';
%     [taps_f{6}, taps_t{6}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180213\tap7_v6.dat';
%     [taps_f{7}, taps_t{7}, x] = read_VNA_compex_v2(filename_real);


    filename_real = 'data\VNA\20180214\external_no_gate\tap2_real_imag.dat';
    [taps_f{1}, taps_t{1}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\external_no_gate\tap3_real_imag.dat';
    [taps_f{2}, taps_t{2}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\external_no_gate\tap4_real_imag.dat';
    [taps_f{3}, taps_t{3}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\external_no_gate\tap5_real_imag.dat';
    [taps_f{4}, taps_t{4}, x] = read_VNA_compex_v2(filename_real);
    
    filename_real = 'data\VNA\20180214\external_no_gate\tap7_real_imag.dat';
    [taps_f{5}, taps_t{5}, x] = read_VNA_compex_v2(filename_real);
    
end

f = x;

NFFT = 16384*2;
taps_f_out = zeros(NFFT,N_taps);
taps_t_out = zeros(N_taps,NFFT);
for idx = 1:N_taps
    temp       = taps_f{idx}.';
    temp_pass  = [temp(1) temp(1) temp(1:end-1)];      % 0,50k,100k
    temp_bsb   = [temp_pass 0 fliplr(conj(temp_pass(2:end)))]; % convert to baseband
    temp_bsb   = temp_bsb.*f_window;
    
    taps_f_out(:,idx) = temp_bsb.';
    taps_t_out(idx,:) = ifft(temp_bsb,NFFT);
end
