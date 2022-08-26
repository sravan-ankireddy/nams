function [taps_f_out,taps_t_out] = read_VNA2_complex_all_taps(f_window,flag_int)


if flag_int == 1 % internal taps  
    N_taps = 7;
    taps_f = cell(1,N_taps);    

    filename = 'data\VNA2\vna2_ch1_0volt.mat';
    echos_f = load(filename);
    taps_f{1} = echos_f.data.';
    
    filename = 'data\VNA2\vna2_ch2_0volt.mat';
    echos_f = load(filename);
    taps_f{2} = echos_f.data.';
    
    filename = 'data\VNA2\vna2_ch3_0volt.mat';
    echos_f = load(filename);
    taps_f{3} = echos_f.data.';

    filename = 'data\VNA2\vna2_ch4_0volt.mat';
    echos_f = load(filename);
    taps_f{4} = echos_f.data.';

    filename = 'data\VNA2\vna2_ch5_0volt.mat';
    echos_f = load(filename);
    taps_f{5} = echos_f.data.';

    filename = 'data\VNA2\vna2_ch6_0volt.mat';
    echos_f = load(filename);
    taps_f{6} = echos_f.data.';
    
    filename = 'data\VNA2\vna2_ch7_0volt.mat';
    echos_f = load(filename);
    taps_f{7} = echos_f.data.';
    
    filename = 'data\VNA2\vna2_ch8_0volt.mat';
    echos_f = load(filename);
    taps_f{8} = echos_f.data.';
    
else % external taps    
%     N_taps = 5;
%     taps_f = cell(1,N_taps);
%     
%     filename_real = 'data\VNA\20180214\external_no_gate\tap2_real_imag.dat';
%     [taps_f{1}, taps_t{1}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180214\external_no_gate\tap3_real_imag.dat';
%     [taps_f{2}, taps_t{2}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180214\external_no_gate\tap4_real_imag.dat';
%     [taps_f{3}, taps_t{3}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180214\external_no_gate\tap5_real_imag.dat';
%     [taps_f{4}, taps_t{4}, x] = read_VNA_compex_v2(filename_real);
%     
%     filename_real = 'data\VNA\20180214\external_no_gate\tap7_real_imag.dat';
%     [taps_f{5}, taps_t{5}, x] = read_VNA_compex_v2(filename_real);
    
end


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
