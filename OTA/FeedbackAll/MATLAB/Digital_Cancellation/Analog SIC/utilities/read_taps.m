function [taps_f, taps_t, x] = read_taps(filename_real,filename_imag)

h_real = load(filename_real);
h_imag = load(filename_imag);


x = h_real(:,1);

N_taps = size(h_real,2)-2;
taps_f = cell(1,N_taps);
taps_t = cell(1,N_taps);
offset = 2;
for idx = 1:N_taps
    temp = h_real(:,idx+offset) + 1i*h_imag(:,idx+offset);
    taps_f{idx} = temp;
    taps_t{idx} = ifft(temp);    
end

% the lowest frequency is actually not from 0, so do ifft to transform back to time domain is not 100% correct   

