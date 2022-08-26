
function update_3ch_freq_time(h,dt) 

N_channel = 3;
NFFT =  4096;

d = cell(1,9);
for idx = 1:N_channel
    d{idx} = 20*log10(abs(fftshift(fft(dt{idx},NFFT))));
end

d{4} = real(dt{1});
d{5} = real(dt{2});
d{6} = real(dt{3});

d{7} = imag(dt{1});
d{8} = imag(dt{2});
d{9} = imag(dt{3});

update_plot(h,d);