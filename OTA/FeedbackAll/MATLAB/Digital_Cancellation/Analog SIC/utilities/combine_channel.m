function hf_combine = combine_channel(hf_bsb,f_start,delta_f)

NFFT = 16384;   % half-band fft

hf_half = zeros(1,NFFT);
for idx_band = 1:length(f_start)
  idx          = floor(f_start(idx_band)/delta_f);
  hf_half(idx : idx + length(hf_bsb{idx_band})-1) = fftshift(hf_bsb{idx_band});
end

hf_half_mirror = fliplr(conj(hf_half(2:end)));
hf_combine  = [hf_half 0 hf_half_mirror];

   



