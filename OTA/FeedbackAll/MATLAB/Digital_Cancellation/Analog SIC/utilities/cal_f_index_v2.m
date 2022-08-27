function [f_window_bsb, f_window_pass,f_idx_bsb,f_idx_pass] = cal_f_index_v2(f_start,BW,delta_f,NFFT)

f_stop = zeros(1,length(f_start));
for idx = 1:length(BW)
 f_stop(idx)  = f_start(idx) + BW(idx);
end

idx = [];
for idx_band = 1:length(f_start)
  idx      = [idx floor(f_start(idx_band)/delta_f):floor(f_stop(idx_band)/delta_f)];
end


f_window = zeros(1,NFFT);
f_window(idx) = ones(1,length(idx));
f_window_mirror = fliplr(f_window(2:end));
f_window_bsb  = [f_window 0 f_window_mirror];
% symmetry as in 0, 1 2 3, 4 ,-3 -2 -1

%scale = NFFT/sum(f_window_bsb); %original
scale = 1;% keep 1 so that it will not change the scale by filtering
f_window_bsb = f_window_bsb*scale; % make sure the total power is still the same 

f_window_pass = fftshift(f_window);


f_idx_bsb = find(f_window_bsb~=0);
f_idx_pass = find(f_window_pass~=0);
