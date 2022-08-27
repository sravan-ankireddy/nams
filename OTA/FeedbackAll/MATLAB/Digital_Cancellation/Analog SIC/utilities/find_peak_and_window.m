% find the peak location and do windowing around the peak 

function dout = find_peak_and_window(din,w_size)

dout = zeros(1,length(din));

[peak, peak_idx] = max(abs(din));

idx_start = peak_idx - w_size/2;
if(idx_start < 1)
  idx_start = 1;
end
idx_end   = idx_start + w_size - 1;

dout(idx_start:idx_end) = din(idx_start:idx_end);



