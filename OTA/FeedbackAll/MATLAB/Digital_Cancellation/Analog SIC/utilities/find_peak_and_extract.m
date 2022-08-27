% find the peak location and extract samples around the peak 
% return the original starting point of the extract signal 

function [dout,idx_start] = find_peak_and_extract(din,w_size)


[peak, peak_idx] = max(abs(din));

idx_start = peak_idx - w_size/2;
if(idx_start < 1)
  idx_start = 1;
end
idx_end   = idx_start + w_size - 1;

dout = din(idx_start:idx_end);