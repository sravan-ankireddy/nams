% find peaks and windowing 

function [ht_bsb_w,hf_bsb_w,desired_peaks,desired_peaks_w] = find_peak_and_window(ht_bsb,w_size)

% echo coarse delay 
params.peak_th = 0.1;
params.idx_neighbors_th = 80;%30; 
desired_peaks = detect_coarse_delay(ht_bsb,params);

desired_peaks_w = desired_peaks;

N_peaks = length(desired_peaks);

ht_bsb_w = cell(1,N_peaks);
hf_bsb_w = cell(1,N_peaks);
for idx_branches = 1:N_peaks
    w_start = desired_peaks(idx_branches) - w_size/2;
    if w_start < 1
        w_start = 1;
    end
    
    ht_bsb_w{idx_branches} = ht_bsb(w_start:w_start + w_size-1 );
    hf_bsb_w{idx_branches} = fft(ht_bsb_w{idx_branches});
    desired_peaks_w(idx_branches) = desired_peaks_w(idx_branches) - w_start + 1;
end