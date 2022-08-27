function [taps_t_out, taps_f_out] = time_gating(taps_t,N_gate)

N_taps = size(taps_t,1);
  NFFT = size(taps_t,2);

taps_f_out = zeros(NFFT,N_taps);
taps_t_out = taps_t;
for idx = 1:N_taps
    taps_t_out(idx,N_gate:end) = 0;
    taps_f_out(:,idx) = fft(taps_t_out(idx,:),NFFT).';
end    