function [H_f H_t] = estimate_channel_LS(params,yt_all,xf_all)

NFFT     = params.NFFT;
NCP      = params.NCP;
idx_data = params.idx_data;
N_ave    = params.N_ave; % number of symbols to average

NOFDM = NFFT + NCP;

H_f = zeros(1,NFFT);
for idx_sym = 1:N_ave
    yt_symbol = yt_all(NOFDM*(idx_sym-1)+NCP+1:NOFDM*(idx_sym-1)+NCP+NFFT-1+1);
    yf = fft(yt_symbol,NFFT);
    xf = xf_all(idx_sym,:);
    H_f(idx_data) = H_f(idx_data) + yf(idx_data)./xf(idx_data);
    
%     show_data(to_pow_dB(yf),'y')
%     show_data(to_pow_dB(xf),'x')
%     keyboard;
end

H_f = H_f/N_ave;
H_t = ifft(H_f,NFFT);

