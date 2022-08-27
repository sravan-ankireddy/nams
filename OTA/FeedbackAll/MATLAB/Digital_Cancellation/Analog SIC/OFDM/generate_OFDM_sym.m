function [xt_all,  xf] = generate_OFDM_sym(params)

NFFT      = params.NFFT;
NCP       = params.NCP;
NRP       = params.NRP;
N_data    = params.N_data;
mod_order = params.mod_order;
N_sym     = params.N_sym;
NOFDM     = params.NOFDM;
idx_data  = params.idx_data;
N_rpt     = params.N_rpt;         % repetition numbers

if params.en_pow_loading == 1
    fc        = params.fc;
    f_low     = params.f_low;
    idx_power = params.idx_data_freq + floor((fc-params.f_low)/params.delta_f);
else
    params.power_loading = ones(1,NFFT);
    idx_power = idx_data;
end
%------ test--------
%randn('state',1234);
%rand('state',12345);
%-------------------

w = calculate_window( params );

xt_all = zeros(1,N_sym*NOFDM);    % buffer for all symbols
xt_cp_rp = zeros(1,NCP+NFFT+NRP); % buffer for windowed symbol
xt_rp_pre = zeros(1,NRP);         % buffer for previous RP period
xf = zeros(N_sym,NFFT);
for idx_sym = 1:N_sym    
    bits = randi(mod_order,1,N_data)-1; 
    qam_data = qammod(bits,mod_order,'UnitAveragePower',true);

    if params.en_pow_loading == 1
        xf(idx_sym,idx_data) = qam_data.*params.power_loading(idx_power); % power loading
    else
        xf(idx_sym,idx_data) = qam_data; 
    end
    xt = ifft(xf(idx_sym,:),NFFT);
    
    % add cp and roll-off period  
    xt_cp_rp(1:NCP)                       = xt(NFFT-NCP+1:NFFT);
    xt_cp_rp(NCP+1:NCP+1+NFFT-1)          = xt(1:NFFT);
    xt_cp_rp(NCP+NFFT+1:NCP+NFFT+1+NRP-1) = xt(1:NRP);
    xt_cp_rp                              = xt_cp_rp.*w;
    
    xt_all( 1+(idx_sym-1)*NOFDM : NOFDM + (idx_sym-1)*NOFDM ) = xt_cp_rp(1:NOFDM);
    xt_all( 1+(idx_sym-1)*NOFDM : NRP + (idx_sym-1)*NOFDM   ) = xt_all( 1+(idx_sym-1)*NOFDM : NRP + (idx_sym-1)*NOFDM )  + xt_rp_pre;    
    xt_rp_pre = xt_cp_rp( NCP+NFFT+1:NCP+NFFT+1+NRP-1 ) ;    
end
xt_all = repmat(xt_all,1,N_rpt);
