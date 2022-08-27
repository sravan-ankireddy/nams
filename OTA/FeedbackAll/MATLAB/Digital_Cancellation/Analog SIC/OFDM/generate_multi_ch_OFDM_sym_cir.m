% generate multi-channel OFDM symbol in different RF band, circular convoluted version
% dt_bsb: raw transmit signal before multiplying symbol coefficient

function [rf_all,bsb_all,dt_bsb] = generate_multi_ch_OFDM_sym_cir(OFDM_params,params)

filter_coe  = params.filter_coe;
sym_coe     = params.sym_coe;
N_sym_total = params.N_sym_total;
fs          = params.fs;
OSR         = params.OSR;
N_channel   = params.N_channel;
fc_1638M    = params.fc_1638M;
fc_819M     = params.fc_819M;
intp_filter = params.intp_filter;
NOFDM       = OFDM_params.NOFDM;

dt_bsb = cell(1,N_channel);

if isfield(params, 'rand_seed')
    if ~isempty(params.rand_seed)
        rand('state',params.rand_seed);
    end
end

rf_all = 0; 
bsb_all = cell(1,N_channel);
for idx_ch = 1:N_channel
    OFDM_params.fc = fc_1638M(idx_ch); % actual carrier frequency for power loading
    fc             = fc_819M(idx_ch);  % complex baseband carrier frequency. Because of up conversion in DAC
    
    dt_bsb{idx_ch} = generate_OFDM_sym(OFDM_params); 
    
    % symbol coefficient
    temp = dt_bsb{idx_ch};
    dt_bsb_sym = zeros(1,length(dt_bsb{idx_ch}));
    for idx_sym = 1:N_sym_total
        dt_bsb_sym(1+(idx_sym-1)*NOFDM: NOFDM+(idx_sym-1)*NOFDM ) = temp(1+(idx_sym-1)*NOFDM: NOFDM+(idx_sym-1)*NOFDM )*sym_coe(idx_sym);
    end
    
    % circular convolution of baseband filter
    [dt_filter,df_filter] = conv_cir(dt_bsb_sym,filter_coe{idx_ch});   
    bsb_all{idx_ch} = dt_filter;
    
    % convert baseband to RF
    [rf ,rf_cpx]= bsb_to_rf_cir(dt_filter,intp_filter,OSR,fc,fs);
    rf_all = rf_all + rf_cpx; % should be "complex" RF signal, since the last stage interpolation is done in the DAC chip
end
