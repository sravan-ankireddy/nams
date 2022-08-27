function dt_bsb = load_signal_from_file(filename)

N_channel = length(filename);
dt_bsb = cell(1,N_channel);

for idx_ch = 1:N_channel
    params.filename = filename{idx_ch};
    params.convert_to_frac = 1;
    params.QF = 15;
    params.iscomplex = 1;
    dt_bsb{idx_ch} = read_from_h_file(params);
end

