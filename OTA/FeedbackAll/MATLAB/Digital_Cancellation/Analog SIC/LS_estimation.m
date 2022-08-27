% @read echo and optical data from PicoZED and estimate coefficient
% @simplified version to reduce the memeory and computational complexity requirement
% @further simplified version for pcb

function code = LS_estimation(taps,Echo)

N_taps = 7;

%---------- load power ratio  -----------%
[pow_max,pow_min] = measure_pow_max_min();


%---------- load code table  ------------%
[code_table,code_to_coe_table] = load_code_and_coe_table(1);

%--------- estimate coefficients --------%
lb = -2*ones(N_taps,1);
ub = 2*ones(N_taps,1);

coe_lin = lsqlin(taps,Echo,[],[],[],[],lb,ub);
coe     = coe_to_cascade_coe(coe_lin,pow_min,pow_max);

% Convert coefficients to DAC values
[code,~] = find_coe_to_code_v2(real(abs(coe)),code_table,code_to_coe_table);

%-------- Cancellation Results-----------%
d_tx_hat = zeros(1,N_taps);
for idx = 1:N_taps
    d_tx_hat = d_tx_hat + taps(idx,:)*coe_lin(idx);
end
e = Echo - d_tx_hat;

figure;
t_new = (1:length(d_tx_hat));
subplot(2,1,1)
plot(t_new,20*log10(abs(d_tx_hat)),t_new,20*log10(abs((d_rx_t))),t_new,20*log10(abs(e)),'g');
legend('reconstructed','self-interference','residual');
xlabel('time');
ylabel('dB');
grid on;

end

