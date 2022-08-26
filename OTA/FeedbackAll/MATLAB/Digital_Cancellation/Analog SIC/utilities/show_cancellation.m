function cancel_amount = show_cancellation(d_tx_hat,d_rx_t,e,fs)

mse    = 10*log10(mean(abs(e).^2));
max_err = 10*log10(max(abs(e).^2));
    
ts = 1/fs;
% time domain
if 1
    figure;
    %t_new = (1:length(d_tx_hat))*ts;
    t_new = (1:length(d_tx_hat));
    subplot(2,1,1)
    plot(t_new,20*log10(abs(d_tx_hat)),t_new,20*log10(abs((d_rx_t))),t_new,20*log10(abs(e)),'g');
    %legend('reconstructed','self-interference',['residual,MSE = ' num2str(mse),',max SE = ' num2str(max_err)]);
    legend('reconstructed','self-interference','residual');    
    xlabel('time');
    ylabel('dB');
    %title([ tap_name ' Number of taps = ' num2str(N_taps) ]);
    grid on;
    
    subplot(2,1,2)
    plot(t_new,20*log10(abs((d_rx_t))) - 20*log10(abs((e))));
    xlabel('time');
    ylabel('dB');
    %title(['cancel amount ' tap_name ' Number of taps = ' num2str(N_taps) ]);
    title('cancel amount')
    grid on;
end

% time domain in linear 
if 0
    figure;
    %t_new = (1:length(d_tx_hat))*ts;
    t_new = (1:length(d_tx_hat));
    %subplot(2,1,1)
    plot(t_new,d_tx_hat,t_new,d_rx_t,t_new,e,'g');    
    %legend('reconstructed','self-interference',['residual,MSE = ' num2str(mse),',max SE = ' num2str(max_err)]);
    legend('reconstructed','self-interference');    
    xlabel('time');
    ylabel('dB');
    %title([ tap_name ' Number of taps = ' num2str(N_taps) ]);
    grid on;
    
    tapst_cell = cell(1,size(taps_t,1));   
    for idx = 1:size(taps_t,1)
        tapst_cell{idx} = coe(idx)*(taps_t(idx,:));
    end
    show_data_para(tapst_cell,{'1','2','3','5','6','7','8','9'});    
    
end

% frequency domain , real
if 0
    NFFT = length(d_tx_hat);
    d_tx_hat_f  = fft(d_tx_hat,NFFT);
    d_rx_f_temp = fft(d_rx_t,NFFT);
    e_f         = fft(e,NFFT);
    
    figure;
    f = (1:NFFT/2)*fs/NFFT*1e-6;
    plot(f,to_pow_dB(d_rx_f_temp(1:NFFT/2)),'.-',f,to_pow_dB(d_tx_hat_f(1:NFFT/2)),f,to_pow_dB(e_f(1:NFFT/2)),'.-g');
    xlabel('frequency(MHz)');
    ylabel('dB');
    legend('echo','reconstruction','residual');    
end


% frequency domain , complex
if 1
    NFFT = length(d_tx_hat);
    d_tx_hat_f  = fftshift(fft(d_tx_hat,NFFT));
    d_rx_f_temp = fftshift(fft(d_rx_t,NFFT));
    e_f         = fftshift(fft(e,NFFT));
    
    figure;
    f = (-NFFT/2:NFFT/2-1)*fs/NFFT*1e-6;
    plot(f,to_pow_dB(d_rx_f_temp),'.-',f,to_pow_dB(d_tx_hat_f),f,to_pow_dB(e_f),'.-g');
    xlabel('frequency(MHz)');
    ylabel('dB');
    legend('echo','reconstruction','residual');
    
    figure;
    f = (-NFFT/2:NFFT/2-1)*fs/NFFT*1e-6;
    plot(f,20*log10(abs((d_rx_f_temp)))-20*log10(abs((e_f))));
    xlabel('frequency(MHz)');
    ylabel('dB');
    legend('cancellation amount');    
    
    cancel_amount = to_pow_dB(d_rx_f_temp)-to_pow_dB(e_f);
end