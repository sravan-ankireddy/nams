function [hf, ht, flag_recapture] = chest_wrapper(dt_bsb,dr,OFDM_params,sys_params,ch_num,isbsb,fix_timing,idx_start)

% ------- decode known data ----
params.NFFT = OFDM_params.NFFT;
params.NCP  = OFDM_params.NCP;
df = get_freq_known_data_bsb(params,dt_bsb);

%------- down convert ------------
if isbsb
    dr_bsb = dr;    
else
    dr_bsb = rf_to_bsb(dr,sys_params.intp_filter,sys_params.DSR,sys_params.fc(ch_num),sys_params.fs);
end

%show_data(to_pow_dB(fftshift(fft(dr_bsb,4096))));

%--------- find starting point ---------
if fix_timing == 0 % do not fix the timing, otherwise, use pre-determined start timing 
     [timing_offset,flag_not_found]= timing_match(dr_bsb,dt_bsb,OFDM_params.th,OFDM_params.N_match);
    %[timing_offset,flag_not_found]= timing_match_v2(dr_bsb,dt_bsb,OFDM_params.th,OFDM_params.N_match);
    
    idx_start = timing_offset - 55 
    if (idx_start < 1) || (length(dr_bsb(idx_start:end)) < (OFDM_params.NFFT  + OFDM_params.NCP)) || (flag_not_found == 1)
        idx_start = 1;
        disp('warning, start index < 1');
        flag_recapture = 1;
        hf = [];
        ht = [];
    else
        flag_recapture = 0;
    end
else
        flag_recapture = 0;
end

  
dr_bsb = dr_bsb(idx_start:end); % advance the real peak to compensate for delay caused by half-band filter 

% -------- FD LS ---------
params.NFFT = OFDM_params.NFFT;
params.NCP  = OFDM_params.NCP;
params.idx_data = OFDM_params.idx_data;
params.N_ave = 1;

[hf, ht] = estimate_channel_LS(params,dr_bsb,df);

%keyboard;

end

%------ show data -------
% if 0
%     %     NFFT = 4096;
%     %     fs   = 204.8e6;
%     %     show_freq_time(dr_bsb,fs,NFFT);
%     
%     Fs = 204.8e6;
%     t = (1:length(ht))/Fs*1e9;
%     figure; plot(t,to_pow_dB(ht));
%     legend('channel in time')
%     
%     %
%     %     Fs = 204.8e6;
%     %     t = (1:length(ht))/Fs*1e9;
%     %     figure; plot(t,abs(hf));  legend('amp');
%     %     figure; plot(t,phase(hf)); legend('phase');
% end