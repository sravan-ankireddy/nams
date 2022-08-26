% read DAC tap response and echo response to optimize the coefficient 
clear all;
close all;
clc;

set_env();
%-------------------------------------------------
%               set parameters  
%-------------------------------------------------
flag_time_domain_opt = 1;
flag_constrained = 1;
   
w_size = 1024;
N_taps = 512;  
coe_max = 1;

%-------------------------------------------------
%                 load echo 
%-------------------------------------------------
filename = '..\VNA_RS\data\20180425\echo.mat';

temp = load(filename);
echo_t = double(temp.dt);
echo_f = temp.df;

%---------- test adding delay ----------
echo_t = adding_delay(50,echo_t);
echo_f = fft(echo_t);
%--------------------------------------
   
show_data(echo_t,'echo in t'); 

%-------------------------------------------------
%                 load taps
%-------------------------------------------------
foldername = '..\\VNA_RS\\data\\20180502_no_external';

filename = sprintf('%s\\tap_1_code_1.mat',foldername);
temp = load(filename);
taps_f = temp.df;
taps_t = ifft(taps_f);

show_data(taps_t,'tap in t'); 

%--------------------------------------------------
%              detect peak location 
%--------------------------------------------------
% echo
params.peak_th = 0.1;
params.idx_neighbors_th = 5;%30; 
params.small_width_th = 0;% not used now

echo_peaks = detect_coarse_delay(echo_t,params);
echo_peaks = round(echo_peaks)



% tap
params.peak_th = 0.06;
params.idx_neighbors_th = 10;%30; 
params.small_width_th = 0;% not used now

tap_peaks = detect_coarse_delay(taps_t,params);
tap_peaks = round(tap_peaks);

%--------------------------------------------------
%                 windowing 
%--------------------------------------------------
% remove the peaks that are covered by the other peaks
idx_peaks = 1;
while( idx_peaks <= length(echo_peaks))
        idx_end = echo_peaks(idx_peaks) + w_size/2-1;    
    idx_covered = find(echo_peaks(idx_peaks+1:end) < idx_end) ;
    echo_peaks(idx_covered+idx_peaks) = [];
    
    idx_peaks = idx_peaks + 1;
end

% echo
[echo_w_t, echo_peaks_w]= windowing_td(echo_t,w_size,echo_peaks );

% tap
[taps_w_t, tap_peaks_w ]= windowing_td(taps_t,w_size,tap_peaks);
taps_w_t = taps_w_t{1};

%--------------------------------------------------
%                   opimization  
%--------------------------------------------------
first_tap_location = zeros(1,length(echo_peaks));

coe = cell(1,length(echo_peaks));
for idx_peaks = 1:length(echo_peaks)
    % form estimation matrix
    
    first_tap_location(idx_peaks) = echo_peaks(idx_peaks) - tap_peaks - N_taps/2; % absolute delay before windowing
    params.first_delay      = echo_peaks_w(idx_peaks) - tap_peaks_w - N_taps/2 % relative delay after windowing

    if (first_tap_location(idx_peaks) < 0)
       if( echo_peaks(idx_peaks) - w_size/2 > 0 )                                            
           first_tap_location(idx_peaks) = echo_peaks(idx_peaks) - w_size/2 - 1; % absolute delay before windowing                  
       else   
           first_tap_location(idx_peaks) = 0; % absolute delay before windowing       
       end
       
       params.first_delay  = 0;
    end
    
    
    params.N_taps           = N_taps;
    params.NFFT             = w_size;
    params.flag_time_domain = flag_time_domain_opt;
    
    [mtx,taps_t_w] = form_tap_matrix(taps_w_t,params);
    
    if flag_time_domain_opt == 1
        echo_w = echo_w_t{idx_peaks}.';
    else
        echo_w = fft(echo_w_t{idx_peaks}).';       
    end
    
    %show_data_para({echo_w_t{idx_peaks},taps_t_w(1,:)},{'echo','first tap'});
    %show_matrix([taps_t;echo_w_t{idx_peaks}],'taps');   
    %show_matrix([to_pow_dB(taps_t_w);to_pow_dB(echo_w_t{idx_peaks})],'taps');   
    
    % optimization
    if flag_constrained == 0 %unconstrained
        coe{idx_peaks} = mtx\echo_w;
    else % constrained
        lb = [];    % negative means no sign change required
        for idx_bnb = 1:N_taps
            %lb = [lb -Inf];
            lb = [lb -coe_max];
        end
        %ub = zeros(1,N_taps);
        ub = ones(1,N_taps);
        
        coe{idx_peaks} = lsqlin(mtx,echo_w_t{idx_peaks},[],[],[],[],lb,ub); 
    end
    coe{idx_peaks}

    %--------------------------------------------------
    %                 show cancellation
    %--------------------------------------------------

    %------ reconstruct echo ----------
    d_tx_hat = reconstruct_t(taps_t_w,coe{idx_peaks});
    
    %------ cancelation ----------
    e = echo_w_t{idx_peaks} - d_tx_hat;
    fs = 1.6384e9;
    show_cancellation(d_tx_hat,echo_w_t{idx_peaks},e,fs,taps_w_t,coe{idx_peaks});
    
end


%--------------------------------------------------
%            combine filter coefficient  
%--------------------------------------------------
N_coe_all = first_tap_location(end) + length(coe{end}) ;
coe_all = zeros(1,N_coe_all);

for idx_peaks = 1:length(echo_peaks)    
    idx_coe = first_tap_location(idx_peaks) + 1 : first_tap_location(idx_peaks) +  length(coe{idx_peaks})  ;    
    coe_all(idx_coe) = coe{idx_peaks};
end

show_data(coe_all,'all tap coefficient');


%--------------------------------------------------
%            show full echo cancellation
%--------------------------------------------------
d_tx_hat_all = conv(coe_all,taps_t);
d_tx_hat_all = d_tx_hat_all(1:length(echo_t));

e = echo_t - d_tx_hat_all;
fs = 1.6384e9;
show_cancellation(d_tx_hat_all,echo_t,e,fs);







