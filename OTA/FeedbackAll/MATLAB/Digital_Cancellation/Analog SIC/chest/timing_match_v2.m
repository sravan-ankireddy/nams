% find the maximum peak instead of just passing the threshold

function [first_sym_offset, flag_not_found] = timing_match_v2(din,dref,th,N_match)

%---- find out the first symbol ---------
A = max(max(real(din)),max(imag(din)));
din = din/A;

A = max(max(real(dref)),max(imag(dref)));
dref = dref/A;

mth = conv(fliplr(conj(dref(1:N_match))),din);
%idx = find(abs(mth).^2 > th);

[max_value,max_idx] = max(abs(mth).^2);

if ( max_value < th )
    flag_not_found = 1;
    first_sym_offset = 0; 
    %figure; plot(abs(mth).^2); legend('correlation')

else
    flag_not_found = 0;    
    first_sym_offset = max_idx - N_match + 1 ;
end

figure; plot(abs(mth).^2); legend('correlation')
%keyboard;
%------------------------------------------