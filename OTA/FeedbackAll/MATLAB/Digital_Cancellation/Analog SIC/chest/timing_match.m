function [first_sym_offset, flag_not_found] = timing_match(din,dref,th,N_match)

%---- find out the first symbol ---------
A = max(max(real(din)),max(imag(din)));
din = din/A;

A = max(max(real(dref)),max(imag(dref)));
dref = dref/A;

mth = conv(fliplr(conj(dref(1:N_match))),din);
idx = find(abs(mth).^2 > th);

if isempty(idx)
    flag_not_found = 1;
    first_sym_offset = 0; 
    %figure; plot(abs(mth).^2); legend('correlation')

else
    flag_not_found = 0;    
    idx = idx(1);
    first_sym_offset = idx - N_match + 1 ;
end
figure; plot(abs(mth).^2); legend('correlation')
%------------------------------------------