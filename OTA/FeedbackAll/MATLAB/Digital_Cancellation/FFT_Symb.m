function [TX,TX_f] = FFT_Symb(N,NFFT,upsample)

NFFT_new = NFFT*upsample;
offset   = (NFFT/2)*(upsample - 1);

A = zeros(NFFT_new,N/upsample);
Int = open('../Python/Int.mat');
Int = Int.I;
for i = 1:N/upsample
    for j = [offset + 7:offset+32 offset + 34: offset + 58]
%         A(j,i) = qammod(randi(4) - 1,4);
        A(j,i) = qammod(Int(i,j),4);
    end
end
cp = NFFT_new/4;

for i = 1:N/upsample
    IFFT_Data = ifft(fftshift(A(1:NFFT_new,i)),NFFT_new);
    A(1:NFFT_new+cp,i) = [IFFT_Data(NFFT_new - cp + 1: NFFT_new);IFFT_Data];
end

TX = A(:);

F  = filt_design(40,17.9/upsample,18/upsample);
TX_f = conv(TX,F);
a  =  fix(length(F)/2);
TX_f = TX_f(a+1:a+N*(NFFT+NFFT/4));


end