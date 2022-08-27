function Y = Convert_to_real(X)

Y = fft(X);
Y = [0;Y;conj(flipud(Y))];
Y = ifft(Y);


end

