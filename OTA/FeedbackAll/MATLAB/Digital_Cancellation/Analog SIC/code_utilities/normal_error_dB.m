function error = normal_error_dB(d1,d2)

error = 10*log10(abs((d1-d2)./d2));

