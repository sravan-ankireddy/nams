function dout = normalize(din)

maxr = max(real(din));
maxi = max(imag(din));
A = max(maxr,maxi);
dout = din/A;
