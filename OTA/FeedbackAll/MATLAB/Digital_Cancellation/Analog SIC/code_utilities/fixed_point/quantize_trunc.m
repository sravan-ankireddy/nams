% quantization by trunction 
% signed , saturation 

function y = quantize_trunc(x,QI,QF)

if isreal(x)
  y = quantize_trunc_sub(x,QI,QF);
else
  y = quantize_trunc_sub(real(x),QI,QF) + 1i*quantize_trunc_sub(imag(x),QI,QF);
end

function y = quantize_trunc_sub(x,QI,QF)

max_x = 2^(QI-1)-2^-QF;
min_x = -2^(QI-1);

y = floor(x/2^-QF)*2^-QF;

y( x > max_x ) = max_x; 
y( x < min_x ) = min_x;

