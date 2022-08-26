function y = q_format(x,QI,QF) 
%  y = quantize_simple(x,QI,QF,1,0); % flag_signed=1 ,flag_floor=0
  y = quantize_trunc(x,QI,QF);