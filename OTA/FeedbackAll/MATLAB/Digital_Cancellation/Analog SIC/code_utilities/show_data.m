function show_data(din,str)
din = din(:);

if nargin < 2
  str = '';  
end

figure;
if(isreal(din))
    plot(din); legend(str)    
else
    subplot(2,1,1); plot(real(din),'.-');  legend(['real,' str]);
    subplot(2,1,2); plot(imag(din),'.-'); legend(['imag,' str]);    
end