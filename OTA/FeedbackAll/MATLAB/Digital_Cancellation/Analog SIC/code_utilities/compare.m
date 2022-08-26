function compare(din1,din2,str1,str2,str3)

if nargin < 3
    str1 = 'data1';
    str2 = 'data2';
    str3 = 'compare';    
elseif nargin == 4
    str3 = '';    
end

din1 = din1(:);
din2 = din2(:);

if isreal(din1) && isreal(din2)
    len = min(length(din1),length(din2));
    x = 1:len;
    figure;
    subplot(2,1,1); plot(x,din1(1:len),'.-',x,din2(1:len),'-o');
    legend([str1],[str2]);
    title(str3)
    
    subplot(2,1,2); plot(x,din1(1:len)-din2(1:len),'.-');
    legend('difference');
    title(str3)
else
    len = min(length(din1),length(din2));
    x = 1:len;
    figure;
    subplot(2,1,1); plot(x,real(din1(1:len)),'-*',x,real(din2(1:len)),'-s');
    legend(['real, ' str1],['real ,' str2]);
    title(str3)
    
    subplot(2,1,2); plot(x,real(din1(1:len))-real(din2(1:len)),'.-');
    legend('real difference');
    title(str3)
    
    figure;
    subplot(2,1,1); plot(x,imag(din1(1:len)),'-*',x,imag(din2(1:len)),'-s');
    legend(['imag, ' str1],['imag ,' str2]);
    title(str3)
    
    subplot(2,1,2); plot(x,imag(din1(1:len))-imag(din2(1:len)),'.-');
    legend('imag difference');
    title(str3)    
end
