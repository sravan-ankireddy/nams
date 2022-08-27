function observe_bit_rage(data,str)

if isreal(data)    
    [y,edges] = histcounts(log2(abs(real(data))), 'Normalization', 'probability');
    
    figure;  plot(edges(1:end-1),y,'-o');
    xlabel('log2')
    legend(str)    
else    
    [y,edges] = histcounts(log2(abs(real(data))), 'Normalization', 'probability');
    
    figure;  
    subplot(2,1,1); plot(edges(1:end-1),y,'-o');
    xlabel('log2')
    legend(sprintf('real %s',str))    

    [y,edges] = histcounts(log2(abs(imag(data))), 'Normalization', 'probability');

    subplot(2,1,2); plot(edges(1:end-1),y,'-o');
    legend(sprintf('imag %s',str))    
    xlabel('log2')
end


