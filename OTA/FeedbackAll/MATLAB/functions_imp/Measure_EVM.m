function pctEVM = Measure_EVM(Data_sent,Data_Received)

Data_EXT = Data_sent;
RX_decoded = Data_Received;
x0 = 1;
step = .5;
evm = comm.EVM('XPercentileEVMOutputPort',true, 'XPercentileValue',90);
pctEVM = evm(Data_EXT,RX_decoded);

xtemp = x0;
for i = 1:20
    xtemp1  = xtemp+step;
    RX_decoded1 = RX_decoded/xtemp1;
    pctEVM1 = evm(Data_EXT,RX_decoded1);
    if pctEVM1 < pctEVM
        pctEVM = pctEVM1;
        xtemp = xtemp1;
        continue;
    end
    xtemp1  = xtemp-step;
    RX_decoded1 = RX_decoded/xtemp1;
    pctEVM1 = evm(Data_EXT,RX_decoded1);
    
    if pctEVM1 < pctEVM
        pctEVM = pctEVM1;
        xtemp = xtemp1;
        continue;
    end
    
    step = step/2;
    
end
RX_decoded = RX_decoded/xtemp;
pctEVM = evm(Data_EXT,RX_decoded);
end

