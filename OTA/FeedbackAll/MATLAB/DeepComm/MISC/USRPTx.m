% USRP Transmitter Object
tx = comm.SDRuTransmitter(tx_address);
tx.InterpolationFactor = 8;
tx_address = '192.168.10.2';
rx_address = '192.168.20.2';
% Receiver Object
rx = comm.SDRuReceiver(rx_address, 'DecimationFactor',8);
info(tx)
info(rx)
% runtime = tic;
% while toc(runtime) < 20
%     tx(TX);
%     RX = rx();
% end
% 
% release(tx)
% release(rx)
% 
    VV = zeros(500000,1);
    for i = 1:500000
        VV(i) = (sum(RX(i:i+window_size - 1).*conj(RX(i+16:i+16+window_size - 1))));
        VV(i) = VV(i)/(sum(RX(i:i+window_size - 1).*conj(RX(i:i+window_size - 1))));
    end
    
    
    plot(abs(VV))
    