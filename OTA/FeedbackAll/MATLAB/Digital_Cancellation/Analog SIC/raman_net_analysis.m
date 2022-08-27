% Reset workspace
close all;
clear all;
clc;

% Scrach area will be deleted!
% x = 0:0.1:10;
% y1 = sin(2*x);
% y2 = cos(2*x);
% tef = figure;
% yyaxis left          % plot against left y-axis  
% f = plot(x,y1)           
% yyaxis right         % plot against right y-axis
% c = plot(x,y2)
% title('Subplot 4')

% Constant header and commands
const.header          = uint32(hex2dec('1020cdab'));
const.header_size = 16; % bytes;

const.cmd_Test_Ramp = uint32(hex2dec('00000009'));
const.cmd_SIC_DMA_Transfer = uint32(hex2dec('00000010'));
const.cmd_Set_Data_Mode    = uint32(hex2dec('00000011'));

const.type_request    = uint32(hex2dec('00000010'));
const.type_reply      = uint32(hex2dec('00000020'));

% Create TCP connection to the board
IP = '192.168.0.157';
PORT = 7;
tcp_obj = tcpclient(IP, PORT, 'Timeout', 600);

% Establish FFT parameters
NFFT = 4096;
N = 10;         % Number of symbols
Fs = 20e6;      % Sample rate
f = (-NFFT/2:NFFT/2-1)*Fs/NFFT;

% Offset
offset_data = 1024;

% Setup the single graph for three sic dmas 
sic_dma_fig = figure();
hold;
title('SIC IO Graph');
sic_tx_in_y = rand(length(f), 1);
sic_tx_in_plt = plot(f, 20*log10(abs(sic_tx_in_y)), '-b');
sic_tx_in_plt.YDataSource = 'sic_tx_in_y';

sic_rx_in_y = rand(length(f), 1);
sic_rx_in_plt = plot(f, 20*log10(abs(sic_rx_in_y)), ':r');
sic_rx_in_plt.YDataSource = 'sic_rx_in_y';

sic_sic_out_y = rand(length(f), 1);
sic_sic_out_plt = plot(f, 20*log10(abs(sic_sic_out_y)), '--c');
sic_sic_out_plt.YDataSource = 'sic_sic_out_y';

legend('TX IN', 'RX IN', 'SIC OUT');
hold;

cmd_sic_dma.SIC_DMA_SICOUT = 0;
cmd_sic_dma.SIC_DMA_TXIN = 1;
cmd_sic_dma.SIC_DMA_RXIN = 2;
n_samples = 16384%8192;

sic_dma_average = 1;
sic_sic_out_sum = 0;
sic_tx_in_sum = 0;
sic_rx_in_sum = 0;

i_itr = 0;
while 1
    % SIC DMA SIC OUT
    data = [uint32(n_samples), uint32(cmd_sic_dma.SIC_DMA_SICOUT)];
    len  = uint32(4*length(data))
    packet  = [const.header, ...
                const.cmd_SIC_DMA_Transfer, ...
                const.type_request, ...
                len, ...
                data];
    write(tcp_obj, packet);
    rcv_data = read(tcp_obj, (n_samples +(const.header_size/4)), 'int32');
    dt = rcv_data((const.header_size/4)+1:end);
    sic_out_16_2 = typecast(dt, 'int16');

    % i and q
    sic_out_i = double(sic_out_16_2(2:2:end));
    sic_out_q = double(sic_out_16_2(1:2:end));
    sic_out_data = sic_out_i + i*sic_out_q;

    sic_sic_out_sum =  20*log10(abs(fftshift(fft(sic_out_data(offset_data+1:NFFT+offset_data))))) + sic_sic_out_sum; 
    
    
    % SIC DMA TX IN
    data = [uint32(n_samples), uint32(cmd_sic_dma.SIC_DMA_TXIN)];
    len  = uint32(4*length(data))
    packet  = [const.header, ...
                const.cmd_SIC_DMA_Transfer, ...
                const.type_request, ...
                len, ...
                data];
    write(tcp_obj, packet);
    rcv_data = read(tcp_obj, (n_samples +(const.header_size/4)), 'int32');
    dt = rcv_data((const.header_size/4)+1:end);
    tx_in_16_2 = typecast(dt, 'int16');

    % i and q
    tx_in_i = double(tx_in_16_2(2:2:end));
    tx_in_q = double(tx_in_16_2(1:2:end));
    tx_in_data = (tx_in_i + i*tx_in_q);% / 2^15;

    sic_tx_in_sum =  20*log10(abs(fftshift(fft(tx_in_data(offset_data+1:NFFT+offset_data))))) + sic_tx_in_sum;
    
    % SIC DMA RX IN
    data = [uint32(n_samples), uint32(cmd_sic_dma.SIC_DMA_RXIN)];
    len  = uint32(4*length(data))
    packet  = [const.header, ...
                const.cmd_SIC_DMA_Transfer, ...
                const.type_request, ...
                len, ...
                data];
    write(tcp_obj, packet);
    rcv_data = read(tcp_obj, (n_samples +(const.header_size/4)), 'int32');
    dt = rcv_data((const.header_size/4)+1:end);
    rx_in_16_2 = typecast(dt, 'int16');

    % i and q
    rx_in_i = double(rx_in_16_2(2:2:end));
    rx_in_q = double(rx_in_16_2(1:2:end));
    rx_in_data = complex(rx_in_i, rx_in_q);%rx_in_i + i*rx_in_q;

    sic_rx_in_sum =  20*log10(abs(fftshift(fft(rx_in_data(offset_data+1:NFFT+offset_data))))) + sic_rx_in_sum;
    
    if(i_itr == sic_dma_average)
        sic_sic_out_y = conv(sic_sic_out_sum ./ sic_dma_average, ones(1,10));
        sic_sic_out_y = sic_sic_out_y(1:end-9);
        sic_tx_in_y = conv(sic_tx_in_sum ./ sic_dma_average,  ones(1,10));
        sic_tx_in_y = sic_tx_in_y(1:end-9);
        sic_rx_in_y = conv(sic_rx_in_sum ./ sic_dma_average,  ones(1,10));
        sic_rx_in_y = sic_rx_in_y(1:end-9);
        
        i_itr = 1;
        sic_sic_out_sum = 0;
        sic_tx_in_sum = 0;
        sic_rx_in_sum = 0;
        
        refreshdata(sic_dma_fig);
    else
        i_itr = i_itr + 1;
    end
    
end
