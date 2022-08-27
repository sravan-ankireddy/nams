function Data_out = Data_capture()

% Constant header and commands
const.header          = uint32(hex2dec('1020cdab'));
const.header_size = 16; % bytes;

const.cmd_Test_Ramp = uint32(hex2dec('00000009'));
const.cmd_SIC_DMA_Transfer = uint32(hex2dec('00000010'));
const.cmd_Set_Data_Mode    = uint32(hex2dec('00000011'));

const.type_request    = uint32(hex2dec('00000010'));
const.type_reply      = uint32(hex2dec('00000020'));


% Create TCP connection to the board
IP = '192.168.1.10';
PORT = 7;
tcp_obj = tcpclient(IP, PORT, 'Timeout', 600);

n_samples = 16384;

% Data packet ready
data     = [uint32(n_samples), uint32(cmd_sic_dma.SIC_DMA_SICOUT)];
len      = uint32(4*length(data));
packet   = [const.header, ...
    const.cmd_SIC_DMA_Transfer, ...
    const.type_request, ...
    len, ...
    data];

write(tcp_obj, packet);

% Capture data
rcv_data   = read(tcp_obj, (n_samples +(const.header_size/4)), 'int32');
dt         = rcv_data((const.header_size/4)+1:end);
Data_out   = typecast(dt, 'int16');

Data_out_i = double(Data_out(2:2:end));
Data_out_q = double(Data_out(1:2:end));
Data_out   = Data_out_i + 1j*Data_out_q;
Data_out   = Data_out/32768;
end
