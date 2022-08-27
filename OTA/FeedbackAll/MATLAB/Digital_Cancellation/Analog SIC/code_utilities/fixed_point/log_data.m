function dout = log_data(din,N_ite,data_len,idx_data)

persistent dout;

if isempty(dout)
dout = zeros(1,N_ite*data_len);
else
dout( 1 + (idx_data-1)*data_len : data_len + (idx_data-1)*data_len) = din;
end