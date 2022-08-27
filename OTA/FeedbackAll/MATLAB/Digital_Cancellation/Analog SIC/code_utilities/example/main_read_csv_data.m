clear all;
close all;
clc

% 
% python_file = '..\read_csv_data.py';
% rd_filename = 'fd_wifi.csv';
% col_name1 = 'i_system_wrapper/wifi_dsn_i/fifo_generator_idata_rx_dout[15:0]';
% col_name2 = 'i_system_wrapper/wifi_dsn_i/fifo_generator_qdata_rx_dout[15:0]';
% 
% data_out = read_csv_data(python_file,rd_filename,col_name1,col_name2);
% show_data(data_out)
% 
% data_out = read_csv_data(python_file,rd_filename,col_name1);
% show_data(data_out)


python_file = '..\read_csv_data.py';
rd_filename = '170419polardata.csv';
col_name1 = 're:Trc4_S21[U]';
col_name2 = 'im:Trc4_S21[U]';

data_out = read_csv_data(python_file,rd_filename,col_name1);
show_data(data_out)

