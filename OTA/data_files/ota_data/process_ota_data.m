tx_pow = 20:-1:12;%-5:0.5:4.5;
rx_pow = 10;

N_samps = 2*250*1e3;

N = 63;
K = 36;

% N = 384;
% K = 320;

demod_input_vec = [];
llr_data_vec = [];
msg_data_vec = [];
enc_data_vec = [];

BERs = zeros(size(tx_pow));

for i_t = 1:length(tx_pow)
    
    filename = sprintf("ota_data_blocks/ota_data_%d_%d_%.1f_%.1f.mat",N,K,tx_pow(i_t),rx_pow);

    disp(filename);

    temp = load(filename,"demod_input","llr_data","msg_data","enc_data");
    temp.demod_input = permute(temp.demod_input,[2 1 3]);
    temp.llr_data = permute(temp.llr_data,[2 1 3]);
    temp.msg_data = permute(temp.msg_data,[2 1 3]);
    temp.enc_data = permute(temp.enc_data,[2 1 3]);

    temp.demod_input = reshape(temp.demod_input,size(temp.demod_input,1),[]);
    temp.llr_data = reshape(temp.llr_data,size(temp.llr_data,1),[]);
    temp.msg_data = reshape(temp.msg_data,size(temp.msg_data,1),[]);
    temp.enc_data = reshape(temp.enc_data,size(temp.enc_data,1),[]);
    
    % randomly select N_samps from full data
    samp_ind = randperm(size(temp.demod_input,2));
    samp_ind = samp_ind(1:N_samps);
    samp_ind = 1:N_samps;

    temp.demod_input = temp.demod_input(:,samp_ind);
    temp.llr_data = temp.llr_data(:,samp_ind);
    temp.msg_data = temp.msg_data(:,samp_ind);
    temp.enc_data = temp.enc_data(:,samp_ind);

    demod_input_vec = cat(3,demod_input_vec,temp.demod_input);
    llr_data_vec = cat(3,llr_data_vec,temp.llr_data);
    msg_data_vec = cat(3,msg_data_vec,temp.msg_data);
    enc_data_vec = cat(3,enc_data_vec,temp.enc_data);

    code_est = temp.llr_data < 0; 
    BERs(i_t) = sum(code_est ~= temp.enc_data,'all')/numel(code_est);

end

% filename = sprintf("final_ota_data_%d_%d_tx_pow_%.1f_to_%.1f_rx_pow_%.1f.mat",N,K,tx_pow(end),tx_pow(1),rx_pow);
% save(filename,"msg_data","enc_data","demod_input","llr_data","raw_undecoded_ber",'-v7.3');

% flip the order of tx pow
start_ind = 1;
end_ind = start_ind + N_samps/2 - 1;
msg_data = msg_data_vec(:,start_ind:end_ind,end:-1:1);
enc_data = enc_data_vec(:,start_ind:end_ind,end:-1:1);
llr_data = llr_data_vec(:,start_ind:end_ind,end:-1:1);
demod_input = demod_input_vec(:,:,end:-1:1);
raw_undecoded_ber = BERs(end:-1:1);
disp(raw_undecoded_ber);

% save in standard format
enc = enc_data;
llr = -1*llr_data;
tx_start = round(tx_pow(end));
tx_end = round(tx_pow(1));
filename = sprintf("BCH_%d_%d_OTA_data_train_%d_%d.mat",N,K,tx_start,tx_end);
save(filename,"enc","llr","raw_undecoded_ber",'-v7.3');

start_ind = end_ind + 1;
end_ind = start_ind + N_samps/2 - 1;
msg_data = msg_data_vec(:,start_ind:end_ind,end:-1:1);
enc_data = enc_data_vec(:,start_ind:end_ind,end:-1:1);
llr_data = llr_data_vec(:,start_ind:end_ind,end:-1:1);
demod_input = demod_input_vec(:,:,end:-1:1);
raw_undecoded_ber = BERs(end:-1:1);
disp(raw_undecoded_ber);

% save in standard format
enc = enc_data;
llr = -1*llr_data;
tx_start = round(tx_pow(end));
tx_end = round(tx_pow(1));
filename = sprintf("BCH_%d_%d_OTA_data_test_%d_%d.mat",N,K,tx_start,tx_end);
save(filename,"enc","llr","raw_undecoded_ber",'-v7.3');


