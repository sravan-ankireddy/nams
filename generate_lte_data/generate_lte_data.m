clear all;

bch = 0;

% num frames
% N = 1200;
N = 2400;

chan = "ETU"; %EPA/EVA/ETU
doppler_freq = 0;

if (bch == 1)
    n = 63;
    k = 36;
    filename = "H_mat/BCH_" + n + "_" + k + ".alist";

    if (chan == "ETU")
        SNRs = 11:24;
    elseif (chan == "EVA")
        SNRs = 10:25;
    end
else
    n = 128;
    k = 64;

    filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    if (chan == "ETU")
    SNRs = 1:10;
    elseif (chan == "EVA")
        SNRs = 10:25;
    end

    % n = 384;
    % k = 320;

    % filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    % if (chan == "ETU")
    %     SNRs = 4:18;
    % elseif (chan == "EVA")
    %     SNRs = 10:25;
    % end

    % n = 384;
    % k = 288;

    % filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    % if (chan == "ETU")
    %     SNRs = 1:16;
    % elseif (chan == "EVA")
    %     SNRs = 10:25;
    % end

    % n = 384;
    % k = 192;

    % filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    % if (chan == "ETU")
    %     SNRs = 3:12;
    % elseif (chan == "EVA")
    %     SNRs = 10:25;
    % end
end

H = alist2full(filename);

% Encoder and Decoder configs to use in built LDPC modules
decodercfg = ldpcDecoderConfig(sparse(logical(H)),'norm-min-sum');
decodercfg_nms = ldpcDecoderConfig(sparse(logical(H)),'norm-min-sum');
decodercfg_oms = ldpcDecoderConfig(sparse(logical(H)),'offset-min-sum');
decodercfg_bp = ldpcDecoderConfig(sparse(logical(H)),'bp');
decodercfg_lbp = ldpcDecoderConfig(sparse(logical(H)),'layered-bp');

encodercfg = ldpcEncoderConfig(decodercfg);

code.N = N;
code.n = n;
code.k = k;

code.H = double(H);

code.encodercfg = encodercfg;
code.decodercfg = decodercfg;

% msg = [];
% enc = [];
% rx = [];
modulation = 'BPSK';
mod_method = "comm"; % comm or lte

% FIX ME
% msg = zeros(k,N*376,length(SNRs));
% enc = zeros(n,N*376,length(SNRs));
msg = zeros(k,N*178,length(SNRs));
enc = zeros(n,N*178,length(SNRs));

rx = zeros(size(enc));

tic;
disp("Generating (" + code.n + ","+ code.k + ") " + chan  + " Doppler " + doppler_freq + " data for SNR " + SNRs(1) + " to " + SNRs(end));
% can be parallelized
parfor i = 1:length(SNRs)
    SNR = SNRs(i);
    
    disp("Generating (" + code.n + ","+ code.k + ") " + chan  + " Doppler " + doppler_freq + " data : SNR " + SNR);
    c_type = 'ETU';
    [msg(:,:,i), enc(:,:,i), rx(:,:,i)] = RUN_lte(SNR, modulation, mod_method, c_type, doppler_freq, code); 
%     [msg_snr, enc_snr, rx_snr] = RUN_lte(SNR, modulation, mod_method, c_type, doppler_freq, code); 
%     msg = cat(3,msg,msg_snr);
%     enc = cat(3,enc,enc_snr);
%     rx = cat(3,rx,rx_snr);
end
disp("Finished generating data");

disp("Demodulating the data ... ");
llr = zeros(size(rx));
for i_l = 1:size(rx,3)
    temp = rx(:,:,i_l);
    if (mod_method == "lte")
        llr(:,:,i_l) = reshape(lteSymbolDemodulate(temp(:),'BPSK','Soft'),size(llr(:,:,i_l)));
    else
        bpskDemodulator = comm.BPSKDemodulator; 
        bpskDemodulator.PhaseOffset = pi/4; 
        bpskDemodulator.DecisionMethod = 'Approximate log-likelihood ratio';
        llr(:,:,i_l) = -1*reshape(bpskDemodulator(temp(:)),size(llr(:,:,i_l)));
    end
end

disp(" ... done");

disp("Saving the data ... ");
lte_path = "lte_data/";
if (bch == 1)
    prefix = lte_path + "BCH_" + n + "_" + k + "_";
else 
    prefix = lte_path + "LDPC_" + n + "_" + k + "_";
end

block_size = size(llr,2)/2;

train_start_ind = 1;
train_end_ind = train_start_ind + block_size - 1;

test_start_ind = train_end_ind + 1;
test_end_ind = test_start_ind + block_size - 1;

msg_ref = msg;
enc_ref = enc;
llr_ref = llr;

msg = msg_ref(:,train_start_ind:train_end_ind,:);
enc = enc_ref(:,train_start_ind:train_end_ind,:);
llr = llr_ref(:,train_start_ind:train_end_ind,:); 

filename = prefix + chan + "_df_" + doppler_freq + "_data_train_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr','-v7.3');

msg = msg_ref(:,test_start_ind:test_end_ind,:);
enc = enc_ref(:,test_start_ind:test_end_ind,:);
llr = llr_ref(:,test_start_ind:test_end_ind,:);

filename = prefix + chan + "_df_" + doppler_freq + "_data_test_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr','-v7.3')

disp(" ... done");

% sanity
enc_est_raw = llr > 0;
BER_raw = (enc ~= enc_est_raw);
BER_raw = squeeze(mean(mean(BER_raw,2),1));
disp("Sanity check - Raw BER : ")
disp(BER_raw');
toc;

% sanity minsum

BER = zeros(4,length(SNRs));

max_iter = 5;

tic;
BER_nms = zeros(size(SNRs));
BER_oms = zeros(size(SNRs));
BER_bp = zeros(size(SNRs));
BER_lbp = zeros(size(SNRs));

for i_SNR = 1:length(SNRs)
    
    msg = squeeze(msg_ref(:,:,i_SNR));
    enc = squeeze(enc_ref(:,:,i_SNR));
    llr = -1*squeeze(llr_ref(:,:,i_SNR));

    disp("Decoding (" + code.n + ","+ code.k + ") " + chan  + " Doppler " + doppler_freq + " data : SNR " + SNRs(i_SNR));

    msg_est = ldpcDecode(llr,decodercfg_nms,max_iter);
    BER_nms(i_SNR) = sum(msg ~= msg_est, 'all');

    msg_est = ldpcDecode(llr,decodercfg_oms,max_iter);
    BER_oms(i_SNR) = sum(msg ~= msg_est, 'all');

    msg_est = ldpcDecode(llr,decodercfg_bp,max_iter);
    BER_bp(i_SNR) = sum(msg ~= msg_est, 'all');

    msg_est = ldpcDecode(llr,decodercfg_lbp,max_iter);
    BER_lbp(i_SNR) = sum(msg ~= msg_est, 'all');
    
end

BER(1,:) = BER_nms;
BER(2,:) = BER_oms;
BER(3,:) = BER_bp;
BER(4,:) = BER_lbp;

BER = BER/(size(msg_ref,2)*size(msg_ref,1));

disp("Sanity check - default MATLAB decoder BER : nms, oms, bp, lbp")
disp(BER);

BER_ms = BER;

if (bch == 1)
    prefix = lte_path + "BERs_ref_BCH_" + n + "_" + k + "_";
else 
    prefix = lte_path + "BERs_ref_LDPC_" + n + "_" + k + "_";
end
filename = prefix + chan + "_df_" + doppler_freq + "_data_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'BER_ms');

toc;

function [msg_data, enc_data, rx_data] = RUN_lte(SNR, modulation, mod_method, c_type, doppler_freq, code)

% Resetting seed for each SNR so that same channel is experienced across SNRs
% Update initTime later to maintain continuity in the fading process
rng('default');

%% Channel Model Configuration
cfg.Seed = 1;                  % Channel seed : 0 = random seed
cfg.NRxAnts = 1;               % 1 receive antenna
cfg.DelayProfile = c_type;     % delay spread
cfg.DopplerFreq = doppler_freq;% Doppler frequency
cfg.MIMOCorrelation = 'Low';   % Low (no) MIMO correlation
cfg.NTerms = 16;               % Oscillators used in fading model
cfg.ModelType = 'GMEDS';       % Rayleigh fading model type
cfg.InitPhase = 'Random';      % Random initial phases
cfg.NormalizePathGains = 'On'; % Normalize delay profile power
cfg.NormalizeTxAnts = 'On';    % Normalize for transmit antennas

%% Cell-Wide Settings

enb.NDLRB = 15;                 % Number of resource blocks
enb.CellRefP = 1;               % One transmit antenna port
enb.NCellID = 10;               % Cell ID
enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
enb.DuplexMode = 'FDD';         % FDD

% Subframe resource grid size
gridsize = lteDLResourceGridSize(enb);
K = gridsize(1);    % Number of subcarriers
L = gridsize(2);    % Number of OFDM symbols in one subframe
P = gridsize(3);    % Number of transmit antenna ports

% Number of bits needed is size of resource grid (K*L*P) * number of bits
% per symbol (2 for QPSK)
numberOfBits = K*L*P*bits_per_symbol(modulation);

% write as many codewords as possible to the subframe and use filler bits
numCwPerSubFrame = floor(numberOfBits/code.n);

% filler bits
num_filler_bits = numberOfBits - numCwPerSubFrame*code.n;

% num sub frames = 10 + 1 buffer
num_sf = 10;

%% Channel Estimator Configuration
cec.PilotAverage = 'UserDefined'; % Pilot averaging method
cec.FreqWindow = 9;               % Frequency averaging window in REs
cec.TimeWindow = 9;               % Time averaging window in REs

cec.InterpType = 'Cubic';         % Cubic interpolation
cec.InterpWinSize = 3;            % Interpolate up to 3 subframes
% simultaneously
cec.InterpWindow = 'Centred';     % Interpolation windowing method

N = code.N;
n = code.n;
k = code.k;

msg_data = randi([0 1],k,numCwPerSubFrame*num_sf,N);

% whos msg_data
enc_data = zeros(n,numCwPerSubFrame*num_sf,N);

rx_data = [];

M = bits_per_symbol(modulation);

msg_vec = [];
enc_vec = [];

% N Frames
for i_N = 1:N
    msgBits = squeeze(msg_data(:,:,i_N));
    enc_data(:,:,i_N) = ldpcEncode(msgBits,code.encodercfg);
    
    encBits = squeeze(enc_data(:,:,i_N));

    frameRx = lte_chan(encBits, num_sf, modulation, mod_method, SNR, i_N, cfg, enb, cec);

    demodSignal = reshape(frameRx,n,[]);

    % Calculate and save the amount of data actually transmitted
    numCw = size(demodSignal,2)*M;
    msg_vec = [msg_vec msgBits(:,1:numCw)];
    enc_vec = [enc_vec encBits(:,1:numCw)];
    rx_data = [rx_data demodSignal];
end
msg_data = msg_vec;
enc_data = enc_vec;
end

function rx = lte_chan(inputBits, num_sf, modulation, mod_method, SNRdB, nFrame, cfg, enb, cec)

%% SNR Configuration
SNR = 10^(SNRdB/20);            % Linear SNR
rng(nFrame);                    % Configure random number generators : for noise

% Update this to maintain continuity in fading process between frames
cfg.InitTime = (nFrame-1)*10e-3;

% Transmit Resource Grid
txGrid = [];

%% Payload Data


%% Frame Generation
code_len = size(inputBits,1);

% final subframe is an extra sub frame that's used as buffer in case any
% delay is experienced

% For all subframes within the frame, generate data based on dataInd 
% PSS and SSS happen only in sf 0,5
end_ind = 0;

M = bits_per_symbol(modulation);

inputSym_vec = [];
for sf = 0:num_sf

    % Set subframe number
    enb.NSubframe = mod(sf,num_sf);

    % Generate empty subframe
    subframe = lteDLResourceGrid(enb);

    % Generate synchronizing signals
    pssSym = ltePSS(enb);
    sssSym = lteSSS(enb);
    pssInd = ltePSSIndices(enb);
    sssInd = lteSSSIndices(enb);
    
    % Map synchronizing signals to the grid
    subframe(pssInd) = pssSym;
    subframe(sssInd) = sssSym;
    
    % Generate cell specific reference signal symbols and indices
    cellRsSym = lteCellRS(enb);
    cellRsInd = lteCellRSIndices(enb);

    % Map cell specific reference signal to grid
    subframe(cellRsInd) = cellRsSym;

    % data indices
    sigInd = [pssInd;sssInd;cellRsInd];
    dataInd = setdiff(1:180*14,sigInd);

    % find suitable number of codewords
    numDataSym = floor(length(dataInd)/(code_len*M));

    if (sf == num_sf)
        inputBits_sf = randi([0,1],length(dataInd),1);

        % Modulate input bits
        if (mod_method == "lte")
            inputSym = lteSymbolModulate(inputBits_sf,modulation);
        else
            bpskModulator = comm.BPSKModulator;
            bpskModulator.PhaseOffset = pi/4;
            inputSym = bpskModulator(inputBits_sf);
        end
    else
        start_ind = end_ind + 1;
        end_ind = start_ind + numDataSym*M - 1;

        num_filler_bits = length(dataInd) - numDataSym*M*code_len;
        % flatten and add any necessary filler bits to fit the sub frame
        inputBits_sf = inputBits(:,start_ind:end_ind);
        inputBits_sf = [inputBits_sf(:); randi([0 1],num_filler_bits,1)];

        % Modulate input bits
        if (mod_method == "lte")
            inputSym = lteSymbolModulate(inputBits_sf,modulation);
        else
            bpskModulator = comm.BPSKModulator;
            bpskModulator.PhaseOffset = pi/4;
            inputSym = bpskModulator(inputBits_sf);
        end

        % store the inputSym in buffer
        inputSym_vec = [inputSym_vec; inputSym]; %#ok
    end
    
    % Map input symbols to grid
    subframe(dataInd) = inputSym;

    % Append subframe to grid to be transmitted
    txGrid = [txGrid subframe]; %#ok
    
end

%% OFDM Modulation

[txWaveform,info] = lteOFDMModulate(enb,txGrid);

% remove the buffer subframe
txGrid = txGrid(:,1:140);

%% Fading Channel
cfg.SamplingRate = info.SamplingRate;

% Pass data through the fading channel model
rxWaveform = lteFadingChannel(cfg,txWaveform);

%% Additive Noise

% Calculate noise gain
N0 = 1/(sqrt(2.0*enb.CellRefP*double(info.Nfft))*SNR);

% Create additive white Gaussian noise
noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));

% Add noise to the received time domain waveform
rxWaveform = rxWaveform + noise;

%% Synchronization
offset = lteDLFrameOffset(enb,rxWaveform);
rxWaveform = rxWaveform(1+offset:end,:);

%% OFDM Demodulation

rxGrid = lteOFDMDemodulate(enb,rxWaveform);

%% Channel Estimation

enb.NSubframe = 0;
[estChannel, noiseEst] = lteDLChannelEstimate(enb,cec,rxGrid);

%% MMSE Equalization

eqGrid = lteEqualizeMMSE(rxGrid, estChannel, noiseEst);

%% Frame Extraction
rx = [];

% For all subframes within the frame
max_sf = size(eqGrid,2)/14-1;
end_ind = 0;
for sf = 0:max_sf
    
    % Set subframe number
    enb.NSubframe = mod(sf,num_sf);
    
    % Generate synchronizing signals indices
    pssInd = ltePSSIndices(enb);
    sssInd = lteSSSIndices(enb);
    
    % Generate cell specific reference signal indices
    cellRsInd = lteCellRSIndices(enb);

    % data indices
    sigInd = [pssInd;sssInd;cellRsInd];
    dataInd = setdiff(1:180*14,sigInd);

    % find suitable number of codewords
    numDataSym = floor(length(dataInd)/code_len);

    start_ind = end_ind + 1;
    end_ind = start_ind + numDataSym*code_len*M - 1;
    
    % extract all symbols from this subframe
    rxSig = eqGrid(:,sf*14 + 1: (sf + 1)*14);
    rxSig = rxSig(:);

    % map to data symbols in current subframe
    rx = [rx; rxSig(dataInd(1:numDataSym*code_len*M))];
end

end
