clear all;

n_blocks = 179;

bch = 0;

chan = "ETU"; %EPA/EVA/ETU
doppler_freq = 0;

if (bch == 1)
    n = 63;
    k = 36;
    filename = "H_mat/BCH_" + n + "_" + k + ".alist";
    N = 2800; % 2GB size is met at N=2800
    if (chan == "ETU")
        SNRs = 15:30;
    elseif (chan == "EVA")
        SNRs = 10:25;
    end
else
    % n = 128;
    % k = 64;
    % N = 2000; % 2GB size is met at N=2000
    % filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    % if (chan == "ETU")
    % SNRs = 7:16;
    % elseif (chan == "EVA")
    %     SNRs = 10:25;
    % end
    n = 384;
    k = 320;
    N = 700; % 2GB size is met at N=2000
    filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
    if (chan == "ETU")
        SNRs = 11:20;
    elseif (chan == "EVA")
        SNRs = 10:25;
    end
end

H = alist2full(filename);

% Encoder and Decoder configs to use in built LDPC modules

decodercfg = ldpcDecoderConfig(sparse(logical(H)),'norm-min-sum');

encodercfg = ldpcEncoderConfig(decodercfg);

code.N = N;
code.n_blocks = n_blocks;
code.n = n;
code.k = k;

code.H = double(H);

code.encodercfg = encodercfg;
code.decodercfg = decodercfg;

msg = zeros(k,N*n_blocks,length(SNRs));
enc = zeros(n,N*n_blocks,length(SNRs));
rx = zeros(size(enc));
llr = zeros(size(enc));

tic;
disp("Generating " + chan  + " Doppler " + doppler_freq + " data for SNR " + SNRs(1) + " to " + SNRs(end));
% can be parallelized
parfor i = 1:length(SNRs)
    SNR = SNRs(i);
    
    disp("Generating " + chan  + " Doppler " + doppler_freq + " data : SNR " + SNR);
    c_type = 'ETU';
    [msg(:,:,i), enc(:,:,i), rx(:,:,i)] = RUN_lte(SNR, c_type, doppler_freq, code);
    
end
disp("Finished generating data");

% demod using matlab bpsk demod
bpskDemod = comm.BPSKDemodulator('DecisionMethod','Log-likelihood ratio');

disp("Demodulating the data ... ");

llr = reshape(bpskDemod(rx(:)),size(llr));

disp(" ... done");

disp("Saving the data ... ");
lte_path = "lte_data/";
if (bch == 1)
    prefix = lte_path + "BCH_" + n + "_" + k + "_";
else 
    prefix = lte_path + "LDPC_" + n + "_" + k + "_";
end

block_size = (N/2)*n_blocks;

start_ind = 1;
end_ind = start_ind + block_size - 1;

msg_ref = msg;
enc_ref = enc;
llr_ref = llr;

msg = msg_ref(:,start_ind:end_ind,:);
enc = enc_ref(:,start_ind:end_ind,:);
llr = -1*llr_ref(:,start_ind:end_ind,:); % modulation consistency

filename = prefix + chan + "_df_" + doppler_freq + "_data_train_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr');

start_ind = end_ind + 1;
end_ind = start_ind + block_size - 1;

msg = msg_ref(:,start_ind:end_ind,:);
enc = enc_ref(:,start_ind:end_ind,:);
llr = -1*llr_ref(:,start_ind:end_ind,:); % modulation consistency

filename = prefix + chan + "_df_" + doppler_freq + "_data_test_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr')

disp(" ... done");

% sanity
enc_est_raw = llr > 0;
BER_raw = (enc ~= enc_est_raw);
BER_raw = squeeze(mean(mean(BER_raw,2),1));
disp("Sanity check - Raw BER : ")
disp(BER_raw');

% sanity minsum

BER = zeros(size(SNRs));

max_iter = 5;

for i_SNR = 1:length(SNRs)
    
    msg = squeeze(msg_ref(:,:,i_SNR));
    enc = squeeze(enc_ref(:,:,i_SNR));
    llr = squeeze(llr_ref(:,:,i_SNR));
    msg_est = ldpcDecode(llr,decodercfg,max_iter);

    BER(i_SNR) = sum(msg ~= msg_est, 'all');
end

BER = BER/(N*n_blocks*k);

disp("Sanity check - default normalised min-sum BER : ")
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

function [msg_data, enc_data, llr_data] = RUN_lte(SNR, c_type, doppler_freq, code)

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

%% Channel Estimator Configuration
cec.PilotAverage = 'UserDefined'; % Pilot averaging method
cec.FreqWindow = 9;               % Frequency averaging window in REs
cec.TimeWindow = 9;               % Time averaging window in REs

cec.InterpType = 'Cubic';         % Cubic interpolation
cec.InterpWinSize = 3;            % Interpolate up to 3 subframes
% simultaneously
cec.InterpWindow = 'Centred';     % Interpolation windowing method

N = code.N;
n_blocks = code.n_blocks;
n = code.n;
k = code.k;

msg_data = randi([0 1],k,n_blocks*N);

% whos msg_data
enc_data = ldpcEncode(msg_data,code.encodercfg);
llr_data = zeros(size(enc_data));

% N Frames
for i_N = 1:N
    In = enc_data(:,1 + (i_N-1)*n_blocks: i_N*n_blocks);
    % round_num = 
    inputBits = [In(:);randi([0 1],26028*3 - n_blocks*n,1)];
    detLLR = lte_chan(inputBits, SNR, i_N, cfg, enb, cec);
    detLLR = detLLR(1:n_blocks*n);
    demodSignal = reshape(detLLR,n,n_blocks);
    llr_data(:,(i_N - 1)*n_blocks + 1: i_N*n_blocks) = demodSignal;
end

end

function detLLR = lte_chan(inputBits, SNRdB, nFrame, cfg, enb, cec)

%% SNR Configuration
SNR = 10^(SNRdB/20);            % Linear SNR
rng(nFrame);                    % Configure random number generators : for noise

% Update this to maintain continuity in fading process between frames
cfg.InitTime = (nFrame-1)*10e-3;

%% Transmit Resource Grid
txGrid = [];

%% Payload Data Generation

% Number of bits needed is size of resource grid (K*L*P) * number of bits
% per symbol (2 for QPSK)
% % numberOfBits = K*L*P*2;

% Modulate input bits
inputSym = lteSymbolModulate(inputBits,'BPSK');

%% Frame Generation
X_new = 0;
% For all subframes within the frame
for sf = 0:30
    
    % Set subframe number
    enb.NSubframe = mod(sf,10);
    
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
    X_old = X_new;
    X_new = X_new+180*14-(length(cellRsInd)+length(sssInd)+length(pssInd));
    % Map cell specific reference signal to grid
    subframe(cellRsInd) = cellRsSym;
    
    % Map input symbols to grid
    dataIdx = [cellRsInd;pssInd;sssInd];
    subframe(setdiff(1:180*14,dataIdx)) = inputSym(X_old+1:X_new);
    
    % Append subframe to grid to be transmitted
    txGrid = [txGrid subframe]; %#ok
    
end

%% OFDM Modulation

[txWaveform,info] = lteOFDMModulate(enb,txGrid);

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
detLLR = zeros(23752*3,1);
X_new = 0;
% For all subframes within the frame

max_sf = size(eqGrid,2)/14-1;
for sf = 0:max_sf
    
    % Set subframe number
    enb.NSubframe = mod(sf,10);
    
    % Generate synchronizing signals indices
    pssInd = ltePSSIndices(enb);
    sssInd = lteSSSIndices(enb);
    
    % Generate cell specific reference signal indices
    cellRsInd = lteCellRSIndices(enb);
    
    X_old = X_new;
    X_new = X_new+180*14-(length(cellRsInd)+length(sssInd)+length(pssInd));
    
    % Map input symbols to grid
    dataIdx = [cellRsInd;pssInd;sssInd];
    RxSig = eqGrid(:,sf*14 + 1: (sf + 1)*14);
    detLLR(X_old+1:X_new) = RxSig(setdiff(1:180*14,dataIdx));
end

detLLR = detLLR(:);

end
