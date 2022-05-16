clear all;

SNRs = 9:18;

N = 10; % 2GB size is met at N=4400
n_blocks = 179;

bch = 1;
chan = "ETU"; %EPA/EVA/ETU
if (bch == 1)
    n = 63;
    k = 36;
    filename = "H_mat/BCH_" + n + "_" + k + ".alist";
else
    n = 128;
    k = 64;
    filename = "H_mat/LDPC_" + n + "_" + k + ".alist";
end
H = alist2full(filename);
G = null2(H);
code.N = N;
code.n_blocks = n_blocks;
code.n = n;
code.k = k;
code.G = double(G);
code.H = double(H);

msg = zeros(N*n_blocks,k,length(SNRs));
rx = zeros(N*n_blocks,n,length(SNRs));
enc = zeros(N*n_blocks,n,length(SNRs));
llr = zeros(size(enc));

tic;
% can be parallelized
for i = 1:length(SNRs)
    SNR = SNRs(i);
    
    disp("Generating " + chan  + " data : SNR " + SNR);
    c_type = 'ETU';
    [msg(:,:,i), enc(:,:,i), rx(:,:,i)] = RUN_lte(SNR, c_type, code);
    
end
delete(gcp('nocreate'));
disp("Finished generating data");

% demod using matlab bpsk demod
bpskDemod = comm.BPSKDemodulator('DecisionMethod','Log-likelihood ratio');

disp("Demodulating the data ... ");

llr = reshape(bpskDemod(rx(:)),size(llr));

disp(" ... done");
lte_path = "lte_data/";
if (bch == 1)
    prefix = lte_path + "BCH_" + n + "_" + k + "_";
else 
    prefix = lte_path + "LDPC_" + n + "_" + k + "_";
end

block_size = (N/2)*n_blocks;

start_ind = 1;
end_ind = start_ind + block_size - 1;

enc_ref = enc;
llr_ref = llr;

enc = permute(enc_ref(start_ind:end_ind,:,:),[2 1 3]);
llr = -1*permute(llr_ref(start_ind:end_ind,:,:),[2 1 3]);

filename = prefix + "ETU_data_train_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr')

start_ind = end_ind + 1;
end_ind = start_ind + block_size - 1;
enc = permute(enc_ref(start_ind:end_ind,:,:),[2 1 3]);
llr = -1*permute(llr_ref(start_ind:end_ind,:,:),[2 1 3]); %modulation consistency

filename = prefix + "ETU_data_test_" + SNRs(1) + "_" + SNRs(end) + ".mat";
save(filename,'enc','llr')

toc;

function [msg_data, enc_data, llr_data] = RUN_lte(SNR, c_type, code)

N = code.N;
n_blocks = code.n_blocks;
n = code.n;
k = code.k;
G = code.G;
msg_data = randi([0 1],n_blocks*N,k);
% whos msg_data
enc_data = mod(msg_data*G',2);
llr_data = zeros(size(enc_data));

for i_N = 1:N
    In = enc_data(1 + (i_N-1)*n_blocks: i_N*n_blocks,:)';
    % round_num = 
    inputBits = [In(:);randi([0 1],26028 - n_blocks*n,1)];
    detLLR = lte_chan(inputBits, SNR, c_type);
    detLLR = detLLR(1:n_blocks*n);
    demodSignal = reshape(detLLR,n,n_blocks);
    llr_data((i_N - 1)*n_blocks + 1: i_N*n_blocks, :) = demodSignal.';
end

end

function detLLR = lte_chan(inputBits, SNRdB, c_type)


%% Cell-Wide Settings

enb.NDLRB = 15;                 % Number of resource blocks
enb.CellRefP = 1;               % One transmit antenna port
enb.NCellID = 10;               % Cell ID
enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
enb.DuplexMode = 'FDD';         % FDD

%% SNR Configuration
SNR = 10^(SNRdB/20);    % Linear SNR
rng('default');         % Configure random number generators


%% Channel Model Configuration
cfg.Seed = ceil(100*rand());                  % Channel seed
cfg.NRxAnts = 1;               % 1 receive antenna
cfg.DelayProfile = c_type;     % EVA delay spread
cfg.DopplerFreq = 0;           % 120Hz Doppler frequency
cfg.MIMOCorrelation = 'Low';   % Low (no) MIMO correlation
cfg.InitTime = 0;              % Initialize at time zero
cfg.NTerms = 16;               % Oscillators used in fading model
cfg.ModelType = 'GMEDS';       % Rayleigh fading model type
cfg.InitPhase = 'Random';      % Random initial phases
cfg.NormalizePathGains = 'On'; % Normalize delay profile power
cfg.NormalizeTxAnts = 'On';    % Normalize for transmit antennas

%% Channel Estimator Configuration

cec.PilotAverage = 'UserDefined'; % Pilot averaging method
cec.FreqWindow = 9;               % Frequency averaging window in REs
cec.TimeWindow = 9;               % Time averaging window in REs

cec.InterpType = 'Cubic';         % Cubic interpolation
cec.InterpWinSize = 3;            % Interpolate up to 3 subframes
% simultaneously
cec.InterpWindow = 'Centred';     % Interpolation windowing method

%% Subframe Resource Grid Size

gridsize = lteDLResourceGridSize(enb);
K = gridsize(1);    % Number of subcarriers
L = gridsize(2);    % Number of OFDM symbols in one subframe
P = gridsize(3);    % Number of transmit antenna ports

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
for sf = 0:10
    
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
detLLR = zeros(23752,1);
X_new = 0;
% For all subframes within the frame
for sf = 0:9
    
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
