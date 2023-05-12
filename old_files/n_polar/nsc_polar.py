import os
import argparse
import matplotlib.pyplot as plt

import time
from time import sleep
import numpy as np
import scipy.io as sio

np.random.seed(0)
import pdb
import sys

import torch
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd.profiler as profiler

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-seed', type=int, default=0)
	parser.add_argument('-snr_lo', type=float, default=3)
	parser.add_argument('-snr_hi', type=float, default=3)
	parser.add_argument('-snr_step', type=float, default=1)
	parser.add_argument('-min_frame_errors', type=int, default=10e2)
	parser.add_argument('-max_frames', type=float, default=10e3)
	parser.add_argument('-output_filename', type=str, default='laskdjhf')
	parser.add_argument('-L', type=float, default=0.5)
	parser.add_argument('-steps', type=int, default=100)
	parser.add_argument('-batch_size', type=int, default=1000)
	parser.add_argument('-decoder_type', type=str, default='NSC')
	parser.add_argument('-force_spa', type=int, default=0)
	parser.add_argument('-force_all_zero', type=int, default=0)
	parser.add_argument('-channel_type', type=str, default='AWGN')
	parser.add_argument('-offset', type=float, default=0)
	parser.add_argument('-norm_fac', type=float, default=1)
	parser.add_argument('-use_gpu', type=int, default=1)
	parser.add_argument('-gpu_index', type=int, default=0)
	parser.add_argument('-rand_init_weights', type=int, default=1)
	parser.add_argument('-entangle_weights', type=int, default=0)
	parser.add_argument('-freeze_b', type=int, default=0)
	parser.add_argument('-freeze_w', type=int, default=0)
	parser.add_argument('-training', type=int, default=1)
	parser.add_argument('-approx', type=int, default=0)

	args = parser.parse_args()

	return args	

args = get_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_index)

if (args.use_gpu == 0):
	device = torch.device("cpu")
else:
	cuda_str = "cuda:" + str(args.gpu_index)
	device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
print("Now running using .... " + str(device))

seed = args.seed
snr_lo = args.snr_lo
snr_hi = args.snr_hi
snr_step = args.snr_step
min_frame_errors = args.min_frame_errors
max_frames = args.max_frames

output_filename = args.output_filename
L = args.L
steps = args.steps
decoder_type = args.decoder_type
batch_size = args.batch_size
SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step)

TRAINING = True
if (args.steps == 0):
	TRAINING = False
ALL_ZEROS_CODEWORD_TRAINING = args.force_all_zero 
ALL_ZEROS_CODEWORD_TESTING = args.force_all_zero

NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
np.set_printoptions(precision=10)

def apply_channel(codewords, noise, channel, FastFading):
	if (channel == 'rayleigh_fast'): ##rayleigh fast
		data_ones = np.ones_like(codewords)
		d0 = data_ones.shape[0]
		d1 = data_ones.shape[1]
		#  Rayleigh Fading Channel, iid
		# fading_h = tf.math.sqrt(tf.random.randn(data_shape)**2 +  tf.random.randn(data_shape)**2)/tf.math.sqrt(3.14/2.0)
		fading_h = np.sqrt(np.random.randn(d0,d1)**2 +  np.random.randn(d0,d1)**2)/np.sqrt(3.14/2.0)
		# fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
		fading_h = torch.tensor(fading_h)
		received_codewords = fading_h*codewords + noise
	elif (channel == 'rayleigh_slow'): ##rayleigh slow
		data_ones = np.ones_like(codewords)
		d0 = data_ones.shape[0]
		d1 = data_ones.shape[1]
		# fading_h = tf.sqrt(tf.random.randn(data_shape[0])**2 +  tf.random.randn(data_shape[0])**2)/tf.sqrt(3.14/2.0)
		fading_h = np.sqrt(np.random.randn(d0)**2 +  np.random.randn(d0)**2)/np.sqrt(3.14/2.0)
		# fading_h = fading_h.type(tf.FloatTensor).to(self.this_device)
		fading_h = torch.tensor(fading_h)
		received_codewords = codewords*fading_h[:,None, None] + noise
	elif (channel == 'rician'):
		K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
		coeffLOS = np.sqrt(K/(K+1))
		coeffNLOS = np.sqrt(1/(K+1))

		if FastFading:
			hLOSReal = np.ones_like(codewords) #Assuming SISO see page 3.108 in Heath and Lazano
			hLOSImag = np.ones_like(codewords)
			hNLOSReal = np.random.randn(hLOSReal.shape[0],hLOSReal.shape[1])
			hNLOSImag = np.random.randn(hLOSImag.shape[0],hLOSImag.shape[1])
		else: #Slow fading case
			hLOSReal = np.ones_like(1) #Assuming SISO see page 3.108 in Heath and Lazano
			hLOSImag = np.ones_like(1)
			hNLOSReal = np.random.randn(1)
			hNLOSImag = np.random.randn(1)
		fading_h = torch.tensor(coeffLOS*(hLOSReal + hLOSImag*1j) + coeffNLOS*(hNLOSReal + hNLOSImag*1j))
		#Assuming phase information at the receiver
		fading_h = torch.abs(fading_h)/np.sqrt(3.14/2.0)
		# fading_h = fading_h.type(torch.FloatTensor).to(self.this_device) 
		received_codewords = fading_h*codewords + noise
	else:
		received_codewords = codewords + noise
	return received_codewords

def f_real(a,b,depth,ind_l,ind_r,cur_len):
	if (args.training==0):
		n_a = a
		n_b = b
	else:
		rep_fac = a.size()[0]
		n_B = decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
		n_W = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
		# breakpoint()
		# n_a = n_W[:,:cur_len]*a + n_B[:,:cur_len]
		# n_b = n_W[:,cur_len:]*b + n_B[:,cur_len:]
		n_a = n_W*a + n_B
		n_b = n_W*b + n_B
		# n_a = decoder.W_cv[0,0]*(a-decoder.B_cv[0,0])
		# n_b = decoder.W_cv[1,0]*(b-decoder.W_cv[1,0])
	out = torch.log( torch.div((1 + torch.exp(n_a+n_b)),(torch.exp(n_a)+torch.exp(n_b))) )
	return out

def f_minsum(a,b):
	mag = torch.min(torch.abs(a),torch.abs(b))
	sign = torch.sign(a)*torch.sign(b)
	# return (sign*mag - decoder.B_cv[0,0])*decoder.W_cv[0,0]
	return sign*mag

def g_func(a,b,c):
	if 1:#(args.training == 0):
		n_a = a
		n_b = b
	# else:
	# 	n_a = decoder.W_cv[2,0]*(a-decoder.B_cv[2,0])
	# 	n_b = decoder.W_cv[3,0]*(b-decoder.W_cv[3,0])
	# breakpoint()
	return n_b + (1-2*c)*n_a

def construct_G(N):
	n = int(np.log2(N))
	F = torch.tensor([[1, 0], [1, 1]])
	G = F
	
	for i in range(n-1):
		G = torch.kron(G,F)
	return G

def p_encode(input,G):
	return torch.remainder(torch.matmul(input,G),2)

def p_decode(LLR):
	# depth of the polar code tree
	n = int(np.log2(N))
	
	# mini-batch size for vectorization
	m = LLR.size()[0]

	# beliefs
	L = torch.zeros((m,n+1,N))

	# decisions
	ucap = torch.zeros((m,n+1,N))
	llr_out = torch.zeros((m,1,N))
	
	# vector to check status of node -- left or right or parent propagation
	ns = np.zeros((1,2*N-1))
	# belief initialisation
	L[:,0,:] = LLR
	Ln = torch.zeros((2,m,n+1,N))
	
	# start at root
	node = 0
	depth = 0
	
	# stopping criteria
	done = 0
	# traverse till all bits are decoded
	while (done == 0):
		# check for leaf node
		# print("depth check " + str(depth))
		if (depth == n):
			# store the llar data of leaf-data nodes
			# if frozen node		
			if (info_check_vec[0,node] == 0):
				ucap[:,n,node] = 0
			else:
				ucap[:,n,node] = L[:,n,node] < 0
				llr_out[:,0,node] = L[:,n,node]
			# check for last leaf node
			if node == (N-1):
				done = 1
			# move back to parent node
			else:
				node = int(np.floor(node/2)); depth = depth - 1
		# non-leaf node
		else:
			# position of node in node state vector
			npos = int((2**depth-1) + node)
			# propagate to left child
			if (ns[0,npos] == 0):
				# length of current node
				cur_len = 2**(n-depth)
				# incoming beliefs
				ind_l = int(cur_len*node)
				ind_r = int(cur_len*(node+1))
				# applying weights ####################################
				Ln = L[:,depth,ind_l:ind_r]
				# rep_fac = L.size()[0]
				# n_W = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
				# n_B = decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
				# print("dep " + str(depth) + " ind_l " + str(ind_l) + " ind_r " + str(ind_r) + " in f ")
				# Ln[0,:,depth,ind_l:ind_r] = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)*L[:,depth,ind_l:ind_r] + decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
				ind_l_old = ind_l; ind_r_old = ind_r; 
				# next node: left child
				node = 2*node; depth = depth + 1
				# incoming belief length for left child
				cur_len = int(cur_len / 2)
				# calculate and store LLRs for left child
				ind_l = int(cur_len*node)
				ind_r = int(cur_len*(node+1))
				if (args.approx == 1):
					L[:,depth,ind_l:ind_r] = f_minsum(Ln[:,:cur_len],Ln[:,cur_len:])
				else:
					# breakpoint()
					L[:,depth,ind_l:ind_r] = f_real(Ln[:,:cur_len],Ln[:,cur_len:],depth,ind_l,ind_r,cur_len)
					# breakpoint()
					# L[:,depth,ind_l:ind_r] = f_real(Ln[0,:,depth-1,ind_l_old:ind_l_old+cur_len],Ln[0,:,depth-1,ind_l_old+cur_len:ind_r_old])
				# mark as left child visited
				ns[0,npos] = 1
			else:
				# propagate to right child
				if (ns[0,npos] == 1):
					# length of current node
					cur_len = 2**(n-depth)
					# incoming beliefs
					ind_l = int(cur_len*node)
					ind_r = int(cur_len*(node+1)) 
					# applying weights ####################################
					Ln = L[:,depth,ind_l:ind_r]
					# rep_fac = L.size()[0]
					# n_W = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
					# n_B = decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
					# print("dep " + str(depth) + " ind_l " + str(ind_l) + " ind_r " + str(ind_r) + " in g ")
					# Ln[1,:,depth,ind_l:ind_r] = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)*L[:,depth,ind_l:ind_r] + decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
					ind_l_old = ind_l; ind_r_old = ind_r; 
					# left child
					lnode = 2*node; ldepth = depth + 1; 
					ltemp = cur_len/2
					# incoming decisions from left child
					ind_l_temp = int(ltemp*lnode)
					ind_r_temp = int(ltemp*(lnode+1))
					ucapn = ucap[:,ldepth,ind_l_temp:ind_r_temp]
					# next node: right child
					node = node *2 + 1; depth = depth + 1
					# incoming belief length for right child
					cur_len = int(cur_len / 2)
					# calculate and store LLRs for right child
					ind_l = int(cur_len*node)
					ind_r = int(cur_len*(node+1))
					# L[:,depth,ind_l:ind_r] = g_func(Ln[:,:cur_len],Ln[:,cur_len:],ucapn)
					# L[:,depth,ind_l:ind_r] = g_func(Ln[1,:,depth-1,ind_l_old:ind_l_old+cur_len],Ln[1,:,depth-1,ind_l_old+cur_len:ind_r_old],ucapn)
					# mark as right child visited
					ns[0,npos] = 2
				# calculate beta propagate to parent node
				else:
					# length of current node
					cur_len = 2**(n-depth)
					# left and right child
					lnode = 2*node; rnode = 2*node + 1; cdepth = depth + 1
					ctemp = int(cur_len/2)
					ind_l_ctemp = int(ctemp*lnode)
					ind_r_ctemp = int(ctemp*(lnode+1))
					# incoming decisions from left child
					ucapl = ucap[:,cdepth,ind_l_ctemp:ind_r_ctemp]
					ind_l_ctemp = int(ctemp*rnode)
					ind_r_ctemp = int(ctemp*(rnode+1))
					# incoming decisions from right child
					ucapr = ucap[:,cdepth,ind_l_ctemp:ind_r_ctemp]
					# combine
					ind_l = int(cur_len*node)
					ind_r = int(cur_len*(node+1))
					ucap[:,depth,ind_l:ind_r] = torch.cat((torch.remainder((ucapl+ucapr),2),ucapr),1)
					# update to index of parent node
					node = int(np.floor(node/2)); depth = depth - 1
	# rep_fac = L.size()[0]
	# n_W = decoder.W_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
	# n_B = decoder.B_cv[depth,ind_l:ind_r].repeat(rep_fac,1)
	return ucap[:,n,data_pos], llr_out[:,0,data_pos]#*n_W + n_B
	
N = 8
G = construct_G(N)

# load reliability order of channels
Q = sio.loadmat('Q_1024.mat')
Q = Q['Q']

# Extracting channel indices from Q for snaller N
Q = Q[Q<N]

# Rate of code
R = 0.5

# Length of message vector
K = int(N*R)

# Number of blocks transmitted
Nblocks = 10e2
Nblocks_base = Nblocks

# Energy per bit
Eb = 1

# No. of samples in EBN0 range
nSam = 8

# start and stop of EbN0dB
EbN0dB_start = 1
EbN0dB_stop = 3.7

# Eb/N0 step size
delta = (EbN0dB_stop - EbN0dB_start)/(nSam-1)

# Eb/N0 range of signal in dB
EbN0dB = np.arange(EbN0dB_start,EbN0dB_stop,delta)

# Varinace of noise in linear scale
var_N = np.power(10,-EbN0dB/10)/(2 * R)

# Vector to store Bit Error Rate
BER_SCD = np.zeros((1,len(var_N)))

# Vector to store Bit Error Rate
FER_SCD = np.zeros((1,len(var_N)))

# Indices corresponding to data channels
data_pos = Q[N-K:]

# Boolean array with information nodes pos = 1
info_check_vec = np.zeros((1,N))
info_check_vec[0,data_pos] = 1

# create a class for the neural decoder
class Decoder:
	def __init__(self, decoder_type="NSC", random_seed=0, learning_rate = 0.001, relaxed = False):
		self.decoder_type = decoder_type
		self.random_seed = random_seed
		self.learning_rate = learning_rate
		self.relaxed = relaxed

starter_learning_rate = 0.01
learning_rate = starter_learning_rate # decoder_type="normal", "FNNMS", "NSC", ...
decoder = Decoder(decoder_type=decoder_type, random_seed=1, learning_rate = learning_rate, relaxed = False)
print("\n\nDecoder type: " + decoder.decoder_type + "\n\n")

if (args.decoder_type == "NSC"):
	n = int(np.log2(N))
	if (args.rand_init_weights == 1):
		var_W = 0.5
		var_B = 0.25	
		# decoder.B_cv = args.norm_fac * (var_B + torch.fmod(torch.randn([4, 1]),var_B))
		# decoder.W_cv = args.offset + var_W + torch.fmod(torch.randn([4, 1]),var_W)
		decoder.B_cv = args.norm_fac * (var_B + torch.fmod(torch.randn([n+1, N]),var_B))
		decoder.W_cv = args.offset + var_W + torch.fmod(torch.randn([n+1, N]),var_W)
		(decoder.B_cv).requires_grad=True
		(decoder.W_cv).requires_grad=True
	else:						
		decoder.B_cv = torch.tensor(args.offset + torch.zeros([n+1, N]), requires_grad=True)
		decoder.W_cv = torch.tensor(args.norm_fac * torch.ones([n+1, N]), requires_grad=True)

if (batch_size % len(SNRs)) != 0:
    print("********************")
    print("********************")
    print("error: batch size must divide by the number of SNRs to train on")
    print("********************")
    print("********************")
BERs = []
SERs = []
FERs = []

torch.autograd.set_detect_anomaly(True)

# Training phase
if args.training :

	# Training loop
	print("***********************")
	print("Training decoder using " + str(steps) + " minibatches...")
	print("***********************")

	criterion = nn.BCEWithLogitsLoss()
	params = [decoder.B_cv, decoder.W_cv]
	optimizer = optim.Adam(params, lr = learning_rate)

	step = 0
	loss_history = []

	while step < steps:
		# generate random codewords

		messages = torch.zeros([K,batch_size])
		codewords = torch.zeros([N,batch_size])
		BPSK_codewords = (0.5 - codewords) * 2.0
		soft_input = torch.zeros_like(BPSK_codewords)
		channel_information = torch.zeros_like(BPSK_codewords)

		# create minibatch with codewords from multiple SNRs
		for i in range(0,len(SNRs)):
			sigma = np.sqrt(1. / (2 * (float(K)/float(N)) * 10**(SNRs[i]/10)))
			noise = sigma * np.random.randn(N,batch_size//len(SNRs))
			start_idx = batch_size*i//len(SNRs)
			end_idx = batch_size*(i+1)//len(SNRs)
			
			# AWGN channel
			if (args.channel_type == 'AWGN'):
				channel_information[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + noise
			# Fading channel
			elif(args.channel_type == 'rayleigh_fast' or args.channel_type == 'rician' ):
				FastFading = False
				channel_information[:,start_idx:end_idx] = apply_channel(BPSK_codewords[:,start_idx:end_idx], noise, args.channel_type, FastFading)
			# Bursty channel
			elif (args.channel_type == 'bursty'):
				p = 0.1
				sigma_bursty = 3*sigma
				temp = np.random.binomial(1,p,np.shape(noise))
				noise_bursty = np.multiply(temp,sigma_bursty*np.random.randn(N,batch_size//len(SNRs)))
				channel_information[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + noise + noise_bursty

			if NO_SIGMA_SCALING_TRAIN:
				soft_input[:,start_idx:end_idx] = channel_information[:,start_idx:end_idx]
			else:
				soft_input[:,start_idx:end_idx] = 2.0*channel_information[:,start_idx:end_idx]/(sigma*sigma)

		# feed minibatch into BP and run SGD
		batch_data = soft_input.to(device)
		batch_labels = codewords[data_pos,:].to(device) #choosing just the messages; not entire codeword
		[message_est, llr_out_mes] = p_decode(torch.transpose(batch_data,0,1))
		llr_out_mes = torch.reshape(llr_out_mes,batch_labels.size()).to(device)
		loss = criterion(-llr_out_mes,batch_labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# freezing weightslo
		if (args.freeze_b == 1):
			if (step == 100):
				decoder.B_cv.requires_grad = False
				decoder.W_cv.requires_grad = True
		if (args.freeze_w == 1):
			if (step == 100):
				decoder.B_cv.requires_grad = True
				decoder.W_cv.requires_grad = False

		if step % 10 == 0:
			print(str(step) + " minibatches completed")
			print ("BCE loss :")
			print (loss)
			print ("B_cv : ")
			print (decoder.B_cv)
			print ("W_cv : ")
			print (decoder.W_cv)
		step += 1
	print("Trained decoder on " + str(step) + " minibatches.\n")
	if (step > 0):
		print (loss)
		print (decoder.B_cv)
		print (decoder.W_cv)

	save_B_cv = decoder.B_cv
	save_W_cv = decoder.W_cv
# breakpoint()
# Testing phase
print("***********************")
print("Testing decoder...")
print("***********************")

for SNR in SNRs:
	# simulate this SNR
	sigma = np.sqrt(1. / (2 * (float(K)/float(N)) * 10**(SNR/10)))
	frame_count = 0
	bit_errors = 0
	frame_errors = 0
	frame_errors_with_HDD = 0
	symbol_errors = 0
	FE = 0
	# print(SNR)
	# simulate frames
	while ((FE < min_frame_errors) or (frame_count < 100000)) and (frame_count < max_frames):
		frame_count += batch_size # use different batch size for test phase?
		print("frame_count " + str(frame_count))
		if not ALL_ZEROS_CODEWORD_TESTING:
			# generate message
			messages = torch.randint(0,2,(batch_size,K)).type(torch.LongTensor)
			data_to_encode = torch.zeros((batch_size,N)).type(torch.LongTensor)
			data_to_encode[:,data_pos] = messages

			# encode message
			codewords = torch.matmul(data_to_encode,G) % 2
			# modulate codeword
			BPSK_codewords = (0.5 - codewords) * 2.0

		if ALL_ZEROS_CODEWORD_TESTING:
			# generate all 0 codewords
			codewords = torch.zeros([N,batch_size])
			BPSK_codewords = torch.zeros([N,batch_size])

		# add Gaussian noise to codeword
		noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
		if (args.channel_type == 'AWGN'):
			channel_information = BPSK_codewords + noise
		elif (args.channel_type == 'rayleigh_fast' or args.channel_type == 'rician'):
			FastFading = False
			channel_information = apply_channel(BPSK_codewords, noise, args.channel_type, FastFading)
		elif (args.channel_type == 'bursty'):
			# add bursty noise
			p = 0.1
			sigma_bursty = 3*sigma
			temp = np.random.binomial(1,p,np.shape(noise))
			noise_bursty = np.multiply(temp,sigma_bursty*np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1]))
			channel_information = BPSK_codewords + noise + noise_bursty

		# convert channel information to LLR format
		if NO_SIGMA_SCALING_TEST:
			soft_input = channel_information
		else:
			soft_input = 2.0*channel_information/(sigma*sigma)

		# run belief propagation
		batch_data = soft_input.to(device)
		[recovered_messages,soft_out] = p_decode(batch_data)

		# update bit error count and frame error count
		errors = messages.to(device) != recovered_messages.to(device)
		# breakpoint()
		bit_errors += errors.sum()
		frame_errors += (errors.sum(0) > 0).sum()

		FE = frame_errors

	# summarize this SNR:
	print("SNR: " + str(SNR))
	print("frame count: " + str(frame_count))

	bit_count = frame_count * K
	BER = float(bit_errors) / float(bit_count)
	BERs.append(BER)
	print(BER)
	print("bit errors: " + str(bit_errors))
	print("BER: " + str(BER))

	FER = float(frame_errors) / float(frame_count)
	FERs.append(FER)
	print("FER: " + str(FER))
	print("")

# print summary
print("BERs : ")
print(BERs)
print("FERs : ")
print(FERs)