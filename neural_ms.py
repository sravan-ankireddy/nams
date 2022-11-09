# Belief propagation using Pytorch

import time
from time import sleep
import numpy as np
import numpy.matlib as mnp
import scipy.io as sio
import pickle 
import mat73
import tracemalloc
import gc

np.random.seed(0)

import pdb
import sys
from utils import load_code, syndrome, convert_dense_to_alist, apply_channel, eb_n0_to_snr, calc_sigma
import os
import argparse
import matplotlib.pyplot as plt

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-seed', type=int, default=0)
	parser.add_argument('-eb_n0_train_lo', type=float, default=15)
	parser.add_argument('-eb_n0_train_hi', type=float, default=30)
	parser.add_argument('-eb_n0_lo', type=float, default=15)
	parser.add_argument('-eb_n0_hi', type=float, default=30)
	parser.add_argument('-eb_n0_step', type=float, default=1)
	parser.add_argument('-min_frame_errors', type=int, default=5e2)
	parser.add_argument('-max_frames', type=float, default=5e5)
	parser.add_argument('-num_iterations', type=int, default=5)
	parser.add_argument('-H_filename', type=str, default='H_G_mat/BCH_63_36.alist')
	parser.add_argument('-G_filename', type=str, default='H_G_mat/G_BCH_63_36.gmat')
	parser.add_argument('-L', type=float, default=0.5)
	parser.add_argument('-steps', type=int, default=100)
	parser.add_argument('-learning_rate', type=float, default=0.01)
	parser.add_argument('-training_batch_size', type=int, default=120)
	parser.add_argument('-testing_batch_size', type=int, default=2400)
	parser.add_argument('-decoder_type', type=str, default='neural_ms')
	parser.add_argument('-force_all_zero', type=int, default=0)
	parser.add_argument('-coding_scheme', type=str, default='LDPC')
	parser.add_argument('-channel_type', type=str, default='AWGN')
	parser.add_argument('-offset', type=float, default=0)
	parser.add_argument('-norm_fac', type=float, default=1)
	parser.add_argument('-use_gpu', type=int, default=1)
	parser.add_argument('-gpu_index', type=int, default=0)
	parser.add_argument('-rand_init_weights', type=int, default=1)
	parser.add_argument('-entangle_weights', type=int, default=2)
	parser.add_argument('-save_torch_model', type=int, default=0)
	parser.add_argument('-use_saved_model', type=int, default=0)
	parser.add_argument('-quantize_weights', type=int, default=0)
	parser.add_argument('-soft_bit_loss', type=int, default=0)
	parser.add_argument('-exact_llr', type=int, default=0)
	parser.add_argument('-nn_eq', type=int, default=0)
	parser.add_argument('-training', type=int, default=1)
	parser.add_argument('-continue_training', type=int, default=0)
	parser.add_argument('-testing', type=int, default=1)
	parser.add_argument('-grid_search', type=int, default=0)
	parser.add_argument('-grid_b', type=float, default=0)
	parser.add_argument('-grid_w', type=float, default=1)
	parser.add_argument('-save_ber_to_mat', type=float, default=0)
	parser.add_argument('-use_offline_training_data', type=int, default=0)
	parser.add_argument('-adaptivity_training', type=int, default=0)
	parser.add_argument('-use_offline_testing_data', type=int, default=0)
	parser.add_argument('-saved_model_path', type=str, default='saved_models/nams_BCH_63_30_st_500_lr_0.01_AWGN_wl_0_bl_0_ent_0_nn_eq_0_max_iter_5_1_6.pt')

	# interference params
	parser.add_argument('-interf', type=int, default=0)
	parser.add_argument('-alpha', type=float, default=0)

	# weight freeze params
	parser.add_argument('-freeze_weights', type=int, default=0)
	parser.add_argument('-freeze_fraction', type=float, default=0.5)

	# params for relu
	parser.add_argument('-relu', type=int, default=0)

	# params for cv/vc model
	parser.add_argument('-cv_model', type=int, default=0)
	parser.add_argument('-vc_model', type=int, default=1)
	
	# params for clipping the grads
	parser.add_argument('-clip_grads', type=int, default=0)
	parser.add_argument('-clip', type=int, default=0.05)

	args = parser.parse_args()

	return args	

args = get_args()

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd.profiler as profiler

# starting time
start = time.time()

seed = args.seed
min_frame_errors = args.min_frame_errors
max_frames = args.max_frames
num_iterations = args.num_iterations
H_filename = args.H_filename
G_filename = args.G_filename
L = args.L
steps = args.steps
training_batch_size = args.training_batch_size
testing_batch_size = args.testing_batch_size

DEBUG = False
TRAINING = args.training
TESTING = args.testing
if (args.steps == 0 or args.use_saved_model == 1 or args.decoder_type == "undec" or args.decoder_type == "spa" or args.decoder_type == "min_sum" or args.grid_search == 1):
	TRAINING = False

np.set_printoptions(precision=10)

print("My PID: " + str(os.getpid()))
if (args.decoder_type == "undec"):
	print("Skipping the decoder")
elif (args.decoder_type == "spa"):
	print("Using Sum-Product algorithm")
elif (args.decoder_type == "min_sum"):
	print("Using Min-Sum algorithm")
else :
	print("Using Neural Min-Sum algorithm")

if args.force_all_zero:
	print("Training using only the all-zeros codeword")
else:
	print("Training using random codewords (not the all-zeros codeword)")
if args.force_all_zero:
	print("Testing using only the all-zeros codeword")
else:
	print("Testing using random codewords (not the all-zeros codeword)")

sleep(2)

if (args.relu == 1):
	print("Using relu .. ")
if (args.clip_grads == 1):
	print("Clipping grads to ", args.clip, " ..")
sleep(2)

if args.force_all_zero: 
	G_filename = ""
code = load_code(H_filename, G_filename)

H = code.H
G = code.G

var_degrees = code.var_degrees
chk_degrees = code.chk_degrees
num_edges = code.num_edges
u = code.u
d = code.d
n = code.n
m = code.m
k = code.k

edges = code.edges
edges_m = code.edges_m
edge_order_vc = code.edge_order_vc
extrinsic_edges_vc = code.extrinsic_edges_vc
edge_order_cv = code.edge_order_cv
extrinsic_edges_cv = code.extrinsic_edges_cv

if (args.freeze_weights == 1 and (args.entangle_weights == 0 or args.entangle_weights == 3)):
	# specify the trainable edges 
	edges_full = np.arange(num_edges)
	# FIX ME
	edges_train = []
	num_edges_train = round((1-args.freeze_fraction)*18)
	for s in range(num_edges):
		if (s%18 < num_edges_train):
			edges_train.append(s)
	# breakpoint()	 
	# np.random.shuffle(edges_train)
	# num_edges_train = round((1-args.freeze_fraction)*num_edges)
	# edges_train = edges_train[0:num_edges_train]
	edges_freeze = []
	for s in edges_full:
		if s not in edges_train:
			edges_freeze.append(s)
# breakpoint()
new_order_cv = np.zeros(num_edges).astype(int)
new_order_cv[edge_order_cv] = np.array(range(0,num_edges)).astype(int)

new_order_vc = np.zeros(num_edges).astype(int)
new_order_vc[edge_order_vc] = np.array(range(0,num_edges)).astype(int)

if (args.use_gpu == 0):
	device = torch.device("cpu")
else:
	cuda_str = "cuda:" + str(args.gpu_index)
	device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
print("Now running using .... " + str(device))

if (args.adaptivity_training == 1):
	args.continue_training = 0

# Chosse the appropriate data_folders
models_folder = "saved_models_sf_df_lte"
# results_folder = "ber_data_sf_df_lte"
results_folder = "ber_data_icc"
# results_folder = "ber_data_icc_from_AWGN"

if (args.cv_model == 0 and args.vc_model == 1):
	models_folder = "saved_models_vc"
	results_folder = "ber_data_vc"

if not os.path.exists(models_folder):
    os.makedirs(models_folder)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    
# results_folder = "ber_data_debug"

if (args.decoder_type == "neural_ms"):
	if (args.training == 1):
		print("Going to save models at : ", models_folder)

	if (args.training == 1 and args.continue_training == 1):
		print("Going to contnue training models at : ", args.saved_model_path)

	if (args.testing == 1):
		print("Using saved models from : ", args.saved_model_path)
		print("Going to save results at : ", results_folder)
sleep(2)

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
		if(args.entangle_weights == 0):
			num_w1 = num_iterations
			num_w2 = num_edges				
		elif(args.entangle_weights == 1):
			num_w1 = 1
			num_w2 = num_edges
		elif(args.entangle_weights == 2):
			num_w1 = 1
			num_w2 = n
		elif(args.entangle_weights == 3):
			num_w1 = 1
			num_w2 = n - k
		# for cyclic codes, repeating the wieghts for check nodes
		elif(args.entangle_weights == 4):
			num_w1 = 1
			num_w2 = chk_degrees[0]
		elif(args.entangle_weights == 5):
			num_w1 = num_iterations
			num_w2 = chk_degrees[0]
		else:
			num_w1 = 1
			num_w2 = 1
		var_W = 1
		var_B = 1

		# generate the weight init vectors : CV
		if(args.cv_model == 1 and args.vc_model == 0):
			B_cv_init = torch.fmod(torch.randn([num_w1, num_w2]),2*var_B)
			W_cv_init = torch.fmod(torch.randn([num_w1, num_w2]),2*var_W)

			if (args.nn_eq == 0):
				self.B_cv = torch.nn.Parameter(B_cv_init)
				self.W_cv = torch.nn.Parameter(W_cv_init)
			elif (args.nn_eq == 1):
				self.B_cv = torch.zeros(num_w1, num_w2).to(device)
				self.W_cv = torch.nn.Parameter(W_cv_init)
			elif (args.nn_eq == 2):
				self.B_cv = torch.nn.Parameter(B_cv_init)
				self.W_cv = torch.ones(num_w1, num_w2).to(device)
			
		# generate the weight init vectors : VC
		if(args.cv_model == 0 and args.vc_model == 1):
			B_vc_init = torch.fmod(torch.randn([num_w1, num_w2]),2*var_B)
			W_vc_init = torch.fmod(torch.randn([num_w1, num_w2]),2*var_W)
			W_ch_init = torch.fmod(torch.randn([1, n]),2*var_W)

			self.B_vc = torch.nn.Parameter(B_vc_init)
			self.W_vc = torch.nn.Parameter(W_vc_init)
			# self.W_ch = torch.nn.Parameter(W_ch_init)
			self.W_ch = torch.ones(1, n).to(device)

			if (args.nn_eq == 0):
				self.B_vc = torch.nn.Parameter(B_vc_init)
				self.W_vc = torch.nn.Parameter(W_vc_init)
			elif (args.nn_eq == 1):
				self.B_vc = torch.nn.Parameter(torch.zeros(num_w1, num_w2))
				self.W_vc = torch.nn.Parameter(W_vc_init)
			elif (args.nn_eq == 2):
				self.B_vc = torch.nn.Parameter(B_vc_init)
				self.W_vc = torch.nn.Parameter(torch.ones(num_w1, num_w2))

model = NeuralNetwork().to(device)

# compute messages from variable nodes to check nodes
def compute_vc(cv, soft_input, iteration, batch_size):
	weighted_soft_input = soft_input		
	reordered_soft_input = weighted_soft_input[edges,:]

	# vc = torch.tensor([]).to(device)
	vc = []
	# for each variable node v, find the list of extrinsic edges
	# fetch the LLR values from cv using these indices
	# find the sum and store in temp, finally store these in vc
	count_vc = 0
	for i in range(0, n): 
		for j in range(0, var_degrees[i]):
			# if the list of extrinsic edges is not empty, add them up
			if extrinsic_edges_vc[count_vc]:
				temp = cv[extrinsic_edges_vc[count_vc],:]
				temp = torch.sum(temp,0)
			else:
				temp = torch.zeros([batch_size])
			vc.append(temp.to(device))
			count_vc = count_vc + 1
	vc = torch.stack(vc)
	vc = vc[new_order_vc,:]

	# apply the weights
	# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
	if (args.cv_model == 0 and args.vc_model == 1):
		if (args.entangle_weights == 0 or args.entangle_weights == 5):
			idx = iteration
		else:
			idx = 0

		# Replicate the weights by repeating for each column : edges are stores column by column i.e, 
  		# for each var node, count all edges and move to next var node
		if (args.entangle_weights == 2):
			B_vc_vec = torch.tensor([]).to(device)
			W_vc_vec = torch.tensor([]).to(device)
			for im in range(n):
				deg = var_degrees[im]
				B_vc_m = model.B_vc[0,im]
				W_vc_m = model.W_vc[0,im]
				B_vc_vec = torch.cat((B_vc_vec, B_vc_m.repeat([1,deg])),1)
				W_vc_vec = torch.cat((W_vc_vec, W_vc_m.repeat([1,deg])),1)
    
		# Replicate the weights by repeating for each row : FIX ME, incorrect : need to read in a different fashion
		elif (args.entangle_weights == 3):
			B_vc_vec = torch.tensor([]).to(device)
			W_vc_vec = torch.tensor([]).to(device)
			for im in range(m):
				deg = chk_degrees[im]
				B_vc_m = model.B_vc[0,im]
				W_vc_m = model.W_vc[0,im]
				B_vc_vec = torch.cat((B_vc_vec, B_vc_m.repeat([1,deg])),1)
				W_vc_vec = torch.cat((W_vc_vec, W_vc_m.repeat([1,deg])),1)

		# Replicate the weights by repeating the entire vec for each row : FIX ME : for warp cases
		elif (args.entangle_weights == 4 or args.entangle_weights == 5):
			B_vc_vec = model.B_vc.repeat([1,m])
			W_vc_vec = model.W_vc.repeat([1,m])

		# Replicate same weight for all edges	
		elif (args.entangle_weights == 6):
			B_vc_vec = model.B_vc.repeat([1,num_edges])
			W_vc_vec = model.W_vc.repeat([1,num_edges])		
		else:
			B_vc_vec = model.B_vc
			W_vc_vec = model.W_vc
			
		# Replicate the offsets and scaling matrix across batch size
		offsets = torch.tile(torch.reshape(B_vc_vec[idx],[-1,1]),[1,batch_size]).to(device)
		scaling = torch.tile(torch.reshape(W_vc_vec[idx],[-1,1]),[1,batch_size]).to(device)

		if 0:
			vc = scaling * torch.nn.functional.relu(vc.to(device) - offsets) + reordered_soft_input
		else:
			vc = scaling * (vc.to(device) - offsets) + reordered_soft_input
	else:
		vc = vc.to(device) + reordered_soft_input
	return vc

# compute messages from check nodes to variable nodes
def compute_cv(vc, iteration, batch_size):
	cv_list = torch.tensor([]).to(device)
	prod_list = torch.tensor([]).to(device)
	min_list = torch.tensor([]).to(device)

	if (args.decoder_type == "spa"):
		vc = torch.clip(vc, -10, 10)
		tanh_vc = torch.tanh(vc / 2.0)
	count_cv = 0
	for i in range(0, m): # for each check node c
		for j in range(0, chk_degrees[i]): #edges per check node
			if (args.decoder_type == "spa"):
				temp = tanh_vc[extrinsic_edges_cv[count_cv],:]
				temp = torch.prod(temp,0)
				temp = torch.log((1+temp)/(1-temp))
				cv_list = torch.cat((cv_list,temp.float()),0)
			elif (args.decoder_type == "min_sum" or args.decoder_type == "neural_ms"):
				if extrinsic_edges_cv[count_cv]:
					temp = vc[extrinsic_edges_cv[count_cv],:]
				else:
					temp = torch.zeros([1,batch_size]).to(device)
				prod_chk_temp = torch.prod(torch.sign(temp),0)
				(sign_chk_temp, min_ind) = torch.min(torch.abs(temp),0)
				prod_list = torch.cat((prod_list,prod_chk_temp.float()),0)
				min_list = torch.cat((min_list,sign_chk_temp.float()),0)
			count_cv = count_cv + 1

	if (args.decoder_type == "spa"):
		cv = torch.reshape(cv_list,vc.size())
	elif (args.decoder_type == "min_sum"):
		prods = torch.reshape(prod_list,vc.size()) #stack across batch size
		mins = torch.reshape(min_list,vc.size())
		cv = prods * mins
	elif (args.decoder_type == "neural_ms"):
		prods = torch.reshape(prod_list,vc.size()) #stack across batch size
		mins = torch.reshape(min_list,vc.size())

		# apply the weights
		# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
		if (args.cv_model == 1 and args.vc_model == 0):
			if (args.entangle_weights == 0 or args.entangle_weights == 5):
				idx = iteration
			else:
				idx = 0
			
			# Replicate the weights by repeating for each column
			if (args.entangle_weights == 2):
				B_cv_vec = torch.tensor([]).to(device)
				W_cv_vec = torch.tensor([]).to(device)
				for im in range(n):
					deg = var_degrees[im]
					B_cv_m = model.B_cv[0,im]
					W_cv_m = model.W_cv[0,im]
					B_cv_vec = torch.cat((B_cv_vec, B_cv_m.repeat([1,deg])),1)
					W_cv_vec = torch.cat((W_cv_vec, W_cv_m.repeat([1,deg])),1)

			# Replicate the weights by repeating for each row
			elif (args.entangle_weights == 3):
				B_cv_vec = torch.tensor([]).to(device)
				W_cv_vec = torch.tensor([]).to(device)
				for im in range(m):
					deg = chk_degrees[im]
					B_cv_m = model.B_cv[0,im]
					W_cv_m = model.W_cv[0,im]
					B_cv_vec = torch.cat((B_cv_vec, B_cv_m.repeat([1,deg])),1)
					W_cv_vec = torch.cat((W_cv_vec, W_cv_m.repeat([1,deg])),1)

			# Replicate the weights by repeating the entire vec for each row : FIX ME : for warp cases
			elif (args.entangle_weights == 4 or args.entangle_weights == 5):
				B_cv_vec = model.B_cv.repeat([1,m])
				W_cv_vec = model.W_cv.repeat([1,m])

			# Replicate same weight for all edges	
			elif (args.entangle_weights == 6):
				B_cv_vec = model.B_cv.repeat([1,num_edges])
				W_cv_vec = model.W_cv.repeat([1,num_edges])		
			else:
				B_cv_vec = model.B_cv
				W_cv_vec = model.W_cv

			# Replicate the offsets and scaling matrix across batch size
			offsets = torch.tile(torch.reshape(B_cv_vec[idx],[-1,1]),[1,batch_size]).to(device)
			scaling = torch.tile(torch.reshape(W_cv_vec[idx],[-1,1]),[1,batch_size]).to(device)
			if (args.relu == 1):
				cv = scaling * prods * torch.nn.functional.relu(mins - offsets)
			else:
				cv = scaling * prods * (mins - offsets)
		else:
			cv = prods * mins
	cv = cv[new_order_cv,:]
	return cv

# combine messages to get posterior LLRs
def marginalize(soft_input, cv, batch_size):
	weighted_soft_input = soft_input

	soft_output =  torch.tensor([]).to(device) 
	for i in range(0,n):
		temp = cv[edges_m[i],:]
		temp = torch.sum(temp,0).to(device) 
		soft_output = torch.cat((soft_output,temp),0)
	soft_output = torch.reshape(soft_output,soft_input.size())
	if 0:
		soft_output = torch.tile(torch.reshape(model.W_ch,[-1,1]),[1,batch_size])*weighted_soft_input + soft_output
	else:
		soft_output = weighted_soft_input + soft_output

	return soft_output

def belief_propagation_iteration(soft_input, iteration, cv, m_t, batch_size):
	# breakpoint()
	# compute vc
	vc = compute_vc(cv, soft_input, iteration, batch_size)
	vc_prime = vc
	# breakpoint()
	# # compute cv
	# tracemalloc.start()
	cv = compute_cv(vc_prime, iteration, batch_size)
	# snapshot = tracemalloc.take_snapshot()
	# top_stats = snapshot.statistics('lineno')
	# print("[ Top 10 ]")
	# for stat in top_stats[:10]:
	# 	print(stat)
	# breakpoint()
	soft_output = marginalize(soft_input, cv, batch_size)
	iteration += 1

	return soft_input, soft_output, iteration, cv, m_t

starter_learning_rate = args.learning_rate
learning_rate = starter_learning_rate 
print("\n\nDecoder type: " + args.decoder_type + "\n\n")

def nn_decode(soft_input,batch_size, batch_labels):
	soft_output = soft_input
	iteration = 0
	cv = torch.zeros([num_edges,batch_size])
	m_t = torch.zeros([num_edges,batch_size])
	loss = 0
	criterion = nn.BCEWithLogitsLoss(reduction='mean')
	while ( iteration < num_iterations ):
		[soft_input, soft_output, iteration, cv, m_t] = belief_propagation_iteration(soft_input, iteration, cv, m_t, batch_size)
		loss = loss + criterion(soft_output,batch_labels.float())
		# FIX ME: Add a stopping criteria based on H*c = 0? for whole batch?
	return soft_output, loss

def nn_decode_test(soft_input,batch_size):
	soft_output = soft_input
	iteration = 0
	cv = torch.zeros([num_edges,batch_size])
	m_t = torch.zeros([num_edges,batch_size])

	while ( iteration < num_iterations ):
		[soft_input, soft_output, iteration, cv, m_t] = belief_propagation_iteration(soft_input, iteration, cv, m_t, batch_size)

	return soft_output

rate = float(k)/float(n)
mod_bits = 1

eb_n0_dB = np.arange(args.eb_n0_train_lo, args.eb_n0_train_hi+args.eb_n0_step, args.eb_n0_step)

SNRs = eb_n0_to_snr(eb_n0_dB,rate,mod_bits)
if (args.decoder_type == "neural_ms" and args.decoder_type == 1):
	if (training_batch_size % len(SNRs)) != 0:
		print("********************")
		print("********************")
		print("error: Training batch size must divide by the number of SNRs to train on")
		print("********************")
		print("********************")
BERs = []
SERs = []
FERs = []
loss_history = []

if (args.adaptivity_training == 1):
	# find the base channel 
	channel_list = ["AWGN", "bursty", "EPA", "EVA", "ETU", "OTA"]
	base_channel = [s for s in channel_list if s in args.saved_model_path ]
	# breakpoint()
	base_channel = base_channel[0]
	
if TRAINING :

	# Training loop
	print("***********************")
	print("Training decoder using " + str(steps) + " minibatches with batch_size " + str(training_batch_size)+ " ... from SNR " + str(int(args.eb_n0_train_lo)) + " to " + str(int(args.eb_n0_train_hi)))
	print("***********************")

	if (args.cv_model == 1 and args.vc_model == 0):
		if(args.nn_eq == 0):
			optimizer = optim.Adam([model.B_cv, model.W_cv], lr = learning_rate)
		elif(args.nn_eq == 1):
			optimizer = optim.Adam([model.W_cv], lr = learning_rate)
		elif(args.nn_eq == 2):
			optimizer = optim.Adam([model.B_cv], lr = learning_rate)
	
	if (args.cv_model == 0 and args.vc_model == 1):
		if(args.nn_eq == 0):
			optimizer = optim.Adam([model.B_vc, model.W_vc], lr = learning_rate)
		elif(args.nn_eq == 1):
			optimizer = optim.Adam([model.W_vc], lr = learning_rate)
		elif(args.nn_eq == 2):
			optimizer = optim.Adam([model.B_vc], lr = learning_rate)
	
	if (args.cv_model == 1 and args.vc_model == 1):
		if(args.nn_eq == 0):
			optimizer = optim.Adam([model.B_cv, model.W_cv, model.B_vc, model.W_vc], lr = learning_rate)
		elif(args.nn_eq == 1):
			optimizer = optim.Adam([model.W_cv, model.W_vc], lr = learning_rate)
		elif(args.nn_eq == 2):
			optimizer = optim.Adam([[model.B_cv, model.B_vc]], lr = learning_rate)

	scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

	if (args.use_offline_training_data == 1):
		lte_data_filename = "generate_lte_data/lte_data/" + args.coding_scheme + "_" + str(n) + "_" + str(k) + "_"  + str(args.channel_type) + "_data_train_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
		data = mat73.loadmat(lte_data_filename)
		enc_data = torch.tensor(data['enc'])
		llr_data = torch.tensor( data['llr'])

		# # Adding singleton dimension for format matching
		# if (enc_data.dim() == 2):
		# 	enc_data = torch.unsqueeze(enc_data, 2)
		# 	llr_data = torch.unsqueeze(llr_data, 2)
		
	# if adapting, load the previous model for further training
	if (args.adaptivity_training == 1 or args.continue_training == 1):
		print("\n\n********* Loading the saved model for further training *********\n\n",args.saved_model_path,"\n\n")
		sleep(2)
		model.load_state_dict(torch.load(args.saved_model_path))
		# reinitialize the optimizer since the model is reloaded
		if (args.cv_model == 1 and args.vc_model == 0):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_cv, model.W_cv], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_cv], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([model.B_cv ], lr = learning_rate)
		
		if (args.cv_model == 0 and args.vc_model == 1):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_vc, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([model.B_vc], lr = learning_rate)
		
		if (args.cv_model == 1 and args.vc_model == 1):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_cv, model.W_cv, model.B_vc, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_cv, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([[model.B_cv, model.B_vc]], lr = learning_rate)
		scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

	step = 0
	if (args.continue_training == 1):
		step = 19000+1
	while step < steps:
		# data selection/generation
		if (args.use_offline_training_data == 1):
			start_idx = int(step*training_batch_size/len(SNRs))
			end_idx = int(start_idx + training_batch_size/len(SNRs))
			snr_start_ind = int(args.eb_n0_train_lo)-int(args.eb_n0_train_lo)
			snr_end_ind = int(args.eb_n0_train_hi)-int(args.eb_n0_train_lo)+1
			codewords = enc_data[:,start_idx:end_idx,snr_start_ind:snr_end_ind].to(device)
			llr_in = llr_data[:,start_idx:end_idx,snr_start_ind:snr_end_ind].to(device)

			# concatenate the data along the SNR dimension
			codewords = torch.reshape(codewords,[codewords.size(0),-1])
			llr_in = torch.reshape(llr_in,[llr_in.size(0),-1])
		else:
			if (args.force_all_zero == 0):
				messages = torch.randint(0,2,(training_batch_size,k))
				codewords = torch.matmul(torch.tensor(G,dtype=int), torch.transpose(messages,0,1)) % 2
				# generate interference data : y_i = x_i + \alpha*x_i-1 + noise
				codewords_interf = codewords
				codewords_interf[:,0] = 0
				codewords_interf[:,:-1] = codewords[:,1:]
			else:
				messages = torch.zeros([k,training_batch_size])
				codewords = torch.zeros([n,training_batch_size])
				codewords_interf = torch.zeros([n,training_batch_size])

			BPSK_codewords = (codewords - 0.5) * 2.0
			# generate interference data : y_i = x_i + \alpha*x_i-1 + noise
			BPSK_codewords_interf = torch.zeros_like(BPSK_codewords)
			BPSK_codewords_interf[1:,:] = BPSK_codewords[:-1,:]
			soft_input = torch.zeros_like(BPSK_codewords)
			received_codewords = torch.zeros_like(BPSK_codewords)
			sigma_vec = torch.zeros_like(BPSK_codewords)
			SNR_vec = torch.zeros_like(BPSK_codewords)
			llr_in = torch.zeros_like(BPSK_codewords)
			
			# create minibatch with codewords from multiple SNRs
			for i in range(0,len(SNRs)):
				sigma = calc_sigma(SNRs[i],rate)
				noise = sigma * np.random.randn(n,training_batch_size//len(SNRs))
				start_idx = training_batch_size*i//len(SNRs)
				end_idx = training_batch_size*(i+1)//len(SNRs)
				
				# Apply channel and noise
				exact_llr = args.exact_llr
				FastFading = False
				if (args.interf == 1):
					BPSK_codewords[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + args.alpha*BPSK_codewords_interf[:,start_idx:end_idx]
				received_codewords[:,start_idx:end_idx], soft_input[:,start_idx:end_idx] = apply_channel(BPSK_codewords[:,start_idx:end_idx], sigma, noise, args.channel_type, FastFading, exact_llr)
				llr_in[:,start_idx:end_idx] = 2*received_codewords[:,start_idx:end_idx]/(sigma**2)
				sigma_vec[:,start_idx:end_idx] = sigma
				SNR_vec[:,start_idx:end_idx] = SNRs[i]

				llr_in = llr_in.to(device)
				codewords = codewords.to(device)

		# training starts
		# perform gradient update for whole snr batch
		soft_output, batch_loss = nn_decode(llr_in,training_batch_size,codewords)
		optimizer.zero_grad()
		batch_loss.backward()

		if (args.clip_grads == 1):
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
		optimizer.step()

		if step % 10 == 0:
			loss_history.append(batch_loss.item())
			print(str(step) + " minibatches completed")
			print ("BCE loss :")
			print (batch_loss)
			if (args.cv_model == 1 and args.vc_model == 0):
				print ("B_cv : ")
				print (model.B_cv)
				print ("W_cv : ")
				print (model.W_cv)
			if (args.cv_model == 0 and args.vc_model == 1):
				print ("B_vc : ")
				print (model.B_vc)
				print ("W_vc : ")
				print (model.W_vc)
				print ("W_ch : ")
				print (model.W_ch)

		saver_step = 500
		if (args.adaptivity_training == 1):
			saver_step = 500
		if (args.save_torch_model == 1 and step % saver_step == 0):
			# store the weights
			print("\n********* Saving the Training weights *********\n")
			models_folder_int = models_folder + "/intermediate_models/"
			models_folder_mat = models_folder + "/model_weights_mat/"
   
			if not os.path.exists(models_folder_int):
				os.makedirs(models_folder_int)
			if not os.path.exists(models_folder_mat):
				os.makedirs(models_folder_mat)
       
			if (args.adaptivity_training == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			else:
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
				
			if (args.freeze_weights == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".pt"
			
			if (args.interf == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_alpha_" + str(args.alpha) + ".pt"
			
			torch.save(model.state_dict(), filename)
			# save to matlab for analysis
			if (args.cv_model == 1 and args.vc_model == 0):
				B_cv = model.B_cv.cpu().data.numpy()
				W_cv = model.W_cv.cpu().data.numpy()
				weights = {'B_cv':B_cv, 'W_cv':W_cv}
			elif (args.cv_model == 0 and args.vc_model == 1):
				B_vc = model.B_vc.cpu().data.numpy()
				W_vc = model.W_vc.cpu().data.numpy()
				W_ch = model.W_ch.cpu().data.numpy()
				weights = {'B_vc':B_vc, 'W_vc':W_vc, 'W_ch':W_ch}
			elif (args.cv_model == 1 and args.vc_model == 1):
				B_cv = model.B_cv.cpu().data.numpy()
				W_cv = model.W_cv.cpu().data.numpy()
				B_vc = model.B_vc.cpu().data.numpy()
				W_vc = model.W_vc.cpu().data.numpy()
				W_ch = model.W_ch.cpu().data.numpy()
				weights = {'B_cv':B_cv, 'W_cv':W_cv, 'B_vc':B_vc, 'W_vc':W_vc, 'W_ch':W_ch}
			if (args.adaptivity_training == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			else:
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			if (args.freeze_weights == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".mat"
			sio.savemat(filename_mat,weights)
		# breakpoint()
		step += 1

	print("Trained decoder on " + str(step) + " minibatches.\n")
	if (step > 0):
		print ("final prints")
		print (batch_loss)
		if (args.cv_model == 1 and args.vc_model == 0):
			print ("B_cv : ")
			print (model.B_cv)
			print ("W_cv : ")
			print (model.W_cv)
		if (args.cv_model == 0 and args.vc_model == 1):
			print ("B_vc : ")
			print (model.B_vc)
			print ("W_vc : ")
			print (model.W_vc)
			print ("W_ch : ")
			print (model.W_ch)

		if (args.save_torch_model == 1):
			# save in intermediate folder
			if (args.adaptivity_training == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			else:
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			
			if (args.freeze_weights == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".pt"
			
			if (args.interf == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_alpha_" + str(args.alpha) + ".pt"
			
			torch.save(model.state_dict(), filename) 
			# save in main folder
			if (args.adaptivity_training == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			else:
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			if (args.freeze_weights == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".pt"
			
			if (args.interf == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_alpha_" + str(args.alpha) + ".pt"
			
			torch.save(model.state_dict(), filename)
			# save to matlab for analysis
			if (args.cv_model == 1 and args.vc_model == 0):
				B_cv_temp = model.B_cv.cpu().data.numpy()
				W_cv_temp = model.W_cv.cpu().data.numpy()
				weights_temp = {'B_cv':B_cv_temp, 'W_cv':W_cv_temp}
			elif (args.cv_model == 0 and args.vc_model == 1):
				B_vc_temp = model.B_vc.cpu().data.numpy()
				W_vc_temp = model.W_vc.cpu().data.numpy()
				W_ch_temp = model.W_ch.cpu().data.numpy()
				weights_temp = {'B_vc':B_vc_temp, 'W_vc':W_vc_temp, 'W_ch':W_ch_temp}
			elif (args.cv_model == 1 and args.vc_model == 1):
				B_cv_temp = model.B_cv.cpu().data.numpy()
				W_cv_temp = model.W_cv.cpu().data.numpy()
				B_vc_temp = model.B_vc.cpu().data.numpy()
				W_vc_temp = model.W_vc.cpu().data.numpy()
				W_ch_temp = model.W_ch.cpu().data.numpy()
				weights_temp = {'B_cv':B_cv_temp, 'W_cv':W_cv_temp, 'B_vc':B_vc_temp, 'W_vc':W_vc_temp, 'W_ch':W_ch_temp}
			if (args.adaptivity_training == 1):
				filename_mat = models_folder + "/model_weights_mat_final/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			else:
				filename_mat = models_folder + "/model_weights_mat_final/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			if (args.freeze_weights == 1):
				filename_mat = models_folder + "/model_weights_mat_final/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".mat"
			sio.savemat(filename_mat,weights_temp)
			if (args.adaptivity_training == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			else:
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			if (args.freeze_weights == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".mat"
			sio.savemat(filename_mat,weights_temp)

eb_n0_dB = np.arange(args.eb_n0_lo, args.eb_n0_hi+args.eb_n0_step, args.eb_n0_step)

SNRs = eb_n0_to_snr(eb_n0_dB,rate,mod_bits)

if TESTING :
	# Testing phase
	print("***********************")
	print("Testing decoder on a max. of " + str(int(max_frames)) + "  frames from SNR : " + str(int(args.eb_n0_lo)) + " to " + str(int(args.eb_n0_hi)) + " for " + str(args.channel_type) + " channel")
	print("***********************")

	if (args.use_saved_model == 1):
		print("\n********* Using saved model for direct inference *********\n")
		print(args.saved_model_path)
		sleep(2)
		model.load_state_dict(torch.load(args.saved_model_path))

		if (args.quantize_weights == 1):
			model.W_cv = torch.nn.Parameter(torch.round(10*model.W_cv)/10)
			model.B_cv = torch.nn.Parameter(torch.round(10*model.B_cv)/10)
			print ("W_cv : ")
			print (model.W_cv)

	# use grid search weights for brute forcing
	if (args.grid_search == 1):
		model.B_cv.requires_grad = False
		model.W_cv.requires_grad = False
		model.B_cv[0] = args.grid_b
		model.W_cv[0] = args.grid_w
	# load offline data
	if (args.use_offline_testing_data == 1):
		lte_data_filename = "generate_lte_data/lte_data/" + args.coding_scheme + "_" + str(n) + "_" + str(k) + "_"  + str(args.channel_type) + "_data_test_" + str(int(args.eb_n0_lo)) + "_" + str(int(args.eb_n0_hi)) + ".mat"
		print("Using saved data from : ", lte_data_filename)

		data = mat73.loadmat(lte_data_filename)

		# # Adding singleton dimension for format matching
		# if (enc_data.dim() == 2):
		# 	enc_data = torch.unsqueeze(enc_data, 2)
		# 	llr_data = torch.unsqueeze(llr_data, 2)
		# 	SNRs = np.arange(args.eb_n0_hi, args.eb_n0_hi+args.eb_n0_step, args.eb_n0_step)

	for SNR in SNRs:
		# simulate this SNR
		############ FIX ME###########
		sigma = calc_sigma(SNR,rate)
		frame_count = 0
		bit_errors = 0
		frame_errors = 0
		frame_errors_with_HDD = 0
		symbol_errors = 0
		FE = 0

		# load data corresponding to this SNR
		if (args.use_offline_testing_data == 1):
			# map snr to ind
			snr_ind = int(SNR)-int(args.eb_n0_lo)

			enc_data = torch.tensor(data['enc'][:,:,snr_ind]).to(device)
			llr_data = torch.tensor(data['llr'][:,:,snr_ind]).to(device)

			# cap max_frames based on size of offline testing dataset
			num_frames = np.shape(enc_data)[1]
			max_frames = min(num_frames - testing_batch_size, max_frames)

		# simulate frames
		while ((FE < min_frame_errors) or (frame_count <300000)) and (frame_count < max_frames) :
			frame_count += testing_batch_size

			if (args.use_offline_testing_data == 1):
				start_ind = frame_count-testing_batch_size
				end_ind = frame_count

				# codewords = enc_data[:,start_ind:end_ind]
				# batch_data = lr_data[:,start_ind:end_ind]

				# breakpoint()
				if (args.decoder_type == "undec"):
					llr_out = (llr_in > 0)
				else:
					llr_out = nn_decode_test(llr_data[:,start_ind:end_ind],testing_batch_size)
				received_codewords = (llr_out > 0)
				# update bit error count and frame error count
				errors = enc_data[:,start_ind:end_ind] != received_codewords
				bit_errors += errors.sum()
				frame_errors += (errors.sum(0) > 0).sum()
				FE = frame_errors

			else: 
				if not args.force_all_zero:
					messages = torch.randint(0,2,(testing_batch_size,k))
					codewords = torch.matmul(torch.tensor(G,dtype=int), torch.transpose(messages,0,1)) % 2

				if args.force_all_zero:
					codewords = torch.zeros([n,testing_batch_size])
					codewords_interf = torch.zeros([n,testing_batch_size])

				# bpsk modulation
				BPSK_codewords = (codewords - 0.5) * 2.0
				# generate interference data : y_i = x_i + \alpha*x_i-1 + noise
				BPSK_codewords_interf = torch.zeros_like(BPSK_codewords)
				BPSK_codewords_interf[1:,:] = BPSK_codewords[:-1,:]
				sigma_vec = sigma*torch.ones_like(BPSK_codewords)

				# Pass through channel
				noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
				FastFading = False
				exact_llr = args.exact_llr
				if (args.interf == 1):
					BPSK_codewords = BPSK_codewords + args.alpha*BPSK_codewords_interf
				received_codewords, soft_input = apply_channel(BPSK_codewords, sigma, noise, args.channel_type, FastFading, exact_llr)
				llr_in = 2.0*received_codewords/(sigma*sigma)

				# Phase 2 : decode using wbp
				batch_data = torch.reshape(llr_in,(received_codewords.size(dim=0),received_codewords.size(dim=1))).to(device)
				if (args.decoder_type == "undec"):
					llr_out = (llr_in > 0).to(device)
				else:
					llr_out, loss_out = nn_decode(batch_data,testing_batch_size,codewords.to(device))
				received_codewords = (llr_out > 0)

				# update bit error count and frame error count
				errors = codewords.to(device) != received_codewords
				bit_errors += errors.sum()
				frame_errors += (errors.sum(0) > 0).sum()

				FE = frame_errors

		# summarize this SNR:
		print("SNR: " + str(SNR))
		print("frame count: " + str(frame_count))

		bit_count = frame_count * n
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
	print("Loss : ")
	loss_history = torch.tensor(loss_history).numpy()
	print(loss_history)

	BERs_prefix = "BERs_" + str(args.decoder_type)
	FERs_prefix = "FERs_" + str(args.decoder_type)
	command = "% **Decoder : " + str(args.relu) + " " + str(args.decoder_type) + " coding_scheme " + str(args.coding_scheme) + " H_file " + str(H_filename) + " channel_type " + str(args.channel_type) + " device " + str(device) + " force_all_zero " + str(args.force_all_zero) + " use_gpu " + str(args.use_gpu) + "\n"
	if (args.decoder_type == "neural_ms"):
		BERs_prefix = BERs_prefix + "_ent" + str(args.entangle_weights) + "_" + str(args.steps)
		FERs_prefix = FERs_prefix + "_ent" + str(args.entangle_weights) + "_" + str(args.steps)
		command = "% **Decoder : " + str(args.decoder_type) + str(args.learning_rate) + " coding_scheme " + str(args.coding_scheme) + " H_file " + str(H_filename) + " channel_type " + str(args.channel_type) + " device " + str(device) + " rand_init_weights " + str(args.rand_init_weights) + " force_all_zero " + str(args.force_all_zero) + " steps " + str(steps) + " training_batch_size " + str(training_batch_size) + " use_gpu " + str(args.use_gpu) + "\n"

	out_file = open('master_out_ms_mod.txt', 'a')
	out_file.write(command)
	out_file.write(BERs_prefix + " = [")
	for element in BERs:
		out_file.write(str(element) + "\t")
	out_file.write(" ];\n")
	out_file.write(FERs_prefix + " = [")
	for element in FERs:
		out_file.write(str(element) + "\t")
	out_file.write(" ];\n\n")
	
	# if (args.decoder_type == "neural_ms"):
	# 	out_file.write("Loss_" + str(args.steps) + " = [")
	# 	for element in loss_history:
	# 		out_file.write(str(element) + "\t")
	# 	out_file.write(" ];\n\n")
	out_file.close()

if (args.channel_type == 'rayleigh_fast'):
	chan_name =  args.channel_type + "_" + str(args.exact_llr)
else:
	chan_name = args.channel_type
# save BERs from grid search
if (args.grid_search == 1):
	ind_var = 0
	res_file_name = results_folder + "/BERs_"+str(args.coding_scheme)+"_"+chan_name+"_"+str(n)+"_"+str(k)+ "_SNR_"+ str(int(args.eb_n0_lo))+"_"+str(int(args.eb_n0_hi))+".mat"
	if os.path.exists(res_file_name):
		prev_data = mat73.loadmat(res_file_name)
	else:
		ind_var = 1
		prev_data = np.zeros((1,len(SNRs)+2))
		prev_data[0,0] = -100
		prev_data[0,1] = -100
		prev_data[0,2:] = SNRs 
	temp = np.zeros((1,len(SNRs)+2))
	temp[0,0] = args.grid_b
	temp[0,1] = args.grid_w
	temp[0,2:] = BERs

	if (ind_var == 0):
		cur_data = np.concatenate([prev_data['BERs_data'],temp])
	else:
		cur_data = np.concatenate([prev_data,temp])
	BERs_data = {'BERs_data':cur_data}
	sio.savemat(res_file_name, BERs_data)

# save BERs for each code per channel
if (args.save_ber_to_mat == 1):
	ind_var = 0
	chan_name = str(args.channel_type)
	if (args.adaptivity_training == 1):
		res_file_name = results_folder + "/BERs_"+str(args.coding_scheme)+"_"+chan_name+"_"+str(n)+"_"+str(k)+"_adapt_from_" + base_channel + "_nn_eq_" + str(args.nn_eq) + "_lr_" + str(args.learning_rate) + "_SNR_"+ str(int(args.eb_n0_lo))+"_"+str(int(args.eb_n0_hi))+".mat"
	else:
		res_file_name = results_folder + "/BERs_"+str(args.coding_scheme)+"_"+chan_name+"_"+str(n)+"_"+str(k)+ "_baseline_nn_eq_" + str(args.nn_eq) + "_lr_" + str(args.learning_rate) +  "_SNR_"+ str(int(args.eb_n0_lo))+"_"+str(int(args.eb_n0_hi))+".mat"

	if os.path.exists(res_file_name):
		prev_data = sio.loadmat(res_file_name)
	else:
		ind_var = 1
		prev_data = np.zeros((1,len(SNRs)+3))
		prev_data[0,0] = -100
		prev_data[0,1] = -100
		prev_data[0,2] = -100
		prev_data[0,3:] = SNRs 
	temp = np.zeros((1,len(SNRs)+3))
	temp = np.zeros((1,len(SNRs)+3))
	temp[0,0] = args.steps
	temp[0,1] = args.learning_rate
	temp[0,2] = args.entangle_weights
	temp[0,3:] = BERs

	if (ind_var == 0):
		cur_data = np.concatenate([prev_data['BERs_data'],temp])
	else:
		cur_data = np.concatenate([prev_data,temp])
	BERs_data = {'BERs_data':cur_data}
	sio.savemat(res_file_name, BERs_data)

	ind_var = 0
	chan_name = str(args.channel_type)
	if (args.adaptivity_training == 1):
		res_file_name = results_folder + "/FERs_"+str(args.coding_scheme)+"_"+chan_name+"_"+str(n)+"_"+str(k)+"_adapt_from_" + base_channel + "_nn_eq_" + str(args.nn_eq) + "_lr_" + str(args.learning_rate) + "_SNR_"+ str(int(args.eb_n0_lo))+"_"+str(int(args.eb_n0_hi))+".mat"
	else:
		res_file_name = results_folder + "/FERs_"+str(args.coding_scheme)+"_"+chan_name+"_"+str(n)+"_"+str(k)+ "_baseline_nn_eq_" + str(args.nn_eq) + "_lr_" + str(args.learning_rate) +  "_SNR_"+ str(int(args.eb_n0_lo))+"_"+str(int(args.eb_n0_hi))+".mat"

	if os.path.exists(res_file_name):
		prev_data = sio.loadmat(res_file_name)
	else:
		ind_var = 1
		prev_data = np.zeros((1,len(SNRs)+3))
		prev_data[0,0] = -100
		prev_data[0,1] = -100
		prev_data[0,2] = -100
		prev_data[0,3:] = SNRs 
	temp = np.zeros((1,len(SNRs)+3))
	temp = np.zeros((1,len(SNRs)+3))
	temp[0,0] = args.steps
	temp[0,1] = args.learning_rate
	temp[0,2] = args.entangle_weights
	temp[0,3:] = FERs

	if (ind_var == 0):
		cur_data = np.concatenate([prev_data['FERs_data'],temp])
	else:
		cur_data = np.concatenate([prev_data,temp])
	FERs_data = {'FERs_data':cur_data}
	sio.savemat(res_file_name, FERs_data)

# end time
end = time.time()
# total time taken
print(f"Runtime of the program is {end - start}")