# Belief propagation using TensorFlow
# Run as follows:
# unit test : python main_nams.py 0 1 10 1 5 500 5 codes/BCH_63_36.alist codes/BCH_63_36.gmat test_out 0.5 10 NAMS_full True 0.01

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
import sys
from tensorflow.python.framework import ops
from helper_functions import load_code, syndrome
import os
import argparse
import scipy.io as sio
import pickle 

np.random.seed(0)

DEBUG = False
TRAINING = True
SUM_PRODUCT = False
MIN_SUM = not SUM_PRODUCT
ALL_ZEROS_CODEWORD_TRAINING = True 
ALL_ZEROS_CODEWORD_TESTING = False
NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
np.set_printoptions(precision=3)

print("My PID: " + str(os.getpid()))

if SUM_PRODUCT:
	print("Using Sum-Product algorithm")
if MIN_SUM:
	print("Using Min-Sum algorithm")

if ALL_ZEROS_CODEWORD_TRAINING:
	print("Training using only the all-zeros codeword")
else:
	print("Training using random codewords (not the all-zeros codeword)")

if ALL_ZEROS_CODEWORD_TESTING:
	print("Testing using only the all-zeros codeword")
else:
	print("Testing using random codewords (not the all-zeros codeword)")

if NO_SIGMA_SCALING_TRAIN:
	print("Not scaling train input by 2/sigma")
else:
	print("Scaling train input by 2/sigma")

if NO_SIGMA_SCALING_TEST:
	print("Not scaling test input by 2/sigma")
else:
	print("Scaling test input by 2/sigma")


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('-seed', type=int, default=0)
	parser.add_argument('-eb_n0_train_lo', type=float, default=1)
	parser.add_argument('-eb_n0_train_hi', type=float, default=10)
	parser.add_argument('-eb_n0_test_lo', type=float, default=1)
	parser.add_argument('-eb_n0_test_hi', type=float, default=10)
	parser.add_argument('-eb_n0_step', type=float, default=1)
	parser.add_argument('-min_frame_errors', type=int, default=5e2)
	parser.add_argument('-max_frames', type=float, default=5e5)
	parser.add_argument('-num_iterations', type=int, default=5)
	parser.add_argument('-H_filename', type=str, default='codes/BCH_63_36.alist')
	parser.add_argument('-G_filename', type=str, default='codes/BCH_63_36.gmat')
	parser.add_argument('-L', type=float, default=0.5)
	parser.add_argument('-total_train_steps', type=int, default=200)
	parser.add_argument('-samples_per_snr', type=int, default=10)
	parser.add_argument('-training', type=int, default=1)
	parser.add_argument('-testing', type=int, default=1)
	parser.add_argument('-adapt_steps', type=int, default=50)
	parser.add_argument('-learning_rate', type=float, default=0.01)
	parser.add_argument('-batch_size', type=int, default=120)
	parser.add_argument('-testing_batch_size', type=int, default=2400)
	parser.add_argument('-chan_train', type=str, default='ETU')
	parser.add_argument('-chan_test', type=str, default='ETU')
	parser.add_argument('-decoder_type', type=str, default='NAMS_full')
	parser.add_argument('-adapt', type=int, default=0)
	parser.add_argument('-use_saved_weights', type=int, default=0)
	parser.add_argument('-saved_model', type=str, default='saved_weights/pkl/weights_FNOMS_ch_tr_AWGN_st_30000_5_lr_0.01.pkl')
	parser.add_argument('-saved_model_for_adapt', type=str, default='saved_weights/pkl/weights_FNOMS_ch_tr_AWGN_st_30000_5_lr_0.01.pkl')

	#modifications
	parser.add_argument('-softplus', type=int, default=0)
	parser.add_argument('-relu', type=int, default=0)
	parser.add_argument('-w_relu', type=int, default=1)

	args = parser.parse_args()

	return args	

args = get_args()

# seed = args.seed
# np.random.seed(seed)
snr_train_lo = args.eb_n0_train_lo
snr_train_hi = args.eb_n0_train_hi
snr_test_lo = args.eb_n0_test_lo
snr_test_hi = args.eb_n0_test_hi
snr_step = args.eb_n0_step
min_frame_errors = args.min_frame_errors
max_frames = args.max_frames
num_iterations = args.num_iterations
H_filename = args.H_filename
G_filename = args.G_filename
L = args.L
total_train_steps = args.total_train_steps
adapt_steps = args.adapt_steps
provided_decoder_type = args.decoder_type
start_lr = args.learning_rate
adapt = args.adapt
chan_train = args.chan_train
chan_test = args.chan_test
if (args.training == 0):
	TRAINING = False
else:
	TRAINING = True
if (provided_decoder_type == "SPA"):
	SUM_PRODUCT = True
	MIN_SUM = not SUM_PRODUCT
	TRAINING = False
if (provided_decoder_type == "MinSum"):
	TRAINING = False

if ALL_ZEROS_CODEWORD_TESTING: G_filename = ""
code = load_code(H_filename, G_filename)

# code.H = np.array([[1, 1, 0, 1, 1, 0, 0],
#        [1, 0, 1, 1, 0, 1, 0],
#        [0, 1, 1, 1, 0, 0, 1]])

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

class Decoder:
	def __init__(self, decoder_type="RNOMS", random_seed=0, learning_rate = 0.001, relaxed = False):
		self.decoder_type = decoder_type
		self.random_seed = random_seed
		self.learning_rate = learning_rate
		self.relaxed = relaxed

# decoder parameters
batch_size = args.batch_size
tf_train_dataset = tf.placeholder(tf.float32, shape=(n,batch_size))
tf_train_labels = tf.placeholder(tf.float32, shape=(n,batch_size))#tf.placeholder(tf.float32, shape=(num_iterations,n,batch_size))

#### decoder functions ####

# compute messages from variable nodes to check nodes
def compute_vc(cv, iteration, soft_input):
	weighted_soft_input = soft_input
	
	edges = []
	for i in range(0, n):
		for j in range(0, var_degrees[i]):
			edges.append(i)
	reordered_soft_input = tf.gather(weighted_soft_input, edges)
	
	vc = []
	edge_order = []
	for i in range(0, n): # for each variable node v
		for j in range(0, var_degrees[i]):
			# edge = d[i][j]
			edge_order.append(d[i][j])
			extrinsic_edges = []
			for jj in range(0, var_degrees[i]):
				if jj != j: # extrinsic information only
					extrinsic_edges.append(d[i][jj])
			# if the list of edges is not empty, add them up
			if extrinsic_edges:
				temp = tf.gather(cv,extrinsic_edges)
				temp = tf.reduce_sum(temp,0)
			else:
				temp = tf.zeros([batch_size])
			if SUM_PRODUCT: temp = tf.cast(temp, tf.float32)#tf.cast(temp, tf.float64)
			vc.append(temp)
	
	vc = tf.stack(vc)
	new_order = np.zeros(num_edges).astype(int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(int)
	vc = tf.gather(vc,new_order)
	vc = vc + reordered_soft_input
	return vc

# compute messages from check nodes to variable nodes
def compute_cv(vc, iteration):
	cv_list = []
	prod_list = []
	min_list = []
	
	if SUM_PRODUCT:
		vc = tf.clip_by_value(vc, -10, 10)
		tanh_vc = tf.tanh(vc / 2.0)
	edge_order = []
	for i in range(0, m): # for each check node c
		for j in range(0, chk_degrees[i]):
			# edge = u[i][j]
			edge_order.append(u[i][j])
			extrinsic_edges = []
			for jj in range(0, chk_degrees[i]):
				if jj != j:
					extrinsic_edges.append(u[i][jj])
			if SUM_PRODUCT:
				temp = tf.gather(tanh_vc,extrinsic_edges)
				temp = tf.reduce_prod(temp,0)
				temp = tf.log((1+temp)/(1-temp))
				cv_list.append(temp)
			if MIN_SUM:
				temp = tf.gather(vc,extrinsic_edges)
				temp1 = tf.reduce_prod(tf.sign(temp),0)
				temp2 = tf.reduce_min(tf.abs(temp),0)
				prod_list.append(temp1)
				min_list.append(temp2)
	
	if SUM_PRODUCT:
		cv = tf.stack(cv_list)
	if MIN_SUM:
		prods = tf.stack(prod_list)
		mins = tf.stack(min_list)
		if decoder.decoder_type == "RNOMS" or decoder.decoder_type == "NAMS_relaxed" or decoder.decoder_type == "NAMS_simple":
			offsets = decoder.B_cv
			if (args.softplus == 1):
				offsets = tf.nn.softplus(decoder.B_cv)
			if (args.relu == 0):
				mins = mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size])
			else:
				mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
		elif decoder.decoder_type == "FNOMS" or decoder.decoder_type == "NAMS_full":
			# check the effect of removing softplus
			offsets = decoder.B_cv[iteration]
			if (args.softplus == 1):
				offsets = tf.nn.softplus(decoder.B_cv[iteration])
			if (args.relu == 0):
				mins = mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size])
			else:
				mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
		cv = prods * mins
	
	new_order = np.zeros(num_edges).astype(int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(int)
	cv = tf.gather(cv,new_order)
	
	scaling = 1
	if decoder.decoder_type == "RNSPA" or decoder.decoder_type == "RNNMS" or decoder.decoder_type == "NAMS_relaxed":
		scaling = tf.tile(tf.reshape(decoder.W_cv,[-1,1]),[1,batch_size])
		if (args.softplus == 1):
			scaling = tf.tile(tf.reshape(tf.nn.softplus(decoder.W_cv),[-1,1]),[1,batch_size])
	elif decoder.decoder_type == "FNSPA" or decoder.decoder_type == "FNNMS" or decoder.decoder_type == "NAMS_full":
		scaling = tf.tile(tf.reshape(decoder.W_cv[iteration],[-1,1]),[1,batch_size])
		if (args.softplus == 1):
			scaling = tf.tile(tf.reshape(tf.nn.softplus(decoder.W_cv[iteration]),[-1,1]),[1,batch_size])
	elif decoder.decoder_type == "NAMS_simple":
		scaling = decoder.W_cv
		if (args.softplus == 1):
			scaling = tf.nn.softplus(decoder.W_cv)
	cv = cv * scaling
	return cv

# combine messages to get posterior LLRs
def marginalize(soft_input, iteration, cv):
	weighted_soft_input = soft_input

	soft_output = []
	for i in range(0,n):
		edges = []
		for e in range(0,var_degrees[i]):
			edges.append(d[i][e])

		temp = tf.gather(cv,edges)
		temp = tf.reduce_sum(temp,0)
		soft_output.append(temp)

	soft_output = tf.stack(soft_output)

	soft_output = weighted_soft_input + soft_output
	return soft_output

def continue_condition(soft_input, soft_output, iteration, cv, m_t, loss, labels):
	condition = (iteration < num_iterations)
	return condition

def belief_propagation_iteration(soft_input, soft_output, iteration, cv, m_t, loss, labels):
	# compute vc
	vc = compute_vc(cv,iteration,soft_input)

	# filter vc
	if decoder.relaxed:
		m_t = R * m_t + (1-R) * vc
		vc_prime = m_t
	else:
		vc_prime = vc

	# compute cv
	cv = compute_cv(vc_prime,iteration)

	# get output for this iteration
	soft_output = marginalize(soft_input, iteration, cv)
	iteration += 1

	# L = 0.5
	print("L = " + str(L))
	CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / num_iterations
	syndrome_loss = tf.reduce_mean(tf.maximum(1. - syndrome(soft_output, code),0) ) / num_iterations
	new_loss = L * CE_loss + (1-L) * syndrome_loss
	loss = loss + new_loss

	return soft_input, soft_output, iteration, cv, m_t, loss, labels

# builds a belief propagation TF graph
def belief_propagation_op(soft_input, labels):
	return tf.while_loop(
		continue_condition, # iteration < max iteration?
		belief_propagation_iteration, # compute messages for this iteration
		[
			soft_input, # soft input for this iteration
			soft_input,  # soft output for this iteration
			tf.constant(0,dtype=tf.int32), # iteration number
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # cv
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # m_t
			tf.constant(0.0,dtype=tf.float32), # loss
			labels
		]
		)

#### end decoder functions ####
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = start_lr
learning_rate = starter_learning_rate # provided_decoder_type="normal", "FNNMS", "FNOMS", ...
decoder = Decoder(decoder_type=provided_decoder_type, random_seed=1, learning_rate = learning_rate, relaxed = False)
print("\n\nDecoder type: " + decoder.decoder_type + "\n\n")
if decoder.relaxed: print("relaxed")
else: print("not relaxed")

## load previous saved models
if (adapt == 1):
	# adapt_pkl_filename = "saved_weights/pkl/weights_" + provided_decoder_type + "_ch_tr_" + chan_train  + "_st_" + str(adapt_steps) + "_" + str(num_iterations) +"_lr_" + str(learning_rate) + ".pkl"
	infile = open(args.saved_model_for_adapt,'rb')
	decoder_saved = pickle.load(infile)
	infile.close()
if (args.use_saved_weights == 1):
	saved_model = args.saved_model + ".pkl"
	print("loading previous saved weights ... ",saved_model)
	infile = open(saved_model,'rb')
	decoder_saved = pickle.load(infile)
	infile.close()
if SUM_PRODUCT:
	if decoder.decoder_type == "FNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "RNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))#tf.Variable(0.0,dtype=tf.float32)#

if MIN_SUM:
	if decoder.decoder_type == "FNNMS":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
	
	if decoder.decoder_type == "NAMS_full":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "NAMS_relaxed":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "NAMS_simple":
		decoder.W_cv = tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		decoder.B_cv = tf.Variable(tf.truncated_normal([1],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
	
	if (decoder.decoder_type == "NAMS_full" or decoder.decoder_type == "NAMS_relaxed" or decoder.decoder_type == "NAMS_simple"):
		if (args.use_saved_weights == 1):
			decoder.W_cv = tf.Variable(decoder_saved['W_cv'])
			decoder.B_cv = tf.Variable(decoder_saved['B_cv'])

	if decoder.decoder_type == "FNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0/num_edges))
		if (args.use_saved_weights == 1):
			decoder.B_cv = tf.Variable(decoder_saved['B_cv'])
	if decoder.decoder_type == "RNNMS":
		decoder.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)))#tf.Variable(0.0,dtype=tf.float32)#
		
	if decoder.decoder_type == "RNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)) #tf.Variable(0.0,dtype=tf.float32)#

if decoder.relaxed:
	decoder.relaxation_factors = tf.Variable(0.0,dtype=tf.float32)
	R = tf.sigmoid(decoder.relaxation_factors)
	# print "single learned relaxation factor"

	# decoder.relaxation_factors = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0))
	# R = tf.tile(tf.reshape(tf.sigmoid(decoder.relaxation_factors),[-1,1]),[1,batch_size])
	# print "multiple relaxation factors"

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
config = tf.ConfigProto(
		device_count = {'CPU': 2,'GPU': 0}
	)
with tf.Session(config=config) as session: #tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
	# simulate each SNR
	SNRs = np.arange(snr_train_lo, snr_train_hi+snr_step, snr_step)
	if (batch_size % args.samples_per_snr) != 0:
		print("********************")
		print("********************")
		print("error: batch size must divide by the number of SNRs to train on")
		print("********************")
		print("********************")
	BERs = []
	BERs_ref = []
	SERs = []
	FERs = []

	print("\nBuilding the decoder graph...")
	belief_propagation = belief_propagation_op(soft_input=tf_train_dataset, labels=tf_train_labels)
	if TRAINING:
		# output 5 corresponds to sum of losses across all iterations
		training_loss = belief_propagation[5]#tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=belief_propagation[1], labels=tf_train_labels))
		loss = training_loss
		print("Learning rate: " + str(starter_learning_rate))
		optimizer = tf.train.AdamOptimizer(learning_rate=decoder.learning_rate).minimize(loss,global_step=global_step)
	print("Done.\n")
	init = tf.global_variables_initializer()

	if ALL_ZEROS_CODEWORD_TRAINING:
		codewords = np.zeros([n,batch_size])
		codewords_repeated = np.zeros([num_iterations,n,batch_size]) # repeat for each iteration (multiloss)
		BPSK_codewords = np.ones([n,batch_size])
		soft_input = np.zeros_like(BPSK_codewords)
		channel_information = np.zeros_like(BPSK_codewords)

	covariance_matrix = np.eye(n)
	eta = 0.99
	for i in range(0,n):
		for j in range(0,n):
			covariance_matrix[i,j] = eta**np.abs(i-j)

	session.run(init)
	
	if TRAINING:
		# total_train_steps = 10001
		print("***********************")
		print("Training decoder using " + str(total_train_steps) + " minibatches...")
		print("***********************")

		if (chan_train == "ETU"):
			#load offline data
			print ("Loading offline ETU data for training ... ")
			lte_data_filename = "lte_data/BCH_63_36_ETU_data_train_enc_9_18.mat"
			data = sio.loadmat(lte_data_filename)
			enc_data = np.array(data['enc'])
			lte_data_filename = "lte_data/BCH_63_36_ETU_data_train_llr_9_18.mat"
			data = sio.loadmat(lte_data_filename)
			llr_data = np.array( data['llr'])
		train_step = 0
		while train_step < total_train_steps:
			# generate random codewords
			if (chan_train == "AWGN"):
				if not ALL_ZEROS_CODEWORD_TRAINING:
					# generate message
					messages = np.random.randint(0,2,[k,batch_size])

					# encode message
					codewords = np.dot(G, messages) % 2
					#codewords_repeated = np.tile(x,(num_iterations,1,1)).shape 

					# modulate codeword
					BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

					soft_input = np.zeros_like(BPSK_codewords)
					channel_information = np.zeros_like(BPSK_codewords)
				else:
					codewords = np.zeros([n,batch_size])
					#codewords_repeated = np.zeros([num_iterations,n,batch_size]) # repeat for each iteration (multiloss)
					BPSK_codewords = np.ones([n,batch_size])
					soft_input = np.zeros_like(BPSK_codewords)
					channel_information = np.zeros_like(BPSK_codewords)

			# create minibatch with codewords from multiple SNRs
			for i in range(0,len(SNRs)):
				if (chan_train == "ETU"):
					start_idx = int(train_step*batch_size/args.samples_per_snr)
					end_idx = int(start_idx + batch_size/args.samples_per_snr)
					snr_start_ind = int(snr_train_lo)-1
					snr_end_ind = int(snr_train_hi)
					# messages = msg_data[:,start_idx:end_idx,snr_start_ind:snr_end_ind]
					codewords = enc_data[:,start_idx:end_idx,snr_start_ind:snr_end_ind]
					soft_input = llr_data[:,start_idx:end_idx,snr_start_ind:snr_end_ind]
					# concatenate the data along the SNR dimension
					# messages = torch.reshape(messages,[messages.size(0),-1])
					# breakpoint()
					codewords = np.reshape(codewords,[np.shape(codewords)[0],-1])
					soft_input = np.reshape(soft_input,[np.shape(soft_input)[0],-1])
				else:
					sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNRs[i]/10)))
					noise = sigma * np.random.randn(n,batch_size//len(SNRs))
					start_idx = batch_size*i//len(SNRs)
					end_idx = batch_size*(i+1)//len(SNRs)
					channel_information[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + noise
					if NO_SIGMA_SCALING_TRAIN:
						soft_input[:,start_idx:end_idx] = channel_information[:,start_idx:end_idx]
					else:
						soft_input[:,start_idx:end_idx] = 2.0*channel_information[:,start_idx:end_idx]/(sigma*sigma)

			# feed minibatch into BP and run SGD
			batch_data = soft_input
			batch_labels = codewords #codewords #codewords_repeated
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

			[_] = session.run([optimizer], feed_dict=feed_dict) #,bp_output,syndrome_output,belief_propagation, soft_syndromes
			# breakpoint()
			if decoder.relaxed and TRAINING: 
				print(session.run(R))
			if train_step % 10 == 0:
				print(str(train_step) + " minibatches completed")

			train_step += 1
		
		print("Trained decoder on " + str(train_step) + " minibatches.\n")

	# testing phase
	print("***********************")
	print("Testing decoder...")
	print("***********************")
	# snr_test_lo = 1
	# snr_test_hi = 12
	SNRs = np.arange(snr_test_lo, snr_test_hi+snr_step, snr_step)
	# batch_size = 120
	if (chan_test == "ETU"):
		#load offline data
		print ("Loading offline ETU data for testing ... ")
		lte_data_filename = "lte_data/BCH_63_36_ETU_data_test_enc_9_18.mat"
		data = sio.loadmat(lte_data_filename)
		enc_data = np.array(data['enc'])
		lte_data_filename = "lte_data/BCH_63_36_ETU_data_test_llr_9_18.mat"
		data = sio.loadmat(lte_data_filename)
		llr_data = np.array( data['llr'])
		num_frames = np.shape(enc_data)[1]
		max_frames = min(num_frames - batch_size, max_frames)
		print ("Max frames modified to ", max_frames)
	for SNR in SNRs:
		# simulate this SNR
		sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR/10)))
		frame_count = 0
		bit_errors = 0
		frame_errors = 0
		bit_errors_ref = 0
		frame_errors_ref = 0
		frame_errors_with_HDD = 0
		symbol_errors = 0
		FE = 0

		# simulate frames
		while ((FE < min_frame_errors) or (frame_count < 100000)) and (frame_count < max_frames):
			frame_count += batch_size # use different batch size for test phase?
			if (chan_test == "ETU"):
				start_ind = frame_count-batch_size
				end_ind = frame_count
				# map snr to ind
				snr_ind = int(SNR) - 1
				codewords = np.squeeze(enc_data[:,start_ind:end_ind,snr_ind])
				# breakpoint()
				soft_input = np.squeeze(llr_data[:,start_ind:end_ind,snr_ind])
				batch_data = soft_input
			else:
				if not ALL_ZEROS_CODEWORD_TESTING:
					# generate message
					messages = np.random.randint(0,2,[batch_size,k])

					# encode message
					codewords = np.dot(G, messages.transpose()) % 2

					# modulate codeword
					BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0
				
				# add Gaussian noise to codeword
				noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
				channel_information = BPSK_codewords + noise

				# convert channel information to LLR format
				if NO_SIGMA_SCALING_TEST:
					soft_input = channel_information
				else:
					soft_input = 2.0*channel_information/(sigma*sigma)
			
			# breakpoint()
			# run belief propagation
			batch_data = soft_input
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : codewords}
			soft_outputs = session.run([belief_propagation], feed_dict=feed_dict)
			soft_output = np.array(soft_outputs[0][1])
			recovered_codewords = (soft_output < 0).astype(int)
			# breakpoint()
			# update bit error count and frame error count
			errors = codewords != recovered_codewords
			bit_errors += errors.sum()
			frame_errors += (errors.sum(0) > 0).sum()

			recovered_codewords_ref = (soft_input < 0).astype(int)
			errors_ref = codewords != recovered_codewords_ref
			bit_errors_ref += errors_ref.sum()
			frame_errors_ref += (errors_ref.sum(0) > 0).sum()
			# if ( np.abs(np.sum(soft_output-soft_input)) > 0.001):
				# breakpoint()
			# if ((errors_ref.sum(0) != errors.sum(0)).sum(0)):
			# 	breakpoint()
			FE = frame_errors
		# breakpoint()
		# summarize this SNR:
		print("SNR: " + str(SNR))
		print("frame count: " + str(frame_count))

		bit_count = frame_count * n
		BER = float(bit_errors) / float(bit_count)
		BERs.append(BER)

		BER_ref = float(bit_errors_ref) / float(bit_count)
		BERs_ref.append(BER_ref)

		print("bit errors: " + str(bit_errors))
		print("BER: " + str(BER))

		print("bit errors ref: " + str(bit_errors_ref))
		print("BER_ref: " + str(BER_ref))

		FER = float(frame_errors) / float(frame_count)
		FERs.append(FER)
		print("FER: " + str(FER))
		print("")

	# print summary
	print("BERs_NOMS = ", end =" ")
	print(BERs)
	print("BERs_ref:")
	print(BERs_ref)
	print("FERs:")
	print(FERs)	

	out_file = open('master_out_nams_paper_random0.txt', 'a')
	command = "%%%% command : chan train " + chan_train + " chan test " + chan_test + " " + str(snr_test_lo) + " " + str(snr_test_hi) + " " + str(max_frames) + " " + str(total_train_steps) + "_" + str(num_iterations) + " lr " + str(learning_rate) + " decoder : " + provided_decoder_type + " sf " + str(args.softplus) + " relu " + str(args.relu) + "\n"

	out_file.write(command)
	if (chan_test == chan_train):
		common_prefix = chan_test + "_" + str(total_train_steps) + "_" + str(num_iterations) + "_sf_" + str(args.softplus) + "_relu_" + str(args.relu) + " = ["
		if (SUM_PRODUCT):
			ber_prefix = "BERs_SPA_" + chan_test + " = ["
			fer_prefix = "FERs_SPA_" + chan_test + " = ["
		elif (provided_decoder_type == "MinSum"):
			ber_prefix = "BERs_MinSum_" + chan_test + " = ["
			fer_prefix = "FERs_MinSum_" + chan_test + " = ["
		elif (provided_decoder_type == "FNOMS"):
			ber_prefix = "BERs_FNOMS_" + common_prefix
			fer_prefix = "FERs_FNOMS_" + common_prefix
		elif (provided_decoder_type == "RNOMS"):
			ber_prefix = "BERs_RNOMS_" + common_prefix
			fer_prefix = "FERs_RNOMS_" + common_prefix
		elif (provided_decoder_type == "FNNMS"):
			ber_prefix = "BERs_FNNMS_" + common_prefix
			fer_prefix = "FERs_FNNMS_" + common_prefix
		elif (provided_decoder_type == "RNNMS"):
			ber_prefix = "BERs_RNNMS_" + common_prefix
			fer_prefix = "FERs_RNNMS_" + common_prefix
		elif (provided_decoder_type == "NAMS_full"):
			ber_prefix = "BERs_NAMS_full_" + common_prefix
			fer_prefix = "FERs_NAMS_full_" + common_prefix
		elif (provided_decoder_type == "NAMS_relaxed"):
			ber_prefix = "BERs_NAMS_relaxed_" + common_prefix
			fer_prefix = "FERs_NAMS_relaxed_" + common_prefix
		elif (provided_decoder_type == "NAMS_simple"):
			ber_prefix = "BERs_NAMS_simple_" + common_prefix
			fer_prefix = "FERs_NAMS_simple_" + common_prefix
	else:
		common_prefix = chan_train + "_" + chan_test + "_" + str(total_train_steps) + "_" + str(num_iterations) + "_sf_" + str(args.softplus) + "_relu_" + str(args.relu) + " = ["
		if (SUM_PRODUCT):
			ber_prefix = "BERs_SPA_" + chan_train + "_" + chan_test + " = ["
			fer_prefix = "FERs_SPA_" + chan_train + "_" + chan_test + " = ["
		elif (provided_decoder_type == "MinSum"):
			ber_prefix = "BERs_MinSum_" + chan_test + " = ["
			fer_prefix = "FERs_MinSum_" + chan_test + " = ["
		elif (provided_decoder_type == "FNOMS"):
			ber_prefix = "BERs_FNOMS_" + common_prefix
			fer_prefix = "FERs_FNOMS_" + common_prefix
		elif (provided_decoder_type == "RNOMS"):
			ber_prefix = "BERs_RNOMS_" + common_prefix
			fer_prefix = "FERs_RNOMS_" + common_prefix
		elif (provided_decoder_type == "FNNMS"):
			ber_prefix = "BERs_FNNMS_" + common_prefix
			fer_prefix = "FERs_FNNMS_" + common_prefix
		elif (provided_decoder_type == "RNNMS"):
			ber_prefix = "BERs_RNNMS_" + common_prefix
			fer_prefix = "FERs_RNNMS_" + common_prefix
		elif (provided_decoder_type == "NAMS_full"):
			ber_prefix = "BERs_NAMS_full_" + common_prefix
			fer_prefix = "FERs_NAMS_full_" + common_prefix
		elif (provided_decoder_type == "NAMS_relaxed"):
			ber_prefix = "BERs_NAMS_relaxed_" + common_prefix
			fer_prefix = "FERs_NAMS_relaxed_" + common_prefix
		elif (provided_decoder_type == "NAMS_simple"):
			ber_prefix = "BERs_NAMS_simple_" + common_prefix
			fer_prefix = "FERs_NAMS_simple_" + common_prefix
	
	out_file.write(ber_prefix)
	for element in BERs:
		out_file.write(str(element) + "\t")
	out_file.write(" ];\n")
	out_file.write(fer_prefix)
	for element in FERs:
		out_file.write(str(element) + "\t")
	out_file.write(" ];\n\n")
	out_file.close()
	if (not (SUM_PRODUCT or provided_decoder_type == "MinSum")):
		offsets = 0
		weights = 0
		if (provided_decoder_type == "NAMS_full" or provided_decoder_type == "NAMS_relaxed" or provided_decoder_type == "NAMS_simple"):
			offsets = session.run(decoder.B_cv)
			weights = session.run(decoder.W_cv)
		elif (provided_decoder_type == "FNNMS" or provided_decoder_type == "RNNMS"):
			weights = session.run(decoder.W_cv)
		elif (provided_decoder_type == "FNOMS" or provided_decoder_type == "RNOMS"):
			offsets = session.run(decoder.B_cv)
		# breakpoint()
		# save weights to pickle and mat
		B_cv = offsets
		W_cv = weights
		weights_arr = {'B_cv':B_cv, 'W_cv':W_cv}
		if (adapt == 0 and args.use_saved_weights == 0):
			mat_filename = args.saved_model + ".mat"#saved_weights/mat/weights_" + provided_decoder_type + "_ch_tr_" + chan_train  + "_st_" + str(total_train_steps) + "_" + str(num_iterations) + "_lr_" + str(learning_rate) + ".mat"
			sio.savemat(mat_filename,weights_arr)
			pkl_filename = args.saved_model + ".pkl"#"saved_weights/pkl/weights_" + provided_decoder_type + "_ch_tr_" + chan_train  + "_st_" + str(total_train_steps) + "_" + str(num_iterations) +"_lr_" + str(learning_rate) + ".pkl"
			outfile = open(pkl_filename, 'wb')
			pickle.dump(weights_arr,outfile)
			outfile.close()
		elif (adapt == 1 and args.use_saved_weights == 0):
			mat_filename = "saved_weights/mat/weights_" + provided_decoder_type + "_ch_tr_" + chan_train  + "_st_" + str(total_train_steps) + "_ch_ts_" + chan_test  + "_st_" + str(test_steps) + "_" + str(num_iterations) + "_lr_" + str(learning_rate) + ".mat"
			sio.savemat(mat_filename,weights_arr)
			pkl_filename = "saved_weights/pkl/weights_" + provided_decoder_type + "_ch_tr_" + chan_train  + "_st_" + str(total_train_steps) + "_ch_ts_" + chan_test  + "_st_" + str(test_steps) + "_" + str(num_iterations) +"_lr_" + str(learning_rate) + ".pkl"
			outfile = open(pkl_filename, 'wb')
			pickle.dump(weights_arr,outfile)
			outfile.close()


