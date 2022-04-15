# Belief propagation using TensorFlow
# Run as follows:
# python main.py 0 1 6 1 100 10000000000000000 5 hamming.alist hamming.gmat laskdjhf 0.5 100 FNOMS
# python main.py 0 1 10 1 500 500000 5 codes/BCH_63_36.alist codes/BCH_63_36.gmat test_out 0.5 30000 FNOMS

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
import sys
from tensorflow.python.framework import ops
from helper_functions import load_code, syndrome
import os
import scipy.io as sio

DEBUG = False
TRAINING = True
SUM_PRODUCT = False
MIN_SUM = not SUM_PRODUCT
ALL_ZEROS_CODEWORD_TRAINING = True 
ALL_ZEROS_CODEWORD_TESTING = False
NO_SIGMA_SCALING_TRAIN = False
NO_SIGMA_SCALING_TEST = False
ETU_data = True
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

seed = int(sys.argv[1])
np.random.seed(seed)
snr_lo = float(sys.argv[2])
snr_hi = float(sys.argv[3])
snr_step = float(sys.argv[4])
min_frame_errors = int(sys.argv[5])
max_frames = float(sys.argv[6])
num_iterations = int(sys.argv[7])
H_filename = sys.argv[8]
G_filename = sys.argv[9]
output_filename = sys.argv[10]
L = float(sys.argv[11])
steps = int(sys.argv[12])
provided_decoder_type = sys.argv[13]

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
batch_size = 120
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
		if decoder.decoder_type == "RNOMS":
			# offsets = tf.nn.softplus(decoder.B_cv)
			# mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
			mins = tf.nn.relu(mins - decoder.B_cv)
		elif decoder.decoder_type == "FNOMS":
			offsets = tf.nn.softplus(decoder.B_cv[iteration])
			mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
		cv = prods * mins
	
	new_order = np.zeros(num_edges).astype(int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(int)
	cv = tf.gather(cv,new_order)
	
	if decoder.decoder_type == "RNSPA" or decoder.decoder_type == "RNNMS":
		cv = cv * tf.tile(tf.reshape(decoder.W_cv,[-1,1]),[1,batch_size])
	elif decoder.decoder_type == "FNSPA" or decoder.decoder_type == "FNNMS":
		cv = cv * tf.tile(tf.reshape(decoder.W_cv[iteration],[-1,1]),[1,batch_size])
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
starter_learning_rate = 0.01
learning_rate = starter_learning_rate # provided_decoder_type="normal", "FNNMS", "FNOMS", ...
decoder = Decoder(decoder_type=provided_decoder_type, random_seed=1, learning_rate = learning_rate, relaxed = False)
print("\n\nDecoder type: " + decoder.decoder_type + "\n\n")
if decoder.relaxed: print("relaxed")
else: print("not relaxed")

if SUM_PRODUCT:
	if decoder.decoder_type == "FNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
		
	if decoder.decoder_type == "RNSPA":
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))#tf.Variable(0.0,dtype=tf.float32)#
		
if MIN_SUM:
	if decoder.decoder_type == "FNNMS":
		# decoder.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)))
		decoder.W_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed))
	
	if decoder.decoder_type == "FNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0/num_edges))

	if decoder.decoder_type == "RNNMS":
		decoder.W_cv = tf.nn.softplus(tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0, seed=decoder.random_seed)))#tf.Variable(0.0,dtype=tf.float32)#
		
	if decoder.decoder_type == "RNOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_edges],dtype=tf.float32,stddev=1.0)) #tf.Variable(0.0,dtype=tf.float32)#

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
	SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step)
	if (batch_size % len(SNRs)) != 0:
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
		# steps = 10001
		print("***********************")
		print("Training decoder using " + str(steps) + " minibatches...")
		print("***********************")

		if (ETU_data):
			#load offline data
			print ("Loading offline ETU data for training ... ")
			lte_data_filename = "lte_data/BCH_63_36_ETU_data_train_enc_9_18.mat"
			data = sio.loadmat(lte_data_filename)
			enc_data = np.array(data['enc'])
			lte_data_filename = "lte_data/BCH_63_36_ETU_data_train_llr_9_18.mat"
			data = sio.loadmat(lte_data_filename)
			llr_data = np.array( data['llr'])

		step = 0
		while step < steps:
			# generate random codewords
			if not ETU_data:
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
				if (ETU_data):
					start_idx = int(step*batch_size/len(SNRs))
					end_idx = int(start_idx + batch_size/len(SNRs))
					snr_start_ind = int(snr_lo)-1
					snr_end_ind = int(snr_hi)
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
				
				# breakpoint()
			# use offline data for ETU


			# feed minibatch into BP and run SGD
			batch_data = soft_input
			batch_labels = codewords #codewords #codewords_repeated
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			[_] = session.run([optimizer], feed_dict=feed_dict) #,bp_output,syndrome_output,belief_propagation, soft_syndromes
			# breakpoint()
			if decoder.relaxed and TRAINING: 
				print(session.run(R))
			if step % 10 == 0:
				print(str(step) + " minibatches completed")

			step += 1
		
		print("Trained decoder on " + str(step) + " minibatches.\n")

	# testing phase
	print("***********************")
	print("Testing decoder...")
	print("***********************")
	# snr_lo = 1
	# snr_hi = 12
	# SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step)
	# batch_size = 120
	if (ETU_data):
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
			if (ETU_data):
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

	# offset = session.run(decoder.B_cv)
	# weights = session.run(decoder.W_cv)
# BER_NOMS_ETU_0  = [0.02174165429561113, 0.0144501858830396, 0.009644506616928679, 0.006275297222539429, 0.004024716734548869, 0.002511482877190311, 0.0014594673467575147, 0.0008175998883432936, 0.000459473690888559]
# BER_NOMS_ETU_500 = [0.022102000938931395, 0.01479387918236839, 0.01007099082638651, 0.006769980840724246, 0.004396800020301219, 0.0027941139152170328, 0.0015551051222514052, 0.00091196883762831, 0.000510861152347963]
# 500 = [0.022636969789247965, 0.015200379379036453, 0.01025132275132275, 0.006693375458363468, 0.004269124383033256, 0.0027260731097661552, 0.0017460634666869679, 0.00103171431109081, 0.000547181302576986]
# 5000 = [0.02294941824318323, 0.015449703729080227, 0.010394700112925533, 0.006884016596246812, 0.0043872838237346634, 0.002761441640338523, 0.001678974280892746, 0.0009887328232651973, 0.0005034067983708271]
# 50000 9-18 = [0.022295338332508595, 0.014921079009808026, 0.010084789311408016, 0.006673550048849809, 0.0042464441145496305, 0.002565725197619682, 0.0014681905269435245, 0.0008064976590156447, 0.0004036453376980955]
# 20000 9-18 = [0.022793511222767817, 0.015209736972326901, 0.010139190235113496, 0.006734929516704097, 0.004279750802532577, 0.0027073579231852614, 0.0015568497582886072, 0.0008536028320200982, 0.00042140890462233387]

# 20000 9-18 0.01 = [0.021126749394135485, 0.014040037811021025, 0.009094311852105618, 0.0058684798193191475, 0.00359125398094223, 0.0020504231535406594, 0.0011251316407191708, 0.0006210904292439065, 0.00028659611992945327]

# 20000 9-17 0.005 = [0.0208926509585982, 0.01384511438468273, 0.00893507416289191, 0.005805038508875439, 0.003555885450369863, 0.002062952812353292, 0.001150508164896654, 0.000621407635796125, 0.000312765660487483]
# 30000 9-17 0.01 = [0.021002404425665817, 0.013979134152995063, 0.009116992120589243, 0.005955553017903138, 0.00365009579637877, 0.0020072830624389376, 0.0010396444748962735, 0.0005189499194295358, 0.00022466184917546784]
# 40000 9-17 0.01 = [0.020944038420057605, 0.013872552751449633, 0.009134597084237372, 0.005910509687488105, 0.003723687716493472, 0.002079764759620875, 0.001162720617157068, 0.0005888939641937244, 0.00021784221686945422]

# with AWGN training
# 20000 9-18 0.01 BER_NOMS = [0.02106473551317676, 0.013998959562508724, 0.009074803649144177, 0.005863563117759761, 0.003566511869869184, 0.0020236191998781925, 0.00110134114930278, 0.0006042784819763237, 0.00025802752293577983, 8.348863612021507e-05]
# 30000 9-18 0.01 BER_NOMS = [0.02102746374329108, 0.013798960831334932, 0.008961719513278267, 0.005788385164883966, 0.0035152830116858894, 0.002045506451981272, 0.0011660512859553627, 0.0006420260616903302, 0.00028231383147450293, 7.884027270862257e-05]
# 40000 9-18 0.01

# 20000 9-18 0.1 BER_NOMS = [0.022843947064570564, 0.015499980967606866, 0.010443391318691079, 0.006869583698120869, 0.004447077258827858, 0.0027478017585931256, 0.0016386890487609912, 0.0009192645883293366, 0.0004412343141359928, 0.00020253638359153946]


### with ETU training
# 0 9-18 0.01 BERs_NOMS =  [0.0213938373111035, 0.01403480390290942, 0.009116040500932588, 0.005833904305127327, 0.003703703703703704, 0.002407914937890957, 0.0014386903175872, 0.0008099869310900485, 0.00042759443239059544, 0.0002423372141682001]
# 10000 9-18 0.01 BERs_NOMS =  [0.019357688452412675, 0.012135053861672566, 0.00755919074264398, 0.0045466801162244806, 0.002694193851268192, 0.0014623222057274815, 0.0008256886554248665, 0.00042442236686841004, 0.00016899841697685364, 8.960427198342865e-05];
# 20000 9-18 0.01 BERs_NOMS =  [0.019363398170352607, 0.012149962569626839, 0.007584250060269245, 0.004588709984393437, 0.0027325758440866353, 0.0014163272556557928, 0.0007530483549668202, 0.00038191668887112534, 0.0001885352538390585, 0.00011581442846441133];


# 0 9-20 0.01 BERs_NOMS =  [0.021566556278786495, 0.01419959270678695, 0.009262272721505335, 0.005834697321507873, 0.0038507289406569984, 0.0024234580589496658, 0.0014865885069722, 0.0008325085962975651, 0.0004163335997868372, 0.0001595917584074505, 4.568518912781208e-05, 2.2127993439468848e-05];
# 0 9-20 0.01 BERs_NOMS =  [0.019440637965817822, 0.012099843934376308, 0.007474972403029957, 0.004534467663964067, 0.0027617588468907416, 0.0014214025604912894, 0.0007657366170555619, 0.0003985700328625988, 0.00019601798732805974, 8.220377072836089e-05, 1.2961385912205585e-05, 6.8995970635314895e-06];


# best NOMS 
# AWGN 1-8 20000 0.1 BERs_NOMS = [0.11994563079694974, 0.0905964117594813, 0.058157283696852045, 0.028494505982515576, 0.009589471280118762, 0.0020667592909799147, 0.0002827242671926065, 2.5458280797853633e-05]
# ETU 9-18 20000 0.01 BERs_NOMS = [0.019440637965817822, 0.012099843934376308, 0.007474972403029957, 0.004534467663964067, 0.0027617588468907416, 0.0014214025604912894, 0.0007657366170555619, 0.0003985700328625988, 0.00019601798732805974, 8.220377072836089e-05, 1.2961385912205585e-05, 6.8995970635314895e-06];

# NAMS
# ETU 9-18 20000 BERs_NOMS =  BERs_NOMS =  [0.018292508850062807, 0.011067336606904952, 0.006605667846675041, 0.0038291588951061373, 0.0021401926078185073, 0.0011425780010911905, 0.0005657378858817708, 0.00029008539200385726, 0.00016021914634096592, 8.026378584754219e-05];



# num training samples NOMS : 20000 x 120 = 2400K 
# num training samples NAMS : 500 x 1500 = 750K
# num training samples BP PAN : 50 x 1000 x 100 = 50000K 