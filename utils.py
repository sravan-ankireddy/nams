import numpy as np
np.random.seed(1)
from scipy.io import loadmat
import tensorflow as tf
import torch.nn.functional as F
import torch
torch.manual_seed(1)

class Code:
	def __init__(self):
		self.num_edges = 0
		# self.n = n
		# self.k = k

def load_code(H_filename, G_filename):
	# parity-check matrix; Tanner graph parameters
	with open(H_filename) as f:
		# get n and m (n-k) from first line
		n,m = [int(s) for s in f.readline().split(' ')]
		k = n-m

		var_degrees = np.zeros(n).astype(np.int) # degree of each variable node
		chk_degrees = np.zeros(m).astype(np.int) # degree of each check node

		# initialize H
		H = np.zeros([m,n]).astype(np.int)
		max_var_degree, max_chk_degree = [int(s) for s in f.readline().split(' ')]
		f.readline() # ignore two lines
		f.readline()

		# create H, sparse version of H, and edge index matrices
		# (edge index matrices used to calculate source and destination nodes during belief propagation)
		var_edges = [[] for _ in range(0,n)]
		for i in range(0,n):
			row_string = f.readline().split(' ')
			var_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			var_degrees[i] = len(var_edges[i])
			H[var_edges[i], i] = 1

		chk_edges = [[] for _ in range(0,m)]
		for i in range(0,m):
			row_string = f.readline().split(' ')
			chk_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			chk_degrees[i] = len(chk_edges[i])

		d = [[] for _ in range(0,n)]
		# for each var node, collect and store edges and move to next var node
		edge = 0
		for i in range(0,n):
			for j in range(0,var_degrees[i]):
				d[i].append(edge)
				edge += 1

		u = [[] for _ in range(0,m)]
		edge = 0
		for i in range(0,m):
			for j in range(0,chk_degrees[i]):
				v = chk_edges[i][j]
				for e in range(0,var_degrees[v]):
					if (i == var_edges[v][e]):
						u[i].append(d[v][e])

		num_edges = H.sum()

	if G_filename == "":
		G = []
	else:
		# if "OLDD" in G_filename:
		# 	G_mat = loadmat(G_filename)
		# 	G = G_mat['G']
		# 	G = G.transpose()
		# elif "LDPC" in G_filename: # dear God please fix this
		# 	G = np.loadtxt(G_filename).astype(np.int)
		# 	G = G.transpose()
		# else:
		# 	P = np.loadtxt(G_filename,skiprows=2)
		# 	G = np.vstack([P.transpose(), np.eye(k)]).astype(np.int)
		G = np.loadtxt(G_filename).astype(np.int)
		# G = G.transpose()
    
	# all edges
	edges = []
	for i in range(0, n):
		for j in range(0, var_degrees[i]):
			edges.append(i)

	# edges for marginalization
	edges_m = []
	for i in range(0, n):
		temp_e = []
		for e in range(0,var_degrees[i]):
			temp_e.append(d[i][e])
		edges_m.append(temp_e)

	# var edges
	edge_order_vc = []
	extrinsic_edges_vc = []
	for i in range(0, n):
		for j in range(0, var_degrees[i]):
			edge_order_vc.append(d[i][j])
			temp_edges = []
			for jj in range(0, var_degrees[i]):
				if jj != j: # extrinsic information only
					temp_edges.append(d[i][jj])
			extrinsic_edges_vc.append(temp_edges)

	# check edges
	edge_order_cv = []
	extrinsic_edges_cv = []
	for i in range(0, m):
		for j in range(0, chk_degrees[i]):
			edge_order_cv.append(u[i][j])
			temp_edges = []
			for jj in range(0, chk_degrees[i]):
				if jj != j: # extrinsic information only
					temp_edges.append(u[i][jj])
			extrinsic_edges_cv.append(temp_edges)
	code = Code()
	code.H = H
	code.G = G
	code.var_degrees = var_degrees
	code.chk_degrees = chk_degrees
	code.num_edges = num_edges
	code.u = u
	code.d = d
	code.n = n
	code.m = m
	code.k = k
	code.edges = edges
	code.edges_m = edges_m
	code.edge_order_vc = edge_order_vc
	code.extrinsic_edges_vc = extrinsic_edges_vc
	code.edge_order_cv = edge_order_cv
	code.extrinsic_edges_cv = extrinsic_edges_cv
	return code

def convert_dense_to_alist(H_filename):
	# parity-check matrix; Tanner graph parameters
	H_mat = loadmat(H_filename)
	H = H_mat['H']
	m = H.shape[0]
	n = H.shape[1]
	k = n-m

	# create a file to write parity check matrix in alist format
	f = open('data_files/LDPC_H.alist', 'w')

	# write n m to first line
	H_list = []
	r = [n, m]
	r_s = ''.join(str(r).split(','))
	H_list.append(r_s)

	# find the max var node and check node degrees
	var_degrees = np.sum(H, axis = 0) # degree of each variable node
	chk_degrees = np.sum(H, axis = 1) # degree of each check node
	vd_max = np.max(var_degrees)
	cd_max = np.max(chk_degrees)
	r = [vd_max, cd_max]
	r_s = ''.join(str(r).split(','))
	H_list.append(r_s)

	# append 2 dummy lines
	H_list.append(r_s)
	H_list.append(r_s)

	# write the locations of non zero elements for each col
	for i in range(n):
		r = (np.where(H[:,i] != 0)[0] + 1).tolist()
		r_s = ''.join(str(r).split(','))
		H_list.append(r_s)
	for j in range(m):
		r = (np.where(H[j,:] != 0)[0] + 1).tolist()
		r_s = ''.join(str(r).split(','))
		H_list.append(r_s)
	count = 0
	for r in H_list:
		count= count + 1
		r = r.replace("[", "")
		# skip adding space at the end of first 2 lines
		if (count < 3):
			r = r.replace("]", "")
		else:
			r = r.replace("]", " ")
		f.write(str(r) + "\n")
	f.close()

def apply_channel(codewords, sigma, noise, channel, FastFading, exact_llr):
	if (channel == 'AWGN'):
		received_codewords = codewords + noise
		soft_input = 2.0*received_codewords/(sigma*sigma)
	elif (channel == 'bursty'):
		# add bursty noise
		p = 0.01
		sigma_bursty = 3*sigma
		temp = np.random.binomial(1,p,np.shape(noise))
		# breakpoint()
		noise_bursty = np.multiply(temp,sigma_bursty*np.random.randn(codewords.shape[0],codewords.shape[1]))
		received_codewords = codewords + noise + noise_bursty
		soft_input = 2.0*received_codewords/(sigma*sigma)
	elif (channel == 'rayleigh_fast'): ##rayleigh fast
		data_ones = np.ones_like(codewords)
		d0 = data_ones.shape[0]
		d1 = data_ones.shape[1]
		#  Rayleigh Fading Channel, iid
		fading_h = np.sqrt(np.random.randn(d0,d1)**2 +  np.random.randn(d0,d1)**2)/np.sqrt(3.14/2.0)
		fading_h = torch.tensor(fading_h)
		received_codewords = fading_h*codewords + noise
		soft_input = 2.0*received_codewords/(sigma*sigma)

		# Pade23 approx implementation
		a1 = np.sqrt(2*np.pi/sigma)
		a3 = -1*np.sqrt(np.pi/2)*(15 - 30*np.pi + 8*(np.pi**2))/(30*(-3+np.pi)*sigma*np.sqrt(sigma))
		b2 = (-35 + 30*np.pi - 6*(np.pi**2))/(20*(-3+np.pi)*sigma)
		y = received_codewords
		lp23 = (a1*y + a3*(y**3))/(1 + b2*(y**2))

		if (exact_llr == 1):
			soft_input = 2.0*fading_h*y/(sigma*sigma)
		elif (exact_llr == 2):
			soft_input = lp23
	elif (channel == 'eva'):
		eva_filter = torch.tensor([8.5192e-04, 2.7762e-03, 6.6923e-03, 1.3914e-02, 2.6517e-02, 4.9323e-02,
		9.9711e-02, 7.1455e-01, 3.1013e-01, 1.8716e-01, 5.9571e-02, 3.6789e-02,
		1.2863e-01, 9.9646e-02, 2.2660e-02, 1.1989e-02, 1.0632e-02, 1.9607e-02,
		1.8184e-01, 1.5879e-02, 1.2813e-02, 2.0223e-02, 3.8519e-02, 1.0080e-01,
		2.9634e-01, 5.6660e-02, 2.7016e-02, 1.4812e-02, 8.6525e-03, 5.9289e-03,
		5.8595e-03, 8.5137e-03, 1.5755e-02, 4.7614e-02, 6.4232e-02, 1.7690e-02,
		8.7245e-03, 4.6893e-03, 2.4575e-03, 1.1798e-03, 4.9426e-04, 2.8556e-04,
		4.8327e-04, 8.9919e-04, 1.7837e-03, 5.5005e-03, 6.8520e-03, 1.9425e-03,
		9.6152e-04, 5.1685e-04, 2.7053e-04, 1.2964e-04, 5.3155e-05, 1.6475e-05,
		0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
		0.0000e+00], dtype=torch.float64)

		peak = torch.argmax(eva_filter)
		input_reshaped = torch.reshape(codewords, (-1, ))
		out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(eva_filter, [0,]).float().view(1, 1, -1), padding = eva_filter.shape[0] - 1).squeeze()[peak:peak+input_reshaped.shape[0]]
		received_codewords = torch.reshape(out, codewords.shape) + noise
		# breakpoint()
		soft_input = 2.0*received_codewords/(sigma*sigma)

	elif (channel == 'rayleigh_slow'): ##rayleigh slow
		data_ones = np.ones_like(codewords)
		d0 = data_ones.shape[0]
		d1 = data_ones.shape[1]
		# fading_h = tf.sqrt(tf.random.randn(data_shape[0])**2 +  tf.random.randn(data_shape[0])**2)/tf.sqrt(3.14/2.0)
		fading_h = np.sqrt(np.random.randn(d0)**2 +  np.random.randn(d0)**2)/np.sqrt(3.14/2.0)
		# fading_h = fading_h.type(tf.FloatTensor).to(self.this_device)
		fading_h = torch.tensor(fading_h)
		received_codewords = codewords*fading_h[:,None, None] + noise
		soft_input = 2.0*received_codewords/(sigma*sigma)
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
		soft_input = 2.0*received_codewords/(sigma*sigma)
	else:
		received_codewords = codewords + noise
		soft_input = 2.0*received_codewords/(sigma*sigma)

	eq = 1
	# if (eq == 1):
	# 	isi_filter = torch.Tensor([7.1455e-01, 3.1013e-01, 1.8716e-01, 5.9571e-02, 3.6789e-02,
	# 	1.2863e-01, 9.9646e-02, 2.2660e-02, 1.1989e-02, 1.0632e-02, 1.9607e-02,
	# 	1.8184e-01, 1.5879e-02, 1.2813e-02, 2.0223e-02, 3.8519e-02, 1.0080e-01,
	# 	2.9634e-01, 5.6660e-02, 2.7016e-02, 1.4812e-02, 8.6525e-03, 5.9289e-03,
	# 	5.8595e-03, 8.5137e-03, 1.5755e-02, 4.7614e-02, 6.4232e-02, 1.7690e-02,
	# 	8.7245e-03, 4.6893e-03, 2.4575e-03, 1.1798e-03, 4.9426e-04, 2.8556e-04,
	# 	4.8327e-04, 8.9919e-04, 1.7837e-03, 5.5005e-03, 6.8520e-03, 1.9425e-03,
	# 	9.6152e-04, 5.1685e-04, 2.7053e-04, 1.2964e-04, 5.3155e-05, 1.6475e-05]).float().to(device)
	# 	noise_type = 'isi_perfect'
	# 	equalizer = Equalizer(isi_filter, device)
    #     w = equalizer.get_equalizer(M = 50)
	return received_codewords, soft_input


# compute the "soft syndrome"
def syndrome(soft_output, code):
	H = code.H
	G = code.G
	n = code.n
	m = code.m
	k = code.k
	soft_syndrome = torch.tensor([]).to(soft_output.device)
	for c in range(0, m): # for each check node
		variable_nodes = []
		for v in range(0, n):
			if H[c,v] == 1: variable_nodes.append(v)
		temp = soft_output[variable_nodes]
		temp1 = torch.prod(torch.sign(temp),0)
		(temp2, min_ind_temp2) = torch.min(torch.abs(temp),0)
		soft_syndrome = torch.cat((soft_syndrome,temp2),0)
	soft_syndrome = torch.reshape(soft_syndrome,(m,soft_output.size(dim=1)))
	return soft_syndrome

# def bler_penalty(soft_output, batch_labels):



def eb_n0_to_snr(eb_n0_dB,rate,mod_bits):
	snr_offset = 0#10*np.log10(rate) + 10*np.log10(mod_bits)
	return eb_n0_dB + snr_offset

def calc_sigma(SNR,rate):
	# rate = 1
	return np.sqrt(10**(-SNR/10) / (2*rate))

def soft_bit_loss(a,b):
	# return torch.sum(a)#torch.sum(-1*(1-b)**a * b**(1-a))
	eps = 1e-12
	log_in = 1-a + eps#torch.clip(1-a,1e-6)
	return torch.sum(-torch.log(log_in))

class Equalizer():
	def __init__(self, isi_filter, device):
		self.w = None
		self.isi_filter = isi_filter
		self.device = device

	def get_equalizer(self, M=50):
		x = torch.randn(100000)
		y = F.conv1d(x.view(1, 1, -1), torch.flip(self.isi_filter.cpu(), [0,]).float().view(1, 1, -1), padding = self.isi_filter.shape[0] - 1).squeeze()[:x.shape[0]]
		e, w = self.Linear_LMS(y, x, M)
		# self.w = w.to(self.device)
		self.filter_len = M

		return w

	def Linear_LMS(self, y, x, M, step = 0.003):

		# input numpy for now

		N = len(y) - M + 1
		# Initialization
		f = torch.zeros(N)  # Filter output
		e = torch.zeros(N)  # Error signal
		w = torch.zeros(M)  # Initialise equaliser

		# Equalise
		for n in range(N):
			yn = torch.flip(y[n : n + M], [0])  #
			f[n] = torch.dot(yn, w)
			e[n] = x[n + M - 1] - f[n]
			w = w + step * yn * e[n]
			#print(w)
		return e, w

	def equalize(self, input_signal):
		input_reshaped = torch.reshape(2.0*input_signal-1.0, (-1, ))
		out = F.conv1d(input_reshaped.view(1, 1, -1), torch.flip(self.w, [0,]).float().view(1, 1, -1), padding = self.filter_len - 1).squeeze()[:input_reshaped.shape[0]]
		x_hat = torch.reshape(out, input_signal.shape)

		return x_hat