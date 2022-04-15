import numpy as np
from scipy.io import loadmat
import tensorflow as tf
# np.random.seed(0)
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
		if "BCH" in H_filename: # dear God please fix this
			G = np.loadtxt(G_filename).astype(np.int)
			G = G.transpose()
		else:
			P = np.loadtxt(G_filename,skiprows=2)
			G = np.vstack([P.transpose(), np.eye(k)]).astype(np.int)
    
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
	f = open('LDPC_H.alist', 'w')

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

def apply_channel(codewords, noise, channel, FastFading):
	received_codewords = np.zeros_like(codewords)
	# if (channel == 'fading_iid'): ##rayleigh fast
	# 	data_shape = codewords.shape
	# 	#  Rayleigh Fading Channel, iid
	# 	fading_h = tf.math.sqrt(tf.random.randn(data_shape)**2 +  tf.random.randn(data_shape)**2)/tf.math.sqrt(3.14/2.0)
	# 	# fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
	# 	received_codewords = fading_h*codewords + noise
	# elif (channel == 'fading_fixed'): ##rayleigh slow
	# 	data_shape = codewords.shape
	# 	fading_h = tf.sqrt(tf.random.randn(data_shape[0])**2 +  tf.random.randn(data_shape[0])**2)/tf.sqrt(3.14/2.0)
	# 	# fading_h = fading_h.type(tf.FloatTensor).to(self.this_device)
	# 	received_codewords = codewords*fading_h[:,None, None] + noise
	if (channel == 'rician'):
		
		K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
		coeffLOS = np.sqrt(K/(K+1))
		coeffNLOS = np.sqrt(1/(K+1))
		if FastFading:
			hLOSReal = np.ones_like(codewords) #Assuming SISO see page 3.108 in Heath and Lazano
			hLOSImag = np.ones_like(codewords)
			hNLOSReal = np.random.randn(np.shape(codewords))
			hNLOSImag = np.random.randn(np.shape(codewords))
		else: #Slow fading case
			hLOSReal = np.ones_like(1) #Assuming SISO see page 3.108 in Heath and Lazano
			hLOSImag = np.ones_like(1)
			hNLOSReal = np.random.randn(1)
			hNLOSImag = np.random.randn(1)
		fading_h = coeffLOS*(hLOSReal + hLOSImag*1j) + coeffNLOS*(hNLOSReal + hNLOSImag*1j) 
		#Assuming phase information at the receiver
		fading_h = np.abs(fading_h)/np.sqrt(3.14/2.0)
		# fading_h = fading_h.type(torch.FloatTensor).to(self.this_device) 
		received_codewords = fading_h*codewords + noise
	else:
		received_codewords = codewords + noise
	return received_codewords


# compute the "soft syndrome"
def syndrome(soft_output, code):
	H = code.H
	G = code.G
	n = code.n
	m = code.m
	k = code.k
	soft_syndrome = []
	for c in range(0, m): # for each check node
		variable_nodes = []
		for v in range(0, n):
			if H[c,v] == 1: variable_nodes.append(v)
		temp = tf.gather(soft_output,variable_nodes)
		temp1 = tf.reduce_prod(tf.sign(temp),0)
		temp2 = tf.reduce_min(tf.abs(temp),0)
		soft_syndrome.append(temp1 * temp2)
	soft_syndrome = tf.stack(soft_syndrome)
	return soft_syndrome