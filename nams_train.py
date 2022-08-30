import os
import numpy as np
import pdb
import sys

def save_weights(h, g, coding_scheme, channel_type, nn_eq, training_batch_size, lr, eb_n0_train_lo, eb_n0_train_hi, offline_tr_data, max_iter, steps, relu, adapt_tr, freeze_wt, ff, models):
	
	gpu_index = ["0", "1", "2", "3"]
	# entanglement type - 0 - 5xedges, 1 - 5x1, 2 - 1x1, 3 - 1xedges, 4 - 1xm, 5 - 1xedges_per_chk_node
	ent_index = [0, 2, 3, 5] 

	os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -learning_rate " + lr + " -testing 0 -gpu_index " + gpu_index[0] +  \
	 " -num_iterations " + max_iter + " -steps " + steps + " -relu " + relu + " -adaptivity_training " + adapt_tr + " -freeze_weights " + freeze_wt + " -freeze_fraction " + ff +  " -exact_llr 0 -H_filename " + h + " -G_filename " + g + " \
	-force_all_zero 0 -channel_type " + channel_type + " -nn_eq " + nn_eq + " -coding_scheme \
	" + coding_scheme + " -training_batch_size " + training_batch_size 
	+ " -eb_n0_train_lo " + eb_n0_train_lo + " -eb_n0_train_hi " + eb_n0_train_hi + " -eb_n0_step 1 -entangle_weights " + str(ent_index[0]) +  \
	" -use_offline_training_data " + offline_tr_data + " -save_torch_model 1 -saved_model_path " + models[ent_index[0]] +\

	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -learning_rate " + lr + " -testing 0 -gpu_index " + gpu_index[1] +  \
	" -num_iterations " + max_iter + " -steps " + steps + " -relu " + relu + " -adaptivity_training " + adapt_tr + " -freeze_weights " + freeze_wt + " -freeze_fraction " + ff +  " -exact_llr 0 -H_filename " + h + " -G_filename " + g + " \
	-force_all_zero 0 -channel_type " + channel_type + " -nn_eq " + nn_eq + " -coding_scheme \
	" + coding_scheme + " -training_batch_size " + training_batch_size 
	+ " -eb_n0_train_lo " + eb_n0_train_lo + " -eb_n0_train_hi " + eb_n0_train_hi + " -eb_n0_step 1 -entangle_weights " + str(ent_index[1]) +  \
	" -use_offline_training_data " + offline_tr_data + " -save_torch_model 1 -saved_model_path " + models[ent_index[1]] +\

	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -learning_rate " + lr + " -testing 0 -gpu_index " + gpu_index[2] +  \
	" -num_iterations " + max_iter + " -steps " + steps + " -relu " + relu + " -adaptivity_training " + adapt_tr + " -freeze_weights " + freeze_wt + " -freeze_fraction " + ff +  " -exact_llr 0 -H_filename " + h + " -G_filename " + g + " \
	-force_all_zero 0 -channel_type " + channel_type + " -nn_eq " + nn_eq + " -coding_scheme \
	" + coding_scheme + " -training_batch_size " + training_batch_size 
	+ " -eb_n0_train_lo " + eb_n0_train_lo + " -eb_n0_train_hi " + eb_n0_train_hi + " -eb_n0_step 1 -entangle_weights " + str(ent_index[2]) +  \
	" -use_offline_training_data " + offline_tr_data + " -save_torch_model 1 -saved_model_path " + models[ent_index[2]] +\

	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -learning_rate " + lr + " -testing 0 -gpu_index " + gpu_index[3] +  \
	" -num_iterations " + max_iter + " -steps " + steps + " -relu " + relu + " -adaptivity_training " + adapt_tr + " -freeze_weights " + freeze_wt + " -freeze_fraction " + ff +  " -exact_llr 0 -H_filename " + h + " -G_filename " + g + " \
	-force_all_zero 0 -channel_type " + channel_type + " -nn_eq " + nn_eq + " -coding_scheme \
	" + coding_scheme + " -training_batch_size " + training_batch_size 
	+ " -eb_n0_train_lo " + eb_n0_train_lo + " -eb_n0_train_hi " + eb_n0_train_hi + " -eb_n0_step 1 -entangle_weights " + str(ent_index[3]) +  \
	" -use_offline_training_data " + offline_tr_data + " -save_torch_model 1 -saved_model_path " + models[ent_index[3]])

adapt_tr = "0"
freeze_wt = "0"
interf = "0"
alpha = "0"
nn_eq = "0"
ff_list = ["0"]
lr_list = ["0.005"]#,"0.01"]

steps = "20000"
if (adapt_tr == "1"):
	steps = "20000"

relu = "1"
coding_scheme_list = ["LDPC"]
# channel_type_list = ["EVA_df_5"]
channel_type_list = ["ETU_df_0"]

# channel_type_list = ["AWGN"]

# H_filename = ['H_G_mat/BCH_63_36.alist']
# G_filename = ['H_G_mat/G_BCH_63_36.gmat']

H_filename = ['H_G_mat/LDPC_128_64.alist']
G_filename = ['H_G_mat/G_LDPC_128_64.gmat']

# H_filename = ['H_G_mat/LDPC_384_192.alist']
# G_filename = ['H_G_mat/G_LDPC_384_192.gmat']

# H_filename = ['H_G_mat/LDPC_384_320.alist']
# G_filename = ['H_G_mat/G_LDPC_384_320.gmat']

# force_idx = 0

max_iter = ["5"]
for mi in max_iter:
	for ff in ff_list:
		for i_f in range(1):
			# i_f = force_idx
			for lr in lr_list:
				for coding_scheme in coding_scheme_list:
					for channel_type in channel_type_list:
						steps_tr = steps
						if (coding_scheme == "BCH"):
							h = H_filename[i_f]
							g = G_filename[i_f]
							# prefix procesisng 
							temp = h.split("/")
							temp = temp[-1]
							prefix = temp.split(".")
							prefix = prefix[0]

							if (channel_type == "ETU_df_0"):
								training_batch_size = "160"
								eb_n0_train_lo_tr = "15"
								eb_n0_train_hi_tr = "30"
								offline_tr_data = "1"
							elif (channel_type == "EVA_df_5"):
								training_batch_size = "160"
								eb_n0_train_lo_tr = "10"
								eb_n0_train_hi_tr = "25"
								offline_tr_data = "1"
							elif (channel_type == "HST"):
								training_batch_size = "100"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "10"
								offline_tr_data = "1"
							elif (channel_type == "AWGN"):
								training_batch_size = "80"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "8"
								offline_tr_data = "0"
							elif(channel_type == "OTA"):
								training_batch_size = "60"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "4"
								offline_tr_data = "1"
							elif(channel_type == "EVA"):
								training_batch_size = "80"
								eb_n0_train_lo_tr = "5"
								eb_n0_train_hi_tr = "12"
								offline_tr_data = "1"

							# adapt config : load model from one channel model; continue training on another channel model
							chan_tr = channel_type
							if (adapt_tr == "1"):
								chan_tr = "AWGN"
								steps_tr = "5000"
								if (chan_tr == "ETU"):
									eb_n0_train_lo_adapt_tr = "18"
									eb_n0_train_hi_adapt_tr = "33"
								elif (chan_tr == "AWGN"):
									eb_n0_train_lo_adapt_tr = "1"
									eb_n0_train_hi_adapt_tr = "8"
							# load appropriate orginal models for futher training (if adaptivity)
							### FIX ME
							if (relu == "1"):
								models_folder = "saved_models_sf_df"
							else:
								models_folder = "saved_models_new_mod"
							saved_models_new = []
							prev_steps = "19000"
							if (adapt_tr == "1"):
								prev_steps = "20000"
							model_prefix = models_folder +  "/intermediate_models/nams_" + prefix + "_st_" + prev_steps + "_lr_" + lr + "_" + chan_tr + "_ent_"
							if (adapt_tr == "1"):
								model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + mi + "_" + eb_n0_train_lo_adapt_tr + "_" + eb_n0_train_hi_adapt_tr + ".pt"
							else:
								model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + mi + "_" + eb_n0_train_lo_tr + "_" + eb_n0_train_hi_tr + ".pt"
							
							for im in range(4):
								filename = model_prefix + str(im) + model_suffix
								saved_models_new.append(filename)
							save_weights(h, g, coding_scheme, channel_type, nn_eq, training_batch_size, lr, eb_n0_train_lo_tr, eb_n0_train_hi_tr, offline_tr_data, mi, steps_tr, relu, adapt_tr, freeze_wt, ff, saved_models_new)
						
						elif (coding_scheme == "LDPC"):
							h = H_filename[i_f]
							g = G_filename[i_f]
							# prefix procesisng 
							temp = h.split("/")
							temp = temp[-1]
							prefix = temp.split(".")
							prefix = prefix[0]

							if (channel_type == "ETU_df_0"):
								# training_batch_size = "100"
								# eb_n0_train_lo_tr = "3"
								# eb_n0_train_hi_tr = "12"
								training_batch_size = "140"
								# eb_n0_train_lo_tr = "11"
								# eb_n0_train_hi_tr = "24"
								eb_n0_train_lo_tr = "7"
								eb_n0_train_hi_tr = "20"
								offline_tr_data = "1"
							elif (channel_type == "EVA_df_5"):
								training_batch_size = "160"
								eb_n0_train_lo_tr = "10"
								eb_n0_train_hi_tr = "25"
								offline_tr_data = "1"
							elif (channel_type == "HST"):
								training_batch_size = "100"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "10"
								offline_tr_data = "1"
							elif (channel_type == "AWGN"):
								training_batch_size = "60"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "6"
								offline_tr_data = "0"
							elif(channel_type == "OTA"):
								training_batch_size = "60"
								eb_n0_train_lo_tr = "1"
								eb_n0_train_hi_tr = "4"
								offline_tr_data = "1"
							elif(channel_type == "EVA"):
								training_batch_size = "80"
								eb_n0_train_lo_tr = "5"
								eb_n0_train_hi_tr = "12"
								offline_tr_data = "1"

							# adapt config : load model from one channel model; continue training on another channel model
							chan_tr = channel_type
							if (adapt_tr == "1"):
								chan_tr = "AWGN"
								steps_tr = "5000"
								if (chan_tr == "ETU"):
									eb_n0_train_lo_adapt_tr = "18"
									eb_n0_train_hi_adapt_tr = "33"
								elif (chan_tr == "AWGN"):
									eb_n0_train_lo_adapt_tr = "1"
									eb_n0_train_hi_adapt_tr = "8"
							# load appropriate orginal models for futher training (if adaptivity)
							### FIX ME
							if (relu == "1"):
								models_folder = "saved_models_sf_df"
							else:
								models_folder = "saved_models_new_mod"
							saved_models_new = []
							prev_steps = "19000"
							if (adapt_tr == "1"):
								prev_steps = "20000"
							model_prefix = models_folder +  "/intermediate_models/nams_" + prefix + "_st_" + prev_steps + "_lr_" + lr + "_" + chan_tr + "_ent_"
							if (adapt_tr == "1"):
								model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + mi + "_" + eb_n0_train_lo_adapt_tr + "_" + eb_n0_train_hi_adapt_tr + ".pt"
							else:
								model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + mi + "_" + eb_n0_train_lo_tr + "_" + eb_n0_train_hi_tr + ".pt"
							
							for im in range(6):
								filename = model_prefix + str(im) + model_suffix
								saved_models_new.append(filename)
							save_weights(h, g, coding_scheme, channel_type, nn_eq, training_batch_size, lr, eb_n0_train_lo_tr, eb_n0_train_hi_tr, offline_tr_data, mi, steps_tr, relu, adapt_tr, freeze_wt, ff, saved_models_new)
