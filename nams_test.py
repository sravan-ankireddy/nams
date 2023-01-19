import os
import time

def run_sim(h,g,set_model,steps,lr,relu,nn_eq,quant_wt,max_iter,coding,chan,alpha,eb_n0_lo,eb_n0_hi,testing_batch_size,offline_test,models,adapt,freeze,ff):


	gpu_id = ["0", "1", "2", "3"]



	# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
	ent_id = [0,1,2,6]
	os.system(" CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[0]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[0]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[0] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[1]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[1]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[1] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[2]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[2]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[2] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[3]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[3]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[3] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g )
	
	time.sleep(60)
	testing_batch_size_bp = "10000"
	max_iter = "5"
	os.system(" CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -entangle_weights -1 -steps " + steps + " -learning_rate " + lr + " -adaptivity_training " + adapt + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[0] + " -decoder_type spa -testing_batch_size " + testing_batch_size_bp + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -entangle_weights -2 -steps " + steps + " -learning_rate " + lr + " -adaptivity_training " + adapt + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[1] + " -decoder_type min_sum -testing_batch_size " + testing_batch_size_bp + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h+" -G_filename "+ g )
 
	time.sleep(60)
	gpu_id = ["0", "1", "2"]
	# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
	ent_id = [3,4,5]
	os.system(" CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[0]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[0]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[0] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[1]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[1]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[1] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -set_model " + set_model + " -saved_model_path " + models[ent_id[2]] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights " + str(ent_id[2]) + " -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -alpha " + alpha + " -gpu_index " + gpu_id[2] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_lo " + eb_n0_lo + " -eb_n0_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g)# +\
	

adapt = "0"
freeze = "0"
ff = "0.2"
interf = "0"
alpha = "0.3"
relu = "1"
coding_scheme_list = ["BCH"]
channel_type_list = ["ETU_df_0"]
# channel_type_list = ["OTA"]
# channel_type_list = ["alpha_interf"]#["interf_4","interf_2"]

H_filename = ['H_G_mat/BCH_63_36.alist']
G_filename = ['H_G_mat/G_BCH_63_36.gmat']

# coding_scheme_list = ["LDPC"]
# channel_type_list = ["ETU_df_0"]
# # H_filename = ['H_G_mat/LDPC_128_64.alist']
# # G_filename = ['H_G_mat/G_LDPC_128_64.gmat']

# H_filename = ['H_G_mat/LDPC_384_320.alist']
# G_filename = ['H_G_mat/G_LDPC_384_320.gmat']

# if (args.channel_type == "alpha_interf"):

set_model = "gw"
models_folder_main = "saved_models_" + set_model

lr_list = ["0.005"]
max_iter = "5"
quant_wt = "0"
nn_eq_list = ["1"]#,"1","2"]#,"1","2"]	
for nn_eq in nn_eq_list:
	if (adapt == "0"):
		steps_list = ["20000","15000","10000","5000","2000"]
		steps_list = ["20000"]#["20000","15000","10000","5000","2000"]
	else:
		steps_list = ["1000"]
	for i_f in range(1):
		for lr in lr_list:
			for steps in steps_list:
				for coding_scheme in coding_scheme_list:
					for channel_type in channel_type_list:
         
						if (coding_scheme == "BCH"):
							h = H_filename[i_f]
							g = G_filename[i_f]

							# prefix procesisng 
							temp = h.split("/")
							temp = temp[-1]
							prefix = temp.split(".")
							prefix = prefix[0]

							# select the parameters based on testing channel
							if (channel_type == "ETU_df_0"):
								testing_batch_size = "2000"
								eb_n0_lo = "5"
								eb_n0_hi = "22"
								offline_ts_data = "1"
							elif (channel_type == "EVA_df_5"):
								testing_batch_size = "2400"
								eb_n0_lo = "10"
								eb_n0_hi = "25"
								offline_ts_data = "1"
							elif (channel_type == "HST"):
								testing_batch_size = "2400"
								eb_n0_lo = "1"
								eb_n0_hi = "10"
								offline_ts_data = "1"
							elif (channel_type == "AWGN"):
								testing_batch_size = "4000"
								eb_n0_lo = "1"
								eb_n0_hi = "8"
								offline_ts_data = "0"
								offline_ts_data = "0"
							elif (channel_type == "alpha_interf"):
								testing_batch_size = "4000"
								eb_n0_lo = "1"
								eb_n0_hi = "12"
								offline_ts_data = "0"
							elif (channel_type == "bursty_p1" or channel_type == "bursty_p2" or channel_type == "bursty_p3" or channel_type == "bursty_p4"):
								testing_batch_size = "4000"
								eb_n0_lo = "1"
								eb_n0_hi = "18"
								offline_ts_data = "0"
							elif (channel_type == "interf_2" or channel_type == "interf_4" or "interf_6" or channel_type == "interf_8"):
								testing_batch_size = "4000"
								eb_n0_lo = "1"
								eb_n0_hi = "16"
								offline_ts_data = "0"
							elif (channel_type == "OTA"):
								testing_batch_size = "1200"
								eb_n0_lo = "0"
								eb_n0_hi = "8"
								offline_ts_data = "1"

							chan_tr = channel_type 
							# select the parameters based on training channel
							if (chan_tr == channel_type):
								eb_n0_lo_tr = eb_n0_lo
								eb_n0_hi_tr = eb_n0_hi
							else:
								if (chan_tr == "ETU"):
									eb_n0_lo_tr = "13"
									eb_n0_hi_tr = "28"
								elif (chan_tr == "HST"):
									eb_n0_lo_tr = "1"
									eb_n0_hi_tr = "10"
								elif (chan_tr == "AWGN"):
									eb_n0_lo_tr = "1"
									eb_n0_hi_tr = "8"
								elif (chan_tr == "bursty"):
									eb_n0_lo_tr = "1"
									eb_n0_hi_tr = "16"
								elif(chan_tr == "OTA"):
									eb_n0_lo_tr = "1"
									eb_n0_hi_tr = "4"

							# load appropriate models for inference
							models = []
							weights_path = models_folder_main

							if (adapt == "1"):
								if (channel_type == "OTA"):
									chan_tr = "ETU"
								elif (channel_type == "ETU_df_0"):
									chan_tr = "AWGN"
								elif (channel_type == "AWGN"):
									chan_tr = "ETU"

							model_prefix_main = models_folder_main + "/" + channel_type + "/arch_" + nn_eq + "/"
							model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + max_iter  + "_" + eb_n0_lo_tr + "_" + eb_n0_hi_tr + ".pt"
							if (adapt == "1"):
								model_prefix_main = models_folder_main + "_adapt_from_" + chan_tr + "_to_" + channel_type + "/" + channel_type + "/arch_" + nn_eq + "/"
							for im in range(7):
								model_prefix = model_prefix_main + "ent_" + str(im) + "/lr_" + lr + "/intermediate_models/nams_" + prefix + "_st_" + steps + "_lr_" + lr + "_" + channel_type 
								if (adapt == "1"):
									model_prefix = model_prefix + "_adapt_from_" + chan_tr
								model_prefix = model_prefix + "_ent_"

								filename = model_prefix + str(im) + model_suffix
								models.append(filename)
							# eb_n0_lo = "13"
							# eb_n0_hi = "16"
							run_sim(h,g,set_model,steps,lr,relu,nn_eq,quant_wt,max_iter,coding_scheme,channel_type,alpha,eb_n0_lo,eb_n0_hi,testing_batch_size,offline_ts_data,models,adapt,freeze,ff)
