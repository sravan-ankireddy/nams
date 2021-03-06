import os

def run_sim(h,g,steps,lr,relu,nn_eq,quant_wt,max_iter,coding,chan,eb_n0_lo,eb_n0_hi,testing_batch_size,offline_test,models,adapt,freeze,ff):
	gpu_id = ["0", "1", "2", "3"]
	os.system(" CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -saved_model_path " + models[0] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights 0 -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[0] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -saved_model_path " + models[1] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights 1 -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[1] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -saved_model_path " + models[2] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights 2 -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[2] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -saved_model_path " + models[3] + " -use_saved_model 1 -adaptivity_training " + adapt + " -freeze_weights " + freeze + " -freeze_fraction " + ff + " -quantize_weights " + quant_wt + " -entangle_weights 3 -steps " + steps + " -learning_rate " + lr + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[3] + " -decoder_type neural_ms -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g )
	testing_batch_size = "24000"
	max_iter = "5"
	os.system(" CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -entangle_weights -1 -steps " + steps + " -learning_rate " + lr + " -adaptivity_training " + adapt + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[1] + " -decoder_type spa -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h + " -G_filename " + g +\
	" & CUDA_VISIBLE_DEVICES=0,1,2,3 python neural_ms.py -save_ber_to_mat 1 -use_offline_testing_data " + offline_test + " -entangle_weights -2 -steps " + steps + " -learning_rate " + lr + " -adaptivity_training " + adapt + " -relu " + relu  + " -nn_eq " + nn_eq + " -num_iterations " + max_iter + " -coding_scheme " + coding + " -channel_type " + chan + " -gpu_index " + gpu_id[2] + " -decoder_type min_sum -testing_batch_size " + testing_batch_size + " -eb_n0_train_lo " + eb_n0_lo + " -eb_n0_train_hi " + eb_n0_hi + " -H_filename "+ h+" -G_filename "+ g )

adapt = "0"
freeze = "0"
ff = "0.2"
interf = "0"
alpha = "0"
relu = "1"
coding_scheme_list = ["BCH"]
channel_type_list = ["AWGN","ETU","OTA"]
H_filename = ['H_G_mat/BCH_63_36.alist']
G_filename = ['H_G_mat/G_BCH_63_36.gmat']
lr_list = ["0.005"]
max_iter = "5"
quant_wt = "0"
nn_eq_list = ["0","1","2"]
for nn_eq in nn_eq_list:
	if (adapt == "0"):
		steps_list = ["2000","5000","10000","15000","20000","25000"]
	else:
		steps_list = ["0","1000","2000","3000","4000","5000"]
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
							if (channel_type == "ETU"):
								testing_batch_size = "2400"
								eb_n0_lo = "9"
								eb_n0_hi = "18"
								offline_ts_data = "1"
							elif (channel_type == "AWGN"):
								testing_batch_size = "2400"
								eb_n0_lo = "1"
								eb_n0_hi = "8"
								offline_ts_data = "0"
							elif (channel_type == "OTA"):
								testing_batch_size = "2400"
								eb_n0_lo = "1"
								eb_n0_hi = "4"
								offline_ts_data = "1"

							chan_tr = channel_type # need not change unless cross testing ## FIX ME
							# select the parameters based on training channel
							if (chan_tr == "ETU"):
								eb_n0_lo_tr = "9"
								eb_n0_hi_tr = "18"
							elif (chan_tr == "AWGN"):
								eb_n0_lo_tr = "1"
								eb_n0_hi_tr = "8"
							elif(chan_tr == "OTA"):
								eb_n0_lo_tr = "1"
								eb_n0_hi_tr = "4"

							# load appropriate models for inference
							models = []
							weights_path = "saved_models"
							model_prefix = weights_path + "/intermediate_models/nams_" + prefix + "_st_" + steps + "_lr_" + lr + "_" + channel_type + "_ent_"
							if (adapt == "1"):
								if (channel_type == "OTA"):
									chan_tr = "ETU"
								elif (channel_type == "ETU"):
									chan_tr = "AWGN"
								elif (channel_type == "AWGN"):
									chan_tr = "ETU"
								weights_path = "saved_models_adapt"
								model_prefix = weights_path + "/intermediate_models/nams_" + prefix + "_st_" + steps + "_lr_" + lr + "_" + channel_type + "_adapt_from_" + chan_tr + "_ent_"
		
							model_suffix = "_nn_eq_" + nn_eq + "_relu_" + relu + "_max_iter_" + max_iter  + "_" + eb_n0_lo_tr + "_" + eb_n0_hi_tr + ".pt"
							

							for im in range(4):
								filename = model_prefix + str(im) + model_suffix
								models.append(filename)

							run_sim(h,g,steps,lr,relu,nn_eq,quant_wt,max_iter,coding_scheme,channel_type,eb_n0_lo,eb_n0_hi,testing_batch_size,offline_ts_data,models,adapt,freeze,ff)