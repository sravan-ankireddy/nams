import matplotlib.pylab as plt
import numpy as np


snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
##added
snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

# CUDA_VISIBLE_DEVICES=1 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0 -init_nw_weight ./tmp/trainable_interleaver_40_circularpadd_rician_fast.pt -channel awgn -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -init_nw_weight default -num_epoch 300 -channel awgn -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -init_nw_weight ./tmp/202532.pt -num_epoch 50 -init_nw_weight ./tmp/875634.pt
uniform_awgntrain_ber = [0.07801837474107742, 0.03662964329123497, 0.014235983602702618, 0.0050713238306343555, 0.0019052757415920496, 0.0007803741609677672, 0.000343599880579859, 0.00014907489821780473, 5.915001020184718e-05, 2.2650034225080162e-05, 6.800004030083073e-06, 1.950000068973168e-06]
#added
uniform_awgntrain_ber = [0.07801837474107742, 0.03662964329123497, 0.014235983602702618, 0.0050713238306343555, 0.0019052757415920496, 0.0007803741609677672, 0.000343599880579859, 0.00014907489821780473]
# CUDA_VISIBLE_DEVICES=2 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0 -init_nw_weight ./tmp/trainable_interleaver_40_circularpadd_rician_fast.pt -channel awgn -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -channel rician -rician_K 10 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 1 -init_nw_weight ./tmp/837697.pt -channel awgn -num_epoch 0
uniform_riciank10train_ber = [0.061299193650484085, 0.02606077678501606, 0.009392677806317806, 0.0031689514871686697, 0.001162724569439888, 0.00046694985940121114, 0.00018719997024163604, 7.055000605760142e-05, 2.805000804073643e-05, 8.625002010376193e-06, 2.175000645365799e-06, 7.499998559978849e-07]
#added
uniform_riciank10train_ber = [0.061299193650484085, 0.02606077678501606, 0.009392677806317806, 0.0031689514871686697, 0.001162724569439888, 0.00046694985940121114, 0.00018719997024163604, 7.055000605760142e-05]
# CUDA_VISIBLE_DEVICES=2 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0 -init_nw_weight ./tmp/trainable_interleaver_40_circularpadd_rician_fast.pt -channel awgn -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12  -channel awgn -num_epoch 300 -init_nw_weight ./new_results/turboae_awgn_trainable_intrlvr.pt
trainable_awgntrain_1dB_ber =[0.04568531736731529, 0.02442820742726326, 0.013236110098659992, 0.007187997922301292, 0.0038860251661390066,
 0.002056199125945568, 0.0010209501488134265, 0.0004796746652573347, 0.0002147749619325623, 8.777500625001267e-05,
 3.0474981031147763e-05, 1.0775017472042236e-05]

trainable_riciank10train_1dB_enc5_ber =[0.10253065079450607, 0.04807106778025627, 0.01699242554605007, 0.0044646780006587505, 0.0009462750749662519,
     0.0002210499660577625, 6.747501174686477e-05, 2.2125041141407564e-05, 7.675003871554509e-06,
     2.3250008780451026e-06, 6.000000212225132e-07, 1.2499999968440534e-07]
#added
trainable_riciank10train_1dB_enc5_ber =[0.10253065079450607, 0.04807106778025627, 0.01699242554605007, 0.0044646780006587505, 0.0009462750749662519,
     0.0002210499660577625, 6.747501174686477e-05, 2.2125041141407564e-05]

# MATLAB venchmarks
snr_matlab = [item - 10 * np.log10(96.0/100) for item in [ -2, -1 ,    0 ,    1 ,    2]]
matlab_ber = [0.2146 , 0.0554,  0.0051,  7.0134e-05, 5.0000e-07]

# plt.figure(1)
# plt.title('AWGN performance')
# plt.yscale('log')
# plt.ylabel('BER')
# plt.xlabel('L=100 SNR')
fig, ax = plt.subplots()
ax.plot(snr_matlab, matlab_ber, 'g>-', label='LTE Turbo')
ax.plot(snrs, uniform_awgntrain_ber, 'm*-', label='TurboAE-UI trained on AWGN')
ax.plot(snrs, uniform_riciank10train_ber, 'y--', label='TurboAE-UI trained on Rician')
ax.set_xlim(left=-1.5,right=1.5)
# c1, = plt.plot(snrs, trainable_awgntrain_1dB_ber, 'y-<', label='Trainable: AWGN Trained')
plt.plot(snrs, trainable_riciank10train_1dB_enc5_ber, 'co-', label='TurboAE-TI trained on Rician')

plt.xlabel('EB/N0',fontsize=15)
plt.ylabel('BER',fontsize=15)
plt.legend(loc='best',fontsize=13)
plt.yscale('log')
plt.grid()
#plt.title("Robustness to AWGN + Chirp (10 consecutive positions)")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig('intrlv.png')
plt.show()