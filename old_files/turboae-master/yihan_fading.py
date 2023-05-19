import matplotlib.pylab as plt
import numpy as np

snrs =  [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# slow fading L=100 Rician K=10, finetuned from RicianK=10 Fast: Fixed interleavers uniform
# without finetune
# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -FastFading True -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 2 -init_nw_weight ./tmp/889844.pt -channel rician -rician_K 10 -FastFading False -block_len 100 -num_epoch 0 -enc_lr 0.00001 -dec_lr 0.00001 -is_interleave 1

'''
Slow Rician Fadings
'''
# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 2 -init_nw_weight ./tmp/889844.pt -channel rician -rician_K 10  -block_len 100 -num_epoch 0 -enc_lr 0.00001 -dec_lr 0.00001 -is_interleave 1 -snr_test_start 0.0 -snr_test_end 15 -snr_points 4
snrs_riciank10 =  [0.0, 5.0, 10.0, 15.0]
# enc5dec5_riciank10_uniform_slow_ber_direct = [0.036813851445913315, 0.0007065750542096794, 7.434999861288816e-05, 0.0]
# enc5dec5_riciank10_uniform_slow_bler_direct =[0.1944200000000003, 0.003959999999999998, 0.0009175, 0.0]

# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 2 -init_nw_weight ./tmp/889844.pt -channel rician -rician_K 10  -block_len 100 -num_epoch 0 -enc_lr 0.00005 -dec_lr 0.00005 -is_interleave 1 -snr_test_start 0.0 -snr_test_end 15 -snr_points 4 -num_epoch 100 -train_enc_channel_low 5 -train_enc_channel_high 5 -train_dec_channel_low 0 -train_dec_channel_high 5 -init_nw_weight ./tmp/939827.pt  -num_epoch 0
enc5dec5_riciank10_uniform_slow_ber_direct = [0.030864493921399117, 0.0008824001997709274, 5.6825007050065324e-05, 1.594999775988981e-05]
enc5dec5_riciank10_uniform_slow_bler_direct = [0.23682249999999955, 0.008094999999999922, 0.00035750000000000023, 0.00010500000000000007]

enc5dec5_riciank10_delta19_slow_ber_direct =[0.036320049315690994, 0.0007977749919518828, 0.00023887499992270023, 0.0]
enc5dec5_riciank10_delta19_slow_bler_direct = [0.1776375000000002, 0.005544999999999998, 0.001245, 0.0]

# CUDA_VISIBLE_DEVICES=3 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -num_epoch 300  -enc_lr 0.00001 -dec_lr 0.00001  -init_nw_weight ./tmp/374284.pt -channel rician -rician_K 10 -snr_test_start 0.0 -snr_test_end 15 -snr_points 4 -num_epoch 0
ti_enc5dec5_riciank10_slow_ber_direct = [0.038391198962926865, 0.001981626031920314, 8.274999345303513e-06, 0.0]
ti_enc5dec5_riciank10_slow_bler_direct =[0.18594500000000033, 0.010835, 0.000165, 0.0]


'''
Slow Rayleigh Fadings
'''
#  CUDA_VISIBLE_DEVICES=1 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 2 -init_nw_weight ./tmp/889844.pt -channel rician -rician_K 0  -block_len 100 -num_epoch 0 -enc_lr 0.00005 -dec_lr 0.00005  -is_interleave 2 -snr_test_start 0.0 -snr_test_end 20 -snr_points 6
snrs_rayleigh =  [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
enc5dec5_rayleigh_delta19_slow_ber_direct =[0.15149271488189697, 0.06951522082090378, 0.034449998289346695, 0.021241946145892143, 0.014346950687468052, 0.009719750843942165]
enc5dec5_rayleigh_delta19_slow_bler_direct =[0.4297625, 0.21841000000000005, 0.117085, 0.076705, 0.051834999999999985, 0.04453]

enc5dec5_rayleigh_uniform_slow_ber_direct = [0.15912745893001556, 0.06478334963321686, 0.029928117990493774, 0.019138723611831665, 0.01004480104893446, 0.010352176614105701]
enc5dec5_rayleigh_uniform_slow_bler_direct =  [0.45733249999999975, 0.20073750000000012, 0.10360500000000006, 0.06763250000000001, 0.048569999999999995, 0.04134]

ti_enc5dec5_rayleigh_slow_ber_direct =[0.16832904517650604, 0.07624880224466324, 0.043753646314144135, 0.02259024791419506, 0.021118301898241043, 0.014772624708712101]
ti_enc5dec5_rayleigh_slow_bler_direct = [0.4759600000000001, 0.2223725000000001, 0.132155, 0.07898250000000002, 0.07327499999999999, 0.0570975]


'''
Fast Fadings
'''
# trainable fast Rayleigh fading: Rician K=10 Trained.
# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -FastFading True -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -num_epoch 300  -enc_lr 0.00001 -dec_lr 0.00001  -init_nw_weight ./tmp/374284.pt -channel rician -rician_K 0 -FastFading True -block_len 100 -num_epoch 0
ti_enc5dec5_rayleigh_fast_ber_direct = [0.2038859874010086, 0.14305102825164795, 0.08741746842861176, 0.045597806572914124, 0.020046144723892212, 0.007152225356549025, 0.0023433510214090347, 0.0007636999944224954, 0.0002762997173704207, 0.00011467483273008838, 5.7750021369429305e-05, 2.7650006813928485e-05]
ti_enc5dec5_rayleigh_fast_bler_direct = [0.9078975000000011, 0.7964074999999987, 0.62768, 0.4335374999999994, 0.25964750000000014, 0.1360300000000001, 0.06787500000000007, 0.03367499999999994, 0.01703249999999993, 0.008574999999999935, 0.004442499999999952, 0.0022575000000000017]

enc5dec5_rayleigh_delta19_fast_ber = [0.18863339722156525, 0.13169100880622864, 0.0804588794708252, 0.041268832981586456, 0.01803663745522499, 0.006453996524214745, 0.0020071996841579676, 0.0005830003065057099, 0.00018847496539819986, 6.797497189836577e-05, 2.7099975341116078e-05, 1.0975021723425016e-05]
enc5dec5_rayleigh_delta19_fast_bler = [0.8989750000000011, 0.7806949999999976, 0.6048425000000003, 0.4048749999999991, 0.23225749999999948, 0.11419749999999977, 0.050904999999999895, 0.021894999999999897, 0.009969999999999929, 0.0044624999999999535, 0.0021425000000000016, 0.0009200000000000007]

enc5dec5_rayleigh_uniform_fast_ber = [0.18479858338832855, 0.1281643509864807, 0.07788124680519104, 0.04006822034716606, 0.017536891624331474, 0.00653547327965498, 0.002234600018709898, 0.0007297751144506037, 0.00027607494848780334, 0.00012174982111901045, 5.912496635573916e-05, 3.217498306185007e-05]

# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -FastFading True -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -num_epoch 300  -enc_lr 0.00001 -dec_lr 0.00001  -init_nw_weight ./tmp/374284.pt -channel rician -rician_K 10 -FastFading True -block_len 100 -num_epoch 0
ti_enc5dec5_riciank10_fast_ber_direct =[0.03435678780078888, 0.011114074848592281, 0.0028318508993834257, 0.0006408747867681086, 0.0001704250171314925, 6.007498086546548e-05, 2.150004729628563e-05, 8.025003808143083e-06, 2.7749993023462594e-06, 9.24999824292172e-07, 1.4999999109477358e-07, 2.4999998515795596e-08]
ti_enc5dec5_riciank10_fast_bler_direct =[0.35140000000000066, 0.17089750000000023, 0.0716875, 0.028612499999999947, 0.011862499999999911, 0.004727499999999955, 0.0017450000000000013, 0.0006700000000000005, 0.00022500000000000016, 6.250000000000004e-05, 1.25e-05, 2.5e-06]

# CUDA_VISIBLE_DEVICES=0 python3.8 main.py -num_train_dec 5 -num_train_enc 1 -num_train_justdec 0 -num_train_justenc 0  -channel rician -rician_K 10 -num_epoch 200 -num_block 50000 -train_enc_channel_low 1 -train_enc_channel_high 1 -train_dec_channel_low -1.5 -train_dec_channel_high 1 -block_len 100 -FastFading True -enc_num_layer 5 -dec_num_layer 5 -snr_test_start -1.5 -snr_test_end 4 -snr_points 12 -is_interleave 2 -init_nw_weight ./tmp/889844.pt -channel rician -rician_K 10 -FastFading False -block_len 100 -num_epoch 0 -enc_lr 0.00001 -dec_lr 0.00001 -rician_K 10 -FastFading True -is_interleave 2
enc5dec5_riciank10_delta19_fast_ber = [0.030257664620876312, 0.009743805043399334, 0.0023303749039769173, 0.00048165011685341597, 0.00011289985559415072, 3.067497891606763e-05, 9.600006706023123e-06, 2.3250013327924535e-06, 8.24999858650699e-07, 2.7500001920088835e-07, 0.0, 0.0]
enc5dec5_riciank10_delta19_fast_bler = [0.3253050000000013, 0.14754000000000042, 0.054662499999999815, 0.018929999999999936, 0.006467499999999942, 0.0023000000000000017, 0.0007925000000000006, 0.00022250000000000015, 7.000000000000005e-05, 2.7500000000000008e-05, 0.0, 0.0]

enc5dec5_riciank10_uniform_fast_ber = [0.029339682310819626, 0.009594602510333061, 0.002495999215170741, 0.0006159496260806918, 0.0001830000983318314, 6.927499634912238e-05, 2.7250009225099348e-05, 1.0800001291499939e-05, 4.0249988160212524e-06, 9.749999207997462e-07, 5.999999643790943e-07, 9.999999406318238e-08]

'''
MATLAB Benchmarks 
'''
matlab_rician_fast_SNR = [-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
matlab_rician10_fast_ber = [48.9130/1000, 26.0417/1000, 8.2117/1000, 0.8655/1000, 0.0764/1000, 0.0051/1000, 0.0013/1000,  0,   0 ,        0]



matlab_rician_slow_SNR = [i for i in range(-1, 11)]
matlab_rician10_slow_ber = [76.3508/1000,41.2227/1000,23.0716/1000, 10.9158/1000 ,5.6874/1000,2.5253/1000, 1.2732/1000,
                            0.6541/1000, 0.3033/1000, 0.1763/1000, 0.1273/1000,  0.0520/1000]


matlab_rayleigh_fast_SNR =  [-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3]
matlab_rayleigh_fast_ber = [219.1658/1000, 162.7604/1000,  120.8551/1000,   70.6638/1000,
                            33.5350/1000,   12.1164/1000,    3.8019/1000,    0.8613/1000,    0.1408/1000,    0.0248/1000]

matlab_rayleigh_slow_SNR = [i for i in range(-1, 11)]
matlab_rayleigh_slow_ber =[175.6605/1000,150.6804/1000, 117.2606/1000,106.1377/1000, 80.5684/1000,73.7706/1000,52.8850/1000,
                           43.3970/1000, 36.6283/1000, 27.6571/1000, 22.7042/1000, 17.4774/1000]



# plt.figure(1)
# plt.subplot(221)
# plt.xlabel('SNR Slow Rician (K=10) Fading L=100')
# plt.yscale('log')
# plt.ylabel('BER')
# b0, = plt.plot(matlab_rician_slow_SNR, matlab_rician10_slow_ber, 'c-*', label='MATLAB benchmark')


# p0, = plt.plot(snrs_riciank10, ti_enc5dec5_riciank10_slow_ber_direct, 'g->', label='TurboAE-TI Rician K=10 Fast Trained')
# #p1, = plt.plot(snrs_riciank10, enc5dec5_riciank10_delta19_slow_ber_direct, 'y-x', label='TurboAE-Delta19 Rician K=10 Fast Trained')
# p2, = plt.plot(snrs_riciank10, enc5dec5_riciank10_uniform_slow_ber_direct, 'r->', label='TurboAE-Uniform Rician K=10 Fast Trained')

# plt.legend(handles= [
#     b0,
#     p0,
#     #p1,
#     p2,
#                      ])
# plt.grid()

# plt.subplot(222)
# plt.yscale('log')
# plt.ylabel('BER')
# plt.xlabel('SNR Slow Rayleigh Fading L=100')

# b0, = plt.plot(matlab_rayleigh_slow_SNR, matlab_rayleigh_slow_ber, 'c-*', label='MATLAB benchmark')

# p0, = plt.plot(snrs_rayleigh, ti_enc5dec5_rayleigh_slow_ber_direct, 'g->', label='TurboAE-TI Rician K=10 Fast Trained')
# #p1, = plt.plot(snrs_rayleigh, enc5dec5_rayleigh_delta19_slow_ber_direct, 'y-x', label='TurboAE-Delta19 Rician K=10 Fast Trained')
# p2, = plt.plot(snrs_rayleigh, enc5dec5_rayleigh_uniform_slow_ber_direct, 'r->', label='TurboAE-Uniform Rician K=10 Fast Trained')

# plt.legend(handles= [
# b0,
#     p0,
#     #p1,
#     p2,
#                      ])
# plt.grid()


##START HERE

# plt.subplot(223)
# plt.xlabel('SNR Fast Rician (K=10) Fading L=100')
# plt.yscale('log')
# plt.ylabel('BER')

# b0, = plt.plot(matlab_rician_fast_SNR, matlab_rician10_fast_ber, 'c-*', label='MATLAB benchmark')


# p0, = plt.plot(snrs[:10], ti_enc5dec5_riciank10_fast_ber_direct[:10], 'g->', label='TurboAE-TI Rician K=10 Fast Trained')
# #p1, = plt.plot(snrs[:10], enc5dec5_riciank10_delta19_fast_ber[:10], 'y-x', label='TurboAE-Delta19 Rician K=10 Fast Trained')
# p2, = plt.plot(snrs[:10], enc5dec5_riciank10_uniform_fast_ber[:10], 'r->', label='TurboAE-Uniform Rician K=10 Fast Trained')

# plt.legend(handles= [
# b0,
#     p0,
#     #p1,
#     p2,
#                      ])
# plt.grid()

# plt.subplot(224)
# plt.yscale('log')
# plt.ylabel('BER')
# plt.xlabel('SNR Fast Rayleigh Fading L=100')

# b0, = plt.plot(matlab_rayleigh_fast_SNR, matlab_rayleigh_fast_ber, 'c-*', label='MATLAB benchmark')


# p0, = plt.plot(snrs, ti_enc5dec5_rayleigh_fast_ber_direct, 'g->', label='TurboAE-TI Rician K=10 Fast Trained')
# #p1, = plt.plot(snrs, enc5dec5_rayleigh_delta19_fast_ber, 'y-x', label='TurboAE-Delta19 Rician K=10 Fast Trained')
# p2, = plt.plot(snrs, enc5dec5_rayleigh_uniform_fast_ber, 'r->', label='TurboAE-Uniform Rician K=10 Fast Trained')

# plt.legend(handles= [
# b0,
#     p0,
#     #p1,
#     p2,
#                      ])
# plt.grid()


# plt.show()

fig, ax = plt.subplots()
ax.plot(matlab_rayleigh_fast_SNR, matlab_rayleigh_fast_ber, 'g>-', label='LTE Turbo')
ax.plot(snrs, ti_enc5dec5_rayleigh_fast_ber_direct, 'rv-', label='TurboAE-TI')
ax.plot(snrs, enc5dec5_rayleigh_uniform_fast_ber, 'm|-', label='TurboAE-UI')
ax.set_xlim(right=3)

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