import matplotlib.pylab as plt
import numpy as np

fading_types = ['Rayleigh', 'Rician K=10', 'Rician K=20', 'AWGN']

# awgn trainable
# minimum euclidean (L2) distance 5.327668190002441
# ave euclidean (L2) distance 8.954754513379765
# minimum L1 distance 14.31308799982071
# ave euclidean (L1) distance 41.594728979605264
# minimum Hamming distance 7.0
# ave Hamming distance 20.52003910068426

# AWGn uniform
# minimum euclidean (L2) distance 5.981672763824463
# ave euclidean (L2) distance 10.635093055046433
# minimum L1 distance 18.2636285610497
# ave euclidean (L1) distance 56.43194071390023
# minimum Hamming distance 6.0
# ave Hamming distance 23.087487781036167

# rician K=20 slow trainable
# minimum euclidean (L2) distance 5.745677947998047
# ave euclidean (L2) distance 11.58922898338221
# minimum L1 distance 13.738670647144318
# ave euclidean (L1) distance 56.70768340401417
# minimum Hamming distance 3.0
# ave Hamming distance 20.023643695014663

# rician K=20 fast trainable
# minimum euclidean (L2) distance 5.757374286651611
# ave euclidean (L2) distance 9.711186652309385
# minimum L1 distance 16.918895289301872
# ave euclidean (L1) distance 49.42031017089008
# minimum Hamming distance 6.0
# ave Hamming distance 21.73281708211144

# rician K=10 slow trainable
# minimum euclidean (L2) distance 4.382943153381348
# ave euclidean (L2) distance 8.742898681879277
# minimum L1 distance 12.313471800647676
# ave euclidean (L1) distance 43.623770433557574
# minimum Hamming distance 1.0
# ave Hamming distance 12.577376588465299

# rician  K=10 fast trainable
# minimum euclidean (L2) distance 6.000196933746338
# ave euclidean (L2) distance 9.99589614644428
# minimum L1 distance 19.669470369815826
# ave euclidean (L1) distance 54.120136843439724
# minimum Hamming distance 10.0
# ave Hamming distance 26.025415444770285
# rician K=10 fast uniform
# minimum euclidean (L2) distance 5.476746559143066
# ave euclidean (L2) distance 10.416772627993646
# minimum L1 distance 15.327994074905291
# ave euclidean (L1) distance 60.60505038129237
# minimum Hamming distance 3.0
# ave Hamming distance 26.970903592375368

# rician K=10 slow uniform
# minimum euclidean (L2) distance 5.3355278968811035
# ave euclidean (L2) distance 11.423483512035679
# minimum L1 distance 11.775017730426043
# ave euclidean (L1) distance 51.47918470940013
# minimum Hamming distance 4.0
# ave Hamming distance 19.068594208211145

# rician K=20 fast uniform
# minimum euclidean (L2) distance 5.435845851898193
# ave euclidean (L2) distance 9.408063752443793
# minimum L1 distance 16.060159504413605
# ave euclidean (L1) distance 46.189319155125254
# minimum Hamming distance 8.0
# ave Hamming distance 21.52101661779081

# rician K=20 slow uniform
# minimum euclidean (L2) distance 4.796055316925049
# ave euclidean (L2) distance 9.944709188660802
# minimum L1 distance 11.835263133049011
# ave euclidean (L1) distance 46.20891865235532
# minimum Hamming distance 4.0
# ave Hamming distance 18.553228861192572

# rayleigh fast trainable
# minimum euclidean (L2) distance 5.5286149978637695
# ave euclidean (L2) distance 9.772794095185729
# minimum L1 distance 17.23502689599991
# ave euclidean (L1) distance 52.227812575645665
# minimum Hamming distance 9.0
# ave Hamming distance 26.025415444770285

# rayleigh slow trainable
# minimum euclidean (L2) distance 4.06403923034668
# ave euclidean (L2) distance 8.12512123503177
# minimum L1 distance 13.730320014059544
# ave euclidean (L1) distance 46.811244487729155
# minimum Hamming distance 5.0
# ave Hamming distance 21.382041177908114

# Turbo LTE
# minimum euclidean (L2) distance 6.928203230275509
# ave euclidean (L2) distance 12.13065332151889
# minimum L1 distance 24.0
# ave euclidean (L1) distance 74.07233626588466
# minimum Hamming distance 12.0
# ave Hamming distance 37.03616813294233​

turbolte_minl2 = [6.92 for _ in range(4)]
turbolte_avel2 = [12.13 for _ in range(4)]
turbolte_minl1 = [24.0 for _ in range(4)]
turbolte_avel1 = [74.07 for _ in range(4)]

# minimum euclidean (L2) distance 6.0
# ave euclidean (L2) distance 12.193094268794137
# minimum L1 distance 18.0
# ave euclidean (L1) distance 75.0733137829912
# minimum Hamming distance 9.0
# ave Hamming distance 37.5366568914956

turbo757_minl2 = [6.0 for _ in range(4)]
turbo757_avel2 = [12.19 for _ in range(4)]
turbo757_minl1 = [18.0 for _ in range(4)]
turbo757_avel1 = [75.07 for _ in range(4)]

trainablem_fastfading_minl2 = [5.286, 6.00,  5.76, 5.32]
trainablem_fastfading_avel2 = [9.773, 9.99, 9.71, 8.95]
trainablem_fastfading_minl1 = [17.23, 19.67, 16.91, 14.31]
trainablem_fastfading_avel1 = [52.22, 54.12, 49.42, 41.59]

uniformm_fastfading_minl2 = [5.31, 5.47,  5.43, 5.98]
uniformm_fastfading_avel2 = [0, 10.41, 9.41, 10.63]
uniformm_fastfading_minl1 = [0, 15.33, 16.06, 18.26]
uniformm_fastfading_avel1 = [0, 60.60, 46.189, 56.43]

trainable_fastfading_minl2 = [5.390,  5.99, 5.75, 5.32]
trainable_fastfading_avel2 = [10.01,  9.90, 9.79, 8.96]
trainable_fastfading_minl1 = [16.31, 19.59, 16.90, 14.27]
trainable_fastfading_avel1 = [54.90, 53.10, 50.28, 41.66]

uniform_fastfading_minl2 = [0, 5.198, 5.13, 5.46]
uniform_fastfading_avel2 = [0, 9.18, 10.22, 10.84]
uniform_fastfading_minl1 = [0, 13.00, 13.01, 13.08]
uniform_fastfading_avel1 = [0, 42.63, 58.71, 55.51]


# plt.figure(1)
# plt.subplot(221)
# plt.title('Minimum Distances')
# plt.ylabel('Code Distance')
# b0, = plt.plot(fading_types[1:], turbo757_minl2[1:], '-', label='Turbo-757 Uniform')
# b1, = plt.plot(fading_types[1:], turbolte_minl2[1:], '-', label='Turbo-LTE Uniform')
# p1, = plt.plot(fading_types[1:], trainablem_fastfading_minl2[1:], '-.', label='Trainable Interleave Min L2')
# p2, = plt.plot(fading_types[1:], uniformm_fastfading_minl2[1:], '->', label='Uniform Interleave Min L2')
# plt.legend(handles = [p1, p2, b0, b1])
# plt.grid()

# plt.subplot(222)
# plt.title('Average Distances')
# #plt.xlabel('Training Channel Type')
# plt.ylabel('Code Distance')

# b0, = plt.plot(fading_types[1:], turbo757_avel2[1:], '-', label='Turbo-757 Uniform')
# b1, = plt.plot(fading_types[1:], turbolte_avel2[1:], '-', label='Turbo-LTE Uniform')
# p1, = plt.plot(fading_types[1:], trainablem_fastfading_avel2[1:], '-.', label='Trainable Interleave Ave L2')
# p2, = plt.plot(fading_types[1:], uniformm_fastfading_avel2[1:], '->', label='Uniform Interleave Ave L2')
# plt.legend(handles=[p1, p2, b0, b1])
# plt.grid()

# plt.subplot(223)
# plt.title('Minimum Distances')
# #plt.xlabel('Training Channel Type')
# plt.ylabel('Code Distance')

# b0, = plt.plot(fading_types[1:], turbo757_minl1[1:], '-', label='Turbo-757 Uniform')
# b1, = plt.plot(fading_types[1:], turbolte_minl1[1:], '-', label='Turbo-LTE Uniform')
# p1, = plt.plot(fading_types[1:], trainablem_fastfading_minl1[1:], '-.', label='Trainable Interleave Min L1')
# p2, = plt.plot(fading_types[1:], uniformm_fastfading_minl1[1:], '->', label='Uniform Interleave Min L1')
# plt.legend(handles=[p1, p2, b0, b1])
# plt.grid()

# plt.subplot(224)
# plt.title('Average Distances')
# #plt.xlabel('Training Channel Type')
# plt.ylabel('Code Distance')
# b0, = plt.plot(fading_types[1:], turbo757_avel1[1:], '-', label='Turbo-757 Uniform')
# b1, = plt.plot(fading_types[1:], turbolte_avel1[1:], '-', label='Turbo-LTE Uniform')
# p1, = plt.plot(fading_types[1:], trainablem_fastfading_avel1[1:], '-.', label='Trainable Interleave Ave L1')
# p2, = plt.plot(fading_types[1:], uniformm_fastfading_avel1[1:], '->', label='Uniform Interleave Ave L1')
# plt.legend(handles=[p1, p2, b0, b1])
# plt.grid()

# plt.figure(2)
# plt.ylabel('Partial Minimum Distances')
# b0, = plt.plot(fading_types[0:], turbo757_minl2[0:], '-*', label='Turbo-757')
# b1, = plt.plot(fading_types[0:], turbolte_minl2[0:], '-', label='Turbo-LTE')
# p1, = plt.plot(fading_types[0:], trainablem_fastfading_minl2[0:], '-.', label='TI')
# p2, = plt.plot(fading_types[0:], uniformm_fastfading_minl2[0:], '->', label='UI')
# plt.legend(handles=[p1, p2, b0, b1])
# plt.grid()


# plt.show()


plt.plot(fading_types[0:], turbo757_minl2[0:], 'rv-', label='757 Turbo')
plt.plot(fading_types[0:], turbolte_minl2[0:], 'm*-', label='LTE Turbo')
plt.plot(fading_types[0:], trainablem_fastfading_minl2[0:], 'co-', label='TurboAE-TI')
plt.plot(fading_types[0:], uniformm_fastfading_minl2[0:], 'y|-', label='TurboAE-UI')

#plt.xlabel('EB/N0',fontsize=12)
plt.ylabel('Partial Minimum Distances',fontsize=16)
plt.legend(loc='best',fontsize=13)
#plt.yscale('log')
plt.grid()
#plt.title("Robustness to AWGN + Chirp (10 consecutive positions)")
plt.xticks(fontsize=16)
plt.yticks(fontsize=13)
plt.savefig('intrlv.png')
plt.show()