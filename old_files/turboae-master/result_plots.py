
import numpy as np
import matplotlib.pyplot as plt
import math


snr_interval = (4.0 + 1.5)* 1.0 /  (12-1)

##Rayleigh fast
snrs = np.array([-1,0,1,2,3,4])

##Rician fast
#snrs = np.array([-2,-1,0,1])
##awgn
# snrs_awgn_turbo_lte_baseline = np.array([-2,-1,0,1,2,3,4])
#snrs = np.array([-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,3.5,4])
snrs = np.array([-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3])
snrs_awgn_turbo_lte_baseline = np.array([-2,-1,0,1,2,3])
# snrs_awgn_757_baselines = np.array([-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3])

##Rayleigh slow
#snrs = np.array([0,4,8,12,16,20])

##bursty awgn
#snrs = np.array([-2,-1,0,1,2,3,4])

##rician slow
#snrs = np.array([0,5,10,15]) old stuff
#snrs = np.array([0,2.5,5,7.5,10])

#rician fast for bursty type 2
#snrs = np.array([-2,-1,0,1,2,3,4,5])
##cut some snr points
snrs = np.array([-2,-1,0,1,2])

##snr chirp
#snrs = np.array([-1,0,1,2,3,4])

##BL=40 rayleigh fading iid ~700 epochs
#uniform=np.array([0.11695440858602524, 0.08530794084072113, 0.05874885618686676, 0.03794756904244423, 0.023485181853175163, 0.013895643875002861, 0.007929530926048756, 0.004444996826350689, 0.0024968713987618685, 0.0014074158389121294, 0.0008137488039210439, 0.0004542078822851181])
# trainable=np.array([0.12237215042114258, 0.08699636161327362, 0.057937055826187134, 0.03564755991101265, 0.02044621855020523, 0.010977153666317463, 0.00555795431137085, 0.0027099137660115957, 0.0013239565305411816, 0.0006400827551260591, 0.0003276245843153447, 0.00016699984553270042])
#baseline=(1/1000)*np.array([190.8467,  148.5015 , 111.7851  , 78.0840  , 49.3291   ,25.4578  , 13.0093  ,  6.3222  ,  2.4203  ,  0.9089  ,  0.2955   , 0.0720])

##BL=40 rayleigh fading non iid ~700 epochs
# uniform=np.array([0.09465871006250381, 0.043250516057014465, 0.01818404719233513, 0.007977863773703575, 0.0040482874028384686, 0.002098623663187027])
# trainable=np.array([0.09613214433193207, 0.04367251321673393, 0.017878806218504906, 0.008061944507062435, 0.00394787173718214, 0.001823123311623931])
# baseline=np.array([0.1517 ,   0.0701   , 0.0296  ,  0.0121  ,  0.0045,    0.0019])

##BL=40 rician slow 
#uniform_rician_slow = np.array([0.02615552581846714, 0.005724697839468718, 0.0010296656982973218, 0.00019099957717116922, 5.266675361781381e-05])
#trainable_rician_slow = np.array([0.03138848394155502, 0.007811586372554302, 0.0014692898839712143, 0.00026474962942302227, 6.504184420919046e-05])

##lte turbo
#baseline1 = 0.1*np.array([0.4040  ,  0.0778   , 0.0131    ,0.0024    ,0.0006])

##conv code
#baseline2_conv =  0.1*np.array([0.3356 ,   0.0664  ,  0.0087    ,0.0029  ,  0.0004])



##BL=40 rician fast 
#uniform_rician_fast = np.array([0.06400810927152634, 0.02065182663500309, 0.004619615618139505, 0.000879707105923444])
# trainable=np.array([0.06595303863286972, 0.018431439995765686, 0.0030901466961950064, 0.0003702751419041306])
#baseline=(1/1000)*np.array([108.3691 ,  44.6619   , 6.0452   , 0.6311])

##robustness to bursty model 1+awgn
# baseline=np.array([0.2366 ,   0.1734  ,  0.0981  ,  0.0424  ,  0.0142  ,  0.0036 ,   0.0007])
# uniform_awgn=np.array([0.19375775754451752, 0.1104138195514679, 0.04817710071802139, 0.016077740117907524, 0.0047311605885624886, 0.0013517069164663553, 0.00039829101297073066])
# trainable_awgn=np.array([0.14693115651607513, 0.07854286581277847, 0.03521816059947014, 0.01449358370155096, 0.005700826179236174, 0.002182915573939681, 0.0008149566710926592])
# trainable_ric_fast = np.array([0.20669567584991455, 0.11659319698810577, 0.04997849836945534, 0.01620461791753769, 0.004498123191297054, 0.001219582511112094, 0.0003367082099430263])

##robustness to bursty model 1+awgn
# baseline=np.array([0.2366 ,   0.1734  ,  0.0981  ,  0.0424  ,  0.0142  ,  0.0036 ,   0.0007])
# uniform_awgn=np.array([0.19375775754451752, 0.1104138195514679, 0.04817710071802139, 0.016077740117907524, 0.0047311605885624886, 0.0013517069164663553, 0.00039829101297073066])
# trainable_awgn=np.array([0.14693115651607513, 0.07854286581277847, 0.03521816059947014, 0.01449358370155096, 0.005700826179236174, 0.002182915573939681, 0.0008149566710926592])
# trainable_ric_fast = np.array([0.20669567584991455, 0.11659319698810577, 0.04997849836945534, 0.01620461791753769, 0.004498123191297054, 0.001219582511112094, 0.0003367082099430263])

##robustness to bursty model 1+rayleigh fast
#baseline=np.array([0.2264  ,  0.2034 ,   0.1738   , 0.1562  ,  0.1234  ,  0.0903  ,  0.0615  ,  0.0445  ,  0.0287  ,  0.0173  ,  0.0099   , 0.0048])
#uniform=np.array([0.19323347508907318, 0.15807221829891205, 0.1246512234210968, 0.09456788748502731, 0.06929711997509003, 0.04912173002958298, 0.03350099176168442, 0.022453468292951584, 0.014525691978633404, 0.009384662844240665, 0.00592945097014308, 0.0038699100259691477])
#trainable=np.array([0.19101786613464355, 0.15378332138061523, 0.11925644427537918, 0.08807606250047684, 0.0627337396144867, 0.04260874167084694, 0.027905359864234924, 0.01765255257487297, 0.010901665315032005, 0.006653164513409138, 0.003991580102592707, 0.0024498701095581055])

##robustness to bursty model 1+rayleigh slow
#baseline=np.array([0.1857 ,   0.0771 ,   0.0312   , 0.0181  ,  0.0060  ,  0.0022])
#uniform=np.array([0.11002497375011444, 0.05347079783678055, 0.02344820648431778, 0.009951039217412472, 0.00484903808683157, 0.0024869164917618036])
#trainable=np.array([0.11546391993761063, 0.055157389491796494, 0.023753568530082703, 0.010158657096326351, 0.005054164677858353, 0.0023740387987345457])

# ##robustness to bursty model 1+rician fast
#baseline=np.array([0.1771 ,   0.1043  ,  0.0590 ,   0.0171])
#uniform=np.array([0.1334906965494156, 0.06875178217887878, 0.028650185093283653, 0.01043681986629963])
#trainable=np.array([0.1419776976108551, 0.06733030825853348, 0.024205608293414116, 0.007144201081246138])


# ##robustness to bursty model 1+rician slow 
#baseline=np.array([0.0911  ,  0.0197  ,  0.0036   , 0.0007   , 0.0002])
#uniform=np.array([0.04943305626511574, 0.014387395232915878, 0.0035299973096698523, 0.0007446237141266465, 0.0001629582402529195])
#trainable=np.array([0.05682986229658127, 0.020122602581977844, 0.005845032632350922, 0.0014126233290880919, 0.00029349984833970666])


##robustness (without training) to bursty model 2 + rician fast
# baseline=(1/1000)*np.array([151.6667 ,  92.0620  , 32.0881  , 13.7486  ,  4.9078  ,  2.9172  ,  1.5950  ,  0.9976])
# uniform = np.array([0.10459791123867035, 0.05038101598620415, 0.020809929817914963, 0.00904011633247137, 0.004882287699729204, 0.003276667557656765, 0.0025273722130805254, 0.002086872002109885])
# trainable = np.array([0.13363108038902283, 0.06462204456329346, 0.025836393237113953, 0.010753368958830833, 0.0058272830210626125, 0.004053493961691856, 0.0031971645075827837, 0.0026871629524976015])
# ##cut some snr points
baseline=(1/1000)*np.array([151.6667 ,  92.0620  , 32.0881  , 13.7486  ,  4.9078  ])
uniform = np.array([0.10459791123867035, 0.05038101598620415, 0.020809929817914963, 0.00904011633247137, 0.004882287699729204])
trainable = np.array([0.13363108038902283, 0.06462204456329346, 0.025836393237113953, 0.010753368958830833, 0.0058272830210626125])

# ##robustness (with training) to bursty model 2 + rician fast
# uniform_trained = np.array([0.09969615191221237, 0.04103301092982292, 0.013473357073962688, 0.0043033696711063385, 0.001748164533637464, 0.0010153315961360931, 0.0006750407628715038, 0.0005095827509649098])
# trainable_trained = np.array([0.10364612191915512, 0.039957866072654724, 0.011552851647138596, 0.003325787140056491, 0.0013703732984140515, 0.0007872076821513474, 0.0005272908019833267, 0.0003963326453231275])
# ##cut some snr points
# uniform_trained = np.array([0.09969615191221237, 0.04103301092982292, 0.013473357073962688, 0.0043033696711063385, 0.001748164533637464])
# trainable_trained = np.array([0.10364612191915512, 0.039957866072654724, 0.011552851647138596, 0.003325787140056491, 0.0013703732984140515])

##robustness (with training) to chirp jamming on awgn
#baseline = (1/1000)*np.array([136.9565 ,  59.5379  , 13.6908  ,  3.0616  ,  0.7341   , 0.2297])
#uniform_trained_awgn=np.array([ 0.06665226072072983, 0.020981764420866966, 0.004448201507329941, 0.0008039158419705927,  0.0001497918419772759,  3.5708511859411374e-05])
#trainable_trained_awgn=np.array([0.04833637923002243, 0.016637666150927544, 0.004839411471039057, 0.0012641240609809756, 0.00029258307768031955, 5.44169670320116e-05])

#trainable_trained_ricfast = np.array([0.07539113610982895, 0.02241596020758152, 0.004348577465862036, 0.0006957076257094741, 0.00015037496632430702, 3.9875198126537725e-05])
#uniform_trained_ricfast = np.array([0.07121836394071579, 0.022773362696170807, 0.004950869362801313, 0.0008797068730928004, 0.0001876664609881118, 4.829190584132448e-05])

##robustness (without training) to chirp jamming on awgn
# uniform_awgn=np.array([0.08162093907594681, 0.030753936618566513, 0.008981030434370041, 0.002814789768308401, 0.0012249975698068738, 0.000702373159583658])
# trainable_awgn=np.array([0.06542099267244339, 0.03013557195663452, 0.014475693926215172, 0.008166246116161346, 0.005432990845292807, 0.004043120890855789])
# uniform_ricfast_on_awgn=np.array([0.0760144293308258, 0.02968907169997692, 0.009743530303239822, 0.003472744021564722, 0.0016786233754828572, 0.001032915897667408])
#trainable_ricfast_on_awgn=np.array([0.09629686176776886, 0.03752879053354263, 0.011919141747057438, 0.0043973722495138645, 0.002366123953834176, 0.0015865389723330736])

# ##Trained on bursty + ric fast, tested on chirp + awgn
#bursty_chirp_trainable = np.array([0.07480894029140472, 0.024524567648768425, 0.006277617532759905, 0.0017824579263105989, 0.0007947072153910995, 0.00046562403440475464])
#bursty_chirp_uniform = np.array([0.0739293023943901, 0.026648398488759995, 0.007544578053057194, 0.00226245797239244, 0.0009905402548611164, 0.0005770823918282986])

##Trained on bursty + ric fast, tested on chirp + ric fast
#bursty_chirp_trainable = np.array([0.096153125166893, 0.03539072722196579, 0.00935424119234085, 0.0022771621588617563])
#bursty_chirp_uniform = np.array([0.0923549011349678, 0.03614828363060951, 0.010976066812872887, 0.0030386648140847683])
#baseline = (1/1000)*np.array([137.7747 ,  78.8281  , 19.8147  ,  4.4469])

##rician fast on awgn trainable (second is cut at 3)
#ric_fast_awgn_trainable=np.array([0.08568878471851349, 0.04964548721909523, 0.02494122087955475, 0.010762392543256283, 0.0040036956779658794, 0.0012422489235177636, 0.0003840495483018458, 0.00011250052193645388, 3.0600069294450805e-05, 8.799993338470813e-06, 2.6999991860066075e-06, 6.499999471998308e-07])
#ric_fast_awgn_trainable=np.array([0.08568878471851349, 0.04964548721909523, 0.02494122087955475, 0.010762392543256283, 0.0040036956779658794, 0.0012422489235177636, 0.0003840495483018458, 0.00011250052193645388, 3.0600069294450805e-05, 8.799993338470813e-06])

##awgn on awgn trainable (second is cut at 3)
#awgn_on_awgn_trainable=np.array([0.05905376747250557, 0.035871054977178574, 0.02057831361889839, 0.01124388724565506, 0.005900294054299593, 0.0029478983487933874, 0.0014227498322725296, 0.0006711993482895195, 0.00028654999914579093, 0.00011155061656609178, 4.1800198232522234e-05, 1.3899983969167806e-05])
#awgn_on_awgn_trainable=np.array([0.05905376747250557, 0.035871054977178574, 0.02057831361889839, 0.01124388724565506, 0.005900294054299593, 0.0029478983487933874, 0.0014227498322725296, 0.0006711993482895195, 0.00028654999914579093, 0.00011155061656609178])

##rician fast on awgn uniform (second is cut at 3)
#ric_fast_awgn_uniform=np.array([0.07796090841293335, 0.047593846917152405, 0.026140138506889343, 0.012882987037301064, 0.005689999554306269, 0.002348001580685377, 0.0009409488411620259, 0.0003515995340421796, 0.0001430502743460238, 5.030023385188542e-05, 1.8600005205371417e-05, 7.449993063346483e-06])
#ric_fast_awgn_uniform=np.array([0.07796090841293335, 0.047593846917152405, 0.026140138506889343, 0.012882987037301064, 0.005689999554306269, 0.002348001580685377, 0.0009409488411620259, 0.0003515995340421796, 0.0001430502743460238, 5.030023385188542e-05])

##rician fast on awgn 3gpp (cut at 3)
#ric_fast_awgn_gpp = np.array([])

##awgn on awgn uniform (second is cut at 3)
#awgn_on_awgn_uniform=np.array([0.08530783653259277, 0.051416244357824326, 0.02743779681622982, 0.012829786166548729, 0.0053354958072304726, 0.0020194489043205976, 0.00072309939423576, 0.0002529500052332878, 8.825040276860818e-05, 2.7600051907938905e-05, 8.049995813053101e-06, 2.550000090195681e-06])
#awgn_on_awgn_uniform=np.array([0.08530783653259277, 0.051416244357824326, 0.02743779681622982, 0.012829786166548729, 0.0053354958072304726, 0.0020194489043205976, 0.00072309939423576, 0.0002529500052332878, 8.825040276860818e-05, 2.7600051907938905e-05])

##lte baseline awgn (second is cut at 3)
#lte_baseline_awgn = 1/1000*np.array([172.9310  , 77.3700  , 21.6826  ,  3.6706  ,  0.1326  ,  0.0049  ,  0.0001] )
#lte_baseline_awgn = 1/1000*np.array([172.9310  , 77.3700  , 21.6826  ,  3.6706  ,  0.1326  ,  0.0049 ] )

##757 baseline awgn (second is cut at 3)
#sev57_baseline_awgn = 1/1000*np.array([80.9695 ,  53.9475   ,32.0170  , 16.7830  ,  7.3545,    2.9415 ,   0.9495  ,  0.2300   , 0.0615,   0.0135 ,   0.0025] )
#sev57_baseline_awgn = 1/1000*np.array([80.9695 ,  53.9475   ,32.0170  , 16.7830  ,  7.3545,    2.9415 ,   0.9495  ,  0.2300   , 0.0615,   0.0135 ] )



#plt.plot(snrs,samsung_delta11,'co-', label='Samsung, delta=11')
#plt.plot(snrs - 10*math.log10(40/120),uniform,'g^-', label='Uniform Interleaver')
# plt.plot(snrs - 10*math.log10(40/120),samsung_delta19_finetuned,'m|-', label='Samsung Interleaver Delta=19')
# plt.plot(snrs - 10*math.log10(40/120),trainable,'y<-', label='Trainable Interleaver')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,uniform_rician_fast,'m|-', label='UI')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable,'c<-', label='TI')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline,'g^-', label='LTE Turbo')
#plt.plot(snrs ,awgn_on_awgn_uniform,'b*-', label='RI Trained on AWGN')
#plt.plot(snrs_awgn_baseline ,lte_baseline_awgn,'r>-', label='LTE Baseline')
#plt.plot(snrs ,baseline1,'k*-', label='LTE Turbo')
#plt.plot(snrs ,baseline2_conv,'r>-', label='Conv Code')
#plt.plot(snrs - 10*math.log10(40/132),baseline,'co-', label='Turbo Baseline')
#plt.plot(snrs,trainable_circular_32,'m|-', label='Trainable Interleaver, circular padd')
#plt.plot(snrs,samsung_delta19_32,'y<-', label='Samsung, delta=19')
# plt.plot(snrs,turbo_h2,'m|-', label='Standard Turbo')
# plt.plot(snrs,avg_h2,'co-', label='DeepIC')
# plt.plot(snrs,sic_ber,'g^-', label='SIC Turbo')

# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,uniform_rician_slow,'m|-', label='UI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable_rician_slow,'c<-', label='TI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline1,'g^-', label='LTE Turbo')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline2_conv,'b*-', label='Conv Code')

# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable,'c<-', label='TI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform,'g^-', label='UI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline,'b*-', label='LTE Turbo')

# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable,'c<-', label='TI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform,'g^-', label='UI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable_trained_awgn,'c>-', label='TI awgn on awgn Finetuned')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline,'g>-', label='LTE Turbo')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,bursty_chirp_trainable,'r*-', label='TurboAE-TI finetuned')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),bursty_chirp_uniform,'b|-', label='TurboAE-UI finetuned')

#plt.plot(snrs + 10*np.log10(3) ,awgn_on_awgn_trainable,'r|-', label='TurboAE-TI trained on AWGN')
#plt.plot(snrs + 10*np.log10(3) ,awgn_on_awgn_uniform,'y>-', label='TurboAE-UI trained on AWGN')
#plt.plot(snrs_awgn_turbo_lte_baseline + 10*np.log10(3) ,lte_baseline_awgn,'g<-', label='LTE Turbo')


# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,trainable_awgn,'r1-', label='TI awgn on awgn')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform_awgn,'y2-', label='UI awgn on awgn')
#plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14) ,ric_fast_awgn_trainable,'c3-', label='TurboAE')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform_ricfast_on_awgn,'kp-', label='UI ricfast on awgn')

# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline,'g>-', label='LTE Turbo')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform_rician_fast,'m|-', label='TurboAE-UI')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),trainable,'r*-', label='TurboAE-TI')

# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform_trained,'bo-', label='TurboAE-UI finetuned')
# plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),trainable_trained,'cd-', label='TurboAE-TI finetuned')

# plt.plot(snrs  + 10*np.log10(3),trainable_awgn,'c<-', label='TI trained on AWGN')
# plt.plot(snrs + 10*np.log10(3),uniform_awgn,'g^-', label='UI trained on AWGN')
# plt.plot(snrs + 10*np.log10(3),trainable_ric_fast,'r>-', label='TI trained on Rician Fast')
# plt.plot(snrs + 10*np.log10(3),baseline,'b*-', label='LTE Turbo')

#plt.plot(snrs_awgn_turbo_lte_baseline + 10*np.log10(3) ,lte_baseline_awgn,'g>-', label='LTE Turbo')
# plt.plot(snrs + 10*np.log10(3) ,awgn_on_awgn_trainable,'rv-', label='TurboAE-TI trained on AWGN')
# plt.plot(snrs + 10*np.log10(3) ,ric_fast_awgn_uniform,'y|-', label='TurboAE-UI trained on Rician')
#plt.plot(snrs + 10*np.log10(3) ,ric_fast_awgn_trainable,'ro-', label='TurboAE')
# plt.plot(snrs + 10*np.log10(3) ,awgn_on_awgn_uniform,'m*-', label='TurboAE-UI trained on AWGN')


plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),baseline,'g>-', label='LTE Turbo')
plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),trainable,'r*-', label='TurboAE-TI')
plt.plot(snrs + 10*np.log10(3) + 10*np.log10(2/3.14),uniform,'m|-', label='TurboAE-UI')


plt.xlabel('EB/N0',fontsize=15)
plt.ylabel('BER',fontsize=15)
plt.legend(loc='best',fontsize=9)
#plt.xlim([3.5, 7.5])
plt.yscale('log')
plt.grid()

#plt.title("Robustness to Rician + Chirp")
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.savefig('intrlv.png')
plt.show()