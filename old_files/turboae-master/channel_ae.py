__author__ = 'yihanjiang'
import torch
import torch.nn.functional as F
import commpy.channelcoding.interleavers as RandInterlv
import numpy as np
import math
from utils import snr_db2sigma, snr_sigma2db

from ste import STEQuantize as MyQuantize


class Channel_AE(torch.nn.Module):
    def __init__(self, args, enc, dec, interleaver):
        super(Channel_AE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec
        self.interleaver = interleaver

    def forward(self, input, fwd_noise, sigma=0):
        
        codes  = self.enc(input,self.interleaver)

        
        if self.args.channel == 'fading_iid': ##rayleigh fast
            data_shape = codes.shape
            #  Rayleigh Fading Channel, iid
            fading_h = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) 
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = fading_h*codes + fwd_noise

        

        elif self.args.channel == 'fading_fixed': ##rayleigh slow
            data_shape = codes.shape
            fading_h = torch.sqrt(torch.randn(data_shape[0])**2 +  torch.randn(data_shape[0])**2)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = codes*fading_h[:,None, None] + fwd_noise

        elif self.args.channel == 'rician':
            data_shape = codes.shape
            K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))
            if self.args.FastFading:
                hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
                hLOSImag = torch.ones(data_shape)
                hNLOSReal = torch.randn(data_shape)
                hNLOSImag = torch.randn(data_shape)
            else: #Slow fading case

                ##Before
                # hLOSReal = torch.ones(1) #Assuming SISO see page 3.108 in Heath and Lazano
                # hLOSImag = torch.ones(1)
                # hNLOSReal = torch.randn(1)
                # hNLOSImag = torch.randn(1)

                ##After with Yihan's modification

                hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
                hLOSImag = torch.ones(data_shape)
                hNLOSReal = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])
                hNLOSImag = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])


            fading_h = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal,hNLOSImag)
            #Assuming phase information at the receiver
            fading_h = torch.abs(fading_h)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)

            received_codes = fading_h*codes + fwd_noise
            #received_codes = codes + fwd_noise/fading_h

        else: ##awgn
            received_codes = codes + fwd_noise

        if self.args.bursty_model_one :
            temp = np.random.binomial(1, .05, codes.shape)
            sigma = snr_db2sigma(sigma)
            sigma_bursty = 3*sigma
            noise_bursty = sigma_bursty*np.random.standard_normal(codes.shape)
            noise_bursty=np.multiply(noise_bursty,temp)
            noise_bursty = torch.from_numpy(noise_bursty)
            noise_bursty=noise_bursty.type(torch.FloatTensor).to(self.this_device)
            received_codes = received_codes + noise_bursty

        elif self.args.bursty_model_two :
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len*self.args.code_rate_n))
            uniform_sample = torch.randint(low=0, high=self.args.block_len*self.args.code_rate_n-9, size=(1,1))
            bursty_noise = self.args.bursty_sigma*torch.randn(self.args.batch_size,10).type(torch.FloatTensor).to(self.this_device)
            received_codes[:,uniform_sample:uniform_sample+10] = received_codes[:,uniform_sample:uniform_sample+10] + bursty_noise
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len,self.args.code_rate_n))

        elif self.args.chirp :
            f0 = 0
            f1 = 0.1
            beta  = f1-f0
            t = torch.arange(0, 10, 0.25/3)

            yvalue = self.args.bursty_sigma*torch.cos(2*math.pi*(beta*0.5)*(t**(2))).type(torch.FloatTensor).to(self.this_device)
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len*self.args.code_rate_n))
            uniform_sample = torch.randint(low=0, high=self.args.block_len*self.args.code_rate_n-9, size=(self.args.batch_size,1))
            ##Not efficient, can be vectorized
            for i in range(self.args.batch_size):
                received_codes[i,uniform_sample[i]:uniform_sample[i]+10] = received_codes[i,uniform_sample[i]:uniform_sample[i]+10] + yvalue[uniform_sample[i]:uniform_sample[i]+10]

            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len,self.args.code_rate_n))

        x_dec          = self.dec(received_codes,self.interleaver)

        return x_dec, codes



##Used with non-trainable interleavers (grid, uniform)
class Channel_AE_non_trainable(torch.nn.Module):
    def __init__(self, args, enc, dec):
        super(Channel_AE_non_trainable, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec

    def forward(self, input, fwd_noise, sigma=0):
        # Setup Interleavers.
        # if self.args.is_interleave == 0:
        #     pass

        # elif self.args.is_same_interleaver == 0:
        #     interleaver = RandInterlv.RandInterlv(self.args.block_len, np.random.randint(0, 1000))

        #     p_array = interleaver.p_array
        #     self.enc.set_interleaver(p_array)
        #     self.dec.set_interleaver(p_array)

        # else:# self.args.is_same_interleaver == 1
        #     interleaver = RandInterlv.RandInterlv(self.args.block_len, 0) # not random anymore!
        #     p_array = interleaver.p_array
        #     self.enc.set_interleaver(p_array)
        #     self.dec.set_interleaver(p_array)

        codes  = self.enc(input)

        # Setup channel mode:
        if self.args.channel in ['awgn', 't-dist', 'radar', 'ge_awgn']:
            received_codes = codes + fwd_noise

  

        elif self.args.channel == 'bec':
            received_codes = codes * fwd_noise

        elif self.args.channel in ['bsc', 'ge']:
            received_codes = codes * (2.0*fwd_noise - 1.0)
            received_codes = received_codes.type(torch.FloatTensor)

        elif self.args.channel == 'fading_iid':
            data_shape = codes.shape
            #  Rayleigh Fading Channel, non-coherent
            fading_h = torch.sqrt(torch.randn(data_shape)**2 +  torch.randn(data_shape)**2)/torch.sqrt(torch.tensor(3.14/2.0)) #np.sqrt(2.0)
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = fading_h*codes + fwd_noise

        elif self.args.channel == 'fading_fixed':
            data_shape = codes.shape
            fading_h = torch.sqrt(torch.randn(data_shape[0])**2 +  torch.randn(data_shape[0])**2)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)
            received_codes = codes*fading_h[:,None, None] + fwd_noise

        elif self.args.channel == 'rician':
            data_shape = codes.shape
            K = 10 #Rician Fading coefficient (Ratio of LOS to NLOS paths)
            coeffLOS = np.sqrt(K/(K+1))
            coeffNLOS = np.sqrt(1/(K+1))
            if self.args.FastFading:
                hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
                hLOSImag = torch.ones(data_shape)
                hNLOSReal = torch.randn(data_shape)
                hNLOSImag = torch.randn(data_shape)
            else: #Slow fading case

                #Before
                # hLOSReal = torch.ones(1) #Assuming SISO see page 3.108 in Heath and Lazano
                # hLOSImag = torch.ones(1)
                # hNLOSReal = torch.randn(1)
                # hNLOSImag = torch.randn(1)

                ##After with Yihan's modification

                hLOSReal = torch.ones(data_shape) #Assuming SISO see page 3.108 in Heath and Lazano
                hLOSImag = torch.ones(data_shape)
                hNLOSReal = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])
                hNLOSImag = torch.randn((data_shape[0], 1, 1)).repeat(1, data_shape[1], data_shape[2])

            fading_h = coeffLOS*torch.complex(hLOSReal,hLOSImag) + coeffNLOS*torch.complex(hNLOSReal,hNLOSImag)
            #Assuming phase information at the receiver
            fading_h = torch.abs(fading_h)/torch.sqrt(torch.tensor(3.14/2.0))
            fading_h = fading_h.type(torch.FloatTensor).to(self.this_device)

            received_codes = fading_h*codes + fwd_noise 
            
        else:
            print('default AWGN channel')
            received_codes = codes + fwd_noise

        if self.args.bursty_model_one :
            temp = np.random.binomial(1, .05, codes.shape)
            sigma = snr_db2sigma(sigma)
            sigma_bursty = 3*sigma
            noise_bursty = sigma_bursty*np.random.standard_normal(codes.shape)
            noise_bursty=np.multiply(noise_bursty,temp)
            noise_bursty = torch.from_numpy(noise_bursty)
            noise_bursty=noise_bursty.type(torch.FloatTensor).to(self.this_device)
            received_codes = received_codes + noise_bursty

        elif self.args.bursty_model_two :
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len*self.args.code_rate_n))
            uniform_sample = torch.randint(low=0, high=self.args.block_len*self.args.code_rate_n-9, size=(1,1))
            bursty_noise = self.args.bursty_sigma*torch.randn(self.args.batch_size,10).type(torch.FloatTensor).to(self.this_device)
            received_codes[:,uniform_sample:uniform_sample+10] = received_codes[:,uniform_sample:uniform_sample+10] + bursty_noise
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len,self.args.code_rate_n))


        elif self.args.chirp :
            f0 = 0
            f1 = 0.1
            beta  = f1-f0
            t = torch.arange(0, 10, 0.25/3)

            yvalue = self.args.bursty_sigma*torch.cos(2*math.pi*(beta*0.5)*(t**(2))).type(torch.FloatTensor).to(self.this_device)
            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len*self.args.code_rate_n))
            uniform_sample = torch.randint(low=0, high=self.args.block_len*self.args.code_rate_n-9, size=(self.args.batch_size,1))
            ##Not efficient, can be vectorized
            for i in range(self.args.batch_size):
                received_codes[i,uniform_sample[i]:uniform_sample[i]+10] = received_codes[i,uniform_sample[i]:uniform_sample[i]+10] + yvalue[uniform_sample[i]:uniform_sample[i]+10]

            received_codes = torch.reshape(received_codes,(self.args.batch_size,self.args.block_len,self.args.code_rate_n))

        x_dec          = self.dec(received_codes)

        return x_dec, codes


class Channel_ModAE(torch.nn.Module):
    def __init__(self, args, enc, dec, mod, demod, modulation = 'qpsk'):
        super(Channel_ModAE, self).__init__()
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self.args = args
        self.enc = enc
        self.dec = dec
        self.mod = mod
        self.demod = demod

    def forward(self, input, fwd_noise):
        # Setup Interleavers.
        if self.args.is_interleave == 0:
            pass

        elif self.args.is_same_interleaver == 0:
            interleaver = RandInterlv.RandInterlv(self.args.block_len, np.random.randint(0, 1000))

            p_array = interleaver.p_array
            self.enc.set_interleaver(p_array)
            self.dec.set_interleaver(p_array)

        else:# self.args.is_same_interleaver == 1
            interleaver = RandInterlv.RandInterlv(self.args.block_len, 0) # not random anymore!
            p_array = interleaver.p_array
            self.enc.set_interleaver(p_array)
            self.dec.set_interleaver(p_array)

        codes  = self.enc(input)
        symbols = self.mod(codes)

        # Setup channel mode:
        if self.args.channel in ['awgn', 't-dist', 'radar', 'ge_awgn']:
            received_symbols = symbols + fwd_noise

        elif self.args.channel == 'fading':
            print('Fading not implemented')

        else:
            print('default AWGN channel')
            received_symbols = symbols + fwd_noise

        if self.args.rec_quantize:
            myquantize = MyQuantize.apply
            received_symbols = myquantize(received_symbols, self.args.rec_quantize_level, self.args.rec_quantize_level)

        x_rec = self.demod(received_symbols)
        x_dec = self.dec(x_rec)

        return x_dec, symbols
