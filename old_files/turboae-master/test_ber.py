from textwrap import indent
import torch
import torch.optim as optim
import numpy as np
from interleavers import Trainable_Interleaver, Trainable_Interleaver_initialized
from utils import errors_ber
import sys
from get_args import get_args
from trainer import train, test
from loss import customized_loss
from channels import generate_noise
from numpy import arange
from numpy.random import mtrand
import torch.nn.functional as F
import matplotlib.pyplot as plt

# utils for logger
class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def import_enc(args):
    # choose encoder

    if args.encoder == 'TurboAE_rate3_rnn':
        from encoders import ENC_interRNN as ENC

    elif args.encoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from encoders import ENC_interCNN as ENC

    elif args.encoder == 'turboae_2int':
        from encoders import ENC_interCNN2Int as ENC

    elif args.encoder == 'rate3_cnn':
        from encoders import CNN_encoder_rate3 as ENC

    elif args.encoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from encoders import ENC_interCNN2D as ENC

    elif args.encoder == 'TurboAE_rate3_rnn_sys':
        from encoders import ENC_interRNN_sys as ENC

    elif args.encoder == 'TurboAE_rate2_rnn':
        from encoders import ENC_turbofy_rate2 as ENC

    elif args.encoder == 'TurboAE_rate2_cnn':
        from encoders import ENC_turbofy_rate2_CNN as ENC  # not done yet

    elif args.encoder in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
        from encoders import ENC_TurboCode as ENC          # DeepTurbo, encoder not trainable.

    elif args.encoder == 'rate3_cnn2d':
        from encoders import ENC_CNN2D as ENC

    else:
        print('Unknown Encoder, stop')

    return ENC

def import_dec(args):

    if args.decoder == 'TurboAE_rate2_rnn':
        from decoders import DEC_LargeRNN_rate2 as DEC

    elif args.decoder == 'TurboAE_rate2_cnn':
        from decoders import DEC_LargeCNN_rate2 as DEC  # not done yet

    elif args.decoder in ['TurboAE_rate3_cnn', 'TurboAE_rate3_cnn_dense']:
        from decoders import DEC_LargeCNN as DEC

    elif args.decoder == 'turboae_2int':
        from decoders import DEC_LargeCNN2Int as DEC

    elif args.encoder == 'rate3_cnn':
        from decoders import CNN_decoder_rate3 as DEC

    elif args.decoder in ['TurboAE_rate3_cnn2d', 'TurboAE_rate3_cnn2d_dense']:
        from decoders import DEC_LargeCNN2D as DEC

    elif args.decoder == 'TurboAE_rate3_rnn':
        from decoders import DEC_LargeRNN as DEC

    elif args.decoder == 'nbcjr_rate3':                # ICLR 2018 paper
        from decoders import NeuralTurbofyDec as DEC

    elif args.decoder == 'rate3_cnn2d':
        from decoders import DEC_CNN2D as DEC

    return DEC

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################

    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = "cuda"

    #################################################
    # Setup Channel AE: Encoder, Decoder, Channel
    #################################################
    # choose encoder and decoder.
    ENC = import_enc(args)
    DEC = import_dec(args)

    
    encoder = ENC(args)
    decoder = DEC(args)
    interleaver = Trainable_Interleaver(args)
    #interleaver = Trainable_Interleaver_initialized(args)

    # choose support channels
    from channel_ae import Channel_AE
    model = Channel_AE(args, encoder, decoder, interleaver).to(device)


    # make the model parallel
    if args.is_parallel == 1:
        model.enc.set_parallel()
        model.dec.set_parallel()




    ##################################################################
    # Setup Optimizers, only Adam and Lookahead for now.
    ##################################################################

    # OPT = optim.Adam

    # enc_optimizer = OPT(model.enc.parameters(),lr=args.enc_lr)
    # dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)
    # intrlv_optimizer = OPT(filter(lambda p: p.requires_grad, model.interleaver.parameters()), lr=args.dec_lr)

    # general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)


    pretrained_model = torch.load("./results_init_intrlvr/trainable_trained_on_rician_bl40.pt")
    #pretrained_model = torch.load("./tmp/trainable_interleaver_40_circularpadd_rician_fast.pt")

    model.load_state_dict(pretrained_model)
    model.args = args
    

    test(model, args, use_cuda = use_cuda)


