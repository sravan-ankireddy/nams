__author__ = 'Karl Chahine'
# THIS FILE IS TO IMPLEMENT A TRAINABLE INTERLEAVER
from interleavers import Trainable_Interleaver, Trainable_Interleaver_initialized
import torch
import torch.optim as optim
import numpy as np
import sys
from get_args import get_args
from trainer import train, validate, test
import matplotlib.pyplot as plt
from numpy import arange
from numpy.random import mtrand

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
    from encoders import ENC_interCNN as ENC
    return ENC

def import_dec(args):
    from decoders import DEC_LargeCNN as DEC
    return DEC

if __name__ == '__main__':
    #################################################
    # load args & setup logger
    #################################################

    args = get_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    OPT = optim.Adam

    enc_optimizer = OPT(model.enc.parameters(),lr=args.enc_lr)
    dec_optimizer = OPT(filter(lambda p: p.requires_grad, model.dec.parameters()), lr=args.dec_lr)
    intrlv_optimizer = OPT(filter(lambda p: p.requires_grad, model.interleaver.parameters()), lr=args.dec_lr)

    general_optimizer = OPT(filter(lambda p: p.requires_grad, model.parameters()),lr=args.dec_lr)

    
    model.args = args
    #################################################
    # Training Processes
    #################################################
    report_loss, report_ber = [], []

    

    # pretrained_model = torch.load("./results_init_intrlvr/trainable_trained_on_rician_bl40.pt")
    # model.load_state_dict(pretrained_model)



    for epoch in range(1, args.num_epoch + 1):

        print("epoch: ",epoch," ")

        temp = model.interleaver.perm_matrix.weight.cpu().detach().numpy()
        temp.astype(int)

        np.savetxt('my_file.txt', temp, fmt='%i')

        plt.figure()
        im = plt.imshow(temp)
        plt.colorbar()

        ax = plt.gca()

        ax.set_xticks(np.arange(-.5, 40, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 40, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

        plt.savefig('matrix.png')

        if args.num_train_enc > 0 and args.encoder not in ['Turbo_rate3_lte', 'Turbo_rate3_757']:
            for idx in range(args.num_train_enc):
                train(epoch, model, enc_optimizer, intrlv_optimizer, args, use_cuda = use_cuda, mode ='encoder')

        if args.num_train_dec > 0:
            for idx in range(args.num_train_dec):
                train(epoch, model, dec_optimizer, intrlv_optimizer, args, use_cuda = use_cuda, mode ='decoder')

        

        
        #this_loss, this_ber  = validate(model, general_optimizer, args, use_cuda = use_cuda)
        #report_loss.append(this_loss)
        #report_ber.append(this_ber)

            


        if epoch%50==0:
            test(model, args, use_cuda = use_cuda)
            torch.save(model.state_dict(), './results_init_intrlvr/trainable_trained_on_rician_bl40.pt')

    
















