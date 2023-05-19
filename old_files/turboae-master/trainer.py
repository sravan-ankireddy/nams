__author__ = 'yihanjiang'
import torch
import time
import torch.nn.functional as F
from torch import linalg as LA

eps  = 1e-6

from utils import snr_sigma2db, snr_db2sigma, code_power, errors_ber_pos, errors_ber, errors_bler
from loss import customized_loss
from channels import generate_noise

import numpy as np
from numpy import arange
from numpy.random import mtrand

######################################################################################
#
# Trainer, validation, and test for AE code design
#
######################################################################################

def train(epoch, model, optimizer,intrlv_optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    train_loss = 0.0
    perm_matrix_loss = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        optimizer.zero_grad()
        intrlv_optimizer.zero_grad()
  
        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
        # train encoder/decoder with different SNR... seems to be a good practice.
        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        output, code = model(X_train, fwd_noise)

        
        loss1 = customized_loss(output, X_train, args, noise=fwd_noise, code = code)
        rows_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=1)
        rows_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=1)

        cols_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=0)
        cols_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=0)


        rows_l1,rows_l2,cols_l1,cols_l2 = rows_l1.to(device), rows_l2.to(device),cols_l1.to(device),cols_l2.to(device)
        loss2 = torch.sum(rows_l1) + torch.sum(cols_l1) - torch.sum(rows_l2) - torch.sum(cols_l2)

        loss = loss1 + 0.002*loss2 ##0.002 is the lambda

        loss.backward()

        train_loss += loss1.item()
        perm_matrix_loss += loss2.item()
        optimizer.step()
        intrlv_optimizer.step()

        ##clip and normalize 
        with torch.no_grad():
            for p in model.interleaver.parameters():
                p.data.clamp_(0) 
            model.interleaver.perm_matrix.weight.div_(torch.sum(model.interleaver.perm_matrix.weight, dim=0, keepdim=True))
            model.interleaver.perm_matrix.weight.div_(torch.sum(model.interleaver.perm_matrix.weight, dim=1, keepdim=True)) 

    train_loss = train_loss /(args.num_block/args.batch_size)
    perm_matrix_loss = perm_matrix_loss /(args.num_block/args.batch_size)

    print("BCE Loss:", train_loss, "Penalty loss:", perm_matrix_loss)


def train_separate(epoch, model, optimizer, args, use_cuda = False, verbose = True, mode = 'encoder'):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.train()
    train_loss = 0.0
    perm_matrix_loss = 0.0


    for batch_idx in range(int(args.num_block/args.batch_size)):

        block_len = args.block_len

        optimizer.zero_grad()
        #intrlv_optimizer.zero_grad()
  
        X_train    = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)

        noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
        # train encoder/decoder with different SNR... seems to be a good practice.
        if mode == 'encoder':
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_enc_channel_low, snr_high=args.train_enc_channel_high, mode = 'encoder')
        else:
            fwd_noise  = generate_noise(noise_shape, args, snr_low=args.train_dec_channel_low, snr_high=args.train_dec_channel_high, mode = 'decoder')

        X_train, fwd_noise = X_train.to(device), fwd_noise.to(device)

        output, code = model(X_train, fwd_noise)
        output = torch.clamp(output, 0.0, 1.0)

        
        loss1 = customized_loss(output, X_train, args, noise=fwd_noise, code = code)
        rows_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=1)
        rows_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=1)

        cols_l1 = LA.norm(model.interleaver.perm_matrix.weight,ord=1, axis=0)
        cols_l2 = LA.norm(model.interleaver.perm_matrix.weight,ord=2, axis=0)


        rows_l1,rows_l2,cols_l1,cols_l2 = rows_l1.to(device), rows_l2.to(device),cols_l1.to(device),cols_l2.to(device)
        loss2 = torch.sum(rows_l1) + torch.sum(cols_l1) - torch.sum(rows_l2) - torch.sum(cols_l2)

        loss = loss1 + 0.002*loss2 ##0.002 is the lambda

        loss.backward()

        train_loss += loss1.item()
        perm_matrix_loss += loss2.item()
        optimizer.step()
        #intrlv_optimizer.step()

        ##clip and normalize 
        with torch.no_grad():
            for p in model.interleaver.parameters():
                p.data.clamp_(0) 
            model.interleaver.perm_matrix.weight.div_(torch.sum(model.interleaver.perm_matrix.weight, dim=0, keepdim=True))
            model.interleaver.perm_matrix.weight.div_(torch.sum(model.interleaver.perm_matrix.weight, dim=1, keepdim=True)) 

    train_loss = train_loss /(args.num_block/args.batch_size)
    perm_matrix_loss = perm_matrix_loss /(args.num_block/args.batch_size)

    print("BCE Loss:", train_loss, "Penalty loss:", perm_matrix_loss)



def validate(model, optimizer, args, use_cuda = False, verbose = True):

    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()
    test_bce_loss, test_custom_loss, test_ber= 0.0, 0.0, 0.0

    with torch.no_grad():
        num_test_batch = int(args.num_block/args.batch_size * args.test_ratio)
        for batch_idx in range(num_test_batch):
            X_test     = torch.randint(0, 2, (args.batch_size, args.block_len, args.code_rate_k), dtype=torch.float)
            noise_shape = (args.batch_size, args.block_len, args.code_rate_n)
            fwd_noise  = generate_noise(noise_shape, args,
                                        snr_low=args.train_enc_channel_low,
                                        snr_high=args.train_enc_channel_low)

            X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

            optimizer.zero_grad()
            output, codes = model(X_test, fwd_noise)

            output = torch.clamp(output, 0.0, 1.0)

            output = output.detach()
            X_test = X_test.detach()

            test_bce_loss += F.binary_cross_entropy(output, X_test)
            test_custom_loss += customized_loss(output, X_test, noise = fwd_noise, args = args, code = codes)
            test_ber  += errors_ber(output,X_test)


    test_bce_loss /= num_test_batch
    test_custom_loss /= num_test_batch
    test_ber  /= num_test_batch

    if verbose:
        print('====> Test set BCE loss', float(test_bce_loss),
              'Custom Loss',float(test_custom_loss),
              'with ber ', float(test_ber),
        )

    report_loss = float(test_bce_loss)
    report_ber  = float(test_ber)

    return report_loss, report_ber


def test(model, args, block_len = 'default',use_cuda = False):

    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    if block_len == 'default':
        block_len = args.block_len
    else:
        pass


    ber_res, bler_res = [], []
    ber_res_punc, bler_res_punc = [], []
    snr_interval = (args.snr_test_end - args.snr_test_start)* 1.0 /  (args.snr_points-1)
    snrs = [snr_interval* item + args.snr_test_start for item in range(args.snr_points)]
    print('SNRS', snrs)
    sigmas = snrs

    for sigma, this_snr in zip(sigmas, snrs):
        test_ber, test_bler = .0, .0
        with torch.no_grad():
            num_test_batch = 1200
            for batch_idx in range(num_test_batch):
                X_test     = torch.randint(0, 2, (args.batch_size, block_len, args.code_rate_k), dtype=torch.float)
                noise_shape = (args.batch_size, args.block_len, args.code_rate_n)

                fwd_noise  = generate_noise(noise_shape, args, test_sigma=sigma)
                X_test, fwd_noise= X_test.to(device), fwd_noise.to(device)

                X_hat_test, the_codes = model(X_test, fwd_noise, sigma)

                test_ber  += errors_ber(X_hat_test,X_test)
                test_bler += errors_bler(X_hat_test,X_test)

                if batch_idx == 0:
                    test_pos_ber = errors_ber_pos(X_hat_test,X_test)
                    codes_power  = code_power(the_codes)
                else:
                    test_pos_ber += errors_ber_pos(X_hat_test,X_test)
                    codes_power  += code_power(the_codes)

            

        test_ber  /= num_test_batch
        test_bler /= num_test_batch
        print('Test SNR',this_snr ,'with ber ', float(test_ber), 'with bler', float(test_bler))
        ber_res.append(float(test_ber))
        bler_res.append( float(test_bler))


    print('final results on SNRs ', snrs)
    print('BER', ber_res)
    print('BLER', bler_res)



 

















