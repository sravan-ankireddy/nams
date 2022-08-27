##Power Norm where we fix the mean and var
#################################
#######  Libraries Used  ########
#################################
import os
import sys
import argparse
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

#################################
#######  Parameters  ########
#################################
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-init_nw_weight", type=str, default="default")
    parser.add_argument("-code_rate", type=int, default=3)
    parser.add_argument("-precompute_stats", type=bool, default=True)  ##########

    parser.add_argument("-learning_rate", type=float, default=0.0001)
    parser.add_argument("-batch_size", type=int, default=500)
    parser.add_argument("-num_epoch", type=int, default=600)

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument("-block_len", type=int, default=1)
    parser.add_argument("-num_block", type=int, default=50000)
    parser.add_argument("-num_rounds", type=int, default=3)
    # parser.add_argument('-delta_snr', type=int, default=15)  ##SNR_FB - SNR_FW

    parser.add_argument("-enc_num_layer", type=int, default=2)
    parser.add_argument("-dec_num_layer", type=int, default=2)
    parser.add_argument("-enc_num_unit", type=int, default=50)
    parser.add_argument("-dec_num_unit", type=int, default=50)

    parser.add_argument("-frwd_snr", type=float, default=2)
    parser.add_argument("-bckwd_snr", type=float, default=16)

    parser.add_argument("-snr_test_start", type=float, default=0.0)
    parser.add_argument("-snr_test_end", type=float, default=5.0)
    parser.add_argument("-snr_points", type=int, default=6)

    parser.add_argument(
        "-channel_mode",
        choices=["normalize", "lazy_normalize", "tanh"],
        default="lazy_normalize",
    )

    parser.add_argument(
        "-enc_act",
        choices=["tanh", "selu", "relu", "elu", "sigmoid", "none"],
        default="elu",
    )
    parser.add_argument(
        "-dec_act",
        choices=["tanh", "selu", "relu", "elu", "sigmoid", "none"],
        default="none",
    )

    args = parser.parse_args()

    return args


class AE(torch.nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.args = args

        # Encoder
        self.enc_rnn_fwd = torch.nn.GRU(
            3 * self.args.block_len,
            self.args.enc_num_unit,
            num_layers=self.args.enc_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.enc_linear = torch.nn.Linear(
            self.args.enc_num_unit,
            int((self.args.code_rate) * (self.args.block_len) / (self.args.num_rounds)),
        )

        # Decoder
        self.dec_rnn = torch.nn.GRU(
            int((self.args.code_rate) * (self.args.block_len) / (self.args.num_rounds)),
            self.args.dec_num_unit,
            num_layers=self.args.dec_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.dec_output = torch.nn.Linear(self.args.dec_num_unit, self.args.block_len)


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        # Encoder
        self.enc_rnn_fwd = torch.nn.GRU(
            3 * self.args.block_len,
            self.args.enc_num_unit,
            num_layers=self.args.enc_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.enc_linear = torch.nn.Linear(
            self.args.enc_num_unit,
            int((self.args.code_rate) * (self.args.block_len) / (self.args.num_rounds)),
        )

    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs)
        this_std = torch.std(inputs)
        outputs = (inputs - this_mean) * 1.0 / this_std
        return outputs

    ##Encoder Activation
    def enc_act(self, inputs):
        if self.args.enc_act == "tanh":
            return F.tanh(inputs)
        elif self.args.enc_act == "elu":
            return F.elu(inputs)
        elif self.args.enc_act == "relu":
            return F.relu(inputs)
        elif self.args.enc_act == "selu":
            return F.selu(inputs)
        elif self.args.enc_act == "sigmoid":
            return F.sigmoid(inputs)
        else:
            return inputs

    def forward(self, input, hidden_state, round):

        output, hidden_state = self.enc_rnn_fwd(input, hidden_state)
        output = self.enc_act(self.enc_linear(output))
        output = self.power_constraint(output)

        return output, hidden_state


class Decoder(torch.nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args = args

        # Encoder
        self.dec_rnn = torch.nn.GRU(
            int((self.args.code_rate) * (self.args.block_len) / (self.args.num_rounds)),
            self.args.dec_num_unit,
            num_layers=self.args.dec_num_layer,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        self.dec_output = torch.nn.Linear(self.args.dec_num_unit, self.args.block_len)

    def power_constraint(self, inputs):
        this_mean = torch.mean(inputs)
        this_std = torch.std(inputs)
        outputs = (inputs - this_mean) * 1.0 / this_std
        return outputs

    def dec_act(self, inputs):
        if self.args.dec_act == "tanh":
            return F.tanh(inputs)
        elif self.args.dec_act == "elu":
            return F.elu(inputs)
        elif self.args.dec_act == "relu":
            return F.relu(inputs)
        elif self.args.dec_act == "selu":
            return F.selu(inputs)
        elif self.args.dec_act == "sigmoid":
            return F.sigmoid(inputs)
        else:
            return inputs

    def forward(self, input, hidden_state, round):

        output, hidden_state = self.dec_rnn(input, hidden_state)

        if round == 1:
            output = self.dec_act(self.dec_output(output))
            output = self.power_constraint(output)

        elif round == 2:
            output = self.dec_act(self.dec_output(output))
            output = self.power_constraint(output)

        elif round == 3:
            output = self.dec_output(output)
            output = F.sigmoid(output)

        return output, hidden_state


###### MAIN
args = get_args()

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

##AE
ae_model = AE(args).to(device)
ae_weights = torch.load("./ric_no_trainable_weights.pt")
ae_model.load_state_dict(ae_weights)
ae_model.args = args


##Encoder load weights
enc_model = Encoder(args).to(device)

enc_model.enc_rnn_fwd = ae_model.enc_rnn_fwd
enc_model.enc_linear = ae_model.enc_linear

##Decoder load weights

dec_model = Decoder(args).to(device)

dec_model.dec_rnn = ae_model.dec_rnn
dec_model.dec_output = ae_model.dec_output


##save the models
torch.save(enc_model.state_dict(), "./encoder.pt")
torch.save(dec_model.state_dict(), "./decoder.pt")


# ##test
# ##encoder
# input        = torch.cat([torch.ones((args.batch_size, 1, args.block_len)).to(device),
#                                     torch.zeros((args.batch_size, 1, 2*args.block_len)).to(device)+0.5], dim=2)

# hidden_state = torch.zeros(2,args.batch_size, 50).to(device)

# output,hidden_state = enc_model(input,hidden_state,1)
# print(output.shape)
# print(hidden_state.shape)

# output,hidden_state = enc_model(input,hidden_state,2)
# print(output.shape)
# print(hidden_state.shape)

# output,hidden_state = enc_model(input,hidden_state,3)
# print(output.shape)
# print(hidden_state.shape)

# ##decoder
# input        = torch.ones((args.batch_size, 1, args.block_len)).to(device)

# hidden_state = torch.zeros(2,args.batch_size, 50).to(device)

# output,hidden_state = dec_model(input,hidden_state,1)
# print(output.shape)
# print(hidden_state.shape)

# output,hidden_state = dec_model(input,hidden_state,2)
# print(output.shape)
# print(hidden_state.shape)

# output,hidden_state = dec_model(input,hidden_state,3)
# print(output.shape)
# print(hidden_state.shape)


# print(list(enc_model.enc_rnn_fwd.parameters()))
