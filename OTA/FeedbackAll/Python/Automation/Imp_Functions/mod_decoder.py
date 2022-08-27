##Power Norm where we fix the mean and var
#################################
#######  Libraries Used  ########
#################################
import os
import sys
import argparse
import random
from turtle import hideturtle
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as sio

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


class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.args = args

        ##Power Norm weights
        self.p1_enc = torch.nn.Parameter(torch.randn(()))
        self.p2_enc = torch.nn.Parameter(torch.randn(()))
        self.p3_enc = torch.nn.Parameter(torch.randn(()))

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
        mu_1_enc = 2.9316
        v_1_enc = 0.0131
        mu_2_enc = 2.9872
        v_2_enc = 0.0229
        mu_3_enc = 2.9965
        v_3_enc = 0.0184

        den_enc = torch.sqrt(self.p1_enc ** 2 + self.p2_enc ** 2 + self.p3_enc ** 2)

        output, hidden_state = self.enc_rnn_fwd(input, hidden_state)
        output = self.enc_act(self.enc_linear(output))

        if round == 1:
            output = (output - mu_1_enc) * 1.0 / v_1_enc
            output = np.sqrt(3) * (self.p1_enc * output) / den_enc

        elif round == 2:
            output = (output - mu_2_enc) * 1.0 / v_2_enc
            output = np.sqrt(3) * (self.p2_enc * output) / den_enc

        elif round == 3:
            output = (output - mu_3_enc) * 1.0 / v_3_enc
            output = np.sqrt(3) * (self.p3_enc * output) / den_enc

        return output, hidden_state


class Decoder(torch.nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.args = args

        ##Power Norm weights
        self.p1_dec = torch.nn.Parameter(torch.randn(()))
        self.p2_dec = torch.nn.Parameter(torch.randn(()))

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

        mu_1_dec = -2.0789
        v_1_dec = 5.0892
        mu_2_dec = -3.6495
        v_2_dec = 5.1956

        den_dec = torch.sqrt(self.p1_dec ** 2 + self.p2_dec ** 2)

        output, hidden_state = self.dec_rnn(input, hidden_state)

        if round == 1:
            output = self.dec_act(self.dec_output(output))
            output = (output - mu_1_dec) * 1.0 / v_1_dec
            output = np.sqrt(2) * (self.p1_dec * output) / den_dec

        elif round == 2:
            output = self.dec_act(self.dec_output(output))
            output = (output - mu_2_dec) * 1.0 / v_2_dec
            output = np.sqrt(2) * (self.p2_dec * output) / den_dec

        elif round == 3:
            output = self.dec_output(output)
            output = F.sigmoid(output)

        return output, hidden_state


def Encoder(round):

    args = get_args()

    enc_model = Encoder(args)
    enc_weights = torch.load("./Imp_Functions/decoder.pt")
    enc_model.load_state_dict(enc_weights)
    enc_model.args = args

    # Obtain input from mat file
    if round == 1:
        input_data = sio.loadmat("to_be_encoded.mat")
        input_data = input_data("Data_Files/to_be_encoded")
        input_arr = torch.from_numpy(input_data)
        input = torch.cat(
            [
                input_arr,
                torch.zeros((input_arr.shape[0], 1, 2 * args.block_len)) + 0.5,
            ],
            dim=2,
        )
        hidden_state = torch.zeros(2, args.batch_size, 50)
    else:
        # obtain hidden state from mat file
        input_data = sio.loadmat("Data_Files/to_be_encoded.mat")
        input_data = input_data("to_be_encoded")
        input_arr = torch.from_numpy(input_data)
        input = torch.cat(
            [
                input_arr,
                torch.zeros((input_arr.shape[0], 1, 2 * args.block_len)) + 0.5,
            ],
            dim=2,
        )

        hidden_state = sio.loadmat("Data_Files/hidden.mat")
        hidden_state = hidden_state("hidden")
        hidden_state = torch.from_numpy(hidden_state)

    output, hidden_state = enc_model(input, hidden_state, 1)

    output = {"output": np.array(output)}
    sio.savemat("Data_Files/Encoded.mat", "output")

    hidden = {"hidden": np.array(output)}
    sio.savemat("Data_Files/Hidden.mat", "hidden")


def Decoder(round):

    args = get_args()

    dec_model = Encoder(args)
    dec_weights = torch.load("./Imp_Functions/decoder.pt")
    dec_model.load_state_dict(dec_weights)
    dec_model.args = args

    input_data = sio.loadmat("Data_Files/to_be_encoded.mat")
    input_data = input_data("to_be_encoded")
    input_arr = torch.from_numpy(input_data)

    if round == 1:
        hidden_state = torch.zeros(2, args.batch_size, 50)
    else:
        # obtain hidden state from mat file
        hidden_state = sio.loadmat("Data_Files/hidden.mat")
        hidden_state = hidden_state("hidden")
        hidden_state = torch.from_numpy(hidden_state)

    output, hidden_state = dec_model(input, hidden_state, 1)

    output = {"output": np.array(output)}
    sio.savemat("Data_Files/Encoded.mat", "output")

    hidden = {"hidden": np.array(output)}
    sio.savemat("Data_Files/Hidden.mat", "hidden")

def 