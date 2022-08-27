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
import scipy.io as sio

#################################
#######  Parameters  ########
#################################

fading = True

if fading:

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
                
        def forward(self, input, hidden_state, round, pow_alloc = False):
            
            output, hidden_state = self.enc_rnn_fwd(input, hidden_state)
            output = self.enc_act(self.enc_linear(output))

            mean, std = torch.mean(output), torch.std(output)
            output = (output - mean) * 1.0 / std


            return output, hidden_state


    args = get_args()
    enc_model = Encoder(args)

    enc_weights = torch.load("./Imp_Functions/encoder_fading.pt", map_location="cpu")

    enc_model.load_state_dict(enc_weights)
    enc_model.args = args

    # Obtain input from mat file
    input_data = sio.loadmat("Data_Files/TX_Encoded3.mat")
    input_data = input_data["encoder_data"]
    input_arr = torch.from_numpy(input_data)
    args.batch_size = input_arr.shape[0]
    input_arr = input_arr.type(torch.FloatTensor)
    input_arr = input_arr.reshape(args.batch_size, 1, 3)
    input = input_arr

    hidden_state = sio.loadmat("Data_Files/TX_Hidden2.mat")
    hidden_state = hidden_state["hidden"]
    hidden_state = torch.from_numpy(hidden_state)
    hidden_state = hidden_state.type(torch.FloatTensor)

    output, _ = enc_model(input, hidden_state, 3)

    output = output.detach().numpy()
    output = output.reshape(args.batch_size, 1)

    hidden_state = hidden_state.detach().numpy()

    output = {"output": output}
    sio.savemat("Data_Files/TX_Modulated3.mat", output)

else:
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
                
        def forward(self, input, hidden_state, round, pow_alloc = False):
            
            output, hidden_state = self.enc_rnn_fwd(input, hidden_state)
            output = self.enc_act(self.enc_linear(output))

            mean, std = torch.mean(output), torch.std(output)
            output = (output - mean) * 1.0 / std

            if pow_alloc:
                den_enc = torch.sqrt(self.p1_enc ** 2 + self.p2_enc ** 2 + self.p3_enc ** 2)
                if round == 1:
                    output = np.sqrt(3) * (self.p1_enc * output) / den_enc

                elif round == 2:
                    output = np.sqrt(3) * (self.p2_enc * output) / den_enc

                elif round == 3:
                    output = np.sqrt(3) * (self.p3_enc * output) / den_enc

            return output, hidden_state


    args = get_args()
    enc_model = Encoder(args)

    enc_weights = torch.load("./Imp_Functions/encoder.pt", map_location="cpu")
    enc_model.load_state_dict(enc_weights)
    enc_model.args = args

    # Obtain input from mat file
    input_data = sio.loadmat("Data_Files/TX_Encoded3.mat")
    input_data = input_data["encoder_data"]
    input_arr = torch.from_numpy(input_data)
    args.batch_size = input_arr.shape[0]
    input_arr = input_arr.type(torch.FloatTensor)
    input_arr = input_arr.reshape(args.batch_size, 1, 3)
    input = input_arr

    hidden_state = sio.loadmat("Data_Files/TX_Hidden2.mat")
    hidden_state = hidden_state["hidden"]
    hidden_state = torch.from_numpy(hidden_state)
    hidden_state = hidden_state.type(torch.FloatTensor)

    output, _ = enc_model(input, hidden_state, 3)

    output = output.detach().numpy()
    output = output.reshape(args.batch_size, 1)

    hidden_state = hidden_state.detach().numpy()

    output = {"output": output}
    sio.savemat("Data_Files/TX_Modulated3.mat", output)
