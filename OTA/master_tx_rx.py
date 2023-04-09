import numpy as np
import matlab.engine
from datetime import date
import os
import scipy.io as sio
import pickle as pkl
# import matplotlib.pyplot as plt
import os
# from Sanity import check_dataset
from tqdm import tqdm
from data_utils import save_datafiles

data_generate = True
sanity_check = True
OTA_data = True
Frame_802_11 = True
save_data = True

N_captures = 1
# tx_gain_vec = np.arange(10,-5,-1)
# rx_gain_vec = np.arange(2,0.5,-0.5)

tx_gain_vec = np.arange(0,-1,-1)
rx_start = 10
rx_end = rx_start - 0.5
rx_gain_vec = np.arange(rx_start,rx_end,-0.5)

for i_r in tqdm(range(rx_gain_vec.size)):
    rx_gain = rx_gain_vec[i_r]
    
    for i_t in tqdm(range(tx_gain_vec.size)):

        if save_data:
            id = "QPSK_FRAMED_OTA_3"

            if OTA_data == True and Frame_802_11 == False:
                raise Exception("OTA Data must be Framed")

            current_date = date.today()
            file_name = "Datasets/Dataset_" + str(id) + "_" + str(current_date)

            # make a folder for today's data
            try:
                try:
                    os.mkdir("Datasets")
                except:
                    pass
                os.mkdir(file_name)
                os.mkdir(file_name + "/Figures")
            except:
                pass

        print(" Capture Process Starts")
        print("=========================\n")
        print("Starting MATLAB")

        eng = matlab.engine.start_matlab()

        if save_data:
            Data_Input, Encoder_Output, Receiver_Output, Demod_Input, Data_Output, Frame_Error = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

        tx_gain = float(tx_gain_vec[i_t])

        for i in tqdm(range(N_captures)):
            # Generate TX
            if Frame_802_11:
                eng.TX_data_generate("802_11 Framed", i, nargout=0)
            else:
                eng.TX_data_generate("Raw Data", nargout=0)
            # Pass through Channel

            if OTA_data:
                os.system("python3 flow_graph.py -tx_gain " + str(tx_gain) + " -rx_gain " + str(rx_gain))
            else:
                eng.awgn_channel("complex_noise", nargout=0)

            # Decode RX
            if Frame_802_11:
                eng.RX_data_extract("802_11 Framed", nargout=0)
            else:
                eng.RX_data_extract("Raw Data", nargout=0)

            if save_data:
                # Prepare Data
                # Input Data
                Data = sio.loadmat("data_files/ota_data/msg_data.mat")
                Data_Input.append(Data["msg_data"])

                # Encoder Output
                Data = sio.loadmat("data_files/ota_data/enc_data.mat")
                Encoder_Output.append(Data["enc_data"])

                # Receiver Output
                Data = sio.loadmat("data_files/ota_data/rx_data.mat")
                Receiver_Output.append(Data["rx_data"])

                # Demod Input
                Data = sio.loadmat("data_files/ota_data/demod_input.mat")
                Demod_Input.append(Data["demod_input"])

                # Output LLR
                Data = sio.loadmat("data_files/ota_data/demod_data.mat")
                Data_Output.append(Data["demod_data"])

                # Frame Error
                Data = sio.loadmat("data_files/ota_data/frame_error.mat")
                Frame_Error.append(Data["frame_error"])

        if save_data:
            Data_Input = np.array(Data_Input)
            Demod_Input = np.array(Demod_Input)
            Data_Output = np.array(Data_Output)
            Receiver_Output = np.array(Receiver_Output)
            Encoder_Output = np.array(Encoder_Output)
            Frame_Error = np.array(Frame_Error)

            if ~Frame_802_11:
                Data_Input = np.expand_dims(Data_Input, -1)
                Encoder_Output = np.expand_dims(Encoder_Output, -1)
                Receiver_Output = np.expand_dims(Receiver_Output, -1)
                Demod_Input = np.expand_dims(Demod_Input, -1)
                Data_Output = np.expand_dims(Data_Output, -1)

        if save_data:
            BERHI = [0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
            BERLO = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00000]
            bin = 1
            for i in range(len(BERHI)):
                berHi = BERHI[i]
                berLo = BERLO[i]
                _, X, _, Z = save_datafiles(
                    file_name,
                    N_captures,
                    Frame_Error,
                    Data_Input,
                    Encoder_Output,
                    Receiver_Output,
                    Demod_Input,
                    Data_Output,
                    berHi,
                    berLo,
                    i,
                    tx_gain,
                    rx_gain,
                    bin
                )

            BERHI = [0.4]
            # BERLO = [0.00001]
            BERLO = [0.000000]
            bin = 0
            for i in range(len(BERHI)):
                berHi = BERHI[i]
                berLo = BERLO[i]
                _, X, _, Z = save_datafiles(
                    file_name,
                    N_captures,
                    Frame_Error,
                    Data_Input,
                    Encoder_Output,
                    Receiver_Output,
                    Demod_Input,
                    Data_Output,
                    berHi,
                    berLo,
                    i,
                    tx_gain,
                    rx_gain,
                    bin
                )

        # # Plot histogram
        # plot_histogram(file_name, X, Z, berHi, berLo)
        # # Sanity Checks
        # check_dataset(
        #     eng, file_name + "/Data_" + str(berHi) + "_" + str(berLo) + ".pkl"
        # )
