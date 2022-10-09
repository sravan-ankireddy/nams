import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.io as sio

def save_datafiles(
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
    r,
    tx_gain,
    rx_gain,
    bin
):
    W, X, Y, Z, ZZ = [], [], [], [], []
    PP = 0

    # At the Tx, we generate 20 frames and keep transmitting multiple times : hence a
    # At the Rx, we capture 20*2 frames and look for the specific frame number : hence i

    for n in range(N_captures):
        for i in range(Frame_Error.shape[1]):
            a = int(Frame_Error[n, i, 1]) # Tx frame_num
            if Frame_Error[n, i, 0] >= berLo and Frame_Error[n, i, 0] < berHi:
                W.append(Data_Input[n, :, :, a])
                X.append(Encoder_Output[n, :, :, a])
                Y.append(Receiver_Output[n, :, i])
                Z.append(Data_Output[n, :, :, i])
                ZZ.append(Demod_Input[n, :, :, i])

    Data = {
        "Data_Input": np.array(W),
        "Encoder_Output": np.array(X),
        "Receiver_Output": np.array(Y),
        "Data_Output": np.array(Z),
        "Demod_Input": np.array(ZZ),
    }

    # saving to mat file
    ota_data = {
        "msg_data": np.array(W),
        "enc_data": np.array(X),
        "rx_data": np.array(Y),
        "demod_input": np.array(ZZ),
        "llr_data": np.array(Z),
    }
    pkl.dump(
        Data, open(file_name + "/Data_" + str(berHi) + "_" + str(berLo) + ".pkl", "wb")
    )
    N_f = str(np.shape(Encoder_Output)[1])
    K_f = str(np.shape(Data_Input)[1])
    if (bin):
        # saving to mat file - per ber err range
        filename = "data_files/ota_data/ota_data_bins/ota_data_" + N_f + "_" + K_f + "_bin_" + str(r) + "_gains_"+ str(tx_gain) + "_" + str(rx_gain) + ".mat"
        sio.savemat(filename,ota_data)
    else:
        # saving to mat file - full data - per tx power range
        filename = "data_files/ota_data/ota_data_blocks/ota_data_" + N_f + "_" + K_f + "_" + str(tx_gain) + "_" + str(rx_gain) + ".mat"
        sio.savemat(filename,ota_data)
    return W, X, Y, Z


def plot_histogram(file_name, X, Z, berHi, berLo):
    plt.figure()
    Z = np.array(Z).flatten()
    E = np.array(X).flatten()
    plt.hist(Z[E == 1], 1000, color="red", alpha=0.3)
    plt.hist(Z[E == 0], 1000, color="blue", alpha=0.3)
    plt.grid(True)
    plt.xlim([-50, 50])
    plt.savefig(file_name + "/Figures/Data_" + str(berHi) + "_" + str(berLo) + ".png")
