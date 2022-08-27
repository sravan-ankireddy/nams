import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time
import subprocess
import matplotlib.pyplot as plt

forward_tx_gain = 5
forward_rx_gain = 12
backward_tx_gain = 10
backward_rx_gain = 15

string1 = "python3 TX_Feedback.py -tx_gain " + str(forward_tx_gain)
string2 = "python3 RX_Feedback.py -rx_gain " + str(forward_rx_gain)
string3 = "python3 TX_Feedback.py"
string4 = "python3 RX_Feedback.py"


def run_scheme(alpha):

    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    subprocess.call(
        string1 + " -dev_type encoder -num 1 -scale " + str(alpha), shell=True
    )
    subprocess.call(string2 + " -dev_type decoder -num 1", shell=True)
    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    subprocess.call(string4 + " -dev_type encoder -num 1 -fdbk noiseless", shell=True)
    subprocess.call(string3 + " -dev_type decoder -num 1 -fdbk noiseless", shell=True)

    subprocess.call(
        string1 + " -dev_type encoder -num 2 -scale " + str(alpha), shell=True
    )
    subprocess.call(string2 + " -dev_type decoder -num 2", shell=True)
    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    subprocess.call(string4 + " -dev_type encoder -num 2 -fdbk noiseless", shell=True)
    subprocess.call(string3 + " -dev_type decoder -num 2 -fdbk noiseless", shell=True)

    subprocess.call(
        string1 + " -dev_type encoder -num 3 -scale " + str(alpha), shell=True
    )
    subprocess.call(string2 + " -dev_type decoder -num 3", shell=True)
    subprocess.call("python3 flowgraph_process_kill.py", shell=True)

    subprocess.call("python3 RX_Feedback.py -dev_type encoder -num 3", shell=True)

    Bit_Input = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Bit_Input.mat"
    )
    Bit_Input = Bit_Input["Bit_Input"]

    Y1_Output = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/Y1_Output.mat"
    )
    Y1_Output = Y1_Output["YL"]

    BB3_Output = sio.loadmat(
        "/home/rajesh/ActiveFeedback/WirelessDL/Python/Automation/Feedback_Files/BB3_Output.mat"
    )
    BB3_Output = BB3_Output["BB_Output"]

    D1_output = np.zeros((len(Y1_Output), 1))
    D2_output = np.zeros((len(BB3_Output), 1))

    D1_output = np.where(Y1_Output < 0, D1_output, 1)
    D2_output = np.where(BB3_Output > 0.5, D2_output, 1)

    BER1 = 1 - sum(abs(D1_output - Bit_Input)) / len(Bit_Input)
    BER2 = 1 - sum(abs(D2_output - Bit_Input)) / len(Bit_Input)
    print(BER1)
    print(BER2)

    return BER1, BER2


alpha_list = np.linspace(15, 3, 20)
BER = np.zeros((len(alpha_list), 2))

for i in range(len(alpha_list)):
    alpha = alpha_list[i]

    BER1, BER2 = run_scheme(alpha)

    BER[i, 0] = BER1
    BER[i, 1] = BER2

a = np.random.randint(0, 100000)
Data = {"BER": BER, "Alpha": alpha_list}
sio.savemat("Datasets/Data_" + str(a) + ".mat", Data)
