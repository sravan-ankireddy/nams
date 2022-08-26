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
forward_rx_gain = 10


string1 = "python3 TX_Feedback.py -tx_gain " + str(forward_tx_gain)
string2 = "python3 RX_Feedback.py -rx_gain " + str(forward_rx_gain)

subprocess.call(string1 + " -dev_type encoder_noise", shell=True)
subprocess.call(string2 + " -dev_type decoder_noise -n_captures 4", shell=True)
subprocess.call("python3 flowgraph_process_kill.py", shell=True)

eng = matlab.engine.start_matlab()
eng.channel_fit("Forward", nargout=0)
subprocess.call(
    "cp ./Channel_Files/Forward_* ../../MATLAB/Implementation/Channel_Files/",
    shell=True,
)
