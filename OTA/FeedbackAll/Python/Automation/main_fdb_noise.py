import numpy as np
import matlab.engine
import os
import signal
import time
import scipy.io as sio
import time
import subprocess
import matplotlib.pyplot as plt

backward_tx_gain = 8
backward_rx_gain = 10


string3 = "python3 TX_Feedback.py -rx_gain " + str(backward_rx_gain)
string4 = "python3 RX_Feedback.py -tx_gain " + str(backward_tx_gain)

subprocess.call(string4 + " -dev_type encoder_noise", shell=True)
subprocess.call(string3 + " -dev_type decoder_noise -n_captures 4", shell=True)
subprocess.call("python3 flowgraph_process_kill.py", shell=True)

eng = matlab.engine.start_matlab()
eng.channel_fit("Backward", nargout=0)
subprocess.call(
    "cp ./Channel_Files/Backward_* ../../MATLAB/Implementation/Channel_Files/",
    shell=True,
)
