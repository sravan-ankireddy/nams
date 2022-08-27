import argparse
import numpy as np
import uhd
import scipy.io as sio

waveforms = {
    "sine": lambda n, tone_offset, rate: np.exp(n * 2j * np.pi * tone_offset / rate),
    "square": lambda n, tone_offset, rate: np.sign(
        waveforms["sine"](n, tone_offset, rate)
    ),
    "const": lambda n, tone_offset, rate: 1 + 1j,
    "ramp": lambda n, tone_offset, rate: 2
    * (n * (tone_offset / rate) - np.floor(float(0.5 + n * (tone_offset / rate)))),
}


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", default="", type=str)
    parser.add_argument(
        "-w", "--waveform", default="sine", choices=waveforms.keys(), type=str
    )
    parser.add_argument("-f", "--freq", default=2.45e9, type=float)
    parser.add_argument("-r", "--rate", default=10e6, type=float)
    parser.add_argument("-d", "--duration", default=100.0, type=float)
    parser.add_argument("-c", "--channels", default=0, nargs="+", type=int)
    parser.add_argument("-g", "--gain", type=int, default=10)
    parser.add_argument("--wave-freq", default=1e4, type=float)
    parser.add_argument("--wave-ampl", default=0.3, type=float)
    return parser.parse_args()


def main():
    """TX samples based on input arguments"""
    # args = parse_args()
    usrp = uhd.usrp.MultiUSRP("type=usrp2,serial=EER11XBUN")

    # if not isinstance(args.channels, list):
    # args.channels = [args.channels]
    # print(args.channels)
    # data = np.array(
    #     list(
    #         map(
    #             lambda n: args.wave_ampl
    #             * waveforms[args.waveform](n, args.wave_freq, args.rate),
    #             np.arange(
    #                 int(10 * np.floor(args.rate / args.wave_freq)), dtype=np.complex64
    #             ),
    #         )
    #     ),
    #     dtype=np.complex64,
    # )  # One period

    data = sio.loadmat("TX.mat")
    data = data["TX"].astype(np.complex64)

    usrp.send_waveform(data, 100, 2.45e9, 10e6, [0], 4)
    print("rajesh")


if __name__ == "__main__":
    main()
