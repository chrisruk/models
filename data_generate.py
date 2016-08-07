from __future__ import division, print_function, absolute_import
import numpy as np
from gnuradio import filter
from gnuradio import gr
from gnuradio import analog
from gnuradio import digital
from gnuradio import blocks
from gnuradio import channels


from multiprocessing import Process, Queue

np.random.seed(1337)  # for reproducibility

channel_model = False
SNR = ["20", "15", "10", "5", "0", "-5", "-10", "-15", "-20"]
SNRV = [
    [
        1, 0.32], [
            1, 0.435], [
                1, 0.56], [
                    1, 0.75], [
                        1, 1], [
                            0.75, 1], [
                                0.56, 1], [
                                    0.435, 1], [
                                        0.32, 1]]

MOD = ["fsk", "qam16", "qam64", "2psk", "4psk", "8psk", "gmsk", "wbfm", "nfm"]

## Generate training data using multiple flow graphs running simultaneously
def getdata(sn,syms,process,train=False):
    mcount = 0

    if train:
        inp = []
        out = []
    else:
        inp = [[] for k in range(0, len(SNR))]
        out = [[] for k in range(0, len(SNR))]

    flow = [None for k in range(len(MOD))]

    for m in MOD:

        z = np.zeros((len(MOD),))
        z[mcount] = 1

        print("MOD ", z)

        q = Queue()  # create a queue object
        plist = []
        for s in sn:
            for sy in syms:
                p = Process(target=process, args=(train, m, s, z, q,sy))
                plist.append(p)
                p.start()

        for p in plist:
            job = q.get()
            if train:
                inp += job[0]
                out += job[1]
            else:
                for i in range(len(inp)):
                    inp[i] += job[0][i]
                    out[i] += job[1][i]

        print("waiting for join")
        for p in plist:
            p.join()
        print("joined")

        mcount += 1

    return np.array(inp), np.array(out)



def create_blocks(self,modulation,sym,sn):
    self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
        interpolation=1,
        decimation=1,
        taps=None,
        fractional_bw=None,
    )

    if modulation == "2psk":
        self.digital_mod = digital.psk.psk_mod(
            constellation_points=2,
            mod_code="gray",
            differential=True,
            samples_per_symbol=sym,
            excess_bw=0.35,
            verbose=False,
            log=False,
        )
    elif modulation == "4psk":
        self.digital_mod = digital.psk.psk_mod(
            constellation_points=4,
            mod_code="gray",
            differential=True,
            samples_per_symbol=sym,
            excess_bw=0.35,
            verbose=False,
            log=False,
        )
    elif modulation == "8psk":
        self.digital_mod = digital.psk.psk_mod(
                constellation_points=8,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
        )
    elif modulation == "fsk":
        self.digital_mod = digital.gfsk_mod(
                samples_per_symbol=sym,
                sensitivity=1.0,
                bt=0.35,
                verbose=False,
                log=False,
        )
    elif modulation == "qam16":
        self.digital_mod = digital.qam.qam_mod(
                constellation_points=16,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
        )
    elif modulation == "qam64":
        self.digital_mod = digital.qam.qam_mod(
                constellation_points=64,
                mod_code="gray",
                differential=True,
                samples_per_symbol=sym,
                excess_bw=0.35,
                verbose=False,
                log=False,
        )
    elif modulation == "gmsk":
        self.digital_mod = digital.gmsk_mod(
                samples_per_symbol=sym,
                bt=0.35,
                verbose=False,
                log=False,
        )

    self.blocks_wavfile_source_0 = blocks.wavfile_source("/home/chris/Desktop/music.wav", False)

    self.analog_wfm_tx_0 = analog.wfm_tx(
        audio_rate=44100,
        quad_rate=44100 * 5,
        tau=75e-6,
        max_dev=75e3,
        fh=-1.0,
    )

    self.analog_nfm_tx_0 = analog.nbfm_tx(
        audio_rate=44100,
        quad_rate=44100 * 2,
        tau=75e-6,
        max_dev=5e3,
        fh=-1.0,
    )

    self.channels_channel_model_0 = channels.channel_model(
        noise_voltage=SNRV[sn][1],
        frequency_offset=100.0,
        epsilon=1.0,
        taps=(1.0 + 1.0j, ),
        noise_seed=0,
        block_tags=False
    )
