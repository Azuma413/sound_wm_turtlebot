import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import time
p = pyaudio.PyAudio()
device = "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:3,0)"
device2 = "TAMAGO-03: USB Audio (hw:3,0)"
index = None

for i in range(p.get_device_count()):
    # print(p.get_device_info_by_index(i))
    if p.get_device_info_by_index(i)['name'] == device2:
        print("Found device [ID:{}]".format(i))
        index = i
        break
    if i == p.get_device_count() - 1:
        print("Device not found")
        exit()

# 定数の定義
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 8 #4 # チャンネル数
samp_rate = 16000 # サンプリングレート
chunk = 8192 # 2^13 バッファーサイズ
record_secs = 20 # seconds to record
dev_index = index # device index
# DOA推定のための設定
c = 343.    # 音速
nfft = 256  # FFTのサイズ
freq_range = [300, 3500]

audio = pyaudio.PyAudio() # create pyaudio instantiation
stream = audio.open(format = form_1,rate = samp_rate, channels = chans, input_device_index=dev_index, input = True, frames_per_buffer=chunk)
print("recording")
frames = []
mic_locs = pra.circular_2D_array(center=[0,0], M=chans, phi0=0, radius=0.05)
fig, ax = plt.subplots(1, 1)

for i in range(0,int((samp_rate/chunk)*record_secs)):
    start_time = time.time()
    data = stream.read(chunk)
    data = np.frombuffer(data, dtype='int16') # int16に変換
    data = data.reshape(-1, chans) # チャンネル数に合わせて整形 (n_samples, n_channels)
    X = pra.transform.stft.analysis(data, nfft, nfft // 2)
    X = X.transpose([2, 1, 0])
    doa = pra.doa.algorithms['MUSIC'](mic_locs, samp_rate, nfft, c=c, num_src=1, max_four=4)
    doa.locate_sources(X, freq_range=freq_range)
    spatial_resp = doa.grid.values
    min_val = spatial_resp.min()
    max_val = spatial_resp.max()
    # spatial_respをプロット
    ax.clear()
    ax.plot(np.rad2deg(doa.grid.azimuth), spatial_resp.T)
    ax.vlines(np.rad2deg(doa.azimuth_recon), 0, 1, color='r', linestyle='--')
    ax.set_title('Azimuth')
    ax.set_xlabel('Angle [degrees]')
    ax.set_ylabel('Probability')
    plt.pause(0.001)
    print("Elapsed Time: ", time.time() - start_time)
    
print("finished recording")
stream.stop_stream()
stream.close()
audio.terminate()