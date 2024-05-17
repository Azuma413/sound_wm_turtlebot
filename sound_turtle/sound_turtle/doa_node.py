# ライブラリのインポート
# *************************************************************************************************
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pyaudio
import numpy as np
import pyroomacoustics as pra
import time

# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
# device = "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:3,0)" # マイク4つのマイクロフォンアレイを使用
device = "TAMAGO-03: USB Audio (hw:3,0)" # 卵型のマイクロフォンアレイを使用
form_1 = pyaudio.paInt16
chans = 8
samp_rate = 16000
chunk = 8192
record_secs = 20
c = 343.
nfft = 256
freq_range = [300, 3500]
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class DoaNode(Node):
    def __init__(self):
        super().__init__('doa_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'spatial_resp', 10)
        p = pyaudio.PyAudio()
        index = None
        for i in range(p.get_device_count()):
            if p.get_device_info_by_index(i)['name'] == device: # デバイスが見つかったらindexを取得
                self.get_logger().info("Found device [ID:{}]".format(i))
                index = i
                break
            if i == p.get_device_count() - 1: # デバイスが見つからなかったら終了
                self.get_logger().info("Device not found")
                exit()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=form_1, rate=samp_rate, channels=chans, input_device_index=index, input=True, frames_per_buffer=chunk)
        self.get_logger().info("Recording")
        radius = None
        if device == "ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:3,0)":
            radius = 0.065
        elif device == "TAMAGO-03: USB Audio (hw:3,0)":
            radius = 0.065
        self.mic_locs = pra.circular_2D_array(center=[0,0], M=chans, phi0=0, radius=radius)
        self.create_timer(0.1, self.timer_callback)
        
    def timer_callback(self):
        msg = Float32MultiArray()
        start_time = time.time()
        data = self.stream.read(chunk)
        data = np.frombuffer(data, dtype='int16')
        data = data.reshape(-1, chans)
        X = pra.transform.stft.analysis(data, nfft, nfft // 2)
        X = X.transpose([2, 1, 0])
        doa = pra.doa.algorithms['MUSIC'](self.mic_locs, samp_rate, nfft, c=c, num_src=1, max_four=4)
        doa.locate_sources(X, freq_range=freq_range)
        spatial_resp = doa.grid.values
        self.get_logger().info("Elapsed Time: {}".format(time.time() - start_time))
        msg.data = spatial_resp.tolist()
        self.publisher_.publish(msg)
        
    def destroy_node(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.get_logger().info("Finished recording")
        super().destroy_node()
# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main(args=None):
    rclpy.init(args=args)
    doa_node = DoaNode()
    rclpy.spin(doa_node)
    doa_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()