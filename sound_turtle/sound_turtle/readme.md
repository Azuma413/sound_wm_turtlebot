# フォルダの説明
ROS2ノードを格納しています。
### control_node.py
強化学習モデルを呼び出して観測と報酬から行動をPublishします。
### wrap_node.py
DoaNodeやSLAMからSubscribeしたデータを利用して強化学習用の観測と報酬をPublishします。
### doa_node.py
マイクロフォンアレイ（spresense?）から送られてきた音のデータから方向ごとの音源の存在確率を計算してPublishします。