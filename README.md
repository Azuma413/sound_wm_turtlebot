# 環境構築
## Ubuntu22.04のインストール
[こちらのページ](https://www.releases.ubuntu.com/22.04/)からDesktop imageをダウンロードする。  
rufus等を用いてUSBに書き込んで，ブータブルUSBを作成すること。
F1, F2, Delキー等を押してBIOSに入ったら，メディアの読み込み順序を変更し，USBの優先度を上げる。
再起動したら指示に従いつつUbuntuのインストールを行う事。
## 初期設定
Dvorak配列と，terminatorを使えるようにする。
```
sudo apt install terminator
```
## ROS2 Humbleのインストール
文字コードの変更
```
sudo apt update
sudo locale-gen ja_JP ja_JP.UTF-8
sudo update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8
export LANG=ja_JP.UTF-8
```
ROS2パッケージをインストールする準備
```
sudo apt install curl gnupg2 lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt upgrade
```
ROS2パッケージのインストール
```
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep python3-argcomplete
sudo rosdep init
rosdep update
```
Gazeboシミュレータとrqtもインストールしておく。
```
sudo apt -y install gazebo
sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-rqt-*
```
環境設定
```
sudo nano ~/.bashrc
```
最後の行に以下の文を追加する。
```
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
export TURTLEBOT3_MODEL=waffle
export LDS_MODEL=LDS-02
```
```
source .bashrc
```
ワークスペースを作成する。
```
mkdir -p ~/ros2_ws/src
cd ros2_ws
colcon build
```
もう一度bashrcを開いて最後の行に以下の文を追加する。
```
source ~/ros2_ws/install/setup.bash
```
```
source .bashrc
```
## ROS2パッケージのインストール
```
sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
```
## このリポジトリのダウンロード
```
cd ~/ros2_ws/src
git clone https://github.com/Azuma413/sound_turtlebot.git
```
## 機械学習環境のセットアップ
```
sudo apt update && sudo apt upgrade -y
```
グラボの確認
```
lspci | grep -i nvidia | grep VGA
```
nouveauドライバの無効化
```
echo 'blacklist nouveau' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo 'options nouveau modeset=0' | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
cat /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
```
NVIDIA ドライバとNVIDIA CUDA ツールキットのアンインストール
```
dpkg -l | grep cuda 
cd /tmp
sudo apt --purge remove -y nvidia-*
sudo apt --purge remove -y cuda-*
sudo apt --purge remove -y libcudnn*
sudo apt --purge remove -y cudnn-*
sudo apt autoremove -y
```
カーネルヘッダーと，カーネル開発用パッケージのインストール
```
sudo apt -y update
sudo apt -y install linux-headers-$(uname -r)
```
NVIDIA ドライバのインストール操作
```
sudo apt -y update
sudo apt -y upgrade
sudo apt dist-upgrade
ubuntu-drivers devices
```
推奨されているドライバをインストールする(例)
```
sudo apt install -y nvidia-driver-550-open
sudo apt install -y cuda-drivers-550
sudo update-initramfs -u
```
システムを再起動する。  
NVIDIAドライバの確認
```
nvidia-smi
```
cuda toolkitをインストールする
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```
## Pythonライブラリのインストール
pytorchのインストール
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
gpuが利用できるかテストする
```
python
>> import torch
>> torch.cuda.is_available()
```
FalseならNVIDIA ドライバとNVIDIA CUDA ツールキットのアンインストールからやり直す  
requirements.txtからライブラリをインストール
```
cd ~/ros2_ws/src/sound_turtlebot
pip install -r requirements.txt
```
## turtlebotをインターネットに接続する
turtlebotとディスプレイを接続し，wifiに接続する。  
ssh接続ができるか確認すること。
```
ssh raspi1@raspi1.local
cd ros2_ws/src/sound_turtlebot
git pull
```
## テスト
turtlebotとPCでROS通信を行う  
- ターミナル１
```
ssh raspi1@raspi1.local
ros2 run demo_nodes_cpp talker
```
- ターミナル2
```
ros2 run demo_nodes_cpp listener
```
# SLAM
- ターミナル１
```
ssh raspi1@raspi1.local
ros2 launch turtlebot3_bringup robot.launch.py
```
- ターミナル2
```
ros2 launch turtlebot3_cartographer cartographer.launch.py
```
- ターミナル3
配列をqwertyにすること。
```
ros2 run turtlebot3_teleop teleop_keyboard
```
- ターミナル4
十分にマップを作成することができたら実行すること。
```
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/sound_turtlebot/sound_turtle/sound_turtle/my_envs/map/main
```
# メインプログラムの実行
- ターミナル１
```
ssh raspi1@raspi1.local
ros2 launch sound_turtle turtle.launch.py
```
- ターミナル2
```
ros2 launch sound_turtle main.launch.py
```
# トレーニングの実行
- DreamerV3
```
python sound_turtle/sound_turtle/train_dreamer.py
```
- DrQv2
```
sound_turtle/sound_turtle/train_drq.py
```
# フォルダについて
・map  
sound_turtle/sound_turtle/my_envs/map/  
・weight  
sound_turtle/sound_turtle/weight/  
・config  
sound_turtle/sound_turtle/my_config/  
# ToDo
動作確認が必要な項目  
・dummy_control_nodeを用いてwrap_nodeが正常に動作する  
	・spatial_respからsound_mapをpublishできる  
	・正常にobs_imageを生成できる  
	・goal_poseを設定できる  
・launchが正常に動作する  
・8/2までに動かせる状態にしておく  
・githubのReadMeにセットアップのコマンドを書いておく（bashファイルを作るのもあり）必要なpythonライブラリを纏めておくこと。（ros runしないのであればvenvで仮想環境を作ってしまうのもあり）  
・当日やることのリストを作成しておく。当日は全体の動作確認と，実験室のMAP生成  
・turtlebotの層の3Dモデルデータを探して3Dプリントしておく。（層を増やす）  
map作成の手順などについてもまとめておく事。  
# 参考文献
