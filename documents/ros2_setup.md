# ROS2 Humbleのセットアップ
## ロケールの設定
```bash
sudo apt update
sudo locale-gen ja_JP ja_JP.UTF-8
sudo update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8
export LANG=ja_JP.UTF-8
```
## ROS2リポジトリの追加
```bash
sudo apt install curl gnupg2 lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg  
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```
## ROS2パッケージのインストール
```bash
sudo apt update && sudo apt upgrade  
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep python3-argcomplete  
sudo rosdep init  
rosdep update
```
## GazeboとRQTのインストール  
```bash
sudo apt -y install gazebo  
sudo apt install ros-humble-gazebo-*  
sudo apt install ros-humble-rqt-*
```
## その他のROS2パッケージのインストール
```bash
sudo apt install ros-humble-cartographer  
sudo apt install ros-humble-cartographer-ros  
sudo apt install ros-humble-navigation2  
sudo apt install ros-humble-nav2-bringup  
sudo apt install ros-humble-dynamixel-sdk  
sudo apt install ros-humble-turtlebot3-msgs  
sudo apt install ros-humble-turtlebot3  
```
## 環境変数の設定
以下を`~/.bashrc`に追加します：
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
export TURTLEBOT3_MODEL=burger
export LDS_MODEL=LDS-02
source ~/ros2_ws/install/setup.bash
```