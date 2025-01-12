# Sound Turtlebot: A ROS2-based Sound Source Localization Robot
<div align="center">
  <!-- You can add a project logo/image here if desired -->
</div>

## Overview
This project implements a sound source localization system using a Turtlebot3 robot with ROS2. The system combines SLAM-based navigation with deep reinforcement learning algorithms (DreamerV3, DrQv2) to enable the robot to autonomously locate and navigate towards sound sources in an environment.

## Installation

### System Requirements
* Ubuntu 22.04
* ROS2 Humble
* Python 3.8+
* NVIDIA GPU with CUDA support

### ROS2 Humble Setup
```bash
# Set up locale
sudo apt update
sudo locale-gen ja_JP ja_JP.UTF-8
sudo update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8
export LANG=ja_JP.UTF-8

# Add ROS2 repository
sudo apt install curl gnupg2 lsb-release -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 packages
sudo apt update && sudo apt upgrade
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep python3-argcomplete
sudo rosdep init
rosdep update

# Install Gazebo and RQT
sudo apt -y install gazebo
sudo apt install ros-humble-gazebo-*
sudo apt install ros-humble-rqt-*

# Install additional ROS2 packages
sudo apt install ros-humble-cartographer
sudo apt install ros-humble-cartographer-ros
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
```

### Environment Setup
Add to ~/.bashrc:
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=30
export TURTLEBOT3_MODEL=waffle
export LDS_MODEL=LDS-02
source ~/ros2_ws/install/setup.bash
```

### NVIDIA Driver and CUDA Setup
```bash
# Check GPU
lspci | grep -i nvidia | grep VGA

# Install NVIDIA driver
ubuntu-drivers devices
sudo apt install -y nvidia-driver-550-open
sudo apt install -y cuda-drivers-550

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

### Project Setup
```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/Azuma413/sound_turtlebot.git

# Install Python dependencies
cd sound_turtlebot
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Build workspace
cd ~/ros2_ws
colcon build
```

## Usage

### Testing ROS Communication
Test the ROS communication between the Turtlebot and PC:

```bash
# Terminal 1 (on Turtlebot)
ssh raspi1@raspi1.local
ros2 run demo_nodes_cpp talker

# Terminal 2 (on PC)
ros2 run demo_nodes_cpp listener
```

### SLAM Mapping
```bash
# Terminal 1 (on Turtlebot)
ros2 launch turtlebot3_bringup robot.launch.py

# Terminal 2
ros2 launch turtlebot3_cartographer cartographer.launch.py

# Terminal 3 (for keyboard control)
ros2 run turtlebot3_teleop teleop_keyboard

# Terminal 4 (save map when finished)
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/sound_turtlebot/sound_turtle/sound_turtle/my_envs/map/main
```

### Main Program Execution
```bash
# Terminal 1 (on Turtlebot)
ros2 launch sound_turtle turtle.launch.py

# Terminal 2
ros2 launch sound_turtle main.launch.py
```

### Training
```bash
# Train DreamerV3
python sound_turtle/sound_turtle/train_dreamer.py

# Train DrQv2
python sound_turtle/sound_turtle/train_drq.py
```

## Directory Structure
```
.
├── sound_turtle/                    # Main ROS2 package
│   ├── launch/                     # Launch files
│   ├── sound_turtle/               # Python package
│   │   ├── my_envs/               # Environment implementations
│   │   ├── my_config/             # Configuration files
│   │   ├── dreamerv3/             # DreamerV3 implementation
│   │   └── drqv2/                 # DrQv2 implementation
│   └── test/                       # Test files
├── sound_turtle_msgs/              # Custom ROS2 messages
├── WorkSpace/                      # Development workspace
└── CADData/                        # CAD models
```

## Important Directories
- Maps: `sound_turtle/sound_turtle/my_envs/map/`
- Weights: `sound_turtle/sound_turtle/weight/`
- Configs: `sound_turtle/sound_turtle/my_config/`

### Additional Tools Setup
```bash
# Install terminator (recommended terminal emulator)
sudo apt install terminator
```

### Turtlebot Network Setup
1. Connect a display to the Turtlebot and configure WiFi connection
2. Verify SSH connectivity:
```bash
ssh raspi1@raspi1.local
cd ros2_ws/src/sound_turtlebot
git pull
```

## License and Citation

### License
This project is licensed under the [LICENSE](LICENSE) file in the repository.

### Citation
If you use this project in your research, please cite:
```
[Citation information to be added]
```
