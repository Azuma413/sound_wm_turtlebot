# TurtleBotを用いた実環境での動作方法（実装中）
## ROS通信のテスト
TurtlebotとPC間の通信をテストします
```bash
# ターミナル1（Turtlebot上）
ssh raspi1@raspi1.local
ros2 run demo_nodes_cpp talker

# ターミナル2（PC上）
ros2 run demo_nodes_cpp listener
```

## SLAMマッピング
```bash
# ターミナル1（Turtlebot上）  
ros2 launch turtlebot3_bringup robot.launch.py

# ターミナル2  
ros2 launch turtlebot3_cartographer cartographer.launch.py

# ターミナル3（キーボード操作用）  
ros2 run turtlebot3_teleop teleop_keyboard

# ターミナル4（マップの保存）
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/sound_turtlebot/sound_turtle/sound_turtle/my_envs/map/main
```

## メインプログラムの実行
```bash
# ターミナル1（Turtlebot上）
ros2 launch sound_turtle turtle.launch.py

# ターミナル2
ros2 launch sound_turtle main.launch.py
```