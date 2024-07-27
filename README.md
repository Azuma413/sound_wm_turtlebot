・mapはsound_turtle/sound_turtle/my_envs/map/
・weightはsound_turtle/sound_turtle/weight/
・configはsound_turtle/sound_turtle/my_config/

トレーニング
・sound_turtle/sound_turtle/train_dreamer.py
・sound_turtle/sound_turtle/train_drq.py

マップ作製
ssh raspi1@raspi1.local
ros2 launch turtlebot3_bringup robot.launch.py
ros2 launch turtlebot3_cartographer cartographer.launch.py
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/sound_turtle/map/test

起動コマンド
# ターミナル1
ssh raspi1@raspi1.local
ros2 launch sound_turtle turtle.launch.py
# ターミナル2
(仮想環境のアクティベート)
(control_nodeの起動)
(wrap_nodeの起動)

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
