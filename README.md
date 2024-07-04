# nav_cloning

## Running simulation
- データ収集（配置）  
![collect_data_resize](https://github.com/YukiTakahashi4690/nav_cloning/assets/72371474/d3e43a62-31b8-4a51-b581-4c9d201a0ebb)  
    - このように目標経路に対してロボットを配置する
    - データ収集後はangフォルダとimgフォルダ内にそれぞれ角速度と画像が保存される  
```
nav_cloning nav_cloning_sim.launch script:=set_collect.py use_waypoint_nav:=false
```
- データ収集（走行）
```
nav_cloning nav_cloning_sim.launch script:=run_collect.py use_waypoint_nav:=true
```
- オフライン学習
    - シェル内で呼び出しているスクリプトのself.proを使用するデータフォルダ名に変更する
    - スクリプト内のself.dataもデータ数に応じて数値を変更させる
    - **angフォルダ名とimgフォルダ名は同一名にする**
    - 配置したデータで学習
    ```
    roscd nav_cloning/sh
    ./learning.sh
    ```  
    - 走行させたデータで学習
    ```
    roscd nav_cloning/sh
    ./learning_3cam.sh
    ```
- 学習したモデルで経路追従できるかテスト
```
./test.sh
```

## install
* 環境 ubuntu20.04, ros noetic

* ワークスペースの用意
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ../
catkin build
```
* nav_cloningの用意
```
cd ~/catkin_ws/src
wget https://raw.githubusercontent.com/YukiTakahashi4690/nav_cloning/master/nav_cloning.install
wstool init
wstool merge nav_cloning.install
wstool up
```
* 依存パッケージのインストール
```
cd ~/catkin_ws/src
rosdep init
rosdep install --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y
cd ../
catkin build
```
* 共通
```
pip3 install scikit-image
pip3 install tensorboard
```
* ＜CPU のみ＞
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
* ＜GPU 使用＞使用しているデバイスを確認し，セットアップします
- nvidia driver
- CUDA
- cuDNN
その後インストールしたCUDAのバージョンに対応したPytorchのバージョンを下記からダウンロードします
https://pytorch.org/get-started/locally/
## Docker
作成次第追加
