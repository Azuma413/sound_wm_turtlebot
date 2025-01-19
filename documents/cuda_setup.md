# NVIDIAドライバーとCUDAのセットアップ
最新の情報はNvidiaのWebページを参照してください．
## GPUの確認
```bash
lspci | grep -i nvidia | grep VGA
```
## NVIDIAドライバーのインストール
```bash
ubuntu-drivers devices
sudo apt install -y nvidia-driver-550-open
sudo apt install -y cuda-drivers-550
```

## CUDAツールキットのインストール
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```