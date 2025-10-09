# 数据集下载指南

## 1. VoxCeleb1 数据集
VoxCeleb1 需要手动注册下载：

**步骤：**
1. 访问：https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
2. 填写注册表格获取下载链接
3. 下载以下文件：
   - vox1_dev_wav.zip (训练集，约 38GB)
   - vox1_test_wav.zip (测试集，约 7GB)
4. 解压到：data/voxceleb1/

**目录结构应该是：**
data/voxceleb1/
├── voxceleb1_complete/
│   ├── vox1_dev_wav/
│   │   └── id10001/...
│   └── vox1_test_wav/
│       └── id10270/...

## 2. MUSAN 噪声语料库
MUSAN 可以直接下载（约 12GB）：

**方法1：直接下载**
访问：https://www.openslr.org/17/
下载：musan.tar.gz

**方法2：使用 wget（如果有）**
wget https://www.openslr.org/resources/17/musan.tar.gz

**解压到：** data/musan/

**目录结构应该是：**
data/musan/
├── music/
├── noise/
└── speech/

## 注意事项
- VoxCeleb1 总计约 45GB
- MUSAN 约 12GB
- 请确保有足够的磁盘空间（至少 60GB）
