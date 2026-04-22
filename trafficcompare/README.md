# Train model 

```bash
# NYC
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 1
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 1
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 0
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 0 --streaming_postprocess 0

# Chicago
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 1 --streaming_postprocess 1
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 0 --streaming_postprocess 1
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 1 --streaming_postprocess 0
python train.py --gpus 0 --dataset chicago --model myplan --evolution_smooth 0 --streaming_postprocess 0

# Baselines
python train.py --gpus 0 --dataset nyc --model lstm
python train.py --gpus 0 --dataset nyc --model gru
python train.py --gpus 0 --dataset nyc --model mlp
```

如果想本地保存模型结果

```bash
python train.py --gpus 0 --dataset nyc --model myplan --evolution_smooth 1 --streaming_postprocess 1 --save_weights weights/myplan_nyc.h5
```

# Requirements

- Python 3.9
- TensorFlow 2.10.0
- 或 TensorFlow 2.10.0（GPU版本需要 CUDA 11.2 + cuDNN 8.1，Windows GPU 最后一个官方支持版本）
- 其他依赖见 `requirement.txt`

安装方式：

```bash
pip install -r requirement.txt

查看数据集文件：
python test.py nyc/data_nyc.npy --max_print 20
python test.py nyc/data_nyc.npy --stats

查看标签文件：
python test.py nyc/label.npy --max_print 20
python test.py nyc/label.npy --stats

```
