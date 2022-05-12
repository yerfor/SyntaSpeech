# SyntaSpeech: Syntax-Aware Generative Adversarial Text-to-Speech

[![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://arxiv.org/abs/2204.11792) | [![GitHub Stars](https://img.shields.io/github/stars/yerfor/SyntaSpeech)](https://github.com/yerfor/SyntaSpeech) | [![downloads](https://img.shields.io/github/downloads/yerfor/SyntaSpeech/total.svg)](https://github.com/yerfor/SyntaSpeech/releases) | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/spaces/yerfor/SyntaSpeech) | [English README](README.md)

这个仓库包含了我们的 IJCAI-2022 [论文](https://arxiv.org/abs/2204.11792) 的官方 PyTorch 实现，我们在其中提出了 **SyntaSpeech** ，一种语法感知的非自回归语音合成算法.

<p align="center">
    <br>
    <img src="assets/SyntaSpeech.png" width="1000"/>
    <br>
</p>

我们的 SyntaSpeech 建立在 [PortaSpeech](https://github.com/NATSpeech/NATSpeech) (NeurIPS 2021) 的基础上，具有三个新功能：

1. 我们提出了**Syntactic Graph Builder (论文的3.1小节)** 和**Syntactic Graph Encoder (论文的3.2小节)**，被证明是提取句法特征以提高韵律建模和持续时间准确性的有效单元 TTS 模型。
2. 我们引入了**Multi-Length Adversarial Training (论文的3.3小节)**，它可以替代PortaSpeech 中基于flow的post-net，加快推理时间的同时提高音频质量的自然度。
3. 我们支持三个数据集：[LJSpeech](https://keithito.com/LJ-Speech-Dataset/)（单人英语数据集）、[Biaobei]() （单人中文数据集）和[LibriTTS](http://www.openslr.org/60)（多人英语数据集）

搭建环境

```
conda create -n synta python=3.7
source activate synta
pip install -U pip
pip install Cython numpy==1.19.1
pip install torch==1.9.0 
pip install -r requirements.txt
# install dgl for graph neural network, dgl-cu102 supports rtx2080, dgl-cu113 support rtx3090
pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html 
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # install force alignment tools

```

## 运行 SyntaSpeech!

请按照以下步骤以运行此仓库。

### 1. 准备数据集和声码器

#### 准备数据集

您可以直接使用我们的处理好的[LJSpeech数据集](https://drive.google.com/file/d/1WfErAxKqMluQU3vupWS6VB6NdehXwCKM/view?usp=sharing)和 [Biaobei](https://drive.google.com/file/d/1-ApEbBrW5kfF0jM18EmW7DCsll-c1ROp/view?usp=sharing)数据集。 从链接给的谷歌云盘里下载它们并将它们解压缩到 `data/binary/` 文件夹中。

至于 LibriTTS，您可以下载原始数据集并使用我们的“data_gen”模块对其进行处理。 详细说明可以在 [dosc/prepare_data](docs/prepare_data.md) 中找到。

#### 准备声码器

我们为三个数据集提供了预训练的声码器模型。 具体来说，Hifi-GAN 用于 [LJSpeech]() 和 [Biaobei]()，ParallelWaveGAN 用于 [LibriTTS]()。 将它们下载并解压到 `checkpoints/`文件夹。

### 2. 开始训练!

然后你可以在三个数据集中训练 SyntaSpeech。

```
cd <the root_dir of your SyntaSpeech folder>
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/lj/synta.yaml --exp_name lj_synta --reset # training in LJSpeech
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/biaobei/synta.yaml --exp_name biaobei_synta --reset # training in Biaobei
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/biaobei/synta.yaml --exp_name libritts_synta --reset # training in LibriTTS
```

### 3. Tensorboard

```
tensorboard --logdir=checkpoints/lj_synta
tensorboard --logdir=checkpoints/biaobei_synta
tensorboard --logdir=checkpoints/libritts_synta
```

### 4. 模型推理

```
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/lj/synta.yaml --exp_name lj_synta --reset --infer # inference in LJSpeech
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/biaobei/synta.yaml --exp_name biaobei_synta --reset --infer # inference in Biaobei
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/tts/biaobei/synta.yaml --exp_name libritts_synta --reset ---infer # inference in LibriTTS
```

## 音频演示

音频样本可以在我们的 [demo page](https://syntaspeech.github.io/) 中找到。

## 引用

```
@article{ye2022syntaspeech,
  title={SyntaSpeech: Syntax-Aware Generative Adversarial Text-to-Speech},
  author={Ye, Zhenhui and Zhao, Zhou and Ren, Yi and Wu, Fei},
  journal={arXiv preprint arXiv:2204.11792},
  year={2022}
}
```

## 致谢Acknowledgement

**我们的代码基于以下仓库：**

* [NATSpeech](https://github.com/NATSpeech/NATSpeech)
* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
* [HifiGAN](https://github.com/jik876/hifi-gan)
* [espnet](https://github.com/espnet/espnet)
* [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
* [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger)
