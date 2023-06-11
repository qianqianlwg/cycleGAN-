<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN和pix2pix的PyTorch实现

**新消息**：请查看我们的新的无配对图像到图像翻译模型 [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation)，它可以实现快速和内存高效的训练。

我们提供了针对非配对和配对图像到图像翻译的PyTorch实现。

代码由[Jun-Yan Zhu](https://github.com/junyanz)和 [Taesung Park](https://github.com/taesungp)编写，并由[Tongzhou Wang](https://github.com/SsnL)提供支持。

这个PyTorch实现产生的结果与我们原来的Torch软件相当或更好。如果您想重现论文中的结果，请查看Lua/Torch中的原始[CycleGAN Torch](https://github.com/junyanz/CycleGAN)和[pix2pix Torch](https://github.com/phillipi/pix2pix) 代码。

**注意**: 当前软件与PyTorch 1.4兼容。查看支持PyTorch 0.1-0.3的旧版本[分支](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1)。

您可以在[training/test tips](docs/tips.md)和[frequently asked questions](docs/qa.md)中找到有用的信息。要实现自定义模型和数据集，请查看我们的[templates](#custom-model-and-dataset)。为了帮助用户更好地理解和适应我们的代码库，我们提供了这个仓库代码结构的[overview](docs/overview.md)。

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)**

<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>

**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004.pdf) |  [Torch](https://github.com/phillipi/pix2pix) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>


**[EdgesCats Demo](https://affinelayer.com/pixsrv/) | [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) | 由[Christopher Hesse](https://twitter.com/christophrhesse)提供支持**

<img src='imgs/edges2cats.jpg' width="400px"/>

如果您将此代码用于您的研究，请引用：

未配对图像到图像翻译使用循环一致的对抗网络。<br>
[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)\*，[Taesung Park](https://taesung.me/)\*，[Phillip Isola](https://people.eecs.berkeley.edu/~isola/)，[Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)在ICCV 2017上发表。(* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


具有条件对抗网络的图像到图像转换。<br>
[Phillip Isola](https://people.eecs.berkeley.edu/~isola)，[Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)，[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz)，[Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)在CVPR 2017上发表。[[Bibtex]](https://www.cs.cmu.edu/~junyanz/projects/pix2pix/pix2pix.bib)

## Talks and Course
pix2pix幻灯片: [keynote](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.key) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/pix2pix.pdf),
CycleGAN幻灯片: [pptx](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pptx) | [pdf](http://efrosgans.eecs.berkeley.edu/CVPR18_slides/CycleGAN.pdf)

由多伦多大学[Roger Grosse](http://www.cs.toronto.edu/~rgrosse/)教授设计的[CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/)“神经网络和机器学习介绍”课程分配[code](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip)和资料手册(handout)，请与教师联系以了解是否可以在您的课程中采用该课程。

## Colab笔记本
TensorFlow Core CycleGAN 教程: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb) | [代码](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/cyclegan.ipynb)

TensorFlow Core pix2pix 教程: [Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb) | [代码](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)

PyTorch Colab笔记本: [CycleGAN](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb) 和 [pix2pix](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/pix2pix.ipynb)

ZeroCostDL4Mic Colab笔记本: [CycleGAN](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/CycleGAN_ZeroCostDL4Mic.ipynb) 和 [pix2pix](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks_Beta/pix2pix_ZeroCostDL4Mic.ipynb)

## 其他实现
### CycleGAN
<p><a href="https://github.com/leehomyc/cyclegan-1"> [Tensorflow]</a> (by Harry Yang),
<a href="https://github.com/architrathore/CycleGAN/">[Tensorflow]</a> (by Archit Rathore),
<a href="https://github.com/vanhuyz/CycleGAN-TensorFlow">[Tensorflow]</a> (by Van Huy),
<a href="https://github.com/XHUJOY/CycleGAN-tensorflow">[Tensorflow]</a> (by Xiaowei Hu),
<a href="https://github.com/LynnHo/CycleGAN-Tensorflow-2"> [Tensorflow2]</a> (by Zhenliang He),
<a href="https://github.com/luoxier/CycleGAN_Tensorlayer"> [TensorLayer1.0]</a> (by luoxier),
<a href="https://github.com/tensorlayer/cyclegan"> [TensorLayer2.0]</a> (by zsdonghao),
<a href="https://github.com/Aixile/chainer-cyclegan">[Chainer]</a> (by Yanghua Jin),
<a href="https://github.com/yunjey/mnist-svhn-transfer">[Minimal PyTorch]</a> (by yunjey),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/CycleGAN">[Mxnet]</a> (by Ldpe2G),
<a href="https://github.com/tjwei/GANotebooks">[lasagne/Keras]</a> (by tjwei),
<a href="https://github.com/simontomaskarlsson/CycleGAN-Keras">[Keras]</a> (by Simon Karlsson),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Oneflow-Python/CycleGAN">[OneFlow]</a> (by Ldpe2G)
</p>


### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>


## 先决条件
- Linux 或 macOS
- Python 3
- CPU 或 NVIDIA GPU + CUDA CuDNN

## 入门
### 安装

- 克隆这个repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix

```

安装 [PyTorch](http://pytorch.org) 和 0.4+及其它依赖 (如 torchvision, [visdom](https://github.com/facebookresearch/visdom) 和 [dominate](https://github.com/Knio/dominate))
- 对于pip用户，请在终端输入命令 `pip install -r requirements.txt`。
- 对于Conda用户，您可以使用命令 `conda env create -f environment.yml` 创建新的Conda环境。
- 对于Docker用户，我们提供了预构建的Docker映像和Dockerfile。请参阅我们的[Docker](docs/docker.md)页面。
- 对于Repl用户，请点击[![Run on Repl.it](https://repl.it/badge/github/junyanz/pytorch-CycleGAN-and-pix2pix)](https://repl.it/github/junyanz/pytorch-CycleGAN-and-pix2pix)。

### CycleGAN 训练/测试
- 下载CycleGAN数据集（例如maps）：
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- 要查看训练结果和损失图，请运行 `python -m visdom.server` 并单击 URL http://localhost:8097 。
- 要将训练进度和测试图像记录到W&B仪表板，请在train和test脚本中设置 `--use_wandb` 标志。
- 训练模型：
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
要查看更多中间结果，请查看 `./checkpoints/maps_cyclegan/web/index.html`。
- 测试模型：
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- 测试结果将保存在此处的html文件中：`./results/maps_cyclegan/latest_test/index.html`。

### pix2pix 训练/测试
- 下载pix2pix数据集（例如[facades](http://cmp.felk.cvut.cz/~tylecr1/facade/)）：
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- 要查看训练结果和损失图，请运行 `python -m visdom.server` 并单击 URL http://localhost:8097 。
- 要将训练进度和测试图像记录到W&B仪表板，请在train和test脚本中设置 `--use_wandb` 标志。
- 训练模型：
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
要查看更多中间结果，请查看 `./checkpoints/facades_pix2pix/web/index.html`。

- 测试模型 ( `bash ./scripts/test_pix2pix.sh` )：
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
```
- 测试结果将保存在此处的html文件中：`./results/facades_pix2pix/test_latest/index.html`。您可以在 `scripts` 目录中找到更多脚本。
- 要训练和测试基于pix2pix的彩色模型，请添加 `--model colorization` 和 `--dataset_mode colorization`。有关更多详情，请参见我们的[培训提示](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#notes-on-colorization)。

### 应用预训练模型（CycleGAN）
- 您可以使用以下脚本下载预训练模型（例如horse2zebra）：
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
- 预先训练的模型保存在 `./checkpoints/{name}_pretrained/latest_net_G.pth`。请查看[这里](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_cyclegan_model.sh#L3) 所有可用的CycleGAN模型。
- 要测试模型，您还需要下载horse2zebra数据集：
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```
```

python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```
选项`--model test`用于仅为CycleGAN生成一侧的结果。此选项将自动设置`--dataset_mode single`，它仅从一个集合中加载图像。相反，使用`--model cycle_gan`需要在两个方向上加载和生成结果，有时是不必要的。结果将保存在`./results/`中。使用`--results_dir {directory_path_to_save_result}`指定结果目录。

对于pix2pix和您自己的模型，您需要明确指定`--netG`，`--norm`，`--no_dropout`以匹配训练模型的生成器体系结构。有关更多详细信息，请参见此[FAQ](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#runtimeerror-errors-in-loading-state_dict-812-671461-296)。

### 应用预训练模型（pix2pix）
使用`./scripts/download_pix2pix_model.sh`下载预训练模型。

- 查看[这里](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/scripts/download_pix2pix_model.sh#L3)以获取所有可用的pix2pix模型。例如，如果您想在Facade数据集上下载label2photo模型，
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```
- 下载pix2pix facades数据集：
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- 然后使用以下命令生成结果：
```bash
python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
- 请注意，我们指定了`--direction BtoA`，因为Facades数据集的A到B方向是照片到标签。

- 如果您想将预训练模型应用于一组输入图像（而不是图像对），请使用`--model test`选项。有关如何将模型应用于Facade标签映射（存储在目录`facades/testB`中）的详细信息，请参见`./scripts/test_single.sh`。

- 在`./scripts/download_pix2pix_model.sh`中查看当前可用模型列表

## [Docker](docs/docker.md)
我们提供预构建的Docker镜像和Dockerfile，可以运行此代码存储库。请参见[docker](docs/docker.md)。

## [数据集](docs/datasets.md)
下载pix2pix/CycleGAN数据集并创建自己的数据集。

## [培训/测试技巧](docs/tips.md)
为培训和测试您的模型的最佳实践。

## [常见问题解答](docs/qa.md)
在发布新问题之前，请首先查看上面的Q&A和现有的GitHub问题。

## 自定义模型和数据集
如果您计划为新的应用程序实现自定义模型和数据集，请使用数据集[模板](data/template_dataset.py)和模型[模板](models/template_model.py)作为起点。

## [代码结构](docs/overview.md)
为了帮助用户更好地理解和使用我们的代码，我们简要概述了每个包和每个模块的功能和实现。

## 拉取请求
您随时可以通过发送[拉取请求](https://help.github.com/articles/about-pull-requests/)为此存储库做出贡献。在提交代码之前，请运行`flake8 --ignore E501 .`和`python ./scripts/test_before_push.py`。如果您添加或删除文件，请同时更新代码结构[概述](docs/overview.md)。

## 引文
如果您将此代码用于您的研究，请引用我们的论文。
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

## 其他语言
[西班牙语](docs/README_es.md)

## 相关项目
**[contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) (CUT)**<br>
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)|
[BicycleGAN](https://github.com/junyanz/BicycleGAN) | [vid2vid](https://tcwang0509.github.io/vid2vid/) | [SPADE/GauGAN](https://github.com/NVlabs/SPADE)**<br>
**[iGAN](https://github.com/junyanz/iGAN) | [GAN Dissection](https://github.com/CSAILVision/GANDissect) | [GAN Paint](http://ganpaint.io/)**

## 猫论文集合
如果您热爱猫，并且喜欢阅读酷炫的图形、视觉和学习论文，请查看猫论文[集合](https://github.com/junyanz/CatPapers)。

## 致谢
我们的代码灵感来自[pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan)。