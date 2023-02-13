# MMDetection Note

官方文档：<https://mmdetection.readthedocs.io/en/v2.21.0/index.html>

一些介绍：

<https://www.zhihu.com/question/315115797/answer/2313523630>

<https://zhuanlan.zhihu.com/p/72719067>

## Installation

1. conda 创建个环境：`conda create -n mmdet python=3.8 -y`

1. 安装 cuda，这里是 1080Ti + 11.3 CUDA

    如果在 wsl 里使用 cuda，首先需要安装 wsl2，目前版本的 windows 不需要使用预览体验计划，直接在应用商店里安装`Ubuntu on Windows`就行了。
    
    或者直接参考微软文档：<https://docs.microsoft.com/zh-cn/windows/wsl/install>

    好像没有专门为 wsl 定制的驱动，只需要安装 windows 的显卡驱动就可以了。在 wsl 不需要安装驱动。

    在 wsl 下需要安装专门为 wsl 定制的 cuda 版本，在 cuda 下载界面选择`Linux -> x86_64 -> WSL-Ubuntu`。

1. 在 torch 官网安装 pytorch，torchvision

1. 安装`cudatoolkit`，这个是必须要装的，但不记得是用 apt 装还是用 pip 装了。

1. 安装`mim`：`pip install openmim`

    然后挂代理，`export http_proxy=<porxy_ip>:<port>`，`export https_proxy=<proxy_ip>:<port>`

    再使用`mim install mmdet`安装 mmdetection。这个时间可能长点。

1. 不知道`mim`把 mmdetection 装到哪里了，所以还需要把官方的 github 再下载一遍，然后在这个项目里魔改：`git clone https://github.com/open-mmlab/mmdetection.git`

1. 验证安装成功

    进入 mmdetection 目录，下载好模型参数，运行下面代码：

    ```python
    from mmdet.apis import init_detector, inference_detector

    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    # url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    inference_detector(model, 'demo/demo.jpg')
    ```

## high level apis

inference example:

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

异步推理：

```python
import asyncio
import torch
from mmdet.apis import init_detector, async_inference_detector
from mmdet.utils.contextmanagers import concurrent

async def main():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 3

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # test a single image and show the results
    img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once

    async with concurrent(streamqueue):
        result = await async_inference_detector(model, img)

    # visualize the results in a new window
    model.show_result(img, result)
    # or save the visualization results to image files
    model.show_result(img, result, out_file='result.jpg')


asyncio.run(main())
```

在 demo 目录里可以直接对单张图片和视频流进行目标检测：<https://github.com/open-mmlab/mmdetection/tree/master/demo>

文档里给出了例子：<https://mmdetection.readthedocs.io/en/v2.21.0/1_exist_data_model.html>

## API

可视化：

1. 使用高层接口`show_result_pyplot`

    ```python
    from mmdet.apis import show_result_pyplot

    show_result_pyplot(
        model,
        args.img,
        result,
        palette=args.palette,
        score_thr=args.score_thr)
    ```

1. 使用模型自带的绘图基类方法

    ```python
    model.show_result(args.img, result, out_file='det.png', score_thr=args.score_thr)
    ```
