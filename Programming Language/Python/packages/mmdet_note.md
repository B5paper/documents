# MMDetection Note

high level apis:

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'xxx'
ckpt_file = 'xxx'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

img = 'test.jpg'
result = inference_detector(model, img)
model.show_result(img, result)
# model.show_result(img, result, out_file='result.jpg')
```

验证已有的模型和数据集：

```bash
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]
```

mmdet 还支持批量 test，但是得先把图片转换成无标注的 coco 格式，挺复杂的。官网里有教程。

训练：

```bash
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

可以用`--work-dir`指定保存训练过程的目录。

官方教程还讲了些多卡并行训练和集群训练，不过没细看。