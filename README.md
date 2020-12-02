# CANet-tf2
this project implements algorithm proposed in paper "CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning".

## dataset preparation

download COCO2017 dataset from [here](https://cocodataset.org/). unzip directory train2017, val2017 and annotations.

## how to train

train with command

```python
python3 train.py </path/to/train2017> </path/to/val2017> </path/to/annotations>
```

## save model

save model with command

```python
python3 save_model.py
```

## experimental results

## how to predict with pretrained model
