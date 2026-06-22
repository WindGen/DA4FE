# EEG-ImageNet PTH 文件参数总结

数据来源：

- `E:\Data\EEG-ImageNet\block_splits_by_image_single.pth`
- `E:\Data\EEG-ImageNet\eeg_55_95_std.pth`

说明：

- `block_splits_by_image_single.pth` 不是 EEG 数据本体，而是数据划分索引文件。
- `eeg_55_95_std.pth` 是实际保存 EEG 样本的数据文件。

## 1. `block_splits_by_image_single.pth`

文件路径：

- `E:\Data\EEG-ImageNet\block_splits_by_image_single.pth`

### 参数表

| 项目 | 数值 / 说明 |
|---|---|
| 文件大小 | 2964 Bytes |
| 顶层类型 | `dict` |
| 顶层键 | `splits` |
| `splits` 类型 | `list` |
| `splits` 长度 | 1 |
| 单个 split 的键 | `train`, `val`, `test` |
| train 数量 | 669 |
| val 数量 | 167 |
| test 数量 | 164 |
| 总索引数 | 1000 |
| 索引范围 | `0 ~ 999` |
| 是否包含 EEG 张量 | 否 |
| 是否包含长度/通道数 | 否 |

### 保存结构

顶层结构如下：

```python
{
    "splits": [
        {
            "train": [...],
            "val": [...],
            "test": [...],
        }
    ]
}
```

含义：

- `train` / `val` / `test` 中保存的是样本索引
- 这些索引完整覆盖 `0 ~ 999`
- 该文件仅用于划分数据集，不包含真实 EEG 信号

## 2. `eeg_55_95_std.pth`

文件路径：

- `E:\Data\EEG-ImageNet\eeg_55_95_std.pth`

### 参数表

| 项目 | 数值 / 说明 |
|---|---|
| 文件大小 | 3135625420 Bytes |
| 顶层类型 | `dict` |
| 顶层键 | `dataset`, `labels`, `images` |
| `dataset` 类型 | `list` |
| 样本总数 | 11965 |
| 类别总数 | 40 |
| 图片索引总数 | 1996 |
| 被试数 | 6 |
| EEG 通道数 | 固定 128 |
| EEG 长度范围 | `491 ~ 1527` |
| EEG 平均长度 | `511.55` |
| EEG 中位长度 | `511` |
| 不同 shape 数量 | 120 |
| 最常见 shape | `[128, 499]`，共 1092 条 |
| 其他高频 shape | `[128, 498]` 936 条；`[128, 500]` 885 条 |

### 顶层结构

```python
{
    "dataset": [...],
    "labels": [...],
    "images": [...]
}
```

### 顶层字段说明

#### `labels`

- 长度：40
- 作用：保存类别索引到真实类别名的映射
- 示例：

```python
[
    "n02389026",
    "n03888257",
    "n03584829",
    "n02607072",
    "n03297495",
    ...
]
```

对应关系：

- 样本中的 `label = 0` 对应 `labels[0]`
- 样本中的 `label = 1` 对应 `labels[1]`

#### `images`

- 长度：1996
- 作用：保存图片索引到真实图片名的映射
- 示例：

```python
[
    "n02951358_31190",
    "n03452741_16744",
    "n04069434_10318",
    "n02951358_34807",
    "n03452741_5499",
    ...
]
```

对应关系：

- 样本中的 `image = 0` 对应 `images[0]`
- 样本中的 `image = 1` 对应 `images[1]`

#### `dataset`

- 长度：11965
- 作用：保存每一个 EEG 样本

单个样本结构如下：

```python
{
    "eeg": Tensor,
    "image": int,
    "label": int,
    "subject": int
}
```

### 单个样本字段说明

#### `eeg`

- 类型：`Tensor`
- shape：`[128, T]`
- 说明：
  - `128` 表示 EEG 通道数
  - `T` 表示时间长度
  - `T` 不是固定值，不同样本长度不同

#### `image`

- 类型：`int`
- 范围：`0 ~ 1995`
- 含义：图片索引，对应顶层 `images[image]`

#### `label`

- 类型：`int`
- 范围：`0 ~ 39`
- 含义：类别索引，对应顶层 `labels[label]`

#### `subject`

- 类型：`int`
- 范围：`1 ~ 6`
- 含义：被试编号

## 3. EEG 长度分布补充

关于时间长度 `T` 的补充统计如下：

| 项目 | 数值 |
|---|---|
| 长度为 1527 的样本数 | 1 |
| 长度 >= 1000 的样本数 | 2 |
| 长度 >= 700 的样本数 | 10 |
| 长度 <= 520 的样本数 | 9597 |

最长的一批长度包括：

```python
[1527, 1044, 912, 827, 765, 751, 734, 727, 719, 710, ...]
```

说明：

- 大多数样本长度在 500 左右
- `1527` 属于极少数长序列样本

## 4. 对当前 TeCh 项目的对应关系

对于当前 `TeCh-improve` 项目，`eeg_55_95_std.pth` 中的 `eeg` 是：

```python
[128, T]
```

而项目中的 loader 通常会整理成：

```python
[T, 128]
```

因此：

- `enc_in = 128`
- `seq_len = T`
- 如果启用自适应长度，则会进一步把变长样本重采样到目标长度

## 5. 总结

- `block_splits_by_image_single.pth`：保存的是 1000 个样本索引的 train/val/test 划分
- `eeg_55_95_std.pth`：保存的是 11965 条真实 EEG 样本
- 每条 EEG 样本包含：
  - `eeg`: EEG 信号，shape 为 `[128, T]`
  - `image`: 图片索引
  - `label`: 类别索引
  - `subject`: 被试编号

