# 字符模型训练工具

所有命令在仓库根目录执行，使用本工位 `.venv/Scripts/python.exe`。
不下载数据，不引入外部预训练权重。字体合成数据只用于验证工程流程，
其验证准确率不能代替 CCPD 或真实分割字符上的正式验收。

## 数据

`synth_chars --out algorithm/data/chars_synth --per-class 200 --seed 42`
按 65 类各生成 200 张，默认每类 train 160 / val 40。
`--font` 默认 `C:/Windows/Fonts/simhei.ttf`，`--val-fraction` 默认 0.2。
输出目录必须为空，避免重跑参数改变后残留旧样本；保留已有结果时请另选目录。

结构为 `{train,val}/<类下标>/*.png`，所有图都是已经过共享
`unify_polarity`、`normalize_char` 和合成增广的 32×32 uint8 灰度图。
训练器不再次归一化缓存图片；CNN 训练阶段再在线增广，验证阶段不在线增广。
共享读取器默认要求全部 65 类并检查图片规格；CNN 的混合来源模式允许单个来源缺类，
但合并训练集必须覆盖全部 65 类，合并验证集只要求非空。

`chars_index.csv` 为 UTF-8，字段：
`path,split,label,char,plate_color,source,font,seed`。
path 相对数据根目录；label 是冻结字符表下标；source 为 font_synthetic。
训练和验证分别使用独立随机流，但共享字体，因此不是独立实景数据分布。

## CNN

`train_cnn --data algorithm/data/chars_synth --epochs 2 --out algorithm/out/models/char_cnn_synth.pt --version char_cnn_synth`

默认 epochs=15、batch-size=256、lr=0.001、weight-decay=0.0001、seed=42、
num-workers=0（可选 0–2）。结构、Dropout 0.3、AdamW、余弦退火、
label smoothing 0.05 来自开发文档 8.1–8.2。
类别损失权重为 `N / (65 * class_count)`，按合并训练集原始计数计算。
每轮有放回抽样 N 次，来源采样比例由来源权重决定，来源内均匀。
CPU Torch 线程数为 2，OpenCV 为 1；random / NumPy / Torch 均固定种子，
在线增广由 seed、epoch、index 确定，多进程通过 main guard 支持 Windows。

默认日志为模型同名 `.train.csv`（可用 `--log` 指定），字段为
`epoch,loss,val_char_acc,lr,epoch_seconds,train_seconds`，每轮立即刷新。
最终模型只保存 CPU state_dict；manifest 的 val_char_acc 对应最终模型，
不把优化器或整模型 pickle 写入 CNN 文件。

增广初值：仿射范围遵循 8.2；模糊概率 0.3、σ 0.2–1.2；
噪声概率 0.5、标准差 1–8；对比度 0.85–1.15、亮度偏移 ±15；
半径 1 px 腐蚀/膨胀概率 0.2；2–5 px 遮挡概率 0.2。
概率和未由文档固定的数值是合成冒烟初值，后续须用真实训练/验证集调节。

## 基线

`build_templates --data algorithm/data/chars_synth --out algorithm/out/models/templates_synth.npz --version templates_synth`
按训练类别取 float32 均值图，用 OpenCV `TM_CCOEFF_NORMED` 匹配，
温度 softmax 默认 `--temperature 0.1`（待真实验证集校准）。

`train_svm --data algorithm/data/chars_synth --out algorithm/out/models/svm_synth.joblib --version svm_synth`
使用 `cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)` 的 324 维特征，
`SVC(kernel="rbf", C=10, probability=True, random_state=42)`，gamma 使用
scikit-learn 默认的 `scale`。默认每类最多 500 张，可用 `--max-per-class`
调低但不可超出 500；抽样由 `--seed` 控制，默认 42。
两种模型都在独立 val 目录计算概率 argmax 准确率并登记 manifest。
运行时每次调用独立创建 HOGDescriptor，模型只读，支持并发推理。

OpenCV 必须提供 `cv2.HOGDescriptor`。工位最初预装的 OpenCV 5.0.0
不提供此 API；代码明确报错，不以其他 HOG 实现替代规定特征。
已按用户明确授权，仅在工位 01 的 `.venv` 安装
`opencv-python==4.14.0.94`，NumPy 保持 2.5.4，全量测试通过。
不修改本分支 requirements；统一依赖约束由架构师维护。
当前 scikit-learn 1.9.1 会提示 `SVC(probability=True)` 将在 1.11 移除；
旧路径继续兼容此选项，第二批已增加 `--calibrated` 切换到文档允许的校准器。

## CCPD 字符样本生成

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.build_chars --index algorithm/data/index/ccpd_index.csv --raw algorithm/data/raw --out algorithm/data/chars --exclude-manifest algorithm/testset/manifest.csv
```

输入采用 `lpr_tools.data.ccpd.read_index` 的 CSV 契约。仅使用
`has_plate=1`、`skip_reason` 为空且 split 为 train/val 的记录，
原图 split 保持不变；GT quad 已为 LT、RT、RB、LB，不再交换点序。
按配置透视到 252×80 或 276×80，再按对应版式中心和槽宽切位，
共享 `unify_polarity` 与 `normalize_char`，输出灰度 32×32 PNG。
所有图片读写经 `lpr.io_utils`，支持中文路径。

`--max-per-class` 默认 8000，限制每类 train+val 合计数量；
按两个 split 的可用数量分配额度，预算至少 2 且两边均有样本时各保留至少 1。
cap=1 优先 train。此规则避免索引中 train 在前时耗尽 val 额度。
测试清单 path 支持绝对路径及清单相对路径，也兼容 raw 相对路径。
跨 split 的重复路径或相同文件内容会在写出字符前报错；不自动重分 split。
输出目录须为空，避免旧样本混入新索引；文件名包含原图名、路径短哈希和字符位置，
防止不同目录下同名图片覆盖。

输出 `chars_index.csv` 字段为
`path,source_path,split,label,char,plate_no,position`；字符 path 相对输出目录，
source_path 为原图绝对路径，position 从 0 开始。
`chars_stats.json` 保存每类数量、split 总数、排除图数和每类截断数，
同时向标准输出打印 65 类数量表。

## 三种字符分类器对比

```powershell
.venv\Scripts\python.exe -m lpr_tools.eval.ablation --chars algorithm/data/chars_synth/val --models cnn=algorithm/out/models/char_cnn_synth.pt template=algorithm/out/models/templates_synth.npz svm=algorithm/out/models/svm_synth.joblib --out algorithm/out/ablation
```

同一批字符在加载后分别交给三种分类器，支持只包含部分类别的目录。
输出 `ablation.md`：总体、省份字、字母数字准确率（同时给出样本数）、
单字符平均推理耗时、模型 MiB 和训练秒数。缺样本的分组显示 N/A。
每个模型先做 5 次单字符预热，再逐字符计时；不计图像读取、模型加载和指标计算，
Torch 固定为 2 线程。此处的单字符延迟和 8 字符批量延迟属于不同测量口径。

训练耗时依次读取 manifest 的 `train_seconds` / `training_seconds`、
模型同名 `.training.json` 的 `train_seconds`、同名 `.train.csv` 的
`epoch_seconds` 之和。旧模型无记录时显示 N/A，不根据文件时间推测。
整牌准确率保留独立章节；后续可调用
`fill_plate_accuracy(report_path, {"clear": 0.95, "night": 0.8})`
回填 `eval_accuracy` 的结果，数值必须有限且位于 [0,1]。
字符准确率不能代替整牌准确率；合成字符对比只作工程冒烟。

## 混合来源和断点续训

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.train_cnn --data algorithm/data/chars --data algorithm/data/provinces_synth --data-weight 3 --data-weight 1 --epochs 15 --out algorithm/out/models/char_cnn_mixed.pt --version char_cnn_mixed
```

重复 `--data` 添加来源；重复 `--data-weight` 时必须与目录数一一对应，
每个权重为正有限数，省略时全部为 1。上例来源总采样概率为 3:1，
不随两个目录的图片数量变化。类别损失加权与来源采样分别作用。
补充来源可只有 train；验证只汇总实际有 val 的来源，且不做在线增广。
正式训练时请让字体合成来源只提供 train，避免把字体合成准确率混作实景验证。

可重复传入 `--class-boost CHAR=FACTOR`，按真值字符提高其训练抽样概率；
加权在各来源内部归一化，因此不会改变 `--data-weight` 指定的来源总占比。
例如 `--class-boost 0=2 --class-boost 皖=2` 会让这些类在各来源内获得
两倍的单样本抽样权重。验证集不应用该权重；权重会写入训练元数据并纳入断点兼容校验。

推理模型仍只保存 state_dict；独立的训练断点默认为同名 `.checkpoint.pt`，
可用 `--checkpoint` 自定义。断点保存模型、优化器、余弦调度器、完成轮数、
Python/NumPy/Torch 随机状态及抽样生成器状态，读取采用 `weights_only=True`。

`--epochs` 是最初规划的总轮数；如需中途停止，可在上例加 `--stop-after 2`，
先运行本次最多 2 轮，再以相同来源、权重和超参数加
`--resume algorithm/out/models/char_cnn_mixed.checkpoint.pt` 续训到总计 15 轮。
断点按 epoch 边界保存，不支持恢复到 batch 中途。
续训会校验数据绝对路径、文件清单、大小和修改时间，以及原计划轮数和超参数；
数据改变或训练计划改变会明确拒绝。文件指纹不是内容哈希。
换用新的日志路径时会创建带表头的日志，只记录该路径实际执行的轮次。

更改来源或采样方案后需要微调时，可用 `--init-weights <本地推理模型.pt>`
初始化 CharCNN，重新创建优化器、调度器与随机状态，并以新输出路径记录训练。
该选项与 `--resume` 互斥；只接受 `weights_only=True` 加载的有限 Tensor
state_dict，严格检查模型键和形状，并在 checkpoint config 与训练 sidecar
记录初始化文件的绝对路径和 SHA256。初始化文件不得与任何训练产物同路径。
`train_seconds` 只计本次训练及逐轮验证，初始化模型既有的训练成本须单列。
重采样/难例目录只能取训练 split，开发集只供验证，不得把验证裁片加入微调。

短程微调可设 `--samples-per-epoch 131072`，每轮按既定来源/类别概率有放回抽样
指定次数；默认仍抽取完整训练集长度。该参数不裁减验证集。训练loss按实际
抽样数归一化，sidecar记录 `actual_draws_per_epoch`；参数进入断点兼容检查，
旧断点缺此字段按默认全长度处理。抽样轮次不等于对每张训练图遍历一次。

manifest 的 `train_data` 是包含所有来源 path/weight 的 JSON 字符串。
CNN 和 SVM 额外生成同名 `.training.json`，其中 `train_seconds` 为明确计时记录：
CNN 累计每轮训练及验证时间（跨断点累计），SVM 记录 fit/校准时间；
不包含数据加载和模型文件保存。旧模型不追补未测量的训练时间。

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.train_svm --data algorithm/data/chars --out algorithm/out/models/svm_calibrated.joblib --version svm_calibrated --calibrated
```

`--calibrated` 使用 `CalibratedClassifierCV(SVC(kernel="rbf", C=10), ensemble=False)`，
sigmoid 校准，折数为 `min(5, 每类最少训练样本数)`，每类不足 2 个样本时报错。
此路径不启用 SVC 的 `probability` 参数，消除 sklearn 1.9 的对应弃用提示；
运行时可加载旧概率 SVC 与新校准器。
