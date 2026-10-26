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
重采样/难例目录只能取训练 split，不得把开发集裁片加入特征微调。

短程微调可设 `--samples-per-epoch 131072`，每轮按既定来源/类别概率有放回抽样
指定次数；默认仍抽取完整训练集长度。该参数不裁减验证集。训练loss按实际
抽样数归一化，sidecar记录 `actual_draws_per_epoch`；参数进入断点兼容检查，
旧断点缺此字段按默认全长度处理。抽样轮次不等于对每张训练图遍历一次。

`calibrate.fit_temperature` 可在明确记录的开发集上拟合单个全局温度，以稳定
NLL为目标；它不按类别或单张图片修正预测。`apply_temperature` 只把最终线性层
的权重和偏置除以温度，保留纯Tensor state_dict，推理仍直接softmax。
应记录原始权重SHA、开发清单、温度、校准前后NLL以及自动通过精度，核验实际
模型argmax及完整流水线输出；不能将同一开发集上的校准结果宣称为独立测试。
冻结测试清单与金样例始终不得用于拟合温度；默认置信度门槛不随校准修改。

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

## CRNN 整牌兜底（第七批）

`lpr.recognize.crnn.CrnnReader` 将 252×80/276×80 BGR uint8 校正图转为
160×40 灰度，CNN 提取 40 个时间步，单层双向 GRU 输出 65 字符与 blank 的
CTC logits。greedy 解码合并连续重复、删除 blank，保留逐位概率；格式、最低
置信度与状态仍由既有流水线处理。权重从随机初始化训练，不加载预训练模型。

下面第一、二阶段的命令保留为历史运行记录：128×40 第一阶段使用
`b5218c6` / `fcead48` 的实现，第二阶段使用 `bc8bf84`；当前代码已是160×40。
复现旧阶段必须固定对应源码版本，不能把同一条命令在160×40代码下运行后
当作旧实验复现。manifest的`input_size`必须与代码的`INPUT_SIZE`一致，
`CrnnReader.from_file`会拒绝不一致的已登记权重。加载器保留调用者设置的Torch线程数；
完整流水线与裁片评测明确使用2线程。


合成工具只写 train，使用 Windows 自带 TrueType 字体；按真实训练 GT 图计数，
把少于 1000 个独立 GT 样本的省份补足到 1500。这个口径避免把同源候选图
算成独立样本。蓝牌 7 位，绿牌按 L8S/L8L 规则生成。字体合成只用于训练，
其清单、字体哈希、种子及数量记录于 `synthetic_manifest.json`。

```powershell
$env:PYTHONPATH = 'algorithm'
$env:PYTHONUTF8 = '1'
.venv\Scripts\python.exe -m lpr_tools.train.synth_plates --help
.venv\Scripts\python.exe -m lpr_tools.train.train_crnn --data D:\keshe2\algorithm\data\plates\v1 --data D:\keshe2\algorithm\data\plates_synth\v2_seed42 --exclude-origins algorithm/out/receipts/batch7/excluded-train-origins.txt --epochs 30 --samples-per-epoch 32768 --batch-size 128 --num-workers 0 --out algorithm/out/models/batch7/crnn_run1/crnn_v1.pt
.venv\Scripts\python.exe -m lpr_tools.train.eval_crnn --data D:\keshe2\algorithm\data\plates\v1 --model algorithm/models/crnn_v1.pt --out algorithm/out/eval/batch7/crnn-crops
```

训练前必须根据测试清单、金样例、`devhard_v1.csv`、`val1000_v1.csv` 做原图
路径和字节哈希隔离；`--exclude-origins` 接受每行一个规范化原图绝对路径，
只排除对应 train 的所有视图，不删改提供的数据。本批预检发现 27 个同源原图、
73 个训练裁片需排除；原始 train 294783，实际保留 294710，真实 val 10000 不变。
排除清单是本批审计产物，复跑时须保留或根据上述清单重新生成，不能省略。

训练采用省份逆频率平方根、上限 8 倍的有放回采样，每轮 32768 次抽样，
不表示遍历全部训练图片。模糊、噪声、亮度、轻透视和遮挡仅用于 train；
源/省份/位数的预期抽样概率记录在 checkpoint metadata。CPU 训练线程为 4，
默认不创建额外数据加载进程。`--stop-after` 在轮次边界停止，用原参数加
`--resume <model.checkpoint.pt>` 续训，校验来源清单、排除文件、源码哈希与配置；
逐轮日志追加，随机抽样和增强按 seed/epoch 确定。新训练拒绝覆盖已有输出。

每轮在完整真实 val 上报告 GT、pipeline、总体、省份及逐位字符准确率，
按总体整牌准确率选择最佳权重。`eval_crnn` 另走真实 `read()` 接口逐图核验，
输出逐图预测、分来源/省份/位数指标和 CPU 2 线程耗时；字符准确率使用
逐位匹配数除以 `sum(max(真值长度, 预测长度))`。其自动通过率只是裁片级指标，
不能代替带四角点网络的完整流水线配对评测。最终模型是否达到 GT 95%、
pipeline 90%、自动通过精度 99% 和 G1 P95 200 ms，见第七批实测回执。

CPU 数据预取可用 `--num-workers 1`（最多 4）；主训练 Torch 4 线程，每个 worker
Torch/OpenCV 各设 1 线程。普通 resume 要求原参数完全匹配。若仅迁移数据加载
worker 数量/对应训练器实现，显式加 `--resume-execution-change` 并使用全新输出
路径；数据、排除清单、模型源码及其他训练配置仍须完全相同。迁移保留优化器与
模型状态、最佳权重和已有历史，并记录源 checkpoint/最佳模型/日志 SHA、旧新
worker 数及训练器 SHA。它不支持模型、训练数据、采样或学习率变更。源断点
必须沿用 `<name>.checkpoint.pt`，同目录有对应 `<name>.pt` 最佳权重；原始目录
保持不动。第七批首轮从 0 worker 迁移到 1 worker 的证据见回执。

第七批第二阶段针对真实绿牌不足增加 `--real-green-boost 8`：只乘真实
`ground_truth` / `pipeline_candidate` 的 8 位车牌采样权重，字体合成数据仍按原
省份权重采样。权重统计用于实际 sampler；每轮日志另外记录实际
`source/layout` 抽样数量，不能把预期概率当作已执行抽样次数。

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.train_crnn --data D:\keshe2\algorithm\data\plates\v1 --data D:\keshe2\algorithm\data\plates_synth\v2_seed42 --exclude-origins algorithm/out/receipts/batch7/excluded-train-origins.txt --init-weights algorithm/out/receipts/batch7/crnn-run1-workers1-epoch7/crnn_v1.pt --init-training-run algorithm/out/models/batch7/crnn_run1_workers1 --version crnn_v1_realgreen --real-green-boost 8 --lr 0.0003 --epochs 12 --samples-per-epoch 32768 --batch-size 128 --num-workers 1 --out algorithm/out/models/batch7/crnn_run2_realgreen_counted/crnn_v1.pt
```

这里的初始化来源是本批从随机权重训练出的第一阶段最佳第 5 轮模型；
快照目录记录第 7 轮停止时的 last checkpoint 和此前保留的 best 权重，两者
不是同一个模型。禁止使用外部预训练权重。`--init-weights` 与
`--init-training-run` 必须同时提供，前者严格加载有限 Tensor state_dict并记录
SHA，后者记录本地来源运行。新阶段使用新的优化器和输出路径；它与普通
`--resume` 互斥。恢复第二阶段时用第二阶段 checkpoint 和原参数，不再次传
初始化参数；初始化来源由 checkpoint 继承。两阶段耗时和完成轮次分别报告。

第二阶段的最佳 checkpoint 为第 11 轮，但 pipeline 候选准确率仍未达到 90%。
第三阶段沿用本地从零训练的第二阶段最佳纯权重，保持网络层和 state_dict 键不变，
只把共享推理/训练输入宽度提高到 160；输入预处理尺寸写入模型 manifest 与
训练 metadata。训练 4 轮、学习率 0.0001、绿牌真实样本权重 8，使用新的输出目录：

```powershell
.venv\Scripts\python.exe -m lpr_tools.train.train_crnn --data D:\keshe2\algorithm\data\plates\v1 --data D:\keshe2\algorithm\data\plates_synth\v2_seed42 --exclude-origins algorithm/out/receipts/batch7/excluded-train-origins.txt --init-weights algorithm/out/models/batch7/crnn_run2_realgreen_counted/crnn_v1.pt --init-training-run algorithm/out/models/batch7/crnn_run2_realgreen_counted --version crnn_v1_realgreen_width160 --real-green-boost 8 --lr 0.0001 --epochs 4 --samples-per-epoch 32768 --batch-size 128 --num-workers 1 --out algorithm/out/models/batch7/crnn_run3_width160/crnn_v1.pt
```

评估时必须以该模型自身的 160×40 manifest 输入尺寸加载；若完整 `read()` 或流水线验收
不优于 128×40 的第二阶段最佳模型，发布候选应回选第二阶段权重及其 128×40 预处理。


### 第七批最终实测与未达标项

发布权重 `models/crnn_v1.pt` 来自第三阶段第4轮，886,653 bytes、218,474参数，
SHA `d50c656f671a882244b2688596e85e97d7eb37a5cadaf947802e96dbfcccca48`。
第三阶段训练源码固定于`ae6c6d9`；后续加载器仅追加manifest尺寸校验，网络与read数值未变。
恢复历史checkpoint须匹配其记录的训练和模型源码SHA。

| 验证 | 结果 | 判定 |
|---|---:|---|
| 真实GT裁片5000张 | 4910/5000，98.20% | ≥95%，通过 |
| 真实流水线裁片5000张 | 4225/5000，84.50% | 未达90% |
| 省份字（完整10000裁片） | 9901/10000，99.01% | 独立列报 |
| val整帧1000张 | 85.30% → 95.70% | 提升10.40个百分点 |
| val自动通过精度 | 99.7611% → 98.6272% | 未达99%，新增11个错误自动通过 |
| dev-hard整帧2100张 | 43.8095% → 53.5714% | 提升9.7619个百分点 |
| dev-hard自动通过精度 | 97.4760% → 94.5437% | 未达99%，新增34个错误自动通过 |
| CPU2 read()，10000次 | 均值3.8133ms / P95 4.5643ms / 最大8.5907ms | ≤10ms，通过 |
| G1，1280×720，100次 | P95 60.7307ms | ≤200ms，通过 |

**本批性能验收未全部通过，权重保留为开发候选，不应据此认定自动通行可上线。**
默认配置未开启CRNN。完整流水线基线已有的错误自动通过全部保留，CRNN另有新增；
继续处理需评估候选缺字问题、CRNN置信度，以及主路线先行OK的复核策略。
本批没有越界修改pipeline或默认配置，未声称已解决这些问题。
原始评测、训练与隔离证据见工位01的`algorithm/out/receipts/第七批.md`及其batch7目录。
