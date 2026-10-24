# 字符模型训练工具

所有命令在仓库根目录执行，使用本工位 `.venv/Scripts/python.exe`。
不下载数据，不使用预训练权重。字体合成数据只用于验证工程流程，
其验证准确率不能代替 CCPD 或真实分割字符上的正式验收。

## 数据

`synth_chars --out algorithm/data/chars_synth --per-class 200 --seed 42`
按 65 类各生成 200 张，默认每类 train 160 / val 40。
`--font` 默认 `C:/Windows/Fonts/simhei.ttf`，`--val-fraction` 默认 0.2。
输出目录必须为空，避免重跑参数改变后残留旧样本；保留已有结果时请另选目录。

结构为 `{train,val}/<类下标>/*.png`，所有图都是已经过共享
`unify_polarity`、`normalize_char` 和合成增广的 32×32 uint8 灰度图。
训练器不再次归一化缓存图片；CNN 训练阶段再在线增广，验证阶段不在线增广。
共享读取器要求 train、val 均包含全部 65 类并检查图片规格。

`chars_index.csv` 为 UTF-8，字段：
`path,split,label,char,plate_color,source,font,seed`。
path 相对数据根目录；label 是冻结字符表下标；source 为 font_synthetic。
训练和验证分别使用独立随机流，但共享字体，因此不是独立实景数据分布。

## CNN

`train_cnn --data algorithm/data/chars_synth --epochs 2 --out algorithm/out/models/char_cnn_synth.pt --version char_cnn_synth`

默认 epochs=15、batch-size=256、lr=0.001、weight-decay=0.0001、seed=42、
num-workers=0（可选 0–2）。结构、Dropout 0.3、AdamW、余弦退火、
label smoothing 0.05 来自开发文档 8.1–8.2。
类别权重为 `N / (65 * class_count)`，每轮全部训练样本遍历一次。
CPU Torch 线程数为 2，OpenCV 为 1；random / NumPy / Torch 均固定种子，
在线增广由 seed、epoch、index 确定，多进程通过 main guard 支持 Windows。

默认日志为模型同名 `.train.csv`（可用 `--log` 指定），字段为
`epoch,loss,val_char_acc,lr`，每轮立即刷新。
最终模型只保存 CPU state_dict；manifest 的 val_char_acc 对应最终模型，
不把优化器或整模型 pickle 写入 CNN 文件。

增广初值：仿射范围遵循 8.2；模糊概率 0.3、σ 0.2–1.2；
噪声概率 0.5、标准差 1–8；对比度 0.85–1.15、亮度偏移 ±15；
半径 1 px 腐蚀/膨胀概率 0.2；2–5 px 遮挡概率 0.2。
概率和未由文档固定的数值是合成冒烟初值，后续须用真实训练/验证集调节。
