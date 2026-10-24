# 派单 01 · 员工 A（WP-A 数据与识别模型）· 首批

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 A / WP-A，工位 01】。
工作目录：D:\keshe2\workers\01    分支：feature/m2-wpa-model（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe，例如 .venv\Scripts\python.exe -m pytest algorithm/tests -q
必读：docs/M2-车牌识别-开发文档.md 第 2、5 章（含 5.9），6.5 节，7.2—7.3 节，第 8 章，11.2 节，第 15 章；
      algorithm/lpr/types.py、charset.py、charnorm.py、recognize/base.py、recognize/__init__.py；
      algorithm/config/default.toml
背景：CCPD 数据集还没到位。本批任务全部用合成数据完成；不要等数据，也不要自己下载任何数据集。
      合成数据上的准确率只用来证明流程跑通，不算正式验收（正式验收等 CCPD）。

本次任务（按顺序做，每做完一项提交一次）：
  A-1 CharCNN 与推理封装（任务 A2 的代码部分）
      - recognize/cnn_model.py：CharCNN，结构严格按 8.1；输入 N×1×32×32（像素/255），输出 N×65 logits。
      - recognize/cnn.py：CnnClassifier.from_file(path, *, version) 用 torch.load(..., weights_only=True)
        加载 state_dict 并置 eval；predict_proba 输入 N×32×32 uint8，输出 N×65 float32（softmax，每行和为 1），
        推理包在 torch.inference_mode() 中；输入 0×32×32 时返回 0×65。
      - tests/test_recognize_cnn.py：参数量在 60 万—76 万之间；输出形状与行和；空输入；
        tmp_path 里保存随机初始化权重 → lpr.recognize.write_manifest_entry 登记 → load_classifier("cnn", path) 能加载；
        8 个字符批量推理（torch 2 线程）中位数 ≤10 ms。
  A-2 增广与字体合成
      - lpr_tools/train/augment.py：只用 OpenCV/NumPy 实现 8.2 表里的增广（旋转 ±8°、缩放 0.9—1.1、平移 ±2 px、
        错切 ±0.1、高斯模糊 σ≤1.2、噪声、亮度对比度抖动、1 px 腐蚀/膨胀、小块遮挡）；
        输入输出都是 32×32 uint8；随机性由传入的 numpy Generator 控制，可复现。
      - lpr_tools/train/synth_chars.py：用 Pillow 和 C:\Windows\Fonts\simhei.ttf（字体路径可传参）渲染全部 65 类字符，
        模拟车牌底色与字色 → charnorm.unify_polarity → charnorm.normalize_char → 增广；
        输出 algorithm/data/chars_synth/{train,val}/<类下标>/*.png、chars_index.csv，并打印每类数量统计表。
        命令：python -m lpr_tools.train.synth_chars --out algorithm/data/chars_synth --per-class 200 --seed 42
      - tests/test_augment.py、tests/test_synth_chars.py（往 tmp_path 生成少量样本即可）。
  A-3 CNN 训练脚本（冒烟）
      - lpr_tools/train/train_cnn.py：按 8.2（AdamW、余弦退火、label smoothing 0.05、类别加权或 WeightedRandomSampler、
        在线增广、种子 42、num_workers 0—2、必须有 if __name__ == "__main__":）；参数至少有
        --data --epochs --batch-size --lr --out --version --seed；每轮把 loss 和验证集字符准确率写日志文件；
        只保存 state_dict；保存后调用 write_manifest_entry 登记（val_char_acc 填验证集字符准确率）。
      - 冒烟：在合成字符上训练 2 轮，模型存到 algorithm/out/models/（不入库）。
  A-4 两种基线（任务 A3 的代码部分）
      - recognize/template.py + lpr_tools/train/build_templates.py：每类均值模板（或 K=5 个代表样本），
        cv2.matchTemplate(TM_CCOEFF_NORMED)，得分经温度 softmax 转概率。
      - recognize/svm.py + lpr_tools/train/train_svm.py：cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)
        得 324 维特征，sklearn SVC(kernel="rbf", C=10)，每类最多 500 个样本；概率用 probability=True 或
        CalibratedClassifierCV。模型用 joblib 保存。
      - 两者都要提供 from_file(path, *, version)，保存后都调用 write_manifest_entry 登记。
      - tests/test_recognize_baselines.py：在 tmp_path 里用少量合成样本训练 → load_classifier 加载 → 输出形状与行和。

只允许新建或修改：
  algorithm/lpr/recognize/cnn_model.py、cnn.py、template.py、svm.py
  algorithm/lpr_tools/train/ 下的文件
  algorithm/tests/test_recognize_cnn.py、test_recognize_baselines.py、test_augment.py、test_synth_chars.py
  algorithm/notebooks/ 下的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、
    recognize/base.py、recognize/__init__.py、tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py
  - 删除或重命名配置键（需要新键时在汇报里申请）
  - 引入白名单外的依赖（见 algorithm/requirements.txt），不得 pip install 任何东西；
    调用任何车牌识别 API 或使用预训练车牌模型（HyperLPR、LPRNet 公开权重、EasyOCR、PaddleOCR、YOLO/ultralytics 等）
  - 在运行时包 algorithm/lpr/ 里 import Pillow（Pillow 只能出现在 lpr_tools/train/synth_chars.py）
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件；本批不往 algorithm/models/ 放模型
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，并贴出完整原始输出）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m lpr_tools.train.synth_chars --out algorithm/data/chars_synth --per-class 200 --seed 42
  .venv\Scripts\python.exe -m lpr_tools.train.train_cnn --data algorithm/data/chars_synth --epochs 2 --out algorithm/out/models/char_cnn_synth.pt --version char_cnn_synth
  .venv\Scripts\python.exe -m lpr_tools.train.build_templates --data algorithm/data/chars_synth --out algorithm/out/models/templates_synth.npz --version templates_synth
  .venv\Scripts\python.exe -m lpr_tools.train.train_svm --data algorithm/data/chars_synth --out algorithm/out/models/svm_synth.joblib --version svm_synth
  .venv\Scripts\python.exe -c "from pathlib import Path; import numpy as np; from lpr.recognize import load_classifier; [print(k, load_classifier(k, Path(p)).predict_proba(np.zeros((8,32,32),np.uint8)).shape) for k,p in [('cnn','algorithm/out/models/char_cnn_synth.pt'),('template','algorithm/out/models/templates_synth.npz'),('svm','algorithm/out/models/svm_synth.joblib')]]"

汇报格式：
  1. 变更文件列表（以及最后一个 commit 号）
  2. 实现要点（不超过 10 行）
  3. 参数初值及来源
  4. 验收命令原始输出
  5. 已知问题与未完成项
  6. 需要架构师决策的事项
```
