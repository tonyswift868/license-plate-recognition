# 派单 01 · 员工 A（WP-A）· 第二批：字符样本生成与三方法对比脚本

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 A / WP-A，工位 01】。
工作目录：D:\keshe2\workers\01    分支：feature/m2-wpa-model（已检出，不要切换分支）
环境：工作目录自带 .venv（Python 3.12、OpenCV 4.14，依赖已装好，lpr 已可编辑安装）。所有命令在工作目录根下运行，
      用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 2、5 章，6.5 节，7.1—7.5 节，第 8 章，11.2 节，12.3 节，第 15 章，附录 A、B；
      algorithm/lpr_tools/data/ccpd.py（员工 C 已实现：parse_ccpd_name、read_index、索引 CSV 的列与 quad/bbox 格式）；
      algorithm/lpr/charnorm.py；algorithm/config/default.toml 的 [rectify] [segment]
背景：CCPD 还没到位。本批目标是“数据一到就能直接跑”：把 A1（字符样本生成）和 A5（三方法对比）的脚本写好，
      用自己构造的小样本跑通。员工 B 同时在造仿 CCPD 数据集，你不依赖它，单测用自己构造的样本。

本次任务（按顺序做，每做完一项提交一次）：
  A-5 字符样本生成脚本（任务 A1）
      - lpr_tools/train/build_chars.py，严格按 7.3：读 ccpd_index.csv → 只取 has_plate=1、split ∈ {train, val}、
        skip_reason 为空的行 → 用真值 quad 透视校正到 252×80（7 位）/ 276×80（8 位）→ 按 default.toml 的版式槽
        （layout7_centers / layout7_char_w，8 位用 layout8_*）切出每一位 → charnorm.unify_polarity（按 plate_color）
        → charnorm.normalize_char → 保存 data/chars/{split}/<类下标>/<图名>_<位>.png，并写 chars_index.csv。
      - 按整图划分（同一张图的字符只进一个 split）；--exclude-manifest 指定测试集清单时，清单里的图一律剔除；
        --max-per-class（默认 8000）限制每类样本数；结束时打印每类数量统计表，并把统计写到输出目录。
      - 图片读写一律用 lpr.io_utils（中文路径安全）。
      - 命令：python -m lpr_tools.train.build_chars --index <ccpd_index.csv> --raw <raw 根目录> --out algorithm/data/chars
      - 单测 tests/test_build_chars.py：在 tmp_path 里画几张已知车牌、按 CCPD 格式命名（可直接调用 ccpd.py 的
        解析函数验证命名），建索引 → build_chars → 检查类别目录、数量、整图不跨 split、排除清单生效。
  A-6 三方法对比脚本（任务 A5 的字符级部分）
      - lpr_tools/eval/ablation.py：在同一个字符测试集上对比 template / svm / cnn 三种分类器：
        字符准确率（总体、省份字、字母数字）、单字符平均耗时、模型文件大小、训练耗时（从 manifest 或日志读取），
        输出 markdown 表格到 --out 目录；整牌准确率部分留接口（后续用 eval_accuracy 的结果填）。
      - 命令：python -m lpr_tools.eval.ablation --chars <字符目录>/val --models cnn=<pt> template=<npz> svm=<joblib> --out algorithm/out/ablation
      - 用你上一批的合成字符和合成模型跑通；单测用小样本。
  A-7 训练脚本补强（为 A4 做准备）
      - train_cnn.py 支持多个 --data 目录混合（如真实字符 + 合成省份字），可给每个目录设采样权重；支持断点续训；
        训练日志写文件；保存时记录训练数据来源到 manifest 的 train_data。
      - train_svm.py 增加 --calibrated 选项改用 CalibratedClassifierCV(SVC(), ensemble=False)，消除 sklearn 1.9 的弃用警告。

只允许新建或修改：
  algorithm/lpr/recognize/cnn_model.py、cnn.py、template.py、svm.py
  algorithm/lpr_tools/train/ 下的文件、algorithm/lpr_tools/eval/ablation.py
  algorithm/tests/ 下以 test_recognize、test_augment、test_synth_chars、test_build_chars、test_train、test_ablation 开头的文件
禁止：
  - 修改 types.py、charset.py、charnorm.py、errors.py、config.py、pipeline.py、testing.py、cli.py、recognize/base.py、
    recognize/__init__.py、lpr_tools/data/ 下任何文件、lpr_tools/eval/ 下除 ablation.py 外的文件、
    tests/conftest.py、tests/test_contracts.py、tests/test_pipeline.py、config/default.toml
  - 引入白名单外的依赖（不允许 torchvision），不得 pip install 任何东西；使用任何车牌识别 API 或预训练模型
  - 在运行时包 algorithm/lpr/ 里 import Pillow
  - 提交 algorithm/data/、algorithm/out/ 下的任何文件；本批不往 algorithm/models/ 放模型
  - 切换或合并分支，改动 D:\keshe2 主检出或其他 workers 目录
协作方式：
  积极派子代理并行完成相互独立的子任务（如不同文件的实现、单测、验收命令），以加快进度；
  子代理同样只能改本派单允许的文件，两个子代理不得同时改同一个文件；汇总、自测与提交由你本人负责。
  需要看图时先缩小到宽 ≤800 px 再打开，一次最多看几张；不要在对话里打开大图或拼图（会拖垮对话界面）。
提交：只在本分支提交，提交信息用 feat(m2): / test(m2): / fix(m2): 前缀。

验收命令（必须实际运行，原始输出保存到 algorithm/out/receipts/ 下）：
  .venv\Scripts\python.exe -m pytest algorithm/tests -q
  .venv\Scripts\python.exe -m lpr_tools.train.build_chars --help
  .venv\Scripts\python.exe -m lpr_tools.eval.ablation --chars algorithm/data/chars_synth/val --models cnn=algorithm/out/models/char_cnn_synth.pt template=algorithm/out/models/templates_synth.npz svm=algorithm/out/models/svm_synth.joblib --out algorithm/out/ablation
  另附：build_chars 在你构造的小样本上的完整运行输出与每类统计表。

汇报方式：
  完整汇报写到 algorithm/out/receipts/第二批.md（含：变更文件列表与最后 commit 号、实现要点≤10 行、参数及来源、
  验收命令原始输出或输出文件路径、已知问题与未完成项、需要架构师决策的事项）。
  聊天里只回一行：回执 algorithm/out/receipts/第二批.md，commit <最后提交号>
```
