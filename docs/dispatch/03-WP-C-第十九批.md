# 派单 03 · 员工 C · 第十九批：把 M2 接进团队仓库（适配函数 + 后端接线 + 全链路测试）

```text
你是 M2 车牌识别模块执行者【员工 C，工位 03】。
我们的仓库：工作目录 D:\keshe2\workers\03，分支 feature/m2-wpc-post-eval（已同步到 develop，含本派单），用 .venv\Scripts\python.exe。
团队仓库本地副本：D:\keshe2\workers\parkinglot。它是独立的本地 git 仓库，没有远程；develop 分支 = 团队 10-30 原样快照（提交 359f3fe）；你在 feature/lpr-integrate 分支上提交。本批只有你写这个目录。

为什么做（架构师 11-03）：组长的后端框架已上团队 develop，给算法留的口子是 algorithm/lpr/pipeline.py 的 recognize(image_bytes) -> dict。
  现在那是空壳，后端调不到就退回假车牌“鲁B12345”。M2b 是团队现在唯一“待交付”的后端依赖，11-06 团队集成。
  用户要求：把我们的部分顶上去，在本地测好，组员拉下来不报错、不用反复改；别人没做完的部分（采集送图等）不管。

必读：
  - 团队副本：backend/app/services/recognition/client.py、backend/app/schemas/recognition.py、backend/app/api/entry.py 与 exit.py、
    backend/app/services/storage/images.py、backend/app/core/config.py、algorithm/README.md、docs/MODULES.md、
    CONTRIBUTING.md（提交信息格式、PR 要求）、pyproject.toml、requirements*.txt、tests/conftest.py、README.md。
  - 我们的仓库：docs/M2-接口说明-M3.md 的“线程安全与运行约定”一节；algorithm/lpr/pipeline.py 的 PlateRecognizer.from_config 与 recognize_bytes。

任务 A（我们的仓库，先做，提交到你的分支）：
  A1. 在 algorithm/lpr/pipeline.py 加模块级函数 recognize(image_bytes: bytes | None, *, lane_id: int | None = None) -> dict，这就是团队契约。要求：
      - 第一次调用时用 PlateRecognizer.from_config() 建实例，之后复用。初始化加锁，多线程并发首调只建一次。加载失败不缓存，下次调用重试。
      - 返回字典恰好 6 个键，与团队 RecognitionResult 一致：
        plate_no：str，没有车牌时为 ""。
        confidence：float。
        image_path：None。
        stage_ms：dict[str, int]，四舍五入成整数（团队 pydantic 字段是 int，带小数会校验失败）。
        success：bool，只有我们的状态是 OK 才为 True。
        message：成功为 None，失败用我们结果里的中文说明。
        非 OK 时 plate_no 照填我们的最佳猜测（可能为 ""），success=False；后端据此报 3001，转人工。
      - image_bytes 为 None、为空或解不开：返回 success=False 的字典，不抛异常。
      - 模型加载失败（缺 torch/torchvision、模型文件缺失、manifest 不符等任何异常）：抛 RuntimeError（from 原异常），
        消息写明原因和“pip install -r requirements-algo.txt”。
        不能抛 ImportError 或 NotImplementedError：团队 client 遇到这两种会悄悄改用假车牌。
  A2. 新建 algorithm/tests/test_backend_contract.py，覆盖：
      - 6 个键与各自类型。
      - 用测试里照抄的 pydantic 模型（6 个字段同团队 schemas/recognition.py，类名别和我们的 RecognitionResult 重名）校验能构造。
      - 从 algorithm/tests/data/golden 挑 1 张默认流程状态为 OK 的图，success=True 且车牌对。
      - 坏字节、None 都返回 success=False，不抛。
      - 加载失败（monkeypatch from_config 抛 ImportError 或 ModelLoadError）时抛 RuntimeError。
      - 多线程并发首调只建一次实例。
  A3. 新建 algorithm/lpr_tools/team_export.py：把我们仓库某个已提交版本（默认 HEAD，用 git archive，不读工作区）导出到团队副本。
      参数 --repo、--ref、--dest。导出清单（全部覆盖，不多不少）：
        algorithm/lpr/、algorithm/lpr_tools/、algorithm/tests/、algorithm/config/、algorithm/reports/scoreboard.md、algorithm/pyproject.toml、
        algorithm/models/ 下的 plate_det_v1.pt、char_cnn_v2.pt、crnn_v4.pt、box_reader_v3.pt、manifest.json（只这 5 个）。
      不导出：algorithm/README.md（团队版由你在 B3 手写）、algorithm/requirements.txt、data/、out/、testset/、其余模型。
      团队副本的 algorithm/lpr/、lpr_tools/、tests/、config/ 下有而我们没有的文件用 git rm 删除（目前只有 algorithm/lpr/recognize.py）；
      另外 git rm 团队的 algorithm/train/ 空壳（由 lpr_tools 取代）；保留 algorithm/__init__.py。
      运行时打印：导出了哪些文件、删了哪些、5 个模型文件的 sha256 与 manifest 是否一致。只写 --dest 下的上述路径。
  A4. 我们仓库全量 pytest 通过：先在改动前跑一次记基线，改完后只允许多出你的新测试。
      本批 A 部分涉及的 3 个文件要在团队 ruff 规则下干净：
        - 装 ruff：.venv\Scripts\python.exe -m pip install ruff（最新版，只装这一个包，授权联网），回执写版本号。
        - 在工位根目录运行：ruff check --config D:\keshe2\workers\parkinglot\pyproject.toml algorithm/lpr/pipeline.py algorithm/tests/test_backend_contract.py algorithm/lpr_tools/team_export.py
      pipeline.py 原有的 ruff 问题也由你修（01、02 本批不碰这个文件），只做不改行为的修改。

任务 B（团队副本，feature/lpr-integrate 分支）：
  B1. 环境：建 D:\keshe2\workers\parkinglot\.venv（Python 3.12，可用 D:\keshe2\workers\03\.venv\Scripts\python.exe -m venv 创建）。
      安装 pip install -r requirements-dev.txt -r requirements-algo.txt；B3 改完 requirements-algo.txt 后再装一次。授权联网安装这些依赖。
      torch、torchvision 装 CPU 版，版本与我们一致（torch 2.14.1、torchvision 0.29.1+cpu）。.venv 不入库（团队 .gitignore 已忽略）。
  B2. 改前基线：在 359f3fe 上跑根目录 pytest，以及 ruff check backend algorithm capture scripts tests，记下结果。团队原有的失败不算我们的，但要列出来。
  B3. 导出与依赖说明：
      - 用 A3 的脚本从你的分支 HEAD 导出。
      - 改 requirements-algo.txt：加 torch>=2.14,<3、torchvision>=0.29、loguru>=0.7；opencv-python 改成 >=4.9,<5。
        注释写明实测版本，以及“Windows 上 PyPI 默认即 CPU 版”。
      - 手写 algorithm/README.md：保留团队契约那一节（改成实际行为）；加安装、运行、测试（cd algorithm && python -m pytest）、模型与配置（LPR_CONFIG）；
        说明对比实验用的旧模型不随团队仓库分发。
  B4. 后端接线。这几处原归刘一凡、贺锦晗，用户已同意由我们顺手改，改动尽量小：
      - client.py：开头把仓库根目录（app.core.config.REPO_DIR）追加到 sys.path，不在才加，用 append。
        否则 cd backend 启动、Docker 里都找不到 algorithm 包，会悄悄退回假车牌。
        行为必须全部满足下表：
        | 给了 plate_no | 图片                                   | 结果 |
        | 是            | 任意                                   | 用给的车牌，不调算法（同现在） |
        | 否            | 没有图                                 | 占位车牌（同现在，保证前端留空演示不坏） |
        | 否            | 有图且能解码                           | 调真算法，success 照算法结果 |
        | 否            | 有图但出任何异常（含 ImportError、RuntimeError） | success=False、message="识别失败"，logger.exception 记原因；不再给假车牌 |
      - storage/images.py：抽出 decode_capture(image_base64) -> bytes | None，解码规则同 save_capture，save_capture 改用它。
        解码失败要能和“没给图”区分开（例如返回 b""），使“有图但解不开”走 3001，而不是占位车牌。
      - entry.py、exit.py：把图片字节传给识别，用 await asyncio.to_thread(recognize, ...) 调用（识别约 100 ms，首调加载模型要几秒，不能卡住事件循环）。
        失败时改为 ApiError(3001, result.message)；其余流程不动。
      - docs/api.md 第 70 行附近：改成实际行为——有图走真识别，非 OK 一律 3001 转人工；不给图也不给车牌时仍是占位识别。
      - README.md 进度表 M2b 一行、docs/TASKS.md 的 F3-T01～T06 六行：按文末“统一口径”改，数字只用那里的，不自己编。
  B5. 测试（放 tests/，与团队测试同风格）：
      - test_recognition_client.py：用假的 algorithm.lpr.pipeline（monkeypatch sys.modules）覆盖行为表 4 行，不依赖 torch。
      - test_recognition_real.py：开头 pytest.importorskip("torch")，用 A2 那张金样例：
        client 直调一次、POST /api/v1/entry/capture 一次（只给 image_base64、不给 plate_no），都要车牌对、返回 200；
        再发一个坏 base64，应返回 422、错误码 3001。
  B6. 验收：全部在“全新 clone”里做，模拟组员拉代码。
      把 D:\keshe2\workers\parkinglot git clone 到你的 out 目录并检出 feature/lpr-integrate（这样才会按团队 .gitattributes 转成 LF），用 B1 的 venv：
      - 根目录 pytest 全过，和 B2 基线比只多不少。
      - ruff check backend algorithm capture scripts tests：backend 与 tests 无新增问题。
        algorithm/lpr/（除 pipeline.py）、lpr_tools/、algorithm/tests/ 的旧问题由 02、01 本批在我们仓库清理，你只记录剩余数量，不改它们。
      - cd algorithm && python -m pytest：记录结果。缺非默认模型或本地数据导致的失败列出来即可，01 本批在修。
      - 按团队 README 的启动方式起后端：先建库，再 cd backend && uvicorn app.main:app --port 8000，不设 PYTHONPATH；端口被占用就换一个。
        用 Python 发一张金样例的 image_base64 到入场接口，确认返回真实车牌而不是“鲁B12345”。记首次请求与第二次请求的耗时。
  B7. 团队副本提交：按 CONTRIBUTING 的格式写中文提交信息（feat(lpr) / fix(api) / docs(docs) / test(...)），分 3～5 个提交。
      不改 develop 分支，不建其他分支，不加远程。

统一口径（README 与 TASKS 只能用这些，出处 algorithm/reports/scoreboard.md）：
  - 冻结测试集只跑过一次：清晰 148/150 = 98.67%，夜间 48/50 = 96.00%，倾斜 49/50 = 98.00%，污损 48/50 = 96.00%；自动通过精度 282/283 = 99.65%。
  - 单帧 P95 105.3 ms（CPU、1280×720、2 线程）；lpr 覆盖率 94.54%；G3 视频端到端待真实视频。
  - 版本 lpr-1.0.0+char_cnn_v2+plate_det_v1+crnn_v4+box_reader_v3。
  - 数据：CCPD2019/2020 公开标注数据（F3-T01 写明是公开数据，不是自采）；开发集 val1000（1000 张）、devhard（2100 张）；冻结测试集 300 张（清晰 150，夜间、倾斜、污损各 50）。
  - F3-T02 定位召回：用积分板 “Top-3 定位召回” 列里默认流程那一行的数，注明口径；找不到就不写数字。
  - F3-T04：写我们实际实现的后处理（格式校验等）与多帧投票（lpr/voting.py），先核对代码确实有，再写。
  - F3-T05 写“进行中”：分场景准确率已有；三方法对比与混淆矩阵随 11-10 算法评测报告交付。
  - F3-T06 写“已完成”（本分支）。积分板里找不到的数字一律不写。

资源与磁盘：不用 GPU；内存 ≤2.5 GB（后端加模型）；测试一次只跑一个进程；新增磁盘 ≤2 GB（venv 约 1.5 GB，clone 约 30 MB）。预计 3～4 小时。

只允许修改：
  - 我们的仓库：algorithm/lpr/pipeline.py；新建 algorithm/tests/test_backend_contract.py、algorithm/lpr_tools/team_export.py；你的回执。
  - 团队副本：B 部分点名的文件、导出清单里的路径、tests/ 下两个新测试文件；.venv（不入库）。

禁止：
  - 改我们仓库的其他文件：config、模型、lpr/lpr_tools/tests 里的其他文件、scoreboard.md。
  - 跑冻结测试集。
  - 改团队副本的 pyproject.toml、Makefile、deploy/、web/、capture/、db/、backend/app/models/。
  - 推送、加远程、改团队副本的 develop 分支；切换或合并我们仓库的分支；改其他工位。
  - 删除文件只用 git rm（团队副本里已跟踪的文件）；若被拒绝，在回执里列出要删的文件，由架构师处理。

汇报方式：完整汇报写到我们仓库你工位的 algorithm/out/receipts/第十九批.md，内容包括：
  - B2 基线；
  - B6 每一项的结果与耗时；
  - 团队副本 git log --oneline develop..feature/lpr-integrate；
  - 按团队 .gitee/PULL_REQUEST_TEMPLATE.md 写好的 PR 说明，可直接粘贴。
聊天里只回一行：回执 algorithm/out/receipts/第十九批.md，commit <我们仓库最后提交号>，团队副本 <最后提交号>
```
