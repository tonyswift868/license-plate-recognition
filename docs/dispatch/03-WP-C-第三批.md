# 派单 03 · 员工 C · 第三批：M3 接口说明、数据到位运行手册、17.3 自查

```text
你是“停车场收费系统”M2 车牌识别模块的执行者【员工 C，工位 03】。
工作目录：D:\keshe2\workers\03    分支：feature/m2-wpc-post-eval（已检出，不要切换分支）
环境：工作目录自带 .venv，用 .venv\Scripts\python.exe。
必读：docs/M2-车牌识别-开发文档.md 第 5 章、7.1 节、9.3 节、13 章、17 章；algorithm/README.md；
      algorithm/lpr/__init__.py、types.py、errors.py、testing.py；algorithm/lpr_tools/ 下各脚本的 --help
任务：
  1. docs/M2-接口说明-M3.md：给刘一凡（M3 出入场控制）的接入说明。内容：安装方式、PlateRecognizer 四个方法与
     调用示例（含 asyncio.to_thread）、RecognitionResult.to_dict() 字段表、状态与 3xxx 错误码及建议处置、
     FakeRecognizer 三种用法（含按演示图文件名返回指定状态）、多图上传走 recognize_many、.env 键 LPR_CONFIG/LPR_FAKE、
     线程安全约定。示例代码必须在本工位实际运行验证（用 FakeRecognizer）。
  2. docs/M2-数据到位运行手册.md：CCPD 下载解压后，从放置目录、建索引、抽测试集、挑金样例、生成字符样本、
     训练 CNN 和两种基线、版式校准、每日验证评测与定位召回、G1/G2/G3 测量，到写积分板的完整命令清单，
     按执行顺序编号，每步写清输入、输出位置、预计耗时和检查点。每条命令必须在 algorithm/data/raw_mock 仿真数据上
     实际跑通（仿真数据不存在就先用 python -m lpr_tools.synth_ccpd --out algorithm/data/raw_mock --per-subset 40 --seed 42 生成）。
  3. docs/M2-自查记录.md（开发文档 17.3 第 2 项）：上网核实 CBLPRD-330k 的仓库地址、许可证、数据内容与是否可用于课程；
     核实可用车牌字体的来源与许可；附来源链接。查不到的写“未查到”，不得编造。
只允许新建或修改：docs/M2-接口说明-M3.md、docs/M2-数据到位运行手册.md、docs/M2-自查记录.md
禁止：修改任何代码、配置、测试和其他文档；引入新依赖或 pip install；下载任何数据集；切换或合并分支；
      改动 D:\keshe2 主检出或其他 workers 目录；提交 algorithm/data/、algorithm/out/ 下的文件。
协作方式：积极派子代理并行完成三份文档，你负责核对与提交；看图先缩小到宽 ≤800 px。
提交：docs(m2): 前缀。
汇报方式：完整汇报写到 algorithm/out/receipts/第三批.md；聊天里只回一行：回执 algorithm/out/receipts/第三批.md，commit <最后提交号>
```
