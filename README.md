# 车牌识别（课设 keshe2 备份）

这是本机 `D:\keshe2` 删除前的完整代码备份（2026-11-12）。

## 里面有什么

- `algorithm/`：车牌识别代码（`lpr/`、`lpr_tools/`）、配置、训练好的模型（`models/`）、测试和小测试集。
- `docs/`：设计、交接文档。
- `archive/parkinglot-patches/`：交给团队仓库（gitee `ankiting/parkinglot`）的 6 个补丁和说明。要用时在团队仓库里执行 `git am archive/parkinglot-patches/*.patch`。
- `archive/parkinglot-demo/demo.html`：演示页面。
- 所有分支都在：`develop`（最新）、`main`、`feature/m2-*`。

## 没有上传的（太大或可以重新生成）

- `algorithm/data/`：数据集、裁剪图和缓存，约 38 GB、281 万个文件。
- `algorithm/out/`：运行输出和员工回执，约 0.8 GB。
- `.venv/`、`.venv-gpu/`：Python 环境，照依赖重新安装即可。
- `workers/01～03`：员工工作副本，提交都已在本仓库。
- `workers/parkinglot`：团队仓库的克隆，没有未推送的提交，原件在 gitee。
