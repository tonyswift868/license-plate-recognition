# 金样例 v1（20 张）

由 CCPD 真值文件名解析车牌标签，不使用模型预测改写标签。图片保留原始像素内容；`labels.csv` 保存识别回归标签，`golden_manifest.csv` 使用标准 manifest 字段并列出原图绝对路径，供验证集排除逻辑并集剔除这 20 张，避免 golden-only 样例进入开发集。

本批构成为：清晰 base 10、CCPD2020 green 2、夜间 db 2、倾斜 tilt 2、模糊 blur 2、无牌 np 2。base 10 全部从正式测试清单中预留的 `ccpd_base` val 清晰 120 张抽取；green/db/tilt/np 从冻结清单抽取。正式清单未包含的两张 `ccpd_blur` 官方 test 图片仅用于 golden 回归，`golden_manifest.csv` 的 `note=golden_only` 标明其来源。其余来自正式清单的样例标为 `frozen_manifest`。

| 类别 | 数量 | 来源 |
| --- | ---: | --- |
| clear/base | 10 | 保留的 ccpd_base val 120 |
| clear/green | 2 | CCPD2020 ccpd_green |
| night | 2 | ccpd_db |
| tilt | 2 | ccpd_tilt |
| dirty/blur | 2 | ccpd_blur 官方 test（golden-only） |
| np | 2 | ccpd_np（无牌，期望 NO_PLATE） |

`tests/test_golden.py` 的通过门槛未改动。
