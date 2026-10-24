# 金样例（20 张）

CCPD 数据到位后由架构师挑选放入本目录，并写 `labels.csv`（列：`file,plate_no,plate_color,category`）。

| 类别 | 数量 | 来源 |
| --- | --- | --- |
| base | 10 | ccpd_base |
| green | 2 | CCPD2020 |
| db | 2 | ccpd_db |
| tilt | 2 | ccpd_tilt |
| blur | 2 | ccpd_blur |
| np | 2 | ccpd_np（无牌，期望 NO_PLATE） |

`tests/test_golden.py` 在 `labels.csv` 不存在时自动跳过。
