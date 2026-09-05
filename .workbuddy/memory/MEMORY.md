# MEMORY.md · laser_cladding_ml_v7 项目长期记忆

## 项目基本事实

- [2026-09-01] - 本项目 `laser_cladding_ml_v7`：机器学习用于激光熔覆，本人（刘玉和）第一作者，目标期刊 Journal of Alloys and Compounds。
- [2026-09-01] - **v7 的 Rockit485 实验基体为 Q355 钢**（用户确认）。
- [2026-09-01] - 数据清洗：215 组文献样本 → 9 条物理准则 → 194 组。
- [2026-09-01] - 基线模型实测（model_summary.xlsx）：硬度 RF R²=0.6178、MAE=94.24；腐蚀 log R² 天花板低（0.13~0.26），数量级准确率 RF 91.79%。

## 两篇稿件边界

- [2026-09-01] - **投稿2 是另一篇独立论文**：本人第二作者，通讯作者夏云浩，题目含 Q355 Iron-Based Alloys + SHAP。
- [2026-09-01] - v7 与投稿2 结论不可混用；涉及摘要/图表/数据必须核对来源。

## 资料库工作台

- [2026-09-02] - 工作台页面「激光熔覆ML论文工作台」：节点 `t6se4NevNuUzIkKeVbRn7f`（web），空间 `aXncruBkoVAWqvJcnJ7ON0`，链接 https://www.workbuddy.cn/space/d/t6se4NevNuUzIkKeVbRn7f
- [2026-09-02] - 4 张子表（已填数据）：实验与模型记录 `h5HWv497Xs35DWhFpb4gJs`、论文章节进度 `VON8CakkzaUVCatPoNZJru`、图表中心 `cldGVJQfy1XPJFqow8m193`、投稿与返修追踪 `vdQiVljliLlo9lM08nZcao`。
- [2026-09-02] - 工作台 HTML 已通过 `__SMART_PAGE__.database` SDK 绑定 4 张表，页面自动读表渲染。

## 禁止出现的表述

- [2026-09-01] - 不得在 v7 摘要/正文中出现投稿2 的数据（276 组、ExtraTrees、BayesianRidge、ElasticNet、Voting/Stacking、448 HV、0.6 m/min、248% 提升等）。
- [2026-09-01] - 严禁编造数据；代码/指标一律以 outputs 实测为准。

## 用户偏好（项目相关）

- [2026-09-01] - 缺失数据（如表征数据）先不写；总结流程图放在论文最开头。
- [2026-09-01] - 中英文沟通均可，文档用中文；图表/论文用英文标签。
