# 阶梯-GRU 策略 (staircase + GRU + 异常检测)

因果、可操作的角球标的线策略。核心:
- **阶梯特征**: 线的单调单步阶梯(<=40分钟成形) -> 暂定方向(上阶梯->低于顶线, 下阶梯->高于底线)。
- **GRU 序列模型** (path1/路径2): 读"截至当前时刻的全部口(因果前缀)", 输出 P(初判成立), 校准良好。
- **可操作窗口**: 最早决定性时刻 T* 到"走势锁定(角球触及锚线)"之间, 窗口>0 才有效。
- **异常检测器** (anomaly_detector.py): 出手后监控 overshoot, 越过阈值报警 -> 反手(对冲)或弃权。

## 脚本
- strategy/gru_operable.py       : GRU + 最早决定性时刻 + 可操作窗口 -> HTML
- strategy/gru_time_compare.py   : 同场不同时刻(45..85') 预测演变 + 各时刻正确率
- strategy/path1_confidence_model.py : 重构目标(HGB)的置信度模型
- strategy/staircase_conditional.py  : 阶梯 + 后段延续幅度 分档正确率
- strategy/anomaly_detector.py   : overshoot 异常报警 (召回/误报/对冲/弃权)

## 结果页
static_stair/index.html (可操作窗口版) / time_compare.html (分时对比)。
部署到 GitHub Pages: /sthfunny/stairgru/

## 关键数字(样本外, 08-02 / 月度)
- 可操作(最早决定性)正确率 ~76%; 越晚越准(45'77% -> 85'93%)。
- 异常报警(overshoot>=1.5~2.0): 召回将错 76-87%, 误报 19-32%, 报警反手 ~78%, 报警弃权 D=1.0->88%(27%覆盖)。
- 提醒: 均为线派生特征, 高准确多在走势临近定型时; 早段edge有限。仅策略验证, 非投注建议。
