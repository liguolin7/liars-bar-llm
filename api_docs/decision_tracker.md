# 决策追踪器 API 文档

## 简介

`decision_tracker.py` 模块提供了决策过程追踪与分析功能，能够从LLM推理文本中提取结构化决策信息，并进行可视化分析。

## 主要功能

1. 提取关键决策组件（考虑因素、证据、替代方案等）
2. 跟踪记录出牌和质疑决策
3. 分析玩家决策模式和策略偏好
4. 提取情绪指标和信心变化
5. 生成决策相关可视化图表

## 核心类

### DecisionComponents

结构化表示决策的组成部分：
- `key_factors`: 关键考虑因素
- `evidence`: 支持决策的证据
- `alternatives`: 考虑的替代方案
- `reasoning_steps`: 推理步骤
- `confidence`: 决策信心程度 (0-1)
- `uncertainty_sources`: 不确定性来源

### DecisionTracker

决策追踪与分析的核心类：

```python
# 初始化
tracker = DecisionTracker(output_dir="decision_analysis")

# 分析推理文本
components = tracker.analyze_reasoning(reasoning_text)

# 跟踪出牌决策
tracker.track_play_decision(
    player_name="玩家A", round_id=1, target_card="K",
    play_cards=["K", "Q"], hand_cards=["A"], reasoning_text="..."
)

# 跟踪质疑决策
tracker.track_challenge_decision(
    player_name="玩家B", round_id=1, challenged_player="玩家A",
    was_challenged=True, challenge_success=False, reasoning_text="..."
)

# 保存分析结果
tracker.save_decision_analysis("game_id")
tracker.export_decision_data_to_csv()

# 分析决策趋势
trends = tracker.generate_decision_trends()

# 分析玩家决策模式
patterns = tracker.analyze_player_decision_patterns()

# 提取情绪指标
emotions = tracker.extract_emotion_indicators()
```

## 集成到游戏流程

在 `game.py` 中：

1. 导入决策追踪器：`from decision_tracker import DecisionTracker`
2. 在Game类初始化时创建追踪器：`self.decision_tracker = DecisionTracker()`
3. 在出牌和质疑环节中记录决策
4. 在游戏结束时保存分析结果

## 离线分析游戏记录

使用 `analyze_decisions.py` 脚本分析现有游戏记录：

```bash
python analyze_decisions.py game_records/ [output_dir]
```

该脚本将：
1. 加载指定目录下的游戏记录
2. 分析所有决策并提取结构化信息
3. 生成各种可视化图表
4. 导出CSV格式的决策数据

## 可视化示例

使用决策追踪系统可以生成以下可视化图表：

### 1. 决策趋势可视化

- **每轮虚张声势率趋势图**：展示各轮次中玩家虚张声势行为的比例变化
- **每轮质疑率趋势图**：展示各轮次中玩家发起质疑的比例变化
- **每轮信心程度趋势图**：展示各轮次中玩家决策信心的平均值变化

### 2. 玩家决策模式可视化

- **玩家决策模式雷达图**：综合展示每个玩家在欺骗率、质疑率、信心程度等多个维度的表现
- **玩家决策复杂度趋势图**：展示玩家决策推理步骤数量随游戏进行的变化趋势

### 3. 情绪和动机分析

- **玩家主导情绪分布图**：展示每个玩家在决策中表现出的主导情绪分布
- **情绪和信心关系散点图**：分析情绪状态与决策信心之间的关系

### 4. 关键因素分析

- **决策关键因素频率分布图**：统计在所有决策中最常被考虑的因素
- **证据类型使用频率图**：展示不同玩家倾向于使用哪些类型的证据支持决策

## 注意事项

1. 确保LLM推理文本包含足够详细的决策信息，以便提取有意义的结构化数据
2. 推理文本最好包含"考虑因素"、"证据"、"推理过程"等关键段落
3. 对于大型游戏记录集，分析过程可能需要较长时间 