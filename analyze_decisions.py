import os
import json
import sys
import glob
from typing import List, Dict, Any, Optional
from decision_tracker import DecisionTracker
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'PingFang SC']  # 优先使用这些中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import pandas as pd
import numpy as np
import seaborn as sns

def load_game_records(game_records_dir: str) -> List[Dict[str, Any]]:
    """加载游戏记录文件
    
    Args:
        game_records_dir: 游戏记录目录
        
    Returns:
        List[Dict[str, Any]]: 游戏记录列表
    """
    game_files = glob.glob(os.path.join(game_records_dir, "*.json"))
    if not game_files:
        print(f"在 {game_records_dir} 目录下未找到游戏记录文件")
        return []
    
    game_records = []
    for file_path in game_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                game_records.append(game_data)
                print(f"已加载游戏记录: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"加载 {file_path} 失败: {str(e)}")
    
    return game_records

def analyze_decisions_from_game_record(game_record: Dict[str, Any], tracker: DecisionTracker) -> None:
    """从游戏记录中分析决策过程
    
    Args:
        game_record: 游戏记录
        tracker: 决策追踪器
    """
    game_id = game_record.get("game_id", "unknown")
    rounds = game_record.get("rounds", [])
    player_names = game_record.get("player_names", [])
    
    # 确保为所有玩家创建记录，即使他们在某些轮次中没有决策
    for player_name in player_names:
        if player_name not in tracker.player_decisions:
            tracker.player_decisions[player_name] = []
    
    # 处理每一轮的记录
    for round_data in rounds:
        round_id = round_data.get("round_id", 0)
        target_card = round_data.get("target_card", "")
        play_history = round_data.get("play_history", [])
        
        for play_action in play_history:
            player_name = play_action.get("player_name", "")
            played_cards = play_action.get("played_cards", [])
            remaining_cards = play_action.get("remaining_cards", [])
            play_thinking = play_action.get("play_thinking", "")
            play_reason = play_action.get("play_reason", "")
            
            # 记录出牌决策，即使思考过程为空也记录
            if player_name in player_names:
                # 如果play_thinking为空，使用play_reason作为备选
                reasoning_text = play_thinking if play_thinking else play_reason
                tracker.track_play_decision(
                    player_name=player_name,
                    round_id=round_id,
                    target_card=target_card,
                    play_cards=played_cards,
                    hand_cards=remaining_cards,
                    reasoning_text=reasoning_text
                )
            
            # 记录质疑决策
            was_challenged = play_action.get("was_challenged", False)
            next_player = play_action.get("next_player", "")
            challenge_reason = play_action.get("challenge_reason", "")
            challenge_result = play_action.get("challenge_result", None)
            challenge_thinking = play_action.get("challenge_thinking", "")
            
            if next_player in player_names:
                # 如果challenge_thinking为空，使用challenge_reason作为备选
                reasoning_text = challenge_thinking if challenge_thinking else challenge_reason
                tracker.track_challenge_decision(
                    player_name=next_player,
                    round_id=round_id,
                    challenged_player=player_name,
                    was_challenged=was_challenged,
                    challenge_success=challenge_result,
                    reasoning_text=reasoning_text
                )
    
    # 为每个玩家单独保存分析结果
    for player_name in player_names:
        player_decisions = [d for d in tracker.get_all_decisions() if d.get("player_name") == player_name]
        # 如果该玩家有决策记录，则处理并保存
        if player_decisions:
            # 确保玩家目录存在
            player_dir = os.path.join(tracker.output_dir, player_name)
            if not os.path.exists(player_dir):
                os.makedirs(player_dir)
            
            # 为玩家保存决策分析
            player_output_file = os.path.join(player_dir, f"{game_id}_decision_analysis.json")
            with open(player_output_file, 'w', encoding='utf-8') as f:
                json.dump(player_decisions, f, ensure_ascii=False, indent=2)
            
            # 为玩家保存CSV数据
            csv_filename = f"{game_id}_decisions.csv"
            tracker.export_player_decision_data_to_csv(player_name, csv_filename, player_decisions)
            
            print(f"已保存 {player_name} 的决策分析结果到 {player_dir}")
        else:
            # 即使没有决策记录，也确保创建玩家目录
            player_dir = os.path.join(tracker.output_dir, player_name)
            if not os.path.exists(player_dir):
                os.makedirs(player_dir)
            print(f"已为玩家 {player_name} 创建目录，但没有找到决策数据")
    
    # 同时保存完整的决策分析结果
    tracker.save_decision_analysis(game_id)
    
    # 为所有玩家导出聚合的CSV数据
    tracker.export_decision_data_to_csv(f"{game_id}_decisions.csv")

def visualize_decision_trends(tracker: DecisionTracker, output_dir: str, player_name: Optional[str] = None) -> None:
    """可视化决策趋势
    
    Args:
        tracker: 决策追踪器
        output_dir: 输出目录
        player_name: 可选，指定玩家名称；如果未提供，则分析所有玩家
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取决策趋势数据
    trends = tracker.generate_decision_trends(player_name)
    
    # 文件名前缀（用于区分不同玩家的图表）
    prefix = f"{player_name}_" if player_name else ""
    
    # 绘制每轮虚张声势率趋势
    if "bluff_rate_by_round" in trends and trends["bluff_rate_by_round"]:
        plt.figure(figsize=(10, 6))
        rounds = sorted(trends["bluff_rate_by_round"].keys())
        bluff_rates = [trends["bluff_rate_by_round"][r] for r in rounds]
        
        plt.plot(rounds, bluff_rates, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('虚张声势率', fontsize=12)
        title = f'{player_name}每轮虚张声势率趋势' if player_name else '每轮虚张声势率趋势'
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}bluff_rate_trend.png'), dpi=300)
        plt.close()
    
    # 绘制每轮质疑率趋势
    if "challenge_rate_by_round" in trends and trends["challenge_rate_by_round"]:
        plt.figure(figsize=(10, 6))
        rounds = sorted(trends["challenge_rate_by_round"].keys())
        challenge_rates = [trends["challenge_rate_by_round"][r] for r in rounds]
        
        plt.plot(rounds, challenge_rates, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange')
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('质疑率', fontsize=12)
        title = f'{player_name}每轮质疑率趋势' if player_name else '每轮质疑率趋势'
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}challenge_rate_trend.png'), dpi=300)
        plt.close()
    
    # 绘制每轮信心趋势
    if "confidence_by_round" in trends and trends["confidence_by_round"]:
        plt.figure(figsize=(10, 6))
        rounds = sorted(trends["confidence_by_round"].keys())
        confidence_values = [trends["confidence_by_round"][r] for r in rounds]
        
        plt.plot(rounds, confidence_values, marker='o', linestyle='-', linewidth=2, markersize=8, color='green')
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('平均信心程度', fontsize=12)
        title = f'{player_name}每轮决策信心趋势' if player_name else '每轮决策信心趋势'
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}confidence_trend.png'), dpi=300)
        plt.close()
    
    # 绘制关键因素频率分布
    if "key_factors_frequency" in trends and trends["key_factors_frequency"]:
        plt.figure(figsize=(12, 8))
        
        # 选取前10个最常见的因素
        top_factors = dict(sorted(trends["key_factors_frequency"].items(), 
                                 key=lambda x: x[1], reverse=True)[:10])
        
        factor_names = list(top_factors.keys())
        factor_counts = list(top_factors.values())
        
        bars = plt.bar(factor_names, factor_counts, color='skyblue')
        
        # 调整x轴标签
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('关键因素', fontsize=12)
        plt.ylabel('出现频率', fontsize=12)
        title = f'{player_name}决策中的关键因素频率分布' if player_name else '决策中的关键因素频率分布'
        plt.title(title, fontsize=14)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}key_factors_frequency.png'), dpi=300)
        plt.close()

def visualize_player_patterns(tracker: DecisionTracker, output_dir: str, player_name: Optional[str] = None) -> None:
    """可视化玩家决策模式
    
    Args:
        tracker: 决策追踪器
        output_dir: 输出目录
        player_name: 可选，指定玩家名称；如果未提供，则分析所有玩家
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取玩家决策模式数据
    player_patterns = tracker.analyze_player_decision_patterns()
    if not player_patterns:
        print("没有足够的数据生成玩家决策模式可视化")
        # 创建一个简单的信息图表表明数据不足
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "没有足够的决策数据生成分析图表", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        
        prefix = f"{player_name}_" if player_name else ""
        plt.savefig(os.path.join(output_dir, f'{prefix}insufficient_data.png'), dpi=300)
        plt.close()
        return
    
    # 如果指定了玩家，只保留该玩家的数据
    if player_name and player_name in player_patterns:
        player_patterns = {player_name: player_patterns[player_name]}
    elif player_name and player_name not in player_patterns:
        print(f"玩家 {player_name} 没有足够的决策数据生成分析图表")
        # 创建一个简单的信息图表表明该玩家数据不足
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"玩家 {player_name} 没有足够的决策数据生成分析图表", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{player_name}_insufficient_data.png'), dpi=300)
        plt.close()
        return
    
    # 准备雷达图数据
    players = list(player_patterns.keys())
    if not players:
        print("没有玩家数据可以生成图表")
        return
        
    metrics = ['bluff_rate', 'pure_bluff_rate', 'semi_bluff_rate', 
               'challenge_rate', 'challenge_success_rate', 'avg_confidence']
    metrics_labels = ['欺骗率', '纯虚张率', '半虚张率', 
                      '质疑率', '质疑成功率', '平均信心']
    
    # 文件名前缀（用于区分不同玩家的图表）
    prefix = f"{player_name}_" if player_name else ""
    
    # 创建雷达图
    plt.figure(figsize=(10, 8))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 设置雷达图的坐标轴
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # 设置坐标轴标签
    plt.xticks(angles[:-1], metrics_labels, fontsize=12)
    
    # 绘制每个玩家的雷达图
    for i, player in enumerate(players):
        # 获取玩家的指标值，确保所有指标都有值
        player_data = player_patterns[player]
        values = []
        for metric in metrics:
            if metric in player_data and player_data[metric] is not None:
                values.append(player_data[metric])
            else:
                values.append(0.0)  # 对于缺失的指标使用0
        
        # 确保有足够的数据点
        if len(values) < len(metrics):
            values.extend([0.0] * (len(metrics) - len(values)))
            
        values += values[:1]  # 闭合雷达图
        
        # 绘制雷达图和填充
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    title = f'{player_name}决策模式雷达图' if player_name else '玩家决策模式雷达图'
    plt.title(title, fontsize=15)
    plt.tight_layout()
    
    # 保存雷达图
    plt.savefig(os.path.join(output_dir, f'{prefix}decision_radar.png'), dpi=300)
    plt.close()
    
    # 创建玩家决策复杂度趋势图
    plt.figure(figsize=(12, 6))
    has_complexity_data = False
    
    for player in players:
        complexity_trend = player_patterns[player].get('reasoning_complexity_trend', [])
        if complexity_trend:
            has_complexity_data = True
            rounds, complexities = zip(*complexity_trend)
            plt.plot(rounds, complexities, marker='o', linestyle='-', linewidth=2, label=player)
    
    if not has_complexity_data:
        plt.text(0.5, 0.5, "没有足够的决策复杂度数据生成趋势图", 
                horizontalalignment='center', verticalalignment='center', 
                transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
    else:
        plt.xlabel('轮次', fontsize=12)
        plt.ylabel('决策复杂度', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    title = f'{player_name}决策复杂度随轮次变化趋势' if player_name else '玩家决策复杂度随轮次变化趋势'
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # 保存趋势图
    plt.savefig(os.path.join(output_dir, f'{prefix}complexity_trend.png'), dpi=300)
    plt.close()

def visualize_emotion_indicators(tracker: DecisionTracker, output_dir: str, player_name: Optional[str] = None) -> None:
    """可视化情绪指标
    
    Args:
        tracker: 决策追踪器
        output_dir: 输出目录
        player_name: 可选，指定玩家名称；如果未提供，则分析所有玩家
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取情绪指标数据
    emotions_by_player = tracker.extract_emotion_indicators()
    if not emotions_by_player:
        print("没有足够的数据生成情绪指标可视化")
        # 创建一个简单的信息图表表明数据不足
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "没有足够的情绪数据生成分析图表", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        
        prefix = f"{player_name}_" if player_name else ""
        plt.savefig(os.path.join(output_dir, f'{prefix}insufficient_emotion_data.png'), dpi=300)
        plt.close()
        return
    
    # 如果指定了玩家，只保留该玩家的数据
    if player_name and player_name in emotions_by_player:
        emotions_by_player = {player_name: emotions_by_player[player_name]}
    elif player_name and player_name not in emotions_by_player:
        print(f"玩家 {player_name} 没有足够的情绪数据生成分析图表")
        # 创建一个简单的信息图表表明该玩家数据不足
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"玩家 {player_name} 没有足够的情绪数据生成分析图表", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{player_name}_insufficient_emotion_data.png'), dpi=300)
        plt.close()
        return
    
    # 文件名前缀（用于区分不同玩家的图表）
    prefix = f"{player_name}_" if player_name else ""
    
    # 汇总不同玩家的主导情绪分布
    emotions_data = []
    for player, decisions in emotions_by_player.items():
        if not decisions:
            continue
            
        for decision in decisions:
            emotions_data.append({
                'player': player,
                'round_id': decision.get('round_id', 0),
                'decision_type': decision.get('decision_type', 'unknown'),
                'dominant_emotion': decision.get('dominant_emotion', '中性'),
                'confidence': decision.get('confidence', 0.5)
            })
    
    if not emotions_data:
        print("情绪数据为空，无法生成图表")
        # 创建一个简单的信息图表表明数据为空
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "情绪数据为空，无法生成图表", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{prefix}empty_emotion_data.png'), dpi=300)
        plt.close()
        return
    
    df = pd.DataFrame(emotions_data)
    
    # 创建主导情绪分布堆叠条形图
    try:
        plt.figure(figsize=(12, 8))
        emotion_counts = df.groupby(['player', 'dominant_emotion']).size().unstack(fill_value=0)
        
        if emotion_counts.empty:
            # 处理数据集为空的情况
            plt.text(0.5, 0.5, "情绪分布数据不足，无法生成堆叠条形图", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.axis('off')
        else:
            emotion_counts.plot(kind='bar', stacked=True, colormap='viridis')
            plt.xlabel('玩家', fontsize=12)
            plt.ylabel('决策次数', fontsize=12)
            plt.legend(title='情绪类型')
            plt.xticks(rotation=45)
        
        title = f'{player_name}主导情绪分布' if player_name else '玩家主导情绪分布'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}dominant_emotions_distribution.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成情绪分布图时出错: {e}")
        # 创建错误信息图表
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"生成情绪分布图时出错: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{prefix}emotion_distribution_error.png'), dpi=300)
        plt.close()
    
    # 创建情绪和信心散点图
    try:
        plt.figure(figsize=(12, 8))
        
        # 为每种情绪类型选择不同的颜色
        emotion_types = df['dominant_emotion'].unique()
        if len(emotion_types) == 0:
            # 处理没有情绪类型的情况
            plt.text(0.5, 0.5, "没有足够的情绪类型数据生成散点图", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=14)
            plt.axis('off')
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(emotion_types)))
            emotion_color_map = dict(zip(emotion_types, colors))
            
            # 按玩家分组绘制
            has_data = False
            for player in df['player'].unique():
                player_data = df[df['player'] == player]
                for emotion in player_data['dominant_emotion'].unique():
                    emotion_data = player_data[player_data['dominant_emotion'] == emotion]
                    if not emotion_data.empty and 'round_id' in emotion_data and 'confidence' in emotion_data:
                        has_data = True
                        plt.scatter(emotion_data['round_id'], emotion_data['confidence'], 
                                    color=emotion_color_map[emotion], 
                                    label=f"{player}_{emotion}" if emotion != "中性" else None,
                                    alpha=0.7, s=100)
            
            if not has_data:
                plt.text(0.5, 0.5, "没有足够的情绪和信心数据生成散点图", 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=plt.gca().transAxes, fontsize=14)
                plt.axis('off')
            else:
                plt.xlabel('轮次', fontsize=12)
                plt.ylabel('信心程度', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.5)
                
                # 只显示非"中性"情绪的图例
                handles, labels = plt.gca().get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), title="玩家_情绪", loc='upper right')
        
        title = f'{player_name}情绪和信心变化趋势' if player_name else '玩家情绪和信心变化趋势'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}emotion_confidence_trend.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"生成情绪和信心趋势图时出错: {e}")
        # 创建错误信息图表
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f"生成情绪和信心趋势图时出错: {str(e)}", 
                horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'{prefix}emotion_trend_error.png'), dpi=300)
        plt.close()

def create_player_comparison_charts(tracker: DecisionTracker, output_dir: str) -> None:
    """创建玩家间决策对比图表
    
    Args:
        tracker: 决策追踪器
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取玩家决策模式数据
    player_patterns = tracker.analyze_player_decision_patterns()
    if not player_patterns or len(player_patterns) < 2:
        print("没有足够的数据生成玩家决策对比图表")
        return
    
    # 准备玩家比较数据
    players = list(player_patterns.keys())
    
    # 1. 创建虚张声势对比图
    plt.figure(figsize=(12, 8))
    
    bluff_metrics = ['bluff_rate', 'pure_bluff_rate', 'semi_bluff_rate']
    bluff_labels = ['总虚张声势率', '纯虚张率', '半虚张率']
    
    # 准备数据
    x = np.arange(len(bluff_labels))  # 标签位置
    width = 0.8 / len(players)  # 柱状图宽度
    
    # 绘制每个玩家的数据
    for i, player in enumerate(players):
        values = [player_patterns[player][metric] for metric in bluff_metrics]
        offset = width * i - width * len(players) / 2 + width / 2
        bars = plt.bar(x + offset, values, width, label=player)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel('比率', fontsize=12)
    plt.title('玩家虚张声势策略对比', fontsize=14)
    plt.xticks(x, bluff_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'player_bluff_comparison.png'), dpi=300)
    plt.close()
    
    # 2. 创建质疑对比图
    plt.figure(figsize=(12, 8))
    
    challenge_metrics = ['challenge_rate', 'challenge_success_rate']
    challenge_labels = ['质疑率', '质疑成功率']
    
    # 准备数据
    x = np.arange(len(challenge_labels))
    width = 0.8 / len(players)
    
    # 绘制每个玩家的数据
    for i, player in enumerate(players):
        values = [player_patterns[player][metric] for metric in challenge_metrics]
        offset = width * i - width * len(players) / 2 + width / 2
        bars = plt.bar(x + offset, values, width, label=player)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.ylabel('比率', fontsize=12)
    plt.title('玩家质疑行为对比', fontsize=14)
    plt.xticks(x, challenge_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'player_challenge_comparison.png'), dpi=300)
    plt.close()
    
    # 3. 创建决策质量对比热力图
    plt.figure(figsize=(10, 8))
    
    # 准备热力图数据
    decision_metrics = ['avg_confidence', 'decision_complexity']
    decision_labels = ['平均信心度', '决策复杂度']
    
    heatmap_data = []
    for player in players:
        row = [player_patterns[player][metric] for metric in decision_metrics]
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=players, columns=decision_labels)
    
    # 绘制热力图
    ax = sns.heatmap(heatmap_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('玩家决策质量对比', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'player_decision_quality_comparison.png'), dpi=300)
    plt.close()
    
    # 4. 创建决策风格热力图
    plt.figure(figsize=(14, 10))
    
    # 准备热力图数据
    all_metrics = [
        'bluff_rate', 'pure_bluff_rate', 'semi_bluff_rate', 
        'challenge_rate', 'challenge_success_rate', 'avg_confidence', 'decision_complexity'
    ]
    metric_labels = [
        '欺骗率', '纯虚张率', '半虚张率', 
        '质疑率', '质疑成功率', '平均信心', '决策复杂度'
    ]
    
    style_data = []
    for player in players:
        row = [player_patterns[player][metric] for metric in all_metrics]
        style_data.append(row)
    
    style_df = pd.DataFrame(style_data, index=players, columns=metric_labels)
    
    # 绘制热力图
    plt.figure(figsize=(14, len(players) * 1.2 + 2))
    ax = sns.heatmap(style_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('玩家决策风格热力图', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'player_decision_style_heatmap.png'), dpi=300)
    plt.close()

def analyze_all_decisions(game_records_path: str, output_dir: str = "decision_analysis_results") -> None:
    """
    分析游戏记录中的所有决策过程
    
    Args:
        game_records_path: 游戏记录文件夹路径
        output_dir: 分析结果输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载游戏记录
    game_records = load_game_records(game_records_path)
    if not game_records:
        print("未找到有效的游戏记录，分析终止")
        return
    
    # 获取所有游戏中的所有玩家
    all_players = set()
    for game_record in game_records:
        player_names = game_record.get("player_names", [])
        for player in player_names:
            all_players.add(player)
    
    print(f"发现的所有玩家: {sorted(all_players)}")
    
    # 创建决策追踪器
    tracker = DecisionTracker(output_dir=output_dir)
    
    # 分析每个游戏记录
    for game_record in game_records:
        analyze_decisions_from_game_record(game_record, tracker)
    
    # 确保所有玩家都有目录，即使没有决策记录
    for player in all_players:
        player_dir = os.path.join(output_dir, player)
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)
            print(f"为玩家 {player} 创建目录")
    
    # 获取所有玩家列表
    player_patterns = tracker.analyze_player_decision_patterns()
    players = list(player_patterns.keys())
    
    # 如果player_patterns为空，使用之前找到的玩家列表
    if not players:
        players = list(all_players)
    
    # 为每个玩家单独生成可视化结果
    for player in players:
        player_dir = os.path.join(output_dir, player)
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)
            
        print(f"正在生成{player}的决策趋势可视化...")
        visualize_decision_trends(tracker, player_dir, player)
        
        print(f"正在生成{player}的决策模式可视化...")
        visualize_player_patterns(tracker, player_dir, player)
        
        print(f"正在生成{player}的情绪指标可视化...")
        visualize_emotion_indicators(tracker, player_dir, player)
    
    # 生成所有玩家的聚合可视化
    print("正在生成所有玩家的决策趋势可视化...")
    visualize_decision_trends(tracker, output_dir)
    
    print("正在生成所有玩家的决策模式可视化...")
    visualize_player_patterns(tracker, output_dir)
    
    print("正在生成所有玩家的情绪指标可视化...")
    visualize_emotion_indicators(tracker, output_dir)
    
    # 生成玩家间对比图表
    print("正在生成玩家间决策对比图表...")
    create_player_comparison_charts(tracker, output_dir)
    
    print(f"决策分析完成，结果保存在 {output_dir} 目录")
    
    return tracker

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python analyze_decisions.py <game_records_dir> [output_dir]")
        return
    
    game_records_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "decision_analysis_results"
    
    # 调用分析函数
    analyze_all_decisions(game_records_dir, output_dir)

if __name__ == "__main__":
    main() 