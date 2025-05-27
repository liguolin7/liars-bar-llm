#!/usr/bin/env python3
"""
量化评估指标系统 - 骗子酒馆多智能体项目

此模块提供一系列量化指标，用于评估不同LLM模型在骗子酒馆游戏中的表现。
指标包括欺骗相关指标、存活指标、决策质量指标等。
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import matplotlib.pyplot as plt
import seaborn as sns

def load_game_records(folder_path: str) -> List[Dict]:
    """
    加载指定文件夹中的所有游戏记录
    
    Args:
        folder_path: 包含游戏记录JSON文件的文件夹路径
        
    Returns:
        游戏记录列表
    """
    game_records = []
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                game_records.append(game_data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    print(f"成功加载 {len(game_records)} 个游戏记录")
    return game_records

# ==================== 欺骗相关指标 ====================

def analyze_deception_metrics(game_records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    分析与欺骗相关的指标
    
    Args:
        game_records: 游戏记录列表
        
    Returns:
        玩家欺骗指标统计，包括：
        - honest_play_rate: 诚实出牌率 (实际出的牌与目标牌相符)
        - deception_rate: 欺骗率 (出牌中含有非目标牌)
        - pure_bluff_rate: 纯虚张声势率 (完全没有目标牌却出牌)
        - half_bluff_rate: 半虚张声势率 (出牌中有部分是目标牌，部分不是)
        - deception_success_rate: 欺骗成功率 (欺骗时没被质疑)
        - challenged_rate: 被质疑率
        - joker_usage_rate: 万能牌使用率 (使用Joker代替目标牌的比例)
    """
    # 初始化统计数据
    deception_stats = {}
    
    # 用于统计各种出牌情况的次数
    for game in game_records:
        player_names = game.get('player_names', [])
        
        # 初始化玩家统计
        for player in player_names:
            if player not in deception_stats:
                deception_stats[player] = {
                    'total_plays': 0,                 # 总出牌次数
                    'honest_plays': 0,                # 诚实出牌次数（所有牌都是目标牌）
                    'deceptive_plays': 0,             # 欺骗出牌次数（至少有一张非目标牌）
                    'pure_bluffs': 0,                 # 纯虚张声势次数（完全没有目标牌）
                    'half_bluffs': 0,                 # 半虚张声势次数（部分是目标牌）
                    'successful_deceptions': 0,       # 成功欺骗次数（欺骗没被质疑）
                    'challenged_plays': 0,            # 被质疑次数
                    'plays_with_joker': 0,            # 使用万能牌次数
                    'joker_count': 0,                 # 使用的万能牌数量
                }
        
        # 分析每一轮的出牌记录
        for round_data in game.get('rounds', []):
            target_card = round_data.get('target_card')
            
            for play in round_data.get('play_history', []):
                player = play.get('player_name')
                played_cards = play.get('played_cards', [])
                was_challenged = play.get('was_challenged', False)
                
                if not played_cards or player not in deception_stats:
                    continue
                
                # 增加总出牌次数
                deception_stats[player]['total_plays'] += 1
                
                # 检查出牌中目标牌和万能牌的数量
                target_cards_count = sum(1 for card in played_cards if card == target_card)
                joker_cards_count = sum(1 for card in played_cards if card == "Joker")
                
                # 判断出牌类型
                is_honest = (target_cards_count + joker_cards_count) == len(played_cards)
                is_pure_bluff = (target_cards_count + joker_cards_count) == 0
                is_half_bluff = not is_honest and not is_pure_bluff
                
                # 更新统计
                if is_honest:
                    deception_stats[player]['honest_plays'] += 1
                else:
                    deception_stats[player]['deceptive_plays'] += 1
                    
                    if is_pure_bluff:
                        deception_stats[player]['pure_bluffs'] += 1
                    elif is_half_bluff:
                        deception_stats[player]['half_bluffs'] += 1
                    
                    # 欺骗是否成功（没被质疑）
                    if not was_challenged:
                        deception_stats[player]['successful_deceptions'] += 1
                
                # 记录被质疑次数
                if was_challenged:
                    deception_stats[player]['challenged_plays'] += 1
                
                # 记录万能牌使用情况
                if joker_cards_count > 0:
                    deception_stats[player]['plays_with_joker'] += 1
                    deception_stats[player]['joker_count'] += joker_cards_count
    
    # 计算比率
    metrics = {}
    for player, stats in deception_stats.items():
        metrics[player] = {}
        
        total_plays = stats['total_plays']
        if total_plays > 0:
            # 诚实出牌率
            metrics[player]['honest_play_rate'] = stats['honest_plays'] / total_plays
            
            # 欺骗率
            metrics[player]['deception_rate'] = stats['deceptive_plays'] / total_plays
            
            # 纯虚张声势率
            metrics[player]['pure_bluff_rate'] = stats['pure_bluffs'] / total_plays
            
            # 半虚张声势率
            metrics[player]['half_bluff_rate'] = stats['half_bluffs'] / total_plays
            
            # 被质疑率
            metrics[player]['challenged_rate'] = stats['challenged_plays'] / total_plays
            
            # 万能牌使用率（使用万能牌的出牌次数占总出牌次数的比例）
            metrics[player]['joker_usage_rate'] = stats['plays_with_joker'] / total_plays
        else:
            # 如果没有出牌记录，设为0
            metrics[player]['honest_play_rate'] = 0
            metrics[player]['deception_rate'] = 0
            metrics[player]['pure_bluff_rate'] = 0
            metrics[player]['half_bluff_rate'] = 0
            metrics[player]['challenged_rate'] = 0
            metrics[player]['joker_usage_rate'] = 0
        
        # 欺骗成功率（欺骗且没被质疑的次数占总欺骗次数的比例）
        if stats['deceptive_plays'] > 0:
            metrics[player]['deception_success_rate'] = stats['successful_deceptions'] / stats['deceptive_plays']
        else:
            metrics[player]['deception_success_rate'] = 0
    
    return metrics

# ==================== 存活指标 ====================

def analyze_survival_metrics(game_records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    分析与存活相关的指标
    
    Args:
        game_records: 游戏记录列表
        
    Returns:
        玩家存活指标统计，包括：
        - average_survival_points: 平均存活积分
        - survival_rate: 存活率 (存活到游戏结束的比例)
        - average_survival_rank: 平均存活排名 (1是最后被淘汰的)
        - early_elimination_rate: 提前淘汰率 (第一个被淘汰的比例)
        - round_survival_rate: 回合存活率 (存活的回合数/总回合数)
    """
    # 初始化统计数据
    survival_stats = defaultdict(lambda: {
        'games_played': 0,              # 参与的游戏数
        'total_survival_points': 0,     # 存活积分总和
        'survived_to_end': 0,           # 存活到游戏结束次数
        'total_survival_rank': 0,       # 存活排名总和
        'first_eliminated': 0,          # 第一个被淘汰次数
        'total_rounds_survived': 0,     # 存活回合总数
        'total_game_rounds': 0,         # 游戏总回合数
    })
    
    # 遍历每个游戏
    for game in game_records:
        player_names = game.get('player_names', [])
        winner = game.get('winner')
        total_rounds = len(game.get('rounds', []))
        
        # 记录游戏参与情况
        for player in player_names:
            survival_stats[player]['games_played'] += 1
            survival_stats[player]['total_game_rounds'] += total_rounds
        
        # 记录胜利者 (胜利者即存活到最后的玩家)
        if winner and winner in survival_stats:
            survival_stats[winner]['survived_to_end'] += 1
            
            # 赢家排名为玩家数
            winner_rank = len(player_names)
            survival_stats[winner]['total_survival_rank'] += winner_rank
            
            # 赢家存活积分 = 玩家数 * 10 + 50(胜利奖励)
            winner_points = winner_rank * 10 + 50
            survival_stats[winner]['total_survival_points'] += winner_points
        
        # 根据游戏轮次重建淘汰顺序
        # 由于游戏记录可能没有明确的淘汰标记，我们根据玩家在轮次中的出现情况来推断
        player_last_seen_round = {}
        for round_idx, round_data in enumerate(game.get('rounds', [])):
            # 获取本轮参与的玩家
            round_players = set()
            for play in round_data.get('play_history', []):
                player = play.get('player_name')
                if player:
                    round_players.add(player)
                    player_last_seen_round[player] = round_idx
        
        # 根据玩家最后出现的回合计算淘汰顺序
        # 不包括胜利者，他是特殊情况
        eliminated_players = []
        for player in player_names:
            if player != winner and player in player_last_seen_round:
                eliminated_players.append((player, player_last_seen_round[player]))
        
        # 按照最后出现的回合排序
        eliminated_players.sort(key=lambda x: x[1])
        
        # 记录第一个被淘汰的玩家
        if eliminated_players and eliminated_players[0][0] in survival_stats:
            survival_stats[eliminated_players[0][0]]['first_eliminated'] += 1
        
        # 计算淘汰排名和积分
        for rank, (player, last_round) in enumerate(eliminated_players):
            # 排名从1开始，1表示最早被淘汰的玩家
            elim_rank = rank + 1
            reversed_rank = len(eliminated_players) - rank  # 反转排名，值越大越好
            
            # 记录排名
            survival_stats[player]['total_survival_rank'] += reversed_rank
            
            # 存活积分 = 存活排名 * 10
            survival_points = reversed_rank * 10
            survival_stats[player]['total_survival_points'] += survival_points
            
            # 记录存活的回合数
            survival_stats[player]['total_rounds_survived'] += (last_round + 1)  # +1因为回合从0开始
        
        # 胜利者存活了所有回合
        if winner in survival_stats:
            survival_stats[winner]['total_rounds_survived'] += total_rounds
    
    # 计算比率和平均值
    metrics = {}
    for player, stats in survival_stats.items():
        metrics[player] = {}
        
        games_played = stats['games_played']
        if games_played > 0:
            # 平均存活积分
            metrics[player]['average_survival_points'] = stats['total_survival_points'] / games_played
            
            # 存活率（存活到游戏结束的比例）
            metrics[player]['survival_rate'] = stats['survived_to_end'] / games_played
            
            # 平均存活排名
            metrics[player]['average_survival_rank'] = stats['total_survival_rank'] / games_played
            
            # 提前淘汰率（第一个被淘汰的比例）
            metrics[player]['early_elimination_rate'] = stats['first_eliminated'] / games_played
            
            # 回合存活率（存活的回合数/总回合数）
            if stats['total_game_rounds'] > 0:
                metrics[player]['round_survival_rate'] = stats['total_rounds_survived'] / stats['total_game_rounds']
            else:
                metrics[player]['round_survival_rate'] = 0
        else:
            # 如果没有参与游戏，设为0
            metrics[player]['average_survival_points'] = 0
            metrics[player]['survival_rate'] = 0
            metrics[player]['average_survival_rank'] = 0
            metrics[player]['early_elimination_rate'] = 0
            metrics[player]['round_survival_rate'] = 0
    
    return metrics

# ==================== 决策质量指标 ====================

def analyze_decision_quality_metrics(game_records: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    分析决策质量相关指标
    
    Args:
        game_records: 游戏记录列表
        
    Returns:
        玩家决策质量指标统计，包括：
        - challenge_precision: 质疑精确度 (质疑成功次数/质疑总次数)
        - challenge_recall: 质疑召回率 (发现的欺骗次数/总欺骗次数)
        - challenge_decision_quality: 质疑决策质量 (2*precision*recall/(precision+recall))
        - optimal_play_ratio: 最优出牌比例 (计算手牌与目标牌重合时是否最优使用)
        - card_efficiency: 用牌效率 (平均每次出牌的牌数)
    """
    # 初始化统计数据
    decision_stats = defaultdict(lambda: {
        'challenges_made': 0,           # 发起的质疑次数
        'challenges_success': 0,        # 质疑成功次数
        'total_deceptions_witnessed': 0, # 目睹的总欺骗次数
        'deceptions_challenged': 0,     # 质疑的欺骗次数
        'optimal_plays': 0,             # 最优出牌次数
        'suboptimal_plays': 0,          # 次优出牌次数
        'total_cards_played': 0,        # 总出牌数量
        'play_count': 0,                # 出牌次数
    })
    
    for game in game_records:
        player_names = game.get('player_names', [])
        
        # 分析每一轮
        for round_data in game.get('rounds', []):
            target_card = round_data.get('target_card')
            play_history = round_data.get('play_history', [])
            
            # 分析每一次出牌
            for i, play in enumerate(play_history):
                player = play.get('player_name')
                played_cards = play.get('played_cards', [])
                
                # 确定下一个玩家（挑战者）
                next_player = None
                if i + 1 < len(play_history):
                    next_player = play_history[i + 1].get('player_name')
                
                if not played_cards or player not in player_names:
                    continue
                
                # 更新出牌统计
                decision_stats[player]['total_cards_played'] += len(played_cards)
                decision_stats[player]['play_count'] += 1
                
                # 检查是否为最优出牌 (如果有目标牌，是否全部使用)
                hand = play.get('hand', [])
                if hand:  # 只有当记录了手牌时才进行分析
                    target_cards_in_hand = sum(1 for card in hand if card == target_card)
                    target_cards_played = sum(1 for card in played_cards if card == target_card)
                    
                    # 如果打出的目标牌数量等于手中所有目标牌，则为最优出牌
                    if target_cards_played == target_cards_in_hand:
                        decision_stats[player]['optimal_plays'] += 1
                    else:
                        decision_stats[player]['suboptimal_plays'] += 1
                
                # 分析质疑决策质量
                was_challenged = play.get('was_challenged', False)
                
                # 检查是否是欺骗行为（出牌中包含非目标牌）
                target_cards_count = sum(1 for card in played_cards if card == target_card)
                joker_cards_count = sum(1 for card in played_cards if card == "Joker")
                is_deception = (target_cards_count + joker_cards_count) < len(played_cards)
                
                # 如果下一个玩家选择质疑，但记录中没有was_challenged字段，
                # 我们可以通过检查是否有challenge_result字段来判断
                if not was_challenged and next_player and 'challenge_result' in play:
                    was_challenged = True
                
                if next_player and is_deception:
                    # 目睹欺骗次数增加
                    decision_stats[next_player]['total_deceptions_witnessed'] += 1
                    
                    # 如果欺骗被质疑，记录质疑的欺骗次数
                    if was_challenged:
                        decision_stats[next_player]['deceptions_challenged'] += 1
                
                # 记录质疑情况
                if was_challenged and next_player:
                    decision_stats[next_player]['challenges_made'] += 1
                    
                    # 尝试确定质疑结果
                    challenge_result = play.get('challenge_result')
                    
                    # 如果没有明确的challenge_result，根据是否欺骗来判断质疑成功与否
                    if challenge_result is None:
                        challenge_result = is_deception
                    
                    if challenge_result:  # 质疑成功（证明对方欺骗）
                        decision_stats[next_player]['challenges_success'] += 1
    
    # 计算指标
    metrics = {}
    for player, stats in decision_stats.items():
        metrics[player] = {}
        
        # 质疑精确度 (质疑成功次数/质疑总次数)
        if stats['challenges_made'] > 0:
            metrics[player]['challenge_precision'] = stats['challenges_success'] / stats['challenges_made']
        else:
            metrics[player]['challenge_precision'] = 0
        
        # 质疑召回率 (发现的欺骗次数/总欺骗次数)
        if stats['total_deceptions_witnessed'] > 0:
            metrics[player]['challenge_recall'] = stats['deceptions_challenged'] / stats['total_deceptions_witnessed']
        else:
            metrics[player]['challenge_recall'] = 0
        
        # 质疑决策质量 (F1分数)
        precision = metrics[player]['challenge_precision']
        recall = metrics[player]['challenge_recall']
        if precision + recall > 0:
            metrics[player]['challenge_decision_quality'] = 2 * precision * recall / (precision + recall)
        else:
            metrics[player]['challenge_decision_quality'] = 0
        
        # 最优出牌比例
        total_plays_analyzed = stats['optimal_plays'] + stats['suboptimal_plays']
        if total_plays_analyzed > 0:
            metrics[player]['optimal_play_ratio'] = stats['optimal_plays'] / total_plays_analyzed
        else:
            metrics[player]['optimal_play_ratio'] = 0
        
        # 用牌效率 (平均每次出牌的牌数)
        if stats['play_count'] > 0:
            metrics[player]['card_efficiency'] = stats['total_cards_played'] / stats['play_count']
        else:
            metrics[player]['card_efficiency'] = 0
    
    return metrics

# ==================== 综合指标计算 ====================

def calculate_overall_metrics(game_records: List[Dict], output_dir: str = None) -> Dict[str, Dict[str, float]]:
    """
    计算综合性能指标
    
    Args:
        game_records: 游戏记录列表
        output_dir: 输出目录，如果指定，将保存指标CSV文件
        
    Returns:
        所有玩家的综合指标
    """
    # 获取所有类别的指标
    deception_metrics = analyze_deception_metrics(game_records)
    survival_metrics = analyze_survival_metrics(game_records)
    decision_quality_metrics = analyze_decision_quality_metrics(game_records)
    
    # 计算胜率
    win_stats = defaultdict(int)
    games_played = defaultdict(int)
    
    for game in game_records:
        player_names = game.get('player_names', [])
        winner = game.get('winner')
        
        for player in player_names:
            games_played[player] += 1
        
        if winner:
            win_stats[winner] += 1
    
    # 合并所有指标
    all_metrics = {}
    players = set(deception_metrics.keys()) | set(survival_metrics.keys()) | set(decision_quality_metrics.keys())
    
    for player in players:
        all_metrics[player] = {}
        
        # 添加胜率
        all_metrics[player]['win_rate'] = win_stats[player] / games_played[player] if games_played[player] > 0 else 0
        
        # 合并欺骗指标
        if player in deception_metrics:
            for key, value in deception_metrics[player].items():
                all_metrics[player][f'deception_{key}'] = value
        
        # 合并存活指标
        if player in survival_metrics:
            for key, value in survival_metrics[player].items():
                all_metrics[player][f'survival_{key}'] = value
        
        # 合并决策质量指标
        if player in decision_quality_metrics:
            for key, value in decision_quality_metrics[player].items():
                all_metrics[player][f'decision_{key}'] = value
    
    # 保存为CSV文件
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 转换为DataFrame便于保存
        df_data = []
        for player, metrics in all_metrics.items():
            player_data = {'player': player}
            player_data.update(metrics)
            df_data.append(player_data)
        
        df = pd.DataFrame(df_data)
        
        # 保存CSV
        csv_path = os.path.join(output_dir, 'player_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"已将玩家指标保存至 {csv_path}")
    
    return all_metrics

# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算骗子酒馆游戏指标')
    parser.add_argument('--input', type=str, default='game_records',
                        help='游戏记录文件夹路径 (默认: game_records)')
    parser.add_argument('--output', type=str, default='metrics_output',
                        help='输出文件夹路径 (默认: metrics_output)')
    args = parser.parse_args()
    
    print(f"从 {args.input} 加载游戏记录...")
    game_records = load_game_records(args.input)
    
    if not game_records:
        print("未找到游戏记录，退出程序")
        return
    
    print("计算综合指标...")
    metrics = calculate_overall_metrics(game_records, args.output)
    
    # 打印每个玩家的部分关键指标
    print("\n===== 玩家关键指标汇总 =====")
    for player, player_metrics in metrics.items():
        print(f"\n{player}:")
        print(f"胜率: {player_metrics['win_rate']:.2f}")
        print(f"诚实出牌率: {player_metrics.get('deception_honest_play_rate', 0):.2f}")
        print(f"欺骗成功率: {player_metrics.get('deception_deception_success_rate', 0):.2f}")
        print(f"平均存活积分: {player_metrics.get('survival_average_survival_points', 0):.2f}")
        print(f"质疑决策质量: {player_metrics.get('decision_challenge_decision_quality', 0):.2f}")
    
    print("\n指标计算完成")

if __name__ == "__main__":
    main() 