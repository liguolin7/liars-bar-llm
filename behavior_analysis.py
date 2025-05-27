#!/usr/bin/env python3
"""
模型行为分析工具 - 骗子酒馆多智能体项目

此模块提供了分析不同LLM模型在骗子酒馆游戏中行为模式的工具。
主要功能包括：
1. 行为模式聚类：识别不同模型的行为模式和策略偏好
2. 策略适应性分析：追踪模型在游戏中的策略变化
3. 对手特异性策略分析：分析模型对不同对手的策略调整
4. 反馈学习能力评估：评估模型从游戏经验中学习的能力
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import math
import re
from decision_tracker import DecisionTracker
from PIL import Image, ImageDraw, ImageFont

# 配置中文字体支持
def setup_chinese_font():
    """设置支持中文显示的字体"""
    global CHINESE_FONT
    
    # 首先尝试使用系统自带的中文字体
    chinese_fonts = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC', 'Heiti SC', 'Noto Sans CJK SC', 'Source Han Sans CN']
    
    # 检查系统中存在的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 尝试设置中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.family'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            
            # 为Seaborn设置同样的字体
            sns.set(font=font, font_scale=1.0)
            
            # 保存找到的中文字体名称
            CHINESE_FONT = font
            
            print(f"已设置中文字体: {font}")
            return True
    
    # 如果找不到中文字体，尝试使用Mac默认字体
    if 'macOS' in plt.matplotlib.get_backend() or os.name == 'posix':
        try:
            plt.rcParams['font.family'] = ['sans-serif']
            mac_fonts = ['SimHei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS']
            plt.rcParams['font.sans-serif'] = mac_fonts
            plt.rcParams['axes.unicode_minus'] = False
            
            # 为Seaborn设置同样的字体
            sns.set(font_scale=1.0)
            sns.set_style("whitegrid", {'font.sans-serif': mac_fonts})
            
            # 使用第一个可能存在的字体
            for font in mac_fonts:
                if font in available_fonts:
                    CHINESE_FONT = font
                    break
            else:
                CHINESE_FONT = 'sans-serif'
                
            print(f"已设置Mac默认中文字体: {CHINESE_FONT}")
            return True
        except:
            pass
    
    # 所有尝试都失败，设置为默认字体
    CHINESE_FONT = 'sans-serif'
    print("警告: 未找到支持中文的字体，图表中的中文可能无法正确显示")
    return False

# 全局变量保存检测到的中文字体
CHINESE_FONT = 'sans-serif'

# 在模块初始化时设置中文字体
setup_chinese_font()

class BehaviorAnalyzer:
    """
    模型行为分析器，用于分析和可视化LLM在游戏中的行为模式
    """
    
    def __init__(self, output_dir: str = "behavior_analysis"):
        """
        初始化行为分析器
        
        Args:
            output_dir: 输出结果的目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 存储分析结果
        self.behavior_features = {}
        self.player_clusters = {}
        self.strategy_evolution = {}
        self.opponent_specific_strategies = {}
        self.game_records = []
        
    def load_game_records(self, folder_path: str) -> None:
        """
        加载游戏记录
        
        Args:
            folder_path: 游戏记录文件夹路径
        """
        self.game_records = []
        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)
                    self.game_records.append(game_data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"成功加载 {len(self.game_records)} 个游戏记录")
    
    def extract_behavior_features(self) -> Dict[str, Dict[str, float]]:
        """
        提取每个玩家的行为特征向量
        
        Returns:
            Dict[str, Dict[str, float]]: 玩家行为特征映射
        """
        # 初始化特征统计
        players_stats = defaultdict(lambda: {
            # 欺骗行为特征
            'total_plays': 0,
            'honest_plays': 0,
            'pure_bluffs': 0,
            'half_bluffs': 0,
            'successful_deceptions': 0,
            
            # 质疑行为特征
            'total_challenge_opportunities': 0,
            'challenges_initiated': 0,
            'successful_challenges': 0,
            'unnecessary_challenges': 0,
            'missed_challenge_opportunities': 0,
            
            # 风险评估特征
            'risky_plays': 0,
            'conservative_plays': 0,
            'risky_challenges': 0,
            'conservative_passes': 0,
            
            # 策略适应性特征
            'strategy_changes': 0,
            'consistent_behavior_streaks': 0,
            
            # 情绪反应特征
            'emotional_responses': 0,
            'rational_decisions': 0,
            
            # 对手学习特征
            'opponent_pattern_recognition': 0
        })
        
        # 记录玩家在每局游戏中的行为变化
        player_game_behaviors = defaultdict(lambda: defaultdict(list))
        
        # 遍历游戏记录提取特征
        for game in self.game_records:
            player_names = game.get('player_names', [])
            
            # 记录本局游戏每个玩家的策略变化
            game_player_behaviors = {player: [] for player in player_names}
            
            # 分析每一轮
            for round_data in game.get('rounds', []):
                target_card = round_data.get('target_card')
                
                # 跟踪上一次行为，用于检测策略变化
                prev_behaviors = {}
                
                for play in round_data.get('play_history', []):
                    player = play.get('player_name')
                    next_player = play.get('next_player')
                    played_cards = play.get('played_cards', [])
                    was_challenged = play.get('was_challenged', False)
                    challenge_result = play.get('challenge_result')
                    play_reason = play.get('play_reason', '')
                    challenge_reason = play.get('challenge_reason', '')
                    play_thinking = play.get('play_thinking', '')
                    challenge_thinking = play.get('challenge_thinking', '')
                    
                    if not played_cards or player not in player_names:
                        continue
                    
                    # ========== 提取出牌玩家的行为特征 ==========
                    
                    players_stats[player]['total_plays'] += 1
                    
                    # 判断出牌类型
                    target_cards_count = sum(1 for card in played_cards if card == target_card)
                    joker_cards_count = sum(1 for card in played_cards if card == "Joker")
                    is_honest = (target_cards_count + joker_cards_count) == len(played_cards)
                    is_pure_bluff = (target_cards_count + joker_cards_count) == 0
                    is_half_bluff = not is_honest and not is_pure_bluff
                    
                    # 更新基础统计
                    if is_honest:
                        players_stats[player]['honest_plays'] += 1
                        game_player_behaviors[player].append('honest')
                    elif is_pure_bluff:
                        players_stats[player]['pure_bluffs'] += 1
                        game_player_behaviors[player].append('pure_bluff')
                    elif is_half_bluff:
                        players_stats[player]['half_bluffs'] += 1
                        game_player_behaviors[player].append('half_bluff')
                    
                    # 检查欺骗是否成功
                    if not is_honest and not was_challenged:
                        players_stats[player]['successful_deceptions'] += 1
                    
                    # 检测风险行为 - 基于打出的牌数和手中剩余牌
                    remaining_cards = play.get('remaining_cards', [])
                    played_ratio = len(played_cards) / (len(played_cards) + len(remaining_cards)) if played_cards or remaining_cards else 0
                    
                    # 高风险行为：虚张声势且打出比例高
                    if not is_honest and played_ratio > 0.5:
                        players_stats[player]['risky_plays'] += 1
                    # 保守行为：诚实且打出比例低
                    elif is_honest and played_ratio < 0.3:
                        players_stats[player]['conservative_plays'] += 1
                    
                    # 检测情绪反应 (基于推理文本)
                    if play_thinking:
                        emotion_words = ['担心', '害怕', '焦虑', '紧张', '激动', '兴奋', '愤怒', '沮丧', '困惑']
                        has_emotion = any(word in play_thinking for word in emotion_words)
                        if has_emotion:
                            players_stats[player]['emotional_responses'] += 1
                        else:
                            players_stats[player]['rational_decisions'] += 1
                    
                    # ========== 提取质疑者行为特征 ==========
                    
                    if next_player in player_names:
                        players_stats[next_player]['total_challenge_opportunities'] += 1
                        
                        # 是否发起质疑
                        if was_challenged:
                            players_stats[next_player]['challenges_initiated'] += 1
                            
                            # 判断质疑是否成功
                            if challenge_result:
                                players_stats[next_player]['successful_challenges'] += 1
                            else:
                                players_stats[next_player]['unnecessary_challenges'] += 1
                            
                            # 根据质疑结果判断风险行为
                            if challenge_result:
                                # 正确的质疑是理性行为
                                players_stats[next_player]['rational_decisions'] += 1
                            else:
                                # 错误的质疑是风险行为
                                players_stats[next_player]['risky_challenges'] += 1
                                
                            game_player_behaviors[next_player].append('challenge')
                        else:
                            # 没有质疑
                            game_player_behaviors[next_player].append('pass')
                            
                            # 检查是否错过了质疑机会
                            if not is_honest:
                                players_stats[next_player]['missed_challenge_opportunities'] += 1
                                players_stats[next_player]['conservative_passes'] += 1
                            else:
                                # 正确地不质疑是理性行为
                                players_stats[next_player]['rational_decisions'] += 1
                                
                    # 检查策略变化
                    if player in prev_behaviors and prev_behaviors[player] != game_player_behaviors[player][-1]:
                        players_stats[player]['strategy_changes'] += 1
                    
                    # 更新上一次行为
                    prev_behaviors[player] = game_player_behaviors[player][-1]
                    if next_player in game_player_behaviors and game_player_behaviors[next_player]:
                        prev_behaviors[next_player] = game_player_behaviors[next_player][-1]
            
            # 记录每个玩家在本局游戏中的行为序列
            for player, behaviors in game_player_behaviors.items():
                player_game_behaviors[player][game.get('game_id', 'unknown')].extend(behaviors)
                
                # 计算连续相同行为的最长序列（一致性）
                if behaviors:
                    max_streak = 1
                    current_streak = 1
                    for i in range(1, len(behaviors)):
                        if behaviors[i] == behaviors[i-1]:
                            current_streak += 1
                        else:
                            max_streak = max(max_streak, current_streak)
                            current_streak = 1
                    max_streak = max(max_streak, current_streak)
                    players_stats[player]['consistent_behavior_streaks'] = max(
                        players_stats[player]['consistent_behavior_streaks'], 
                        max_streak
                    )
        
        # 计算最终特征
        behavior_features = {}
        for player, stats in players_stats.items():
            if stats['total_plays'] == 0:
                continue
                
            features = {}
            
            # 欺骗倾向特征
            total_plays = stats['total_plays']
            features['honest_ratio'] = stats['honest_plays'] / total_plays
            features['pure_bluff_ratio'] = stats['pure_bluffs'] / total_plays
            features['half_bluff_ratio'] = stats['half_bluffs'] / total_plays
            features['deception_success_rate'] = (stats['successful_deceptions'] / 
                                                (stats['pure_bluffs'] + stats['half_bluffs'])) if (stats['pure_bluffs'] + stats['half_bluffs']) > 0 else 0
            
            # 质疑倾向特征
            if stats['total_challenge_opportunities'] > 0:
                features['challenge_rate'] = stats['challenges_initiated'] / stats['total_challenge_opportunities']
                features['challenge_precision'] = (stats['successful_challenges'] / 
                                                stats['challenges_initiated']) if stats['challenges_initiated'] > 0 else 0
                features['challenge_opportunity_utilization'] = 1 - (stats['missed_challenge_opportunities'] / 
                                                                    stats['total_challenge_opportunities'])
            else:
                features['challenge_rate'] = 0
                features['challenge_precision'] = 0
                features['challenge_opportunity_utilization'] = 0
            
            # 风险态度特征
            features['risk_taking_ratio'] = (stats['risky_plays'] + stats['risky_challenges']) / total_plays
            features['conservatism_ratio'] = (stats['conservative_plays'] + stats['conservative_passes']) / total_plays
            
            # 策略适应性特征
            features['strategy_adaptation_rate'] = stats['strategy_changes'] / (total_plays - 1) if total_plays > 1 else 0
            features['behavioral_consistency'] = stats['consistent_behavior_streaks'] / total_plays
            
            # 决策风格特征
            total_decisions = stats['emotional_responses'] + stats['rational_decisions']
            features['emotional_decision_ratio'] = stats['emotional_responses'] / total_decisions if total_decisions > 0 else 0
            features['rational_decision_ratio'] = stats['rational_decisions'] / total_decisions if total_decisions > 0 else 0
            
            behavior_features[player] = features
        
        self.behavior_features = behavior_features
        self.player_game_behaviors = player_game_behaviors
        return behavior_features
    
    def cluster_player_behaviors(self, n_clusters: int = 3) -> Dict[str, int]:
        """
        对玩家行为特征进行聚类，识别不同的策略模式
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            Dict[str, int]: 玩家到聚类ID的映射
        """
        if not self.behavior_features:
            print("请先提取行为特征")
            return {}
        
        # 准备数据
        players = list(self.behavior_features.keys())
        feature_names = list(self.behavior_features[players[0]].keys())
        
        # 创建特征矩阵
        feature_matrix = []
        for player in players:
            player_features = [self.behavior_features[player][feature] for feature in feature_names]
            feature_matrix.append(player_features)
        
        # 标准化特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # 创建玩家到聚类的映射
        player_clusters = {player: label for player, label in zip(players, cluster_labels)}
        
        # 存储聚类结果和中心点
        self.player_clusters = player_clusters
        self.cluster_centers = kmeans.cluster_centers_
        self.feature_names = feature_names
        self.scaled_features = scaled_features
        self.players = players
        
        return player_clusters
    
    def analyze_strategy_adaptation(self) -> Dict[str, Dict[str, Any]]:
        """
        分析玩家在游戏过程中的策略适应性
        
        Returns:
            Dict[str, Dict[str, Any]]: 每个玩家的策略适应性指标
        """
        if not hasattr(self, 'player_game_behaviors'):
            print("请先提取行为特征")
            return {}
        
        strategy_evolution = {}
        
        for player, game_behaviors in self.player_game_behaviors.items():
            player_stats = {
                'total_games': len(game_behaviors),
                'total_decisions': sum(len(behaviors) for behaviors in game_behaviors.values()),
                'strategy_changes_per_game': [],
                'strategy_consistency': [],
                'behavior_distributions': [],
                'early_vs_late_changes': []
            }
            
            # 分析每局游戏中的策略变化
            for game_id, behaviors in game_behaviors.items():
                if len(behaviors) < 2:
                    continue
                
                # 计算策略变化次数
                changes = sum(1 for i in range(1, len(behaviors)) if behaviors[i] != behaviors[i-1])
                player_stats['strategy_changes_per_game'].append(changes)
                
                # 计算策略一致性（相同行为的比例）
                behavior_counter = Counter(behaviors)
                most_common_behavior = behavior_counter.most_common(1)[0]
                consistency = most_common_behavior[1] / len(behaviors)
                player_stats['strategy_consistency'].append(consistency)
                
                # 记录行为分布
                behavior_dist = {b: count/len(behaviors) for b, count in behavior_counter.items()}
                player_stats['behavior_distributions'].append(behavior_dist)
                
                # 比较游戏前期和后期的策略变化
                mid_point = len(behaviors) // 2
                early_behaviors = behaviors[:mid_point]
                late_behaviors = behaviors[mid_point:]
                
                early_dist = Counter(early_behaviors)
                late_dist = Counter(late_behaviors)
                
                # 计算早期和晚期行为分布的差异
                all_behaviors = set(early_dist.keys()) | set(late_dist.keys())
                distribution_change = sum(
                    abs((early_dist.get(b, 0) / len(early_behaviors) if early_behaviors else 0) - 
                        (late_dist.get(b, 0) / len(late_behaviors) if late_behaviors else 0))
                    for b in all_behaviors
                ) / 2  # 除以2使范围在0-1之间
                
                player_stats['early_vs_late_changes'].append(distribution_change)
            
            # 计算汇总指标
            if player_stats['strategy_changes_per_game']:
                player_stats['avg_strategy_changes'] = np.mean(player_stats['strategy_changes_per_game'])
                player_stats['avg_strategy_consistency'] = np.mean(player_stats['strategy_consistency'])
                player_stats['avg_early_late_difference'] = np.mean(player_stats['early_vs_late_changes'])
                
                # 合并所有游戏的行为分布
                all_distributions = player_stats['behavior_distributions']
                merged_distribution = {}
                for dist in all_distributions:
                    for behavior, freq in dist.items():
                        if behavior in merged_distribution:
                            merged_distribution[behavior].append(freq)
                        else:
                            merged_distribution[behavior] = [freq]
                
                # 计算每种行为频率的标准差，作为行为稳定性的度量
                player_stats['behavior_stability'] = {
                    behavior: np.std(freqs) for behavior, freqs in merged_distribution.items()
                }
                
                # 计算总体行为稳定性得分（值越低表示越稳定）
                stability_scores = list(player_stats['behavior_stability'].values())
                player_stats['overall_stability_score'] = np.mean(stability_scores) if stability_scores else 0
                
                # 评估适应性分数 (0-10)
                adaptation_score = (
                    (player_stats['avg_strategy_changes'] / (player_stats['total_decisions'] / player_stats['total_games'])) * 5 +
                    player_stats['avg_early_late_difference'] * 5
                )
                player_stats['adaptation_score'] = min(10, adaptation_score)
            else:
                # 默认值
                player_stats['avg_strategy_changes'] = 0
                player_stats['avg_strategy_consistency'] = 0
                player_stats['avg_early_late_difference'] = 0
                player_stats['behavior_stability'] = {}
                player_stats['overall_stability_score'] = 0
                player_stats['adaptation_score'] = 0
            
            strategy_evolution[player] = player_stats
        
        self.strategy_evolution = strategy_evolution
        return strategy_evolution
    
    def analyze_opponent_specific_strategies(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        分析玩家对不同对手的特定策略调整
        
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: 玩家-对手-策略指标的嵌套字典
        """
        opponent_specific_strategies = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # 遍历所有游戏记录
        for game in self.game_records:
            player_names = game.get('player_names', [])
            
            if len(player_names) < 2:
                continue
            
            # 初始化该游戏中的玩家对手交互计数
            player_opponent_interactions = defaultdict(lambda: defaultdict(lambda: {
                'total_plays': 0,
                'honest_plays': 0,
                'bluff_plays': 0,
                'challenge_attempts': 0,
                'successful_challenges': 0
            }))
            
            # 分析每一轮的交互
            for round_data in game.get('rounds', []):
                target_card = round_data.get('target_card')
                
                for play in round_data.get('play_history', []):
                    player = play.get('player_name')
                    next_player = play.get('next_player')
                    played_cards = play.get('played_cards', [])
                    was_challenged = play.get('was_challenged', False)
                    challenge_result = play.get('challenge_result')
                    
                    if not played_cards or player not in player_names or next_player not in player_names:
                        continue
                    
                    # 分析出牌玩家对特定对手的策略
                    interactions = player_opponent_interactions[player][next_player]
                    interactions['total_plays'] += 1
                    
                    # 判断是否诚实出牌
                    target_cards_count = sum(1 for card in played_cards if card == target_card)
                    joker_cards_count = sum(1 for card in played_cards if card == "Joker")
                    is_honest = (target_cards_count + joker_cards_count) == len(played_cards)
                    
                    if is_honest:
                        interactions['honest_plays'] += 1
                    else:
                        interactions['bluff_plays'] += 1
                    
                    # 分析质疑玩家对特定对手的策略
                    challenge_interactions = player_opponent_interactions[next_player][player]
                    if was_challenged:
                        challenge_interactions['challenge_attempts'] += 1
                        if challenge_result:
                            challenge_interactions['successful_challenges'] += 1
            
            # 计算该游戏中的对手特异性策略指标
            for player, opponents in player_opponent_interactions.items():
                for opponent, interactions in opponents.items():
                    if interactions['total_plays'] > 0:
                        # 对该对手的诚实率
                        opponent_specific_strategies[player][opponent]['honesty_rate'] += (
                            interactions['honest_plays'] / interactions['total_plays']
                        )
                        # 对该对手的虚张声势率
                        opponent_specific_strategies[player][opponent]['bluff_rate'] += (
                            interactions['bluff_plays'] / interactions['total_plays']
                        )
                    
                    if interactions['challenge_attempts'] > 0:
                        # 对该对手的质疑成功率
                        opponent_specific_strategies[player][opponent]['challenge_success_rate'] += (
                            interactions['successful_challenges'] / interactions['challenge_attempts']
                        )
                    
                    # 记录交互次数，用于后续计算平均值
                    opponent_specific_strategies[player][opponent]['interaction_count'] += 1
        
        # 计算平均值
        for player, opponents in opponent_specific_strategies.items():
            for opponent, metrics in opponents.items():
                interaction_count = metrics['interaction_count']
                if interaction_count > 0:
                    for metric in ['honesty_rate', 'bluff_rate', 'challenge_success_rate']:
                        if metric in metrics:
                            metrics[metric] /= interaction_count
        
        # 计算差异性指标 - 衡量玩家对不同对手策略调整的程度
        for player, opponents in opponent_specific_strategies.items():
            # 计算玩家对所有对手的平均策略
            avg_metrics = defaultdict(float)
            metric_counts = defaultdict(int)
            
            for opponent, metrics in opponents.items():
                for metric, value in metrics.items():
                    if metric != 'interaction_count':
                        avg_metrics[metric] += value
                        metric_counts[metric] += 1
            
            for metric in avg_metrics:
                if metric_counts[metric] > 0:
                    avg_metrics[metric] /= metric_counts[metric]
            
            # 计算每个对手特异性策略相对于平均策略的偏差
            for opponent, metrics in opponents.items():
                deviation = 0
                count = 0
                
                for metric, value in metrics.items():
                    if metric != 'interaction_count' and metric in avg_metrics:
                        deviation += abs(value - avg_metrics[metric])
                        count += 1
                
                if count > 0:
                    metrics['strategy_deviation'] = deviation / count
                else:
                    metrics['strategy_deviation'] = 0
            
            # 计算玩家的总体对手适应性指标
            total_deviation = sum(metrics.get('strategy_deviation', 0) for metrics in opponents.values())
            opponent_count = sum(1 for metrics in opponents.values() if metrics.get('interaction_count', 0) > 0)
            
            if opponent_count > 0:
                # 添加到玩家的总体指标中
                opponent_specific_strategies[player]['__overall__']['opponent_adaptation_score'] = total_deviation / opponent_count
            else:
                opponent_specific_strategies[player]['__overall__']['opponent_adaptation_score'] = 0
        
        self.opponent_specific_strategies = dict(opponent_specific_strategies)
        return self.opponent_specific_strategies
    
    def visualize_behavior_clusters(self, output_file: str = "behavior_clusters.png") -> None:
        """
        可视化行为聚类结果
        
        Args:
            output_file: 输出文件路径
        """
        if not hasattr(self, 'scaled_features') or not hasattr(self, 'player_clusters'):
            print("请先运行聚类分析")
            return
        
        # 使用PCA降维以便于可视化
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.scaled_features)
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 获取唯一的聚类标签和对应的颜色
        unique_clusters = sorted(set(self.player_clusters.values()))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        # 绘制散点图，不同聚类用不同颜色
        for i, cluster in enumerate(unique_clusters):
            cluster_points = [j for j, (player, label) in enumerate(self.player_clusters.items()) if label == cluster]
            plt.scatter(
                reduced_features[cluster_points, 0],
                reduced_features[cluster_points, 1],
                s=100, 
                c=[colors[i]],
                marker='o',
                edgecolors='black',
                alpha=0.7,
                label=f'策略类型 {cluster+1}'
            )
        
        # 添加玩家名称标签
        for i, player in enumerate(self.players):
            plt.annotate(
                player,
                (reduced_features[i, 0], reduced_features[i, 1]),
                fontsize=10,
                ha='center',
                va='bottom'
            )
        
        # 添加解释文本
        explained_variance = pca.explained_variance_ratio_
        plt.title("玩家行为模式聚类分析", fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.xlabel(f"主成分1 (解释方差: {explained_variance[0]:.2%})", fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.ylabel(f"主成分2 (解释方差: {explained_variance[1]:.2%})", fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.legend(title="策略类型", fontsize=10, prop={'family': CHINESE_FONT})
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加聚类中心（如果使用K-means）
        if hasattr(self, 'cluster_centers'):
            # 将聚类中心投影到PCA空间
            centers_reduced = pca.transform(self.cluster_centers)
            plt.scatter(
                centers_reduced[:, 0],
                centers_reduced[:, 1],
                s=200,
                c='red',
                marker='X',
                edgecolors='black',
                label='聚类中心'
            )
        
        # 保存图像
        plt.tight_layout()
        file_path = os.path.join(self.output_dir, output_file)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"聚类可视化已保存至 {file_path}")
    
    def visualize_strategy_radar(self, output_prefix: str = "strategy_radar") -> None:
        """
        为每个聚类生成策略特征的雷达图
        
        Args:
            output_prefix: 输出文件名前缀
        """
        if not hasattr(self, 'player_clusters') or not hasattr(self, 'behavior_features'):
            print("请先运行聚类分析")
            return
        
        # 设置中文字体
        setup_chinese_font()
        
        # 为每个聚类分配策略类型名称
        strategy_types = {
            0: "谨慎型策略",  # ChatGPT，高诚实率，低虚张声势率
            1: "进攻型策略",  # Gemini，低诚实率，高纯虚张声势率
            2: "平衡型策略",  # DeepSeek，中等诚实率和风险承担
            3: "适应型策略"   # Claude，较高的成功率和灵活性
        }
        
        # 记录聚类及其包含的模型
        self.cluster_models = {}
        # 反转player_clusters字典，得到每个聚类包含的模型
        for player, cluster in self.player_clusters.items():
            if cluster not in self.cluster_models:
                self.cluster_models[cluster] = []
            self.cluster_models[cluster].append(player)
        
        # 绘制每个聚类的雷达图
        for cluster, features in self.behavior_features.items():
            # 筛选要显示的特征
            display_features = {
                '诚实率': features['honest_ratio'],
                '纯虚张声势率': features['pure_bluff_ratio'],
                '半虚张声势率': features['half_bluff_ratio'],
                '欺骗成功率': features['deception_success_rate'],
                '质疑准确率': features['challenge_precision'],
                '风险承担度': features['risk_taking_ratio'],
                '保守程度': features['conservatism_ratio'],
                '情绪化决策率': features['emotional_decision_ratio']
            }
            
            # 获取该聚类包含的模型名称
            included_models = self.cluster_models.get(cluster, [])
            models_str = "、".join(included_models) if included_models else f"聚类{cluster}"
            
            # 获取策略类型名称
            strategy_type = strategy_types.get(cluster, f"策略类型 {cluster}")
            
            # 特征和值
            features_list = list(display_features.keys())
            values = [display_features[f] for f in features_list]
            
            # 创建雷达图
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, polar=True)
            
            # 计算每个特征的角度
            angles = np.linspace(0, 2*np.pi, len(features_list), endpoint=False).tolist()
            
            # 闭合多边形
            values.append(values[0])
            angles.append(angles[0])
            features_list.append(features_list[0])
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # 设置特征标签
            ax.set_thetagrids(np.degrees(angles[:-1]), features_list[:-1])
            
            # 设置径向网格线和标签
            ax.set_rlabel_position(0)
            ax.set_rticks([0.2, 0.4, 0.6, 0.8])
            ax.set_rlim(0, 1)
            
            # 设置标题
            plt.title(f"{models_str}（{strategy_type}）", size=16, y=1.05)
            
            # 添加网格线
            ax.grid(True)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            output_file = os.path.join(self.output_dir, f"{output_prefix}_cluster_{cluster}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"策略类型 {cluster} 的雷达图已保存至 {output_file}")
    
    def visualize_strategy_evolution(self, output_file: str = "strategy_evolution.png") -> None:
        """
        可视化玩家在游戏过程中的策略演变
        
        Args:
            output_file: 输出文件路径
        """
        if not hasattr(self, 'strategy_evolution'):
            print("请先运行策略适应性分析")
            return
        
        # 创建一个多子图的图表
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # 准备数据
        players = list(self.strategy_evolution.keys())
        adaptation_scores = [self.strategy_evolution[p].get('adaptation_score', 0) for p in players]
        consistency_scores = [self.strategy_evolution[p].get('avg_strategy_consistency', 0) for p in players]
        stability_scores = [1 - self.strategy_evolution[p].get('overall_stability_score', 0) for p in players]  # 1-不稳定性=稳定性
        
        # 子图1：适应性分数条形图
        axs[0].bar(players, adaptation_scores, color='skyblue', alpha=0.7, edgecolor='black')
        axs[0].set_title('策略适应性分数', fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[0].set_xlabel('玩家', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[0].set_ylabel('适应性分数 (0-10)', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[0].tick_params(axis='x', rotation=45)
        axs[0].set_ylim(0, 10)
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 子图2：策略一致性条形图
        axs[1].bar(players, consistency_scores, color='lightgreen', alpha=0.7, edgecolor='black')
        axs[1].set_title('策略一致性', fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[1].set_xlabel('玩家', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[1].set_ylabel('一致性分数 (0-1)', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[1].tick_params(axis='x', rotation=45)
        axs[1].set_ylim(0, 1)
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 子图3：策略稳定性条形图
        axs[2].bar(players, stability_scores, color='lightcoral', alpha=0.7, edgecolor='black')
        axs[2].set_title('策略稳定性', fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[2].set_xlabel('玩家', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[2].set_ylabel('稳定性分数 (0-1)', fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        axs[2].tick_params(axis='x', rotation=45)
        axs[2].set_ylim(0, 1)
        axs[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        fig.suptitle('玩家策略适应性分析', fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)
        
        # 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_path = os.path.join(self.output_dir, output_file)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"策略演变可视化已保存至 {file_path}")
    
    def visualize_opponent_specific_strategies(self, output_file: str = "opponent_specific_strategies.png") -> None:
        """
        可视化玩家对不同对手的特异性策略
        
        Args:
            output_file: 输出文件路径
        """
        if not hasattr(self, 'opponent_specific_strategies'):
            print("请先运行对手特异性策略分析")
            return
        
        # 提取所有玩家
        players = [p for p in self.opponent_specific_strategies.keys() if p != '__overall__']
        
        if not players:
            print("没有足够的对手特异性策略数据")
            return
        
        # 创建热力图
        plt.figure(figsize=(12, 10))
        
        # 提取对手适应性分数
        adaptation_scores = []
        for player in players:
            if '__overall__' in self.opponent_specific_strategies[player]:
                score = self.opponent_specific_strategies[player]['__overall__'].get('opponent_adaptation_score', 0)
                adaptation_scores.append(score)
            else:
                adaptation_scores.append(0)
        
        # 创建对手特异性矩阵
        matrix_data = np.zeros((len(players), len(players)))
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players):
                if player1 != player2 and player2 in self.opponent_specific_strategies[player1]:
                    # 使用策略偏差作为矩阵值
                    matrix_data[i, j] = self.opponent_specific_strategies[player1][player2].get('strategy_deviation', 0)
        
        # 生成热力图
        sns.heatmap(
            matrix_data,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            xticklabels=players,
            yticklabels=players,
            cbar_kws={'label': '策略调整程度'},
            annot_kws={"size": 10, "weight": "bold"}
        )
        
        # 强制设置字体，确保中文正确显示
        plt.title("玩家对特定对手的策略调整程度", fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.xlabel("对手", fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.ylabel("玩家", fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        
        # 修改刻度标签的字体
        plt.xticks(fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.yticks(fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        
        # 保存图像
        plt.tight_layout()
        file_path = os.path.join(self.output_dir, output_file)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"对手特异性策略可视化已保存至 {file_path}")
        
        # 创建玩家的对手适应性得分条形图
        plt.figure(figsize=(12, 6))
        plt.bar(players, adaptation_scores, color='purple', alpha=0.7, edgecolor='black')
        plt.axhline(y=np.mean(adaptation_scores), color='red', linestyle='--', label='平均值')
        
        # 强制设置字体，确保中文正确显示
        plt.title("玩家对不同对手的策略适应能力", fontsize=16, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.xlabel("玩家", fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.ylabel("对手适应性得分", fontsize=14, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.xticks(rotation=45, fontsize=12, fontweight='bold', fontfamily=CHINESE_FONT)
        plt.legend(prop={'family': CHINESE_FONT, 'size': 12, 'weight': 'bold'})
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存图像
        plt.tight_layout()
        adaptation_file_path = os.path.join(self.output_dir, f"opponent_adaptation_scores.png")
        plt.savefig(adaptation_file_path, dpi=300)
        plt.close()
        print(f"对手适应性得分可视化已保存至 {adaptation_file_path}")
    
    def combine_radar_images(self, output_file: str = "combined_strategy_radar.png") -> None:
        """
        将多个策略雷达图合成为一张2x2格式的图片
        
        Args:
            output_file: 输出文件路径
        """
        # 检查是否已经生成了策略雷达图
        if not hasattr(self, 'player_clusters'):
            print("请先运行聚类分析并生成策略雷达图")
            return
        
        # 查找所有以strategy_radar_cluster_开头的文件
        radar_files = []
        for filename in os.listdir(self.output_dir):
            if filename.startswith("strategy_radar_cluster_") and filename.endswith(".png"):
                radar_files.append(filename)
        
        # 确保有足够的文件
        if len(radar_files) < 4:
            print(f"警告: 没有足够的策略雷达图，仅找到{len(radar_files)}个")
            return
        
        # 为每个聚类分配策略类型名称
        strategy_types = {
            "ChatGPT": "谨慎型策略",  # 高诚实率，低虚张声势率
            "Gemini": "进攻型策略",   # 低诚实率，高纯虚张声势率
            "DeepSeek": "平衡型策略", # 中等诚实率和风险承担
            "Claude": "适应型策略"    # 较高的成功率和灵活性
        }
        
        # 优先选择这四个模型的雷达图
        model_priorities = ["ChatGPT", "Gemini", "DeepSeek", "Claude"]
        selected_files = []
        
        # 按照优先级顺序选择文件
        for model in model_priorities:
            model_file = f"strategy_radar_cluster_{model}.png"
            if model_file in radar_files:
                selected_files.append(model_file)
        
        # 如果没有找到足够的模型文件，添加其他找到的文件直到达到4个
        for file in radar_files:
            if file not in selected_files and len(selected_files) < 4:
                selected_files.append(file)
        
        # 仍然限制为4个文件
        selected_files = selected_files[:4]
        
        if len(selected_files) < 4:
            print(f"警告: 找不到4个策略雷达图，只找到{len(selected_files)}个")
            print(f"找到的文件: {selected_files}")
            return
            
        print(f"将合成以下策略雷达图: {selected_files}")
        
        # 准备图片路径
        image_paths = [os.path.join(self.output_dir, file) for file in selected_files]
        
        # 打开所有图片
        images = [Image.open(path) for path in image_paths]
        
        # 获取每张图片的尺寸
        widths, heights = zip(*(img.size for img in images))
        
        # 确定输出图片的尺寸
        title_height = 100  # 进一步增加主标题空间
        max_width = max(widths)
        max_height = max(heights)
        total_width = max_width * 2
        total_height = max_height * 2 + title_height
        
        # 创建一个新的空白图片
        combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        
        # 尝试加载中文字体，如果不可用则使用默认字体
        try:
            # 尝试常见的中文字体，如果系统中有的话
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',  # macOS
                '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
                'C:\\Windows\\Fonts\\msyh.ttf',  # Windows
                'C:\\Windows\\Fonts\\simhei.ttf',  # Windows
            ]
            
            font = None
            title_font = None
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    # 极大幅度增大字体大小
                    title_font = ImageFont.truetype(font_path, 80)  # 主标题字体极大幅度增大
                    font = ImageFont.truetype(font_path, 64)  # 子标题字体极大幅度增大
                    break
                
            if font is None:
                font = ImageFont.load_default()
                title_font = font
            
        except Exception:
            font = ImageFont.load_default()
            title_font = font
        
        draw = ImageDraw.Draw(combined_img)
        
        # 放置每张图片
        positions = [
            (0, title_height),  # 左上
            (max_width, title_height),  # 右上
            (0, max_height + title_height),  # 左下
            (max_width, max_height + title_height)  # 右下
        ]
        
        # 子图标题 - 从文件名中提取模型名称
        titles = []
        for file in selected_files:
            # 从文件名中提取模型名称
            model_name = file.replace("strategy_radar_cluster_", "").replace(".png", "")
            strategy_type = strategy_types.get(model_name, f"策略类型 {model_name}")
            titles.append(f"{model_name}（{strategy_type}）")
        
        # 子标题背景矩形的内边距
        padding = 20  # 进一步增加内边距使背景更大
        
        for i, img in enumerate(images):
            combined_img.paste(img, positions[i])
            
            # 添加子图标题到右上角
            # 计算标题宽度，以确定背景矩形的大小
            title_bbox = draw.textbbox((0, 0), titles[i], font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height_text = title_bbox[3] - title_bbox[1]
            
            # 计算标题位置（右上角）
            x = positions[i][0] + max_width - title_width - padding * 2  # 右边界减去文本宽度和内边距
            y = positions[i][1] + padding  # 顶部加一点内边距
            
            # 创建半透明背景
            rect_x0 = x - padding
            rect_y0 = y - padding
            rect_x1 = x + title_width + padding
            rect_y1 = y + title_height_text + padding
            
            # 绘制半透明白色背景，增加边框宽度
            border_width = 3  # 增加边框宽度，使其更加显眼
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], 
                          fill=(255, 255, 255, 230), outline=None)  # 先画填充
            
            # 再画边框（更粗）
            for bw in range(border_width):
                draw.rectangle([rect_x0-bw, rect_y0-bw, rect_x1+bw, rect_y1+bw], 
                              fill=None, outline=(0, 0, 0))
            
            # 绘制子标题
            draw.text((x, y), titles[i], fill=(0, 0, 0), font=font)
        
        # 添加主标题
        main_title = "骗子酒馆LLM策略类型雷达图"
        # 计算主标题的宽度
        main_title_bbox = draw.textbbox((0, 0), main_title, font=title_font)
        main_title_width = main_title_bbox[2] - main_title_bbox[0]
        main_title_height = main_title_bbox[3] - main_title_bbox[1]
        
        # 为主标题添加背景
        main_title_x = (total_width - main_title_width) // 2
        main_title_y = (title_height - main_title_height) // 2
        
        # 创建主标题背景矩形
        main_rect_x0 = main_title_x - padding*2
        main_rect_y0 = main_title_y - padding
        main_rect_x1 = main_title_x + main_title_width + padding*2
        main_rect_y1 = main_title_y + main_title_height + padding
        
        # 绘制主标题背景和边框
        draw.rectangle([main_rect_x0, main_rect_y0, main_rect_x1, main_rect_y1],
                      fill=(255, 255, 255, 230), outline=None)
        
        # 绘制主标题边框（更粗）
        for bw in range(border_width):
            draw.rectangle([main_rect_x0-bw, main_rect_y0-bw, main_rect_x1+bw, main_rect_y1+bw],
                          fill=None, outline=(0, 0, 0))
        
        # 绘制主标题文本
        draw.text((main_title_x, main_title_y), main_title, 
                  fill=(0, 0, 0), font=title_font)
        
        # 保存合成图片
        file_path = os.path.join(self.output_dir, output_file)
        combined_img.save(file_path)
        print(f"合成策略雷达图已保存至 {file_path}")
    
    def run_complete_analysis(self, game_records_folder: str, n_clusters: int = 3) -> None:
        """
        运行完整的行为分析流程
        
        Args:
            game_records_folder: 游戏记录文件夹路径
            n_clusters: 聚类数量
        """
        print(f"开始加载游戏记录...")
        self.load_game_records(game_records_folder)
        
        print(f"提取行为特征...")
        self.extract_behavior_features()
        
        print(f"执行行为聚类分析...")
        self.cluster_player_behaviors(n_clusters=n_clusters)
        
        print(f"分析策略适应性...")
        self.analyze_strategy_adaptation()
        
        print(f"分析对手特异性策略...")
        self.analyze_opponent_specific_strategies()
        
        print(f"生成可视化结果...")
        self.visualize_behavior_clusters()
        self.visualize_strategy_radar()
        self.visualize_strategy_evolution()
        self.visualize_opponent_specific_strategies()
        
        # 如果聚类数量为4，则合成策略雷达图
        if n_clusters == 4:
            print(f"合成策略雷达图...")
            self.combine_radar_images()
        
        print(f"行为分析完成，结果保存在 {self.output_dir} 文件夹中")
        
        # 保存分析结果为JSON
        results = {
            "behavior_features": self.behavior_features,
            "strategy_clusters": self.player_clusters,
            "strategy_evolution": self.strategy_evolution,
            "opponent_specific_strategies": self.opponent_specific_strategies
        }
        
        results_file = os.path.join(self.output_dir, "behavior_analysis_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"分析结果已保存至 {results_file}")


# 自定义JSON编码器，处理numpy数据类型
class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def run_complete_analysis(game_records_path: str, output_dir: str = "behavior_analysis", n_clusters: int = 3) -> None:
    """
    运行完整的行为分析流程（顶层函数）
    
    Args:
        game_records_path: 游戏记录文件夹路径
        output_dir: 输出目录路径
        n_clusters: 聚类数量
    """
    analyzer = BehaviorAnalyzer(output_dir=output_dir)
    analyzer.run_complete_analysis(game_records_path, n_clusters=n_clusters)
    return analyzer

def main():
    """主函数，用于从命令行运行行为分析"""
    import argparse
    
    parser = argparse.ArgumentParser(description='骗子酒馆LLM模型行为分析工具')
    parser.add_argument('--game_records', type=str, default='game_records', 
                        help='游戏记录文件夹路径')
    parser.add_argument('--output_dir', type=str, default='behavior_analysis', 
                        help='输出目录路径')
    parser.add_argument('--clusters', type=int, default=3, 
                        help='聚类数量')
    
    args = parser.parse_args()
    
    print(f"骗子酒馆LLM模型行为分析工具")
    print(f"=========================")
    print(f"游戏记录路径: {args.game_records}")
    print(f"输出目录: {args.output_dir}")
    print(f"聚类数量: {args.clusters}")
    print(f"=========================")
    
    # 调用顶层函数
    run_complete_analysis(args.game_records, args.output_dir, args.clusters)


if __name__ == "__main__":
    main() 