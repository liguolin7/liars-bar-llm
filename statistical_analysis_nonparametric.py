#!/usr/bin/env python3
"""
非参数统计显著性测试模块 - 骗子酒馆多智能体项目

此模块提供非参数统计检验功能，用于分析不同LLM模型在骗子酒馆游戏中的表现差异。
适用于数据不符合正态分布或样本量较小的情况。
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'PingFang SC']  # 优先使用这些中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
from metrics import load_game_records, calculate_overall_metrics

# 指标名称中英文映射
METRIC_NAME_MAPPING = {
    'win': '胜利',
    'total_plays': '总出牌数',
    'honest_plays': '诚实出牌',
    'deceptive_plays': '欺骗出牌',
    'successful_deceptions': '成功欺骗',
    'challenges_initiated': '发起质疑',
    'successful_challenges': '成功质疑',
    'challenge_quality': '质疑质量',
    'card_efficiency': '卡牌效率',
    'challenge_precision': '质疑精确度',
    'challenge_recall': '质疑召回率',
    'survival_rate': '存活率',
    'average_rank': '平均排名',
    'average_points': '平均积分',
    'round_rate': '轮次率',
    'early_elimination_rate': '早期淘汰率',
    'deception_rate': '欺骗率',
    'honest_play_rate': '诚实出牌率',
    'pure_bluff_rate': '纯虚张率',
    'half_bluff_rate': '半虚张率',
    'challenged_rate': '被质疑率',
    'joker_usage_rate': '王牌使用率',
    'success_rate': '成功率'
}

class NonparametricAnalyzer:
    """非参数统计分析器，用于执行Mann-Whitney U检验和Kruskal-Wallis H检验"""
    
    def __init__(self, output_dir: str = "statistical_analysis_nonparametric"):
        """
        初始化非参数统计分析器
        
        Args:
            output_dir: 输出结果的目录
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 存储分析结果
        self.mannwhitney_results = {}
        self.kruskal_results = {}
        self.game_level_df = None
    
    def load_data(self, game_records_path: str) -> None:
        """
        加载游戏记录并准备用于分析的数据框
        
        Args:
            game_records_path: 游戏记录文件夹路径
        """
        print(f"从 {game_records_path} 加载游戏记录...")
        self.game_records = load_game_records(game_records_path)
        
        if not self.game_records:
            print("未找到游戏记录，分析失败")
            return
        
        print(f"计算指标数据...")
        metrics_data = calculate_overall_metrics(self.game_records)
        
        # 创建按玩家汇总的指标数据框
        metrics_rows = []
        for player, metrics in metrics_data.items():
            row = {'player': player}
            row.update(metrics)
            metrics_rows.append(row)
        
        self.metrics_df = pd.DataFrame(metrics_rows)
        print(f"已准备 {len(self.metrics_df)} 个玩家的指标数据")
        
        # 计算游戏级别的数据框
        print("准备游戏级别的表现数据...")
        self._prepare_game_level_data()
    
    def _prepare_game_level_data(self) -> None:
        """准备游戏级别的数据框，每行表示一个玩家在一场游戏中的表现"""
        game_data_rows = []
        
        for game in self.game_records:
            game_id = game.get('game_id', 'unknown')
            players = game.get('player_names', [])
            winner = game.get('winner')
            
            # 记录玩家在该游戏中的表现
            for player in players:
                # 初始化基本数据
                row = {
                    'game_id': game_id,
                    'player': player,
                    'win': 1 if player == winner else 0,
                    'total_plays': 0,
                    'honest_plays': 0,
                    'deceptive_plays': 0,
                    'successful_deceptions': 0,
                    'challenges_initiated': 0,
                    'successful_challenges': 0
                }
                
                # 分析轮次数据
                for round_data in game.get('rounds', []):
                    for play in round_data.get('play_history', []):
                        play_player = play.get('player_name')
                        next_player = play.get('next_player')
                        
                        if play_player == player:
                            # 当前玩家是出牌者
                            row['total_plays'] += 1
                            
                            # 判断出牌类型
                            target_card = round_data.get('target_card')
                            played_cards = play.get('played_cards', [])
                            target_cards_count = sum(1 for card in played_cards if card == target_card)
                            joker_cards_count = sum(1 for card in played_cards if card == "Joker")
                            is_honest = (target_cards_count + joker_cards_count) == len(played_cards)
                            
                            if is_honest:
                                row['honest_plays'] += 1
                            else:
                                row['deceptive_plays'] += 1
                                if not play.get('was_challenged', False):
                                    row['successful_deceptions'] += 1
                        
                        elif next_player == player:
                            # 当前玩家是质疑者
                            if play.get('was_challenged', False):
                                row['challenges_initiated'] += 1
                                if play.get('challenge_result', False):
                                    row['successful_challenges'] += 1
                
                game_data_rows.append(row)
        
        self.game_level_df = pd.DataFrame(game_data_rows)
        print(f"已准备 {len(self.game_level_df)} 行游戏级别的表现数据")
    
    def run_mannwhitney_tests(self, metrics: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        对选定的指标执行Mann-Whitney U检验（两两比较）
        
        Args:
            metrics: 要测试的指标名称列表
            
        Returns:
            包含每对玩家在每个指标上的Mann-Whitney U检验p值的字典
        """
        if self.game_level_df is None or self.game_level_df.empty:
            print("请先加载数据")
            return {}
        
        results = {}
        players = self.game_level_df['player'].unique()
        
        for metric in metrics:
            results[metric] = {}
            
            # 确保指标在数据中
            if metric not in self.game_level_df.columns:
                print(f"指标 {metric} 在游戏级别数据中不可用，跳过")
                continue
            
            for i, player1 in enumerate(players):
                results[metric][player1] = {}
                
                for player2 in players[i+1:]:  # 只比较不重复的配对
                    p1_data = self.game_level_df[self.game_level_df['player'] == player1][metric]
                    p2_data = self.game_level_df[self.game_level_df['player'] == player2][metric]
                    
                    # 只在有足够数据时进行检验
                    if len(p1_data) < 2 or len(p2_data) < 2:
                        results[metric][player1][player2] = {'p_value': np.nan, 'significant': False}
                        continue
                    
                    # 进行Mann-Whitney U检验
                    try:
                        u_stat, p_value = stats.mannwhitneyu(p1_data, p2_data, alternative='two-sided')
                        significant = p_value < 0.05
                        
                        results[metric][player1][player2] = {
                            'u_statistic': u_stat,
                            'p_value': p_value,
                            'significant': significant
                        }
                    except Exception as e:
                        print(f"Mann-Whitney U检验失败 ({player1} vs {player2}, {metric}): {e}")
                        results[metric][player1][player2] = {'p_value': np.nan, 'significant': False}
            
        self.mannwhitney_results = results
        return results
    
    def run_kruskal_tests(self, metrics: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        对选定的指标执行Kruskal-Wallis H检验
        
        Args:
            metrics: 要测试的指标名称列表
            
        Returns:
            包含每个指标的Kruskal-Wallis检验结果的字典
        """
        if self.game_level_df is None or self.game_level_df.empty:
            print("请先加载数据")
            return {}
        
        results = {}
        
        for metric in metrics:
            # 确保指标在数据中
            if metric not in self.game_level_df.columns:
                print(f"指标 {metric} 在游戏级别数据中不可用，跳过")
                continue
            
            # 创建用于Kruskal-Wallis的数据
            samples = []
            player_groups = []
            
            for player in self.game_level_df['player'].unique():
                player_data = self.game_level_df[self.game_level_df['player'] == player][metric].dropna()
                if len(player_data) > 0:
                    samples.append(player_data)
                    player_groups.append(player)
            
            # 只在有足够数据时进行检验
            if len(samples) < 2:
                print(f"指标 {metric} 没有足够的分组，跳过Kruskal-Wallis H检验")
                results[metric] = {'h_value': np.nan, 'p_value': np.nan, 'significant': False}
                continue
            
            # 运行Kruskal-Wallis H检验
            try:
                h_val, p_val = stats.kruskal(*samples)
                
                # 如果Kruskal-Wallis显著，进行Dunn事后检验
                posthoc_results = None
                if p_val < 0.05:
                    try:
                        # 准备Dunn事后检验的数据
                        all_data = []
                        all_groups = []
                        
                        for i, sample in enumerate(samples):
                            all_data.extend(sample)
                            all_groups.extend([player_groups[i]] * len(sample))
                        
                        # 创建多重比较对象
                        mc = MultiComparison(all_data, all_groups)
                        
                        # 执行Dunn检验
                        posthoc = mc.allpairtest(stats.mannwhitneyu, method='bonf')
                        posthoc_results = posthoc[0]
                    except Exception as e:
                        print(f"Dunn事后检验失败 ({metric}): {e}")
                
                results[metric] = {
                    'h_value': h_val,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'posthoc_results': posthoc_results
                }
            except Exception as e:
                print(f"Kruskal-Wallis H检验失败 ({metric}): {e}")
                results[metric] = {'h_value': np.nan, 'p_value': np.nan, 'significant': False}
        
        self.kruskal_results = results
        return results
    
    def visualize_results(self) -> None:
        """可视化非参数统计检验结果"""
        if not hasattr(self, 'mannwhitney_results') or not hasattr(self, 'kruskal_results'):
            print("请先运行统计检验")
            return
        
        # 可视化Mann-Whitney U检验结果
        for metric, results in self.mannwhitney_results.items():
            plt.figure(figsize=(12, 10))
            
            # 提取p值矩阵
            players = list(results.keys())
            p_values = np.zeros((len(players), len(players)))
            p_values.fill(np.nan)  # 填充NaN
            
            for i, player1 in enumerate(players):
                for j, player2 in enumerate(players[i+1:], i+1):
                    if player2 in results[player1]:
                        p_values[i, j] = results[player1][player2]['p_value']
                        p_values[j, i] = p_values[i, j]  # 对称矩阵
            
            # 生成热力图
            mask = np.triu(np.ones_like(p_values, dtype=bool))
            
            # 创建带自定义颜色的热力图
            cmap = plt.cm.YlOrRd_r.copy()
            cmap.set_bad('white')  # 将NaN显示为白色
            
            sns.heatmap(p_values, mask=mask, annot=True, fmt=".3f", 
                       xticklabels=players, yticklabels=players, 
                       cmap=cmap, vmin=0, vmax=0.05)
            
            # 使用中文指标名称
            metric_name_zh = METRIC_NAME_MAPPING.get(metric, metric)
            plt.title(f"{metric_name_zh} 曼-惠特尼U检验 P值矩阵", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"mannwhitney_{metric}.png"), dpi=300)
            plt.close()
        
        # 可视化Kruskal-Wallis H检验结果
        metrics = list(self.kruskal_results.keys())
        h_values = [self.kruskal_results[m]['h_value'] for m in metrics]
        p_values = [self.kruskal_results[m]['p_value'] for m in metrics]
        significant = [self.kruskal_results[m]['significant'] for m in metrics]
        
        # 使用中文指标名称
        metrics_zh = [METRIC_NAME_MAPPING.get(m, m) for m in metrics]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(metrics_zh, h_values, color=[('green' if sig else 'red') for sig in significant])
        
        plt.title("克鲁斯卡尔-沃利斯 H值 (绿色表示p < 0.05)", fontsize=16)
        plt.xlabel("指标", fontsize=14)
        plt.ylabel("H值", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "kruskal_h_values.png"), dpi=300)
        plt.close()
        
        # 创建各指标的箱线图
        for metric in metrics:
            if metric in self.game_level_df.columns:
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='player', y=metric, data=self.game_level_df)
                # 使用中文指标名称
                metric_name_zh = METRIC_NAME_MAPPING.get(metric, metric)
                plt.title(f"{metric_name_zh} - 玩家间差异箱线图", fontsize=16)
                # 添加中文Y轴标签
                plt.ylabel(metric_name_zh, fontsize=14)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f"boxplot_{metric}.png"), dpi=300)
                plt.close()
    
    def generate_report(self) -> None:
        """生成非参数统计分析报告"""
        if not hasattr(self, 'kruskal_results') or not hasattr(self, 'mannwhitney_results'):
            print("没有可用的分析结果")
            return
        
        report_file = os.path.join(self.output_dir, "nonparametric_analysis_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 骗子酒馆LLM模型表现非参数统计分析\n\n")
            
            # Kruskal-Wallis结果摘要
            f.write("## Kruskal-Wallis H检验结果\n\n")
            f.write("以下指标在不同模型间存在统计显著差异 (p < 0.05)：\n\n")
            
            significant_metrics = [metric for metric, result in self.kruskal_results.items() 
                                  if result['significant']]
            
            if significant_metrics:
                f.write("| 指标 | H值 | p值 | 显著性 |\n")
                f.write("|------|-----|-----|--------|\n")
                
                for metric in significant_metrics:
                    result = self.kruskal_results[metric]
                    f.write(f"| {METRIC_NAME_MAPPING[metric]} | {result['h_value']:.3f} | {result['p_value']:.3f} | {'是' if result['significant'] else '否'} |\n")
            else:
                f.write("*没有找到统计显著的差异*\n")
            
            f.write("\n## Mann-Whitney U检验结果\n\n")
            
            # 显示每个指标的显著配对
            for metric, results in self.mannwhitney_results.items():
                f.write(f"### {METRIC_NAME_MAPPING[metric]}\n\n")
                
                significant_pairs = []
                for player1, player_results in results.items():
                    for player2, test_result in player_results.items():
                        if test_result.get('significant', False):
                            significant_pairs.append((player1, player2, test_result['p_value']))
                
                if significant_pairs:
                    f.write("| 玩家1 | 玩家2 | p值 |\n")
                    f.write("|-------|-------|-----|\n")
                    
                    for player1, player2, p_value in sorted(significant_pairs, key=lambda x: x[2]):
                        f.write(f"| {player1} | {player2} | {p_value:.3f} |\n")
                else:
                    f.write("*没有找到统计显著的两两差异*\n")
                
                f.write("\n")
            
            f.write("\n## 结论\n\n")
            
            if significant_metrics:
                f.write("根据非参数统计分析，不同模型在以下方面表现出显著差异：\n\n")
                for metric in significant_metrics:
                    f.write(f"- {METRIC_NAME_MAPPING[metric]}\n")
                
                f.write("\n这表明不同的LLM模型在骗子酒馆游戏中确实采取了不同的策略，并且这些差异具有统计学意义。非参数检验对于分布不符合正态假设的数据提供了稳健的分析结果。\n")
            else:
                f.write("根据非参数统计分析，在测试的指标中没有发现不同模型之间存在统计显著的差异。这可能是由于样本量不足，或者所有模型在骗子酒馆游戏中采取了相似的策略。\n")
        
        print(f"非参数分析报告已保存到 {report_file}")
    
    def run_analysis(self, game_records_path: str, metrics_to_test: List[str] = None) -> None:
        """
        运行完整的非参数统计分析流程
        
        Args:
            game_records_path: 游戏记录文件夹路径
            metrics_to_test: 要测试的指标列表，如果为None则使用默认指标
        """
        # 设置默认指标
        if metrics_to_test is None:
            metrics_to_test = ['win', 'total_plays', 'honest_plays', 'deceptive_plays', 
                              'successful_deceptions', 'challenges_initiated', 'successful_challenges']
        
        print(f"开始非参数统计显著性分析...")
        self.load_data(game_records_path)
        
        if self.game_level_df is None or self.game_level_df.empty:
            print("数据加载失败，无法进行分析")
            return
        
        print(f"运行Mann-Whitney U检验分析...")
        self.run_mannwhitney_tests(metrics_to_test)
        
        print(f"运行Kruskal-Wallis H检验分析...")
        self.run_kruskal_tests(metrics_to_test)
        
        print(f"生成可视化结果...")
        self.visualize_results()
        
        print(f"生成分析报告...")
        self.generate_report()
        
        print(f"非参数统计分析完成，结果保存在 {self.output_dir} 目录中")


def main():
    """主函数，用于从命令行运行非参数统计分析"""
    import argparse
    
    parser = argparse.ArgumentParser(description='骗子酒馆LLM模型表现非参数统计分析')
    parser.add_argument('--game_records', type=str, default='game_records', 
                        help='游戏记录文件夹路径')
    parser.add_argument('--output_dir', type=str, default='statistical_analysis_nonparametric', 
                        help='输出目录路径')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        help='要分析的指标列表，用空格分隔')
    
    args = parser.parse_args()
    
    analyzer = NonparametricAnalyzer(output_dir=args.output_dir)
    analyzer.run_analysis(args.game_records, args.metrics)


if __name__ == "__main__":
    main() 