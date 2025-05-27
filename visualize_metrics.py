#!/usr/bin/env python3
"""
骗子酒馆指标可视化脚本

将metrics.py计算的指标数据进行可视化展示
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from metrics import load_game_records, calculate_overall_metrics
import argparse

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS适用
plt.rcParams['axes.unicode_minus'] = False

# 指标名称中英文映射
METRIC_NAME_MAPPING = {
    # 欺骗相关指标
    'honest_play_rate': '诚实出牌率',
    'deception_rate': '欺骗率',
    'pure_bluff_rate': '纯虚张率',
    'half_bluff_rate': '半虚张率',
    'challenged_rate': '被质疑率',
    'joker_usage_rate': '王牌使用率',
    'success_rate': '成功率',
    'deception_success_rate': '欺骗成功率',
    # 存活相关指标
    'average_points': '平均积分',
    'average_survival_points': '平均存活积分',
    'survival_rate': '存活率',
    'average_rank': '平均排名',
    'early_elimination_rate': '早期淘汰率',
    'round_rate': '轮次率',
    'round_survival_rate': '轮次存活率',
    # 决策质量相关指标
    'challenge_precision': '质疑精确度',
    'challenge_recall': '质疑召回率',
    'challenge_quality': '质疑质量',
    'challenge_decision_quality': '质疑决策质量',
    'card_efficiency': '卡牌效率',
    'optimal_play_ratio': '最优出牌比例',
    'win_rate': '胜率',
    'rate': '比率',  # 通用后缀
    # 特殊处理的完整列名
    'deception_deception_rate': '欺骗率',
    'deception_honest_play_rate': '诚实出牌率',
    'deception_pure_bluff_rate': '纯虚张率',
    'deception_half_bluff_rate': '半虚张率',
    'deception_challenged_rate': '被质疑率',
    'deception_joker_usage_rate': '王牌使用率',
    'deception_success_rate': '成功率',
    'deception_deception_success_rate': '欺骗成功率',
    'survival_average_points': '平均积分',
    'survival_average_survival_points': '平均存活积分',
    'survival_survival_rate': '存活率',
    'survival_average_rank': '平均排名',
    'survival_early_elimination_rate': '早期淘汰率',
    'survival_round_rate': '轮次率',
    'survival_round_survival_rate': '轮次存活率',
    'decision_challenge_decision_quality': '质疑决策质量',
    'decision_challenge_precision': '质疑精确度',
    'decision_challenge_recall': '质疑召回率',
    'decision_card_efficiency': '卡牌效率',
    'decision_optimal_play_ratio': '最优出牌比例',
    'decision_challenge_quality': '质疑质量'
}

def plot_deception_metrics(metrics_df, output_dir):
    """
    生成欺骗相关指标的可视化图表
    
    Args:
        metrics_df: 包含指标数据的DataFrame
        output_dir: 输出目录
    """
    # 提取欺骗相关指标
    deception_cols = [col for col in metrics_df.columns if col.startswith('deception_')]
    
    if not deception_cols:
        print("未找到欺骗相关指标，跳过可视化")
        return
    
    # 为每个指标创建条形图
    for metric in deception_cols:
        # 尝试直接从映射表中查找完整列名
        metric_name_zh = METRIC_NAME_MAPPING.get(metric)
        
        if not metric_name_zh:
            # 从列名中提取简短的指标名称
            metric_name = metric.replace('deception_', '')
            # 尝试查找简短名称的映射
            metric_name_zh = METRIC_NAME_MAPPING.get(metric_name, f'{metric_name}(未翻译)')
        
        # 排序以便展示
        sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_df['player'], sorted_df[metric], color=sns.color_palette("viridis", len(sorted_df)))
        
        # 设置标题和标签
        # 对于欺骗率特殊处理，使用更精确的标题
        if metric == 'deception_deception_rate':
            plt.title(f'玩家欺骗比率比较', fontsize=16)
        else:
            plt.title(f'玩家{metric_name_zh}比较', fontsize=16)
        
        plt.xlabel('玩家', fontsize=12)
        plt.ylabel(metric_name_zh, fontsize=12)
        plt.ylim(0, max(sorted_df[metric]) * 1.2)  # 留出空间给标签
        
        # 在条形图上添加数值标签
        for bar, value in zip(bars, sorted_df[metric]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存欺骗指标图表: {output_file}")
    
    # 创建欺骗综合雷达图
    plt.figure(figsize=(10, 8))
    
    # 选择要展示的关键指标
    key_metrics = [
        'deception_honest_play_rate', 
        'deception_deception_success_rate',
        'deception_pure_bluff_rate', 
        'deception_half_bluff_rate',
        'deception_challenged_rate'
    ]
    
    # 准备雷达图数据
    categories_orig = [m.replace('deception_', '') for m in key_metrics]
    categories = [METRIC_NAME_MAPPING.get(cat, cat) for cat in categories_orig]
    players = metrics_df['player'].tolist()
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 设置图表
    ax = plt.subplot(111, polar=True)
    
    # 为每个玩家绘制雷达图
    for i, player in enumerate(players):
        values = metrics_df.loc[metrics_df['player'] == player, key_metrics].values.flatten().tolist()
        values += values[:1]  # 闭合多边形
        
        # 绘制线条
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图标签
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 设置标题
    plt.title('各玩家欺骗策略雷达图', size=20, y=1.05)
    
    # 保存图表
    output_file = os.path.join(output_dir, "deception_radar.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存欺骗雷达图: {output_file}")

def plot_survival_metrics(metrics_df, output_dir):
    """
    生成存活相关指标的可视化图表
    
    Args:
        metrics_df: 包含指标数据的DataFrame
        output_dir: 输出目录
    """
    # 提取存活相关指标
    survival_cols = [col for col in metrics_df.columns if col.startswith('survival_')]
    
    if not survival_cols:
        print("未找到存活相关指标，跳过可视化")
        return
    
    # 为每个指标创建条形图
    for metric in survival_cols:
        # 尝试直接从映射表中查找完整列名
        metric_name_zh = METRIC_NAME_MAPPING.get(metric)
        
        if not metric_name_zh:
            # 从列名中提取简短的指标名称
            metric_name = metric.replace('survival_', '')
            # 尝试查找简短名称的映射
            metric_name_zh = METRIC_NAME_MAPPING.get(metric_name, f'{metric_name}(未翻译)')
        
        # 排序以便展示
        sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_df['player'], sorted_df[metric], color=sns.color_palette("viridis", len(sorted_df)))
        
        # 设置标题和标签
        # 对于存活率和平均存活积分特殊处理，使用更精确的标题
        if metric == 'survival_survival_rate':
            plt.title(f'玩家存活比率比较', fontsize=16)
        elif metric == 'survival_average_points' or metric == 'survival_average_survival_points':
            plt.title(f'玩家平均存活积分比较', fontsize=16)
        else:
            plt.title(f'玩家{metric_name_zh}比较', fontsize=16)
            
        plt.xlabel('玩家', fontsize=12)
        plt.ylabel(metric_name_zh, fontsize=12)
        plt.ylim(0, max(sorted_df[metric]) * 1.2)  # 留出空间给标签
        
        # 在条形图上添加数值标签
        for bar, value in zip(bars, sorted_df[metric]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存存活指标图表: {output_file}")
    
    # 创建存活能力比较图 (存活率vs.平均存活积分)
    plt.figure(figsize=(10, 8))
    
    # 提取关键指标
    survival_rate = metrics_df['survival_survival_rate']
    survival_points = metrics_df['survival_average_survival_points']
    
    # 绘制散点图
    scatter = plt.scatter(
        survival_rate, 
        survival_points, 
        c=np.arange(len(metrics_df)), 
        cmap='viridis', 
        s=200, 
        alpha=0.7
    )
    
    # 添加玩家标签
    for i, player in enumerate(metrics_df['player']):
        plt.annotate(
            player, 
            (survival_rate[i], survival_points[i]),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=12
        )
    
    # 设置坐标轴标签和标题
    plt.xlabel('存活率', fontsize=14)
    plt.ylabel('平均存活积分', fontsize=14)
    plt.title('玩家存活能力矩阵', fontsize=18)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    output_file = os.path.join(output_dir, "survival_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存存活能力矩阵图: {output_file}")

def plot_decision_metrics(metrics_df, output_dir):
    """
    生成决策质量相关指标的可视化图表
    
    Args:
        metrics_df: 包含指标数据的DataFrame
        output_dir: 输出目录
    """
    # 提取决策相关指标
    decision_cols = [col for col in metrics_df.columns if col.startswith('decision_')]
    
    if not decision_cols:
        print("未找到决策质量相关指标，跳过可视化")
        return
    
    # 为每个指标创建条形图
    for metric in decision_cols:
        # 跳过optimal_play_ratio指标，因为数据全为0，没有分析价值
        if 'optimal_play_ratio' in metric:
            print(f"跳过 {metric} 指标，因为其数据全为0，没有分析价值")
            continue
            
        # 尝试直接从映射表中查找完整列名
        metric_name_zh = METRIC_NAME_MAPPING.get(metric)
        
        if not metric_name_zh:
            # 从列名中提取简短的指标名称
            metric_name = metric.replace('decision_', '')
            # 尝试查找简短名称的映射
            metric_name_zh = METRIC_NAME_MAPPING.get(metric_name, f'{metric_name}(未翻译)')
        
        # 排序以便展示
        sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_df['player'], sorted_df[metric], color=sns.color_palette("viridis", len(sorted_df)))
        
        # 设置标题和标签
        plt.title(f'玩家{metric_name_zh}比较', fontsize=16)
        plt.xlabel('玩家', fontsize=12)
        plt.ylabel(metric_name_zh, fontsize=12)
        plt.ylim(0, max(max(sorted_df[metric]) * 1.2, 0.1))  # 留出空间给标签，且处理全为0的情况
        
        # 在条形图上添加数值标签
        for bar, value in zip(bars, sorted_df[metric]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(output_dir, f"{metric}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存决策质量指标图表: {output_file}")
    
    # 创建质疑决策质量散点图 (精确度vs.召回率)
    if 'decision_challenge_precision' in metrics_df and 'decision_challenge_recall' in metrics_df:
        plt.figure(figsize=(10, 8))
        
        # 提取关键指标
        precision = metrics_df['decision_challenge_precision']
        recall = metrics_df['decision_challenge_recall']
        
        # 绘制散点图
        scatter = plt.scatter(
            recall, 
            precision, 
            c=np.arange(len(metrics_df)), 
            cmap='viridis', 
            s=200, 
            alpha=0.7
        )
        
        # 添加玩家标签
        for i, player in enumerate(metrics_df['player']):
            plt.annotate(
                player, 
                (recall[i], precision[i]),
                xytext=(7, 7),
                textcoords='offset points',
                fontsize=12
            )
        
        # 设置坐标轴标签和标题
        plt.xlabel('质疑召回率', fontsize=14)
        plt.ylabel('质疑精确度', fontsize=14)
        plt.title('玩家质疑决策质量', fontsize=18)
        
        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        output_file = os.path.join(output_dir, "decision_challenge_quality.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存质疑决策质量散点图: {output_file}")

def plot_overall_ranking(metrics_df, output_dir):
    """
    生成综合排名可视化图表
    
    Args:
        metrics_df: 包含指标数据的DataFrame
        output_dir: 输出目录
    """
    # 选择要展示的关键指标
    key_metrics = [
        'win_rate', 
        'deception_deception_success_rate',
        'survival_survival_rate', 
        'decision_challenge_decision_quality'
    ]
    
    # 只保留存在的列
    key_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if not key_metrics:
        print("未找到足够的关键指标，跳过综合排名可视化")
        return
    
    # 提取这些列
    ranking_df = metrics_df[['player'] + key_metrics].copy()
    
    # 创建热力图
    plt.figure(figsize=(12, len(metrics_df) * 0.8 + 2))
    
    # 转换为适合热力图的格式
    heatmap_df = ranking_df.set_index('player')
    
    # 使用更具可读性的列名
    renamed_cols = {}
    for col in heatmap_df.columns:
        # 尝试从映射表中获取中文名称
        zh_name = METRIC_NAME_MAPPING.get(col)
        if zh_name:
            renamed_cols[col] = zh_name
    
    heatmap_df = heatmap_df.rename(columns=renamed_cols)
    
    # 绘制热力图
    ax = sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu", 
        cbar_kws={'label': '分数'},
        linewidths=0.5
    )
    
    # 设置标题
    plt.title('玩家综合能力评分', fontsize=18)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, "overall_ranking.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存综合排名热力图: {output_file}")
    
    # 创建综合能力雷达图
    plt.figure(figsize=(10, 8))
    
    # 准备雷达图数据
    categories = []
    for m in key_metrics:
        # 尝试从映射表中获取中文名称
        zh_name = METRIC_NAME_MAPPING.get(m)
        if zh_name:
            categories.append(zh_name)
        else:
            categories.append(m)
    
    players = metrics_df['player'].tolist()
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 设置图表
    ax = plt.subplot(111, polar=True)
    
    # 为每个玩家绘制雷达图
    for i, player in enumerate(players):
        values = metrics_df.loc[metrics_df['player'] == player, key_metrics].values.flatten().tolist()
        values += values[:1]  # 闭合多边形
        
        # 绘制线条
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=player)
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图标签
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 设置标题
    plt.title('玩家综合能力雷达图', size=20, y=1.05)
    
    # 保存图表
    output_file = os.path.join(output_dir, "overall_radar.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存综合能力雷达图: {output_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='骗子酒馆指标可视化')
    parser.add_argument('--input', type=str, default='game_records',
                        help='游戏记录文件夹路径 (默认: game_records)')
    parser.add_argument('--metrics', type=str, default=None,
                        help='指标CSV文件路径 (如果不指定，将通过游戏记录计算)')
    parser.add_argument('--output', type=str, default='metrics_output',
                        help='输出文件夹路径 (默认: metrics_output)')
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 获取指标数据
    if args.metrics and os.path.exists(args.metrics):
        # 从CSV文件加载指标
        print(f"从文件 {args.metrics} 加载指标数据...")
        metrics_df = pd.read_csv(args.metrics)
    else:
        # 从游戏记录计算指标
        print(f"从 {args.input} 加载游戏记录并计算指标...")
        game_records = load_game_records(args.input)
        
        if not game_records:
            print("未找到游戏记录，退出程序")
            return
        
        # 计算指标并保存到临时目录
        metrics_temp_dir = os.path.join(args.output, 'temp')
        metrics = calculate_overall_metrics(game_records, metrics_temp_dir)
        
        # 从生成的CSV文件加载指标
        metrics_csv = os.path.join(metrics_temp_dir, 'player_metrics.csv')
        metrics_df = pd.read_csv(metrics_csv)
    
    # 生成可视化图表
    print(f"\n开始生成可视化图表，将保存到 {args.output}...")
    
    # 生成欺骗指标图表
    print("\n生成欺骗指标图表...")
    plot_deception_metrics(metrics_df, args.output)
    
    # 生成存活指标图表
    print("\n生成存活指标图表...")
    plot_survival_metrics(metrics_df, args.output)
    
    # 生成决策质量指标图表
    print("\n生成决策质量指标图表...")
    plot_decision_metrics(metrics_df, args.output)
    
    # 生成综合排名图表
    print("\n生成综合排名图表...")
    plot_overall_ranking(metrics_df, args.output)
    
    print(f"\n所有图表已生成完毕，保存在 {args.output} 目录")

if __name__ == "__main__":
    main() 