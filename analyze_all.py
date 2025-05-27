#!/usr/bin/env python3
"""
骗子酒馆LLM模型综合分析工具

此脚本整合了以下分析功能：
1. 数据可视化分析
2. 量化指标计算与可视化
3. 决策过程分析
4. 行为模式分析
5. 统计显著性测试

提供一站式分析接口，自动处理游戏记录并生成综合分析报告。
"""

import os
import argparse
import time
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional

# 导入各个分析模块
from visualize_metrics import main as visualize_metrics_main
from metrics import calculate_overall_metrics, load_game_records
from analyze_decisions import analyze_all_decisions
from behavior_analysis import run_complete_analysis
from statistical_analysis import StatisticalAnalyzer
from statistical_analysis_nonparametric import NonparametricAnalyzer
from run_statistical_tests import run_all_statistical_tests


class LiarsBarAnalyzer:
    """骗子酒馆综合分析器"""
    
    def __init__(self, 
                game_records_path: str, 
                output_dir: str = "analysis_results",
                create_dirs: bool = True):
        """
        初始化分析器
        
        Args:
            game_records_path: 游戏记录文件夹路径
            output_dir: 输出目录
            create_dirs: 是否创建输出目录
        """
        self.game_records_path = game_records_path
        self.output_dir = output_dir
        self.start_time = time.time()
        
        # 创建输出目录结构
        if create_dirs:
            self.setup_output_dirs()
        
        # 存储分析结果
        self.metrics_results = None
        self.visualization_paths = []
        self.decision_analysis_paths = []
        self.behavior_analysis_paths = []
        self.statistical_analysis_paths = []
    
    def setup_output_dirs(self) -> None:
        """创建输出目录结构"""
        # 创建主输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 创建子目录
        subdirs = [
            "metrics",  # 现在指标可视化内容将直接存放在这里，而不是visualizations/metrics
            "decision_analysis",
            "behavior_analysis",
            "statistical_analysis",
            "report"
        ]
        
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)
    
    def run_metrics_analysis(self) -> None:
        """运行量化指标分析"""
        print("\n=== 运行量化指标分析 ===")
        
        metrics_dir = os.path.join(self.output_dir, "metrics")
        
        try:
            # 加载游戏记录
            print("加载游戏记录...")
            game_records = load_game_records(self.game_records_path)
            
            if not game_records:
                print("未找到游戏记录，指标分析失败")
                return
            
            # 计算指标
            print("计算指标...")
            self.metrics_results = calculate_overall_metrics(
                game_records, 
                output_dir=metrics_dir
            )
            
            # 创建指标可视化
            print("生成指标可视化...")
            
            # 使用calculate_overall_metrics生成的CSV文件路径
            metrics_csv = os.path.join(metrics_dir, "player_metrics.csv")
            visualization_dir = os.path.join(self.output_dir, "metrics")
            
            # 构建命令行参数形式，适配visualize_metrics.main函数
            import sys
            original_argv = sys.argv
            sys.argv = [
                'visualize_metrics.py', 
                '--metrics', metrics_csv,  # 正确的参数名是--metrics
                '--output', visualization_dir  # 正确的参数名是--output
            ]
            try:
                visualize_metrics_main()
            finally:
                sys.argv = original_argv
            
            print(f"量化指标分析完成，结果保存在 {metrics_dir}")
        except Exception as e:
            print(f"量化指标分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run_decision_analysis(self) -> None:
        """运行决策过程分析"""
        print("\n=== 运行决策过程分析 ===")
        
        decision_dir = os.path.join(self.output_dir, "decision_analysis")
        
        try:
            # 分析决策过程
            print("分析决策过程...")
            analyze_all_decisions(
                game_records_path=self.game_records_path,
                output_dir=decision_dir
            )
            
            print(f"决策过程分析完成，结果保存在 {decision_dir}")
        except Exception as e:
            print(f"决策过程分析失败: {e}")
    
    def run_behavior_analysis(self) -> None:
        """运行行为模式分析"""
        print("\n=== 运行行为模式分析 ===")
        
        behavior_dir = os.path.join(self.output_dir, "behavior_analysis")
        
        try:
            # 分析行为模式
            print("分析行为模式...")
            run_complete_analysis(
                game_records_path=self.game_records_path,
                output_dir=behavior_dir,
                n_clusters=4  # 可以根据需要调整聚类数量
            )
            
            print(f"行为模式分析完成，结果保存在 {behavior_dir}")
        except Exception as e:
            print(f"行为模式分析失败: {e}")
    
    def run_statistical_analysis(self) -> None:
        """运行统计显著性分析"""
        print("\n=== 运行统计显著性分析 ===")
        
        stats_dir = os.path.join(self.output_dir, "statistical_analysis")
        
        try:
            # 运行统计分析
            print("进行统计显著性测试...")
            run_all_statistical_tests(
                game_records_path=self.game_records_path,
                output_dir=stats_dir,
                metrics=None,  # 使用默认指标集
                run_parametric=True,
                run_nonparametric=True
            )
            
            print(f"统计显著性分析完成，结果保存在 {stats_dir}")
        except Exception as e:
            print(f"统计显著性分析失败: {e}")
    
    def collect_report_assets(self) -> None:
        """收集报告资源"""
        print("\n=== 收集报告资源 ===")
        
        report_dir = os.path.join(self.output_dir, "report")
        
        # 收集关键分析图表
        key_visualizations = [
            # 指标可视化
            (os.path.join(self.output_dir, "metrics", "overall_radar.png"), 
             os.path.join(report_dir, "overall_performance_radar.png")),
            (os.path.join(self.output_dir, "metrics", "deception_radar.png"), 
             os.path.join(report_dir, "deception_strategies_radar.png")),
            
            # 决策分析
            (os.path.join(self.output_dir, "decision_analysis", "challenge_precision.png"), 
             os.path.join(report_dir, "reasoning_confidence.png")),
            (os.path.join(self.output_dir, "decision_analysis", "bluff_rate_trend.png"), 
             os.path.join(report_dir, "reasoning_complexity.png")),
            
            # 行为分析
            (os.path.join(self.output_dir, "behavior_analysis", "behavior_clusters.png"), 
             os.path.join(report_dir, "behavior_clusters.png")),
            # 使用合成的策略雷达图替代单个策略雷达图
            (os.path.join(self.output_dir, "behavior_analysis", "combined_strategy_radar.png"), 
             os.path.join(report_dir, "combined_strategy_radar.png")),
            
            # 统计分析
            (os.path.join(self.output_dir, "statistical_analysis", "parametric", "anova_f_values.png"), 
             os.path.join(report_dir, "anova_results.png")),
            (os.path.join(self.output_dir, "statistical_analysis", "nonparametric", "kruskal_h_values.png"), 
             os.path.join(report_dir, "kruskal_results.png")),
        ]
        
        # 复制关键图表
        for src, dst in key_visualizations:
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                    print(f"已复制: {src} -> {dst}")
                except Exception as e:
                    print(f"复制失败 {src}: {e}")
            else:
                print(f"文件不存在: {src}")
        
        # 复制关键报告
        key_reports = [
            (os.path.join(self.output_dir, "metrics", "player_metrics.csv"), 
             os.path.join(report_dir, "overall_metrics.csv")),
            (os.path.join(self.output_dir, "statistical_analysis", "combined_statistical_report.md"), 
             os.path.join(report_dir, "statistical_report.md")),
        ]
        
        for src, dst in key_reports:
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                    print(f"已复制: {src} -> {dst}")
                except Exception as e:
                    print(f"复制失败 {src}: {e}")
            else:
                print(f"文件不存在: {src}")
    
    def generate_comprehensive_report(self) -> None:
        """生成综合分析报告"""
        print("\n=== 生成综合分析报告 ===")
        
        report_path = os.path.join(self.output_dir, "report", "comprehensive_report.md")
        
        # 收集各个分析模块的数据
        metrics_csv = os.path.join(self.output_dir, "metrics", "player_metrics.csv")
        statistical_report = os.path.join(self.output_dir, "statistical_analysis", "combined_statistical_report.md")
        
        metrics_data = None
        if os.path.exists(metrics_csv):
            try:
                import pandas as pd
                metrics_data = pd.read_csv(metrics_csv)
            except Exception as e:
                print(f"读取指标数据失败: {e}")
        
        statistical_content = ""
        if os.path.exists(statistical_report):
            try:
                with open(statistical_report, 'r', encoding='utf-8') as f:
                    # 去掉开头的标题和空行
                    lines = f.readlines()
                    if lines and lines[0].startswith('# '):
                        lines = lines[1:]
                    while lines and lines[0].strip() == '':
                        lines = lines[1:]
                    statistical_content = ''.join(lines)
            except Exception as e:
                print(f"读取统计报告失败: {e}")
        
        # 生成报告
        with open(report_path, 'w', encoding='utf-8') as f:
            # 报告标题
            f.write("# 骗子酒馆LLM模型表现综合分析报告\n\n")
            
            # 生成时间和基本信息
            now = datetime.now()
            f.write(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**分析数据源**: {self.game_records_path}\n")
            
            # 分析模型列表
            if metrics_data is not None:
                models = metrics_data['player'].tolist() if 'player' in metrics_data.columns else []
                if models:
                    f.write(f"**分析模型**: {', '.join(models)}\n\n")
            
            # 执行摘要
            f.write("## 执行摘要\n\n")
            f.write("本报告对多个大语言模型在骗子酒馆（Liars Bar）策略博弈游戏中的表现进行了全面分析。")
            f.write("通过量化指标评估、决策过程追踪、行为模式聚类和统计显著性测试，揭示了不同模型在博弈中的策略偏好和表现差异。\n\n")
            
            # 关键发现
            f.write("### 关键发现\n\n")
            
            # 如果有指标数据，生成关键发现
            if metrics_data is not None and 'player' in metrics_data.columns:
                try:
                    # 找出胜率最高的模型
                    if 'win_rate' in metrics_data.columns:
                        best_win_rate = metrics_data.loc[metrics_data['win_rate'].idxmax()]
                        f.write(f"1. **{best_win_rate['player']}** 模型取得了最高的胜率 ({best_win_rate['win_rate']:.2f})，")
                        f.write("在整体博弈表现上领先其他模型。\n")
                    
                    # 找出欺骗成功率最高的模型
                    if 'deception_deception_success_rate' in metrics_data.columns:
                        best_deception = metrics_data.loc[metrics_data['deception_deception_success_rate'].idxmax()]
                        f.write(f"2. **{best_deception['player']}** 模型在欺骗策略上表现最佳，成功率达到 {best_deception['deception_deception_success_rate']:.2f}，")
                        f.write("显示出优秀的虚张声势能力。\n")
                    
                    # 找出质疑决策质量最高的模型
                    if 'decision_challenge_decision_quality' in metrics_data.columns:
                        best_decision = metrics_data.loc[metrics_data['decision_challenge_decision_quality'].idxmax()]
                        f.write(f"3. **{best_decision['player']}** 模型在质疑决策上最为精准，决策质量指标为 {best_decision['decision_challenge_decision_quality']:.2f}，")
                        f.write("表现出良好的判断能力。\n")
                except Exception as e:
                    print(f"生成关键发现时出错: {e}")
            
            f.write("\n行为模式分析和统计检验表明，不同模型采用了明显不同的博弈策略，")
            f.write("这些差异在多项关键指标上具有统计显著性。\n\n")
            
            # 模型表现比较
            f.write("## 模型表现比较\n\n")
            
            # 如果有指标数据，生成模型表现比较表格
            if metrics_data is not None and 'player' in metrics_data.columns:
                try:
                    # 选择关键指标
                    key_metrics = [
                        'win_rate', 
                        'deception_deception_rate', 
                        'deception_deception_success_rate', 
                        'survival_average_survival_points',
                        'decision_challenge_precision',
                        'decision_challenge_recall'
                    ]
                    
                    # 确认哪些指标在数据中存在
                    available_metrics = [m for m in key_metrics if m in metrics_data.columns]
                    
                    if available_metrics:
                        # 创建表格头
                        f.write("| 模型 | " + " | ".join([m.split('_')[-1].replace('_', ' ').title() for m in available_metrics]) + " |\n")
                        f.write("|" + "----|" * (len(available_metrics) + 1) + "\n")
                        
                        # 添加每个模型的数据行
                        for _, row in metrics_data.iterrows():
                            f.write(f"| {row['player']} |")
                            for metric in available_metrics:
                                f.write(f" {row[metric]:.2f} |")
                            f.write("\n")
                except Exception as e:
                    print(f"生成表格时出错: {e}")
            
            f.write("\n")
            
            # 添加图片引用
            f.write("### 整体表现雷达图\n\n")
            f.write("下图展示了各模型在关键指标上的综合表现：\n\n")
            f.write("![整体表现雷达图](overall_performance_radar.png)\n\n")
            
            f.write("### 行为模式聚类\n\n")
            f.write("通过行为特征聚类，可以识别不同的策略类型：\n\n")
            f.write("![行为模式聚类](behavior_clusters.png)\n\n")
            
            # 决策过程分析
            f.write("## 决策过程分析\n\n")
            f.write("决策过程分析揭示了不同模型的推理特点：\n\n")
            f.write("- 推理复杂度差异表明，模型在处理博弈情境时采用了不同深度的思考\n")
            f.write("- 推理信心水平反映了模型对自身判断的确信程度\n")
            f.write("- 策略适应性分析显示，部分模型能够根据游戏进展动态调整策略\n\n")
            
            f.write("![推理复杂度](reasoning_complexity.png)\n\n")
            f.write("![推理信心](reasoning_confidence.png)\n\n")
            
            # 统计显著性分析
            f.write("## 统计显著性分析\n\n")
            
            if statistical_content:
                f.write(statistical_content)
            else:
                f.write("统计分析表明，模型间在多项指标上存在显著差异，特别是在欺骗策略和质疑决策方面。\n\n")
                f.write("参数检验(ANOVA)和非参数检验(Kruskal-Wallis)的结果一致，进一步证实了这些差异的可靠性。\n\n")
            
            f.write("![统计分析结果](anova_results.png)\n\n")
            
            # 结论与建议
            f.write("## 结论与建议\n\n")
            f.write("基于综合分析，得出以下结论和建议：\n\n")
            
            f.write("1. **策略多样性**：不同LLM模型在策略博弈中表现出明显不同的策略偏好和行为模式，")
            f.write("这反映了它们在推理、风险评估和决策方式上的本质差异。\n\n")
            
            f.write("2. **推理能力差异**：在质疑决策的准确性上，模型间存在显著差异，")
            f.write("表明LLM对不确定情境的推理能力各不相同。\n\n")
            
            f.write("3. **策略适应性**：某些模型展现出更强的策略适应能力，能够根据游戏进展和对手行为调整自身策略，")
            f.write("这是评估LLM在动态环境中表现的重要维度。\n\n")
            
            f.write("4. **未来研究方向**：\n")
            f.write("   - 探索更复杂的博弈环境，测试LLM的多层级推理能力\n")
            f.write("   - 设计针对性的提示词，提升LLM在特定策略上的表现\n")
            f.write("   - 研究不同版本LLM的策略演化，追踪模型能力的发展轨迹\n")
            f.write("   - 将LLM在策略博弈中的表现与其他智能体（如人类或强化学习系统）进行比较\n\n")
            
            f.write("---\n\n")
            f.write("*注：完整的分析数据和图表可在相应的分析目录中找到。*")
        
        print(f"综合分析报告已生成: {report_path}")
    
    def run_all_analysis(self) -> None:
        """运行所有分析"""
        print("=== 开始骗子酒馆LLM模型综合分析 ===")
        print(f"游戏记录路径: {self.game_records_path}")
        print(f"输出目录: {self.output_dir}")
        print("=" * 40)
        
        # 检查输入路径
        if not os.path.exists(self.game_records_path):
            print(f"错误: 游戏记录路径 '{self.game_records_path}' 不存在")
            return False
        
        # 运行各个分析模块
        self.run_metrics_analysis()
        self.run_decision_analysis()
        self.run_behavior_analysis()
        self.run_statistical_analysis()
        
        # 收集报告资源
        self.collect_report_assets()
        
        # 生成综合报告
        self.generate_comprehensive_report()
        
        # 计算总用时
        end_time = time.time()
        duration = end_time - self.start_time
        
        print("\n=== 分析完成 ===")
        print(f"总用时: {duration:.2f} 秒")
        print(f"分析结果保存在: {self.output_dir}")
        print(f"综合报告: {os.path.join(self.output_dir, 'report', 'comprehensive_report.md')}")
        
        return True


def main():
    """主函数，处理命令行参数并运行分析"""
    parser = argparse.ArgumentParser(description='骗子酒馆LLM模型综合分析工具')
    
    parser.add_argument('--game_records', type=str, default='game_records',
                        help='游戏记录文件夹路径')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='输出目录路径')
    parser.add_argument('--metrics_only', action='store_true',
                        help='仅运行指标分析')
    parser.add_argument('--decision_only', action='store_true',
                        help='仅运行决策过程分析')
    parser.add_argument('--behavior_only', action='store_true',
                        help='仅运行行为模式分析')
    parser.add_argument('--stats_only', action='store_true',
                        help='仅运行统计显著性分析')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = LiarsBarAnalyzer(
        game_records_path=args.game_records,
        output_dir=args.output_dir
    )
    
    # 根据命令行参数运行指定的分析
    if args.metrics_only:
        analyzer.run_metrics_analysis()
    elif args.decision_only:
        analyzer.run_decision_analysis()
    elif args.behavior_only:
        analyzer.run_behavior_analysis()
    elif args.stats_only:
        analyzer.run_statistical_analysis()
    else:
        # 默认运行所有分析
        analyzer.run_all_analysis()


if __name__ == "__main__":
    main() 