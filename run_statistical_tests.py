#!/usr/bin/env python3
"""
统计显著性测试执行脚本 - 骗子酒馆多智能体项目

该脚本整合运行多种统计分析方法，包括参数检验和非参数检验，生成综合的分析报告
"""

import os
import argparse
import time
from statistical_analysis import StatisticalAnalyzer
from statistical_analysis_nonparametric import NonparametricAnalyzer

def run_all_statistical_tests(game_records_path, output_dir, metrics=None, run_parametric=True, run_nonparametric=True):
    """
    运行所有统计检验
    
    Args:
        game_records_path: 游戏记录文件夹路径
        output_dir: 结果输出目录
        metrics: 要分析的指标列表
        run_parametric: 是否运行参数检验
        run_nonparametric: 是否运行非参数检验
    """
    start_time = time.time()
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用默认指标列表（如果未指定）
    if metrics is None:
        metrics = [
            'win', 'total_plays', 'honest_plays', 'deceptive_plays',
            'successful_deceptions', 'challenges_initiated', 'successful_challenges'
        ]
    
    print(f"=== 骗子酒馆LLM模型统计显著性分析 ===")
    print(f"游戏记录目录: {game_records_path}")
    print(f"输出目录: {output_dir}")
    print(f"分析指标: {', '.join(metrics)}")
    print(f"参数检验: {'启用' if run_parametric else '禁用'}")
    print(f"非参数检验: {'启用' if run_nonparametric else '禁用'}")
    print("=" * 40)
    
    # 运行参数检验（t检验和ANOVA）
    if run_parametric:
        print("\n运行参数统计检验...")
        parametric_dir = os.path.join(output_dir, "parametric")
        
        analyzer = StatisticalAnalyzer(output_dir=parametric_dir)
        analyzer.run_analysis(game_records_path, metrics)
    
    # 运行非参数检验（Mann-Whitney U检验和Kruskal-Wallis H检验）
    if run_nonparametric:
        print("\n运行非参数统计检验...")
        nonparametric_dir = os.path.join(output_dir, "nonparametric")
        
        nonparametric_analyzer = NonparametricAnalyzer(output_dir=nonparametric_dir)
        nonparametric_analyzer.run_analysis(game_records_path, metrics)
    
    # 生成综合报告
    generate_combined_report(output_dir, run_parametric, run_nonparametric)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n统计分析完成！耗时 {duration:.2f} 秒")
    print(f"分析结果保存在: {output_dir}")

def generate_combined_report(output_dir, parametric_included, nonparametric_included):
    """
    生成综合统计分析报告
    
    Args:
        output_dir: 结果输出目录
        parametric_included: 是否包含参数检验
        nonparametric_included: 是否包含非参数检验
    """
    report_path = os.path.join(output_dir, "combined_statistical_report.md")
    
    parametric_report = os.path.join(output_dir, "parametric", "statistical_analysis_report.md")
    nonparametric_report = os.path.join(output_dir, "nonparametric", "nonparametric_analysis_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 骗子酒馆LLM模型表现统计分析综合报告\n\n")
        f.write("本报告汇总了参数和非参数统计检验的分析结果，用于评估不同LLM模型在骗子酒馆游戏中的表现差异。\n\n")
        
        # 添加报告生成时间
        from datetime import datetime
        now = datetime.now()
        f.write(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 分析方法概述\n\n")
        
        if parametric_included:
            f.write("### 参数检验方法\n\n")
            f.write("- **t检验**：用于比较两个模型之间的表现差异\n")
            f.write("- **方差分析(ANOVA)**：用于比较多个模型之间的整体表现差异\n")
            f.write("- **Tukey HSD事后检验**：在ANOVA显著时用于识别具体哪些模型之间存在差异\n\n")
        
        if nonparametric_included:
            f.write("### 非参数检验方法\n\n")
            f.write("- **Mann-Whitney U检验**：用于比较两个模型之间的表现差异，不要求数据服从正态分布\n")
            f.write("- **Kruskal-Wallis H检验**：用于比较多个模型之间的整体表现差异，不要求数据服从正态分布\n")
            f.write("- **Dunn事后检验**：在Kruskal-Wallis H检验显著时用于识别具体哪些模型之间存在差异\n\n")
        
        # 合并参数检验结果
        if parametric_included and os.path.exists(parametric_report):
            f.write("## 参数统计检验结果\n\n")
            
            with open(parametric_report, 'r', encoding='utf-8') as pr:
                content = pr.read()
                # 去掉标题（第一行）
                lines = content.split('\n')
                if lines and lines[0].startswith('# '):
                    content = '\n'.join(lines[1:])
                f.write(content)
            
            f.write("\n\n")
        
        # 合并非参数检验结果
        if nonparametric_included and os.path.exists(nonparametric_report):
            f.write("## 非参数统计检验结果\n\n")
            
            with open(nonparametric_report, 'r', encoding='utf-8') as npr:
                content = npr.read()
                # 去掉标题（第一行）
                lines = content.split('\n')
                if lines and lines[0].startswith('# '):
                    content = '\n'.join(lines[1:])
                f.write(content)
            
            f.write("\n\n")
        
        # 添加综合结论部分
        f.write("## 综合结论\n\n")
        
        if parametric_included and nonparametric_included:
            f.write("比较参数检验和非参数检验的结果，我们可以发现：\n\n")
            f.write("- 对于样本量较大且分布近似正态的指标，参数检验提供了更高的统计检验功效\n")
            f.write("- 对于分布偏斜或样本量较小的指标，非参数检验提供了更加稳健的结果\n")
            f.write("- 当两种检验方法都显示显著差异时，我们可以更加确信不同模型之间存在真实的表现差异\n\n")
        
        f.write("### 研究发现\n\n")
        f.write("基于统计分析结果，我们得出以下关于LLM模型在骗子酒馆游戏中表现的关键发现：\n\n")
        f.write("1. *各模型之间是否存在显著的策略差异？*\n")
        f.write("2. *哪些模型展示了更好的欺骗能力？*\n")
        f.write("3. *哪些模型在质疑决策上表现更优？*\n")
        f.write("4. *模型的表现与其基础架构或训练方法是否相关？*\n\n")
        
        f.write("这些问题的答案需要结合统计结果和模型特性进行综合分析。统计显著性只是表明差异在数学上是真实的，")
        f.write("而实际意义还需要结合游戏机制和模型设计进行解释。\n\n")
        
        f.write("### 后续研究方向\n\n")
        f.write("统计分析结果提示了以下有价值的后续研究方向：\n\n")
        f.write("1. 深入分析表现出色的模型的决策过程，提取其策略优势\n")
        f.write("2. 调整游戏规则或提示词，测试模型在不同约束条件下的适应性\n")
        f.write("3. 对比不同版本的同一模型，研究版本迭代对策略博弈能力的影响\n")
        f.write("4. 设计针对性的对抗策略，测试模型的鲁棒性和应对能力\n\n")
        
        f.write("---\n\n")
        f.write("*注：完整的数据分析图表和详细结果可在相应的子目录中找到。*")
    
    print(f"综合报告已生成: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='骗子酒馆LLM模型统计显著性分析')
    parser.add_argument('--game_records', type=str, default='game_records', 
                        help='游戏记录文件夹路径')
    parser.add_argument('--output_dir', type=str, default='statistical_analysis_results', 
                        help='输出目录路径')
    parser.add_argument('--metrics', type=str, nargs='+', 
                        help='要分析的指标列表，用空格分隔')
    parser.add_argument('--skip_parametric', action='store_true',
                        help='跳过参数检验')
    parser.add_argument('--skip_nonparametric', action='store_true',
                        help='跳过非参数检验')
    
    args = parser.parse_args()
    
    run_all_statistical_tests(
        args.game_records,
        args.output_dir,
        args.metrics,
        not args.skip_parametric,
        not args.skip_nonparametric
    )

if __name__ == "__main__":
    main() 