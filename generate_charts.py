#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成Liars Bar LLM项目PPT所需的图表
包括项目结构图、分析流程图、游戏流程图和模型对战示意图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# 确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建存储目录
os.makedirs('analysis_results/report', exist_ok=True)

# 定义四个模型的统一颜色编码
MODEL_COLORS = {
    'Claude': '#3366CC',      # 蓝色
    'DeepSeek': '#33AA55',    # 绿色
    'Gemini': '#9966CC',      # 紫色
    'ChatGPT': '#33CCCC'      # 青色
}

# 通用图表设置
def setup_figure(width=12, height=8, dpi=100):
    """设置图表基本参数"""
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, ax

def draw_rounded_box(ax, x, y, width, height, color, alpha=1.0, label=None, fontsize=10, text_color='white'):
    """绘制圆角矩形框"""
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.2, rounding_size=0.15),
        facecolor=color,
        alpha=alpha,
        edgecolor='gray',
        linewidth=1
    )
    ax.add_patch(box)
    
    # 添加标签文本
    if label:
        ax.text(x + width/2, y + height/2, label, color=text_color,
                fontsize=fontsize, ha='center', va='center')
    
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='gray', width=0.01, head_width=0.03, label=None, label_offset=(0, 0)):
    """绘制带标签的箭头"""
    ax.arrow(x1, y1, x2-x1, y2-y1, color=color, width=width, 
             head_width=head_width, head_length=0.04, length_includes_head=True, zorder=1)
    
    if label:
        mid_x = (x1 + x2) / 2 + label_offset[0]
        mid_y = (y1 + y2) / 2 + label_offset[1]
        ax.text(mid_x, mid_y, label, color='black', fontsize=8, ha='center', va='center', 
                backgroundcolor='white', zorder=2)

def generate_project_structure():
    """生成项目结构图，展示游戏核心与分析系统的模块关系"""
    print("生成项目结构图...")
    
    fig, ax = setup_figure(12, 8)
    
    # 设置标题
    ax.set_title('Liars Bar LLM项目架构', fontsize=18, pad=20)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 创建游戏核心模块（左侧）
    game_core_x, game_core_y = 0.1, 0.7
    draw_rounded_box(ax, game_core_x, game_core_y, 0.35, 0.1, '#4477AA', label="游戏核心", fontsize=14)
    
    # 游戏核心子模块
    modules = [
        ("game.py", "游戏主程序", 0.15, 0.55),
        ("player.py", "LLM智能体", 0.15, 0.45),
        ("game_record.py", "对局记录", 0.15, 0.35),
        ("llm_client.py", "模型接口", 0.15, 0.25),
        ("multi_game_runner.py", "批量运行", 0.15, 0.15),
    ]
    
    for mod_file, mod_desc, x, y in modules:
        draw_rounded_box(ax, x, y, 0.25, 0.06, '#6699CC', label=f"{mod_file}\n{mod_desc}", fontsize=9)
    
    # 创建分析系统模块（右侧）
    analysis_x, analysis_y = 0.55, 0.7
    draw_rounded_box(ax, analysis_x, analysis_y, 0.35, 0.1, '#44AA77', label="分析系统", fontsize=14)
    
    # 分析系统子模块
    analysis_modules = [
        ("analyze_all.py", "一站式分析入口", 0.6, 0.55),
        ("metrics.py", "指标计算与可视化", 0.6, 0.45),
        ("decision_tracker.py", "决策追踪", 0.6, 0.35),
        ("behavior_analysis.py", "行为聚类", 0.6, 0.25),
        ("statistical_analysis.py", "统计检验", 0.6, 0.15),
    ]
    
    for mod_file, mod_desc, x, y in analysis_modules:
        draw_rounded_box(ax, x, y, 0.25, 0.06, '#66CCAA', label=f"{mod_file}\n{mod_desc}", fontsize=9)
    
    # 绘制模块间的数据流向
    # 游戏核心到游戏记录
    draw_arrow(ax, 0.275, 0.38, 0.4, 0.38, color='#888888', label="JSON游戏记录")
    
    # 游戏记录到分析系统
    draw_arrow(ax, 0.4, 0.38, 0.6, 0.38, color='#888888')
    
    # 分析系统内部流向
    draw_arrow(ax, 0.725, 0.575, 0.85, 0.575, color='#888888', label="分析结果")
    draw_arrow(ax, 0.85, 0.575, 0.85, 0.4, color='#888888')
    draw_arrow(ax, 0.85, 0.4, 0.725, 0.4, color='#888888', label="可视化图表")
    
    # 添加图例
    legend_items = [
        patches.Patch(facecolor='#6699CC', edgecolor='gray', label='游戏核心组件'),
        patches.Patch(facecolor='#66CCAA', edgecolor='gray', label='分析系统组件'),
    ]
    ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              frameon=True, ncol=2, fontsize=10)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('analysis_results/report/project_structure.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("项目结构图生成完成！")

def generate_analysis_flow():
    """生成分析流程图，展示从游戏记录到分析结果的处理流程"""
    print("生成分析流程图...")
    
    fig, ax = setup_figure(12, 8)
    
    # 设置标题
    ax.set_title('Liars Bar LLM分析流程', fontsize=18, pad=20)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 定义流程步骤
    steps = [
        ("游戏记录生成", "JSON格式对局数据", 0.1, 0.7, '#d1e5f0'),
        ("数据预处理", "格式转换与数据清洗", 0.3, 0.7, '#b8d4e9'),
        ("多维度分析", "指标计算、决策分析", 0.5, 0.7, '#8bb1d7'),
        ("可视化生成", "图表与报告生成", 0.7, 0.7, '#5a9bcf'),
        ("结果整合", "综合报告与发现", 0.9, 0.7, '#2070b4'),
    ]
    
    # 绘制主流程步骤
    for i, (step_name, step_desc, x, y, color) in enumerate(steps):
        draw_rounded_box(ax, x-0.07, y, 0.14, 0.1, color, label=step_name, fontsize=12)
        ax.text(x, y-0.05, step_desc, fontsize=9, ha='center', va='center', color='#444444')
        
        # 添加连接箭头
        if i < len(steps) - 1:
            next_x = steps[i+1][2]
            draw_arrow(ax, x+0.07, y+0.05, next_x-0.07, y+0.05, color='#555555', width=0.005)
    
    # 添加多维度分析的详细步骤（在主流程下方）
    analysis_types = [
        ("指标计算", 0.35, 0.45, '#8cb3d9'),
        ("决策分析", 0.45, 0.45, '#7da7cf'),
        ("行为分析", 0.55, 0.45, '#6e9bc5'),
        ("统计分析", 0.65, 0.45, '#5f8fbb')
    ]
    
    for name, x, y, color in analysis_types:
        draw_rounded_box(ax, x-0.05, y, 0.1, 0.08, color, label=name, fontsize=10)
        # 连接到多维度分析步骤
        if name == "指标计算":
            draw_arrow(ax, x, y+0.08, 0.5, 0.7, color='#777777', width=0.003)
        elif name == "统计分析":
            draw_arrow(ax, x, y+0.08, 0.5, 0.7, color='#777777', width=0.003)
        else:
            draw_arrow(ax, x, y+0.08, 0.5, 0.7, color='#777777', width=0.003)
    
    # 添加分析输出（在多维度分析下方）
    outputs = [
        ("欺骗能力指标", 0.35, 0.25, '#a6c5e1'),
        ("存活能力指标", 0.45, 0.25, '#97b9d7'),
        ("决策能力指标", 0.55, 0.25, '#88accd'),
        ("策略分类结果", 0.65, 0.25, '#79a0c3')
    ]
    
    for i, (name, x, y, color) in enumerate(outputs):
        draw_rounded_box(ax, x-0.05, y, 0.1, 0.08, color, label=name, fontsize=9)
        # 连接到对应的分析类型
        draw_arrow(ax, x, y+0.08, analysis_types[i][1], analysis_types[i][2], color='#777777', width=0.003)
    
    # 添加一站式分析命令说明
    ax.text(0.5, 0.1, "一站式分析命令：python analyze_all.py --game_records game_records --output_dir analysis_results", 
            fontsize=10, ha='center', va='center', 
            bbox=dict(facecolor='#eeeeee', edgecolor='#cccccc', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # 添加图例
    legend_items = [
        patches.Patch(facecolor='#d1e5f0', edgecolor='gray', label='数据输入'),
        patches.Patch(facecolor='#8bb1d7', edgecolor='gray', label='处理分析'),
        patches.Patch(facecolor='#2070b4', edgecolor='gray', label='结果输出'),
    ]
    ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, 0.05),
              frameon=True, ncol=3, fontsize=10)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('analysis_results/report/analysis_flow.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("分析流程图生成完成！")

def generate_game_flow():
    """生成游戏流程图，展示骗子酒馆游戏的基本流程"""
    print("生成游戏流程图...")
    
    fig, ax = setup_figure(10, 8)
    
    # 设置标题
    ax.set_title('骗子酒馆游戏流程', fontsize=18, pad=20)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 定义游戏流程步骤（环形布局）
    steps = [
        ("游戏初始化", "发牌阶段", 0.5, 0.85, '#e9f3db'),
        ("玩家回合", "选择牌、决定诚实/欺骗", 0.8, 0.65, '#c6e2af'),
        ("其他玩家决策", "质疑/不质疑", 0.8, 0.35, '#97ca82'),
        ("结果处理", "质疑成功/失败后果", 0.5, 0.15, '#64b053'),
        ("胜负判定", "是否有玩家被淘汰", 0.2, 0.35, '#3c8e3f'),
        ("回合结束", "进入下一位玩家回合", 0.2, 0.65, '#276b29'),
    ]
    
    # 绘制主流程步骤（环形布局）
    for i, (step_name, step_desc, x, y, color) in enumerate(steps):
        draw_rounded_box(ax, x-0.1, y-0.05, 0.2, 0.1, color, label=step_name, fontsize=12)
        ax.text(x, y+0.08, step_desc, fontsize=9, ha='center', va='center', color='#444444')
        
        # 添加环形连接箭头
        next_i = (i + 1) % len(steps)
        next_x, next_y = steps[next_i][2], steps[next_i][3]
        
        # 计算箭头控制点，使其呈现弧形
        if i == 0:  # 从上到右
            draw_arrow(ax, x+0.1, y-0.02, next_x-0.07, next_y, color='#555555', width=0.005)
        elif i == 1:  # 从右到下
            draw_arrow(ax, x, y-0.05, next_x, next_y+0.05, color='#555555', width=0.005)
        elif i == 2:  # 从下到下
            draw_arrow(ax, x-0.1, y-0.02, next_x+0.1, next_y-0.02, color='#555555', width=0.005)
        elif i == 3:  # 从下到左
            draw_arrow(ax, x-0.1, y+0.02, next_x+0.07, next_y, color='#555555', width=0.005)
        elif i == 4:  # 从左到左上
            draw_arrow(ax, x, y+0.05, next_x, next_y-0.05, color='#555555', width=0.005)
        else:  # 从左上到上
            draw_arrow(ax, x+0.1, y+0.02, next_x-0.1, next_y+0.02, color='#555555', width=0.005)
    
    # 添加关键决策点说明
    decision_points = [
        ("诚实出牌", 0.65, 0.7, '#e1f0e5'),
        ("欺骗出牌", 0.75, 0.55, '#f4d9d9'),
        ("不质疑", 0.65, 0.3, '#e1f0e5'),
        ("质疑", 0.75, 0.4, '#f4d9d9'),
    ]
    
    for name, x, y, color in decision_points:
        draw_rounded_box(ax, x-0.06, y-0.03, 0.12, 0.06, color, label=name, fontsize=9)
    
    # 添加结果说明
    results = [
        ("质疑成功: 质疑者+1分，被质疑者-1分", 0.5, 0.27, '#f0f0d8'),
        ("质疑失败: 质疑者-1分，游戏继续", 0.5, 0.22, '#d8d8c0'),
    ]
    
    for text, x, y, color in results:
        ax.text(x, y, text, fontsize=8, ha='center', va='center', color='#444444',
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='#cccccc', boxstyle='round,pad=0.3'))
    
    # 添加中心游戏说明
    ax.text(0.5, 0.5, "骗子酒馆\n策略博弈游戏", fontsize=14, ha='center', va='center', color='#333333',
            bbox=dict(facecolor='#f0f0f0', alpha=0.8, edgecolor='#cccccc', boxstyle='round,pad=1.0'))
    
    # 添加游戏规则简述
    rules = [
        "游戏目标: 通过明智的出牌和质疑决策，尽可能长时间存活",
        "玩家可以选择诚实出牌或欺骗(虚张声势)",
        "其他玩家可以选择质疑或不质疑前一位玩家",
        "积分耗尽或条件达成时玩家被淘汰",
        "最后存活的玩家获胜"
    ]
    
    rule_text = '\n'.join([f"• {rule}" for rule in rules])
    ax.text(0.15, 0.9, rule_text, fontsize=8, ha='left', va='top', 
            bbox=dict(facecolor='#f9f9f9', alpha=0.8, edgecolor='#dddddd', boxstyle='round,pad=0.5'))
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('analysis_results/report/game_flow.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("游戏流程图生成完成！")

def generate_model_interaction():
    """生成模型对战示意图，展示多个LLM如何在游戏环境中交互"""
    print("生成模型对战示意图...")
    
    fig, ax = setup_figure(10, 8)
    
    # 设置标题
    ax.set_title('多LLM模型对战机制', fontsize=18, pad=20)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 绘制中心游戏环境
    central_circle = plt.Circle((0.5, 0.5), 0.15, fill=True, facecolor='#f0f0f0', 
                                edgecolor='#999999', linewidth=2, alpha=0.7)
    ax.add_patch(central_circle)
    ax.text(0.5, 0.5, "骗子酒馆\n游戏环境", fontsize=12, ha='center', va='center')
    
    # 绘制卡牌背景
    for i in range(4):
        angle = i * np.pi/2
        x = 0.5 + 0.1 * np.cos(angle)
        y = 0.5 + 0.1 * np.sin(angle)
        card = patches.Rectangle((x-0.03, y-0.04), 0.06, 0.08, 
                                angle=i*45, facecolor='#ffffff', 
                                edgecolor='#666666', linewidth=1, alpha=0.9)
        ax.add_patch(card)
    
    # 定义四个模型的位置和颜色
    models = [
        ("Claude", 0.5, 0.85, MODEL_COLORS['Claude']),
        ("DeepSeek", 0.85, 0.5, MODEL_COLORS['DeepSeek']),
        ("Gemini", 0.5, 0.15, MODEL_COLORS['Gemini']),
        ("ChatGPT", 0.15, 0.5, MODEL_COLORS['ChatGPT']),
    ]
    
    # 绘制模型节点和连接
    for name, x, y, color in models:
        # 绘制模型节点
        model_circle = plt.Circle((x, y), 0.1, fill=True, facecolor=color, 
                                 edgecolor='#999999', linewidth=1, alpha=0.8)
        ax.add_patch(model_circle)
        ax.text(x, y, name, fontsize=10, ha='center', va='center', color='white', weight='bold')
        
        # 绘制模型与游戏环境的连接
        draw_arrow(ax, x, y, 0.5, 0.5, color='#aaaaaa', width=0.003)
        draw_arrow(ax, 0.5, 0.5, x, y, color='#aaaaaa', width=0.003)
    
    # 添加模型间的互动箭头（对抗关系）
    interactions = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 相邻模型
        (0, 2), (1, 3)  # 对角模型
    ]
    
    for i, j in interactions:
        x1, y1 = models[i][1], models[i][2]
        x2, y2 = models[j][1], models[j][2]
        
        # 计算曲线控制点，使互动箭头呈弧形
        # 避免穿过中心
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 向外偏移
        cx = mid_x + (mid_x - 0.5) * 0.2
        cy = mid_y + (mid_y - 0.5) * 0.2
        
        # 使用Path和PathPatch创建弧形箭头
        verts = [
            (x1, y1),  # 起点
            (cx, cy),  # 控制点
            (x2, y2),  # 终点
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', edgecolor='#999999', 
                                lw=1, alpha=0.4, zorder=1)
        ax.add_patch(patch)
    
    # 添加交互说明
    interactions = [
        ("模型输入: 游戏状态，可见信息", 0.5, 0.7),
        ("模型输出: 决策（诚实/欺骗/质疑）", 0.5, 0.65),
        ("信息不完全性: 只能看到自己的手牌", 0.5, 0.3),
        ("策略对抗: 预测对手决策，进行反制", 0.5, 0.25),
    ]
    
    for text, x, y in interactions:
        ax.text(x, y, text, fontsize=9, ha='center', va='center', color='#333333',
               bbox=dict(facecolor='#f8f8f8', alpha=0.7, edgecolor='#dddddd', boxstyle='round,pad=0.3'))
    
    # 添加运行方式说明
    run_cmds = [
        "单局模式: python game.py",
        "多局模式: python multi_game_runner.py -n 次数"
    ]
    
    cmd_text = '\n'.join([f"• {cmd}" for cmd in run_cmds])
    ax.text(0.2, 0.1, cmd_text, fontsize=9, ha='left', va='center',
           bbox=dict(facecolor='#eaeaea', alpha=0.7, edgecolor='#cccccc', boxstyle='round,pad=0.5'))
    
    # 添加图例
    legend_items = [
        patches.Patch(facecolor=MODEL_COLORS['Claude'], label='Claude'),
        patches.Patch(facecolor=MODEL_COLORS['DeepSeek'], label='DeepSeek'),
        patches.Patch(facecolor=MODEL_COLORS['Gemini'], label='Gemini'),
        patches.Patch(facecolor=MODEL_COLORS['ChatGPT'], label='ChatGPT'),
    ]
    ax.legend(handles=legend_items, loc='upper right', frameon=True, fontsize=9)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('analysis_results/report/model_interaction.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("模型对战示意图生成完成！")

if __name__ == "__main__":
    # 生成所有图表
    generate_project_structure()
    generate_analysis_flow()
    generate_game_flow()
    generate_model_interaction()
    
    print("所有图表生成完成！存储在 analysis_results/report/ 目录中") 