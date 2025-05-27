#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合成策略雷达图
将四张策略雷达图合成为一张2x2格式的图片
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def combine_images(image_paths, output_path, titles=None, main_title=None):
    """
    将多张图片合成为一张2x2的图片
    
    参数:
    image_paths: 四张图片的路径列表
    output_path: 输出图片的路径
    titles: 每张子图的标题列表
    main_title: 整张图的主标题
    """
    # 打开所有图片
    images = [Image.open(path) for path in image_paths]
    
    # 获取每张图片的尺寸
    widths, heights = zip(*(img.size for img in images))
    
    # 确定输出图片的尺寸
    max_width = max(widths)
    max_height = max(heights)
    
    # 计算输出图片的总尺寸
    title_height = 50 if main_title else 0
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
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 24)
                break
                
        if font is None:
            font = ImageFont.load_default()
            
    except Exception:
        font = ImageFont.load_default()
    
    draw = ImageDraw.Draw(combined_img)
    
    # 放置每张图片
    positions = [
        (0, title_height),  # 左上
        (max_width, title_height),  # 右上
        (0, max_height + title_height),  # 左下
        (max_width, max_height + title_height)  # 右下
    ]
    
    for i, img in enumerate(images):
        combined_img.paste(img, positions[i])
        
        # 如果提供了标题，则添加子图标题
        if titles and i < len(titles):
            x = positions[i][0] + max_width // 2
            y = positions[i][1] + 10
            draw.text((x, y), titles[i], fill=(0, 0, 0), font=font, anchor="mt")
    
    # 添加主标题
    if main_title:
        draw.text((total_width // 2, title_height // 2), main_title, 
                  fill=(0, 0, 0), font=font, anchor="mm")
    
    # 保存合成图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_img.save(output_path)
    print(f"合成图片已保存至: {output_path}")
    return output_path

if __name__ == "__main__":
    # 策略雷达图路径
    radar_dir = "analysis_results/behavior_analysis"
    image_paths = [
        os.path.join(radar_dir, f"strategy_radar_cluster_{i}.png")
        for i in range(4)
    ]
    
    # 确保所有文件都存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"错误: 文件 {path} 不存在")
            exit(1)
    
    # 输出路径
    output_path = os.path.join(radar_dir, "combined_strategy_radar.png")
    
    # 子图标题
    titles = [f"策略类型 {i}" for i in range(4)]
    
    # 主标题
    main_title = "骗子酒馆LLM策略类型雷达图"
    
    # 合成图片
    combine_images(image_paths, output_path, titles, main_title) 