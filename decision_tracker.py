import json
import re
import os
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class DecisionComponents:
    """决策组成部分的结构化表示"""
    key_factors: List[str]  # 关键考虑因素
    evidence: List[str]  # 支持决策的证据
    alternatives: List[str]  # 考虑的替代方案
    reasoning_steps: List[str]  # 推理步骤
    confidence: float  # 决策信心程度 (0-1)
    uncertainty_sources: List[str]  # 不确定性来源
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        return {
            "key_factors": self.key_factors,
            "evidence": self.evidence,
            "alternatives": self.alternatives,
            "reasoning_steps": self.reasoning_steps,
            "confidence": self.confidence,
            "uncertainty_sources": self.uncertainty_sources
        }

class DecisionTracker:
    """决策过程跟踪与分析工具"""
    
    def __init__(self, output_dir: str = "decision_analysis"):
        """初始化决策跟踪器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 决策组件模式匹配正则表达式
        self.patterns = {
            "key_factors": r"考虑(?:因素|的是)[:：]?\s*(.*?)(?:\n|$)",
            "evidence": r"(?:证据|根据)[:：]?\s*(.*?)(?:\n|$)",
            "alternatives": r"(?:替代方案|其他选择)[:：]?\s*(.*?)(?:\n|$)",
            "reasoning_steps": r"(?:推理过程|思考过程)[:：]?\s*(.*?)(?:\n|$)",
            "confidence": r"(?:信心程度|确信度)[:：]?\s*(\d+(?:\.\d+)?)%?",
            "uncertainty": r"(?:不确定性|顾虑|担忧)[:：]?\s*(.*?)(?:\n|$)"
        }
        
        # 存储所有分析的决策
        self.analyzed_decisions = []
        self.player_decisions = {}  # 玩家决策记录: {player_name: [decision_records]}
    
    def analyze_reasoning(self, reasoning_text: str) -> DecisionComponents:
        """分析推理文本并提取结构化的决策组件
        
        Args:
            reasoning_text: LLM生成的推理文本
            
        Returns:
            DecisionComponents: 结构化的决策组件
        """
        # 确保输入文本是字符串
        if not isinstance(reasoning_text, str):
            reasoning_text = str(reasoning_text) if reasoning_text is not None else ""
        
        # 对于空文本，返回默认的决策组件
        if not reasoning_text.strip():
            return DecisionComponents(
                key_factors=["未提供推理文本"],
                evidence=["未提供证据"],
                alternatives=["未考虑替代方案"],
                reasoning_steps=["未描述推理过程"],
                confidence=0.5,  # 默认中等信心
                uncertainty_sources=["未描述不确定性来源"]
            )
        
        # 提取关键因素
        key_factors = self._extract_pattern(reasoning_text, self.patterns["key_factors"]) or ["未明确指出"]
        
        # 提取证据
        evidence = self._extract_pattern(reasoning_text, self.patterns["evidence"]) or ["未提供明确证据"]
        
        # 提取替代方案
        alternatives = self._extract_pattern(reasoning_text, self.patterns["alternatives"]) or ["未考虑替代方案"]
        
        # 提取推理步骤
        reasoning_steps = []
        # 尝试查找数字编号的步骤（如1. 2. 3.）
        steps_match = re.findall(r'\d+\.\s*(.*?)(?:\n|$)', reasoning_text)
        if steps_match:
            reasoning_steps = steps_match
        else:
            # 否则尝试找明确标记的推理过程
            reasoning_match = re.search(self.patterns["reasoning_steps"], reasoning_text)
            if reasoning_match:
                # 以句号分割得到步骤
                steps = reasoning_match.group(1).split('。')
                reasoning_steps = [s.strip() for s in steps if s.strip()]
            else:
                # 如果都没找到，则尝试划分段落
                paragraphs = reasoning_text.split('\n')
                reasoning_steps = [p.strip() for p in paragraphs if p.strip()]
                
                # 如果段落太多，取前3段作为推理步骤
                if len(reasoning_steps) > 3:
                    reasoning_steps = reasoning_steps[:3]
                    reasoning_steps.append("...")
        
        # 提取信心程度
        confidence = 0.5  # 默认中等信心
        confidence_match = re.search(self.patterns["confidence"], reasoning_text)
        if confidence_match:
            try:
                confidence_value = float(confidence_match.group(1))
                # 如果是百分比形式，转换到0-1范围
                if confidence_value > 1:
                    confidence = confidence_value / 100
                else:
                    confidence = confidence_value
            except ValueError:
                pass
        else:
            # 基于关键词判断可能的信心水平
            confidence_keywords = {
                0.8: ['确信', '肯定', '绝对', '确定', '确切', '相信', '有把握'],
                0.6: ['很可能', '很有可能', '应该', '相当', '比较', '相信', '觉得'],
                0.5: ['或许', '可能', '也许', '估计', '猜测', '假设'],
                0.3: ['不确定', '不太确定', '有些怀疑', '存疑', '不太清楚', '不太肯定']
            }
            
            max_confidence = 0.5  # 默认中等信心
            for conf, keywords in confidence_keywords.items():
                for keyword in keywords:
                    if keyword in reasoning_text:
                        if (conf > 0.5 and conf > max_confidence) or (conf < 0.5 and conf < max_confidence):
                            max_confidence = conf
                            break
            confidence = max_confidence
        
        # 提取不确定性来源
        uncertainty_sources = self._extract_pattern(reasoning_text, self.patterns["uncertainty"]) or ["未提及不确定性"]
        
        # 如果推理文本很短，则基于文本长度推断组件质量
        if len(reasoning_text) < 50:  # 短文本
            if len(reasoning_steps) <= 1:
                reasoning_steps = ["简短推理"] if reasoning_steps else ["未详细说明推理过程"]
            if len(key_factors) <= 1 and key_factors[0] == "未明确指出":
                # 尝试从文本中提取可能的关键因素
                # 使用正则表达式中常见的关键词
                potential_factors = re.findall(r'(考虑|因为|由于|基于|根据|决定)(?:了|的|我)?(.*?)(?:\.|。|，|,|\n|$)', reasoning_text)
                if potential_factors:
                    key_factors = [f[1].strip() for f in potential_factors if f[1].strip()]
        
        # 创建并返回结构化决策组件
        return DecisionComponents(
            key_factors=key_factors,
            evidence=evidence,
            alternatives=alternatives,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            uncertainty_sources=uncertainty_sources
        )
    
    def _extract_pattern(self, text: str, pattern: str) -> List[str]:
        """从文本中提取匹配正则表达式的内容
        
        Args:
            text: 要分析的文本
            pattern: 正则表达式模式
            
        Returns:
            List[str]: 提取的内容列表
        """
        matches = re.findall(pattern, text)
        if matches:
            # 分割多个项目（通常以逗号、分号或顿号分隔）
            results = []
            for match in matches:
                items = re.split(r'[,，;；、]', match)
                results.extend([item.strip() for item in items if item.strip()])
            return results
        return []
    
    def track_play_decision(self, 
                           player_name: str, 
                           round_id: int, 
                           target_card: str,
                           play_cards: List[str], 
                           hand_cards: List[str],
                           reasoning_text: str) -> Dict[str, Any]:
        """跟踪并分析出牌决策
        
        Args:
            player_name: 玩家名称
            round_id: 轮次ID
            target_card: 目标牌
            play_cards: 打出的牌
            hand_cards: 手中的牌
            reasoning_text: 推理文本
            
        Returns:
            Dict[str, Any]: 结构化的决策分析
        """
        # 分析推理过程
        decision_components = self.analyze_reasoning(reasoning_text)
        
        # 创建决策记录
        decision_record = {
            "decision_type": "play",
            "player_name": player_name,
            "round_id": round_id,
            "target_card": target_card,
            "played_cards": play_cards,
            "remaining_cards": hand_cards,
            "is_bluffing": not all(card == target_card or card == 'Joker' for card in play_cards),
            "bluff_type": self._determine_bluff_type(play_cards, target_card),
            "decision_components": decision_components.to_dict(),
            "raw_reasoning": reasoning_text
        }
        
        # 添加到分析记录
        self.analyzed_decisions.append(decision_record)
        
        # 将决策记录存储到玩家决策列表中
        if player_name not in self.player_decisions:
            self.player_decisions[player_name] = []
        
        self.player_decisions[player_name].append(decision_record)
        return decision_record
    
    def track_challenge_decision(self,
                                player_name: str,
                                round_id: int,
                                challenged_player: str,
                                was_challenged: bool,
                                challenge_success: Optional[bool],
                                reasoning_text: str) -> Dict[str, Any]:
        """跟踪并分析质疑决策
        
        Args:
            player_name: 玩家名称
            round_id: 轮次ID
            challenged_player: 被质疑的玩家
            was_challenged: 是否发起质疑
            challenge_success: 质疑是否成功
            reasoning_text: 推理文本
            
        Returns:
            Dict[str, Any]: 结构化的决策分析
        """
        # 分析推理过程
        decision_components = self.analyze_reasoning(reasoning_text)
        
        # 创建决策记录
        decision_record = {
            "decision_type": "challenge",
            "player_name": player_name,
            "round_id": round_id,
            "challenged_player": challenged_player,
            "was_challenged": was_challenged,
            "challenge_success": challenge_success,
            "decision_components": decision_components.to_dict(),
            "raw_reasoning": reasoning_text
        }
        
        # 添加到分析记录
        self.analyzed_decisions.append(decision_record)
        
        # 将决策记录存储到玩家决策列表中
        if player_name not in self.player_decisions:
            self.player_decisions[player_name] = []
        
        self.player_decisions[player_name].append(decision_record)
        return decision_record
    
    def track_observation_decision(self, player_name: str, round_id: int, observed_player: str, 
                                 reasoning_text: str) -> Dict[str, Any]:
        """记录观察决策
        
        Args:
            player_name: 观察者
            round_id: 轮次ID
            observed_player: 被观察玩家
            reasoning_text: 推理文本
            
        Returns:
            Dict[str, Any]: 结构化的决策分析
        """
        # 分析推理过程
        decision_components = self.analyze_reasoning(reasoning_text)
        
        # 创建决策记录
        decision_record = {
            "decision_type": "observation",
            "player_name": player_name,
            "round_id": round_id,
            "observed_player": observed_player,
            "decision_components": decision_components.to_dict(),
            "raw_reasoning": reasoning_text
        }
        
        # 添加到分析记录
        self.analyzed_decisions.append(decision_record)
        
        # 将决策记录存储到玩家决策列表中
        if player_name not in self.player_decisions:
            self.player_decisions[player_name] = []
        
        self.player_decisions[player_name].append(decision_record)
        return decision_record
    
    def _determine_bluff_type(self, played_cards: List[str], target_card: str) -> str:
        """确定虚张声势的类型
        
        Args:
            played_cards: 打出的牌
            target_card: 目标牌
            
        Returns:
            str: 虚张声势类型（'诚实', '半虚张', '纯虚张'）
        """
        # 检查是否所有牌都符合目标
        if all(card == target_card or card == 'Joker' for card in played_cards):
            return "诚实"
        
        # 检查是否存在符合目标的牌
        if any(card == target_card or card == 'Joker' for card in played_cards):
            return "半虚张"
        
        # 所有牌都不符合目标
        return "纯虚张"
    
    def save_decision_analysis(self, game_id: str) -> None:
        """保存决策分析结果到文件
        
        Args:
            game_id: 游戏ID，用于文件命名
        """
        # 合并所有玩家的决策记录
        all_decisions = self.get_all_decisions()
        
        output_file = os.path.join(self.output_dir, f"{game_id}_decision_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_decisions, f, ensure_ascii=False, indent=2)
        print(f"决策分析已保存到: {output_file}")
    
    def get_all_decisions(self) -> List[Dict[str, Any]]:
        """获取所有玩家的决策记录
        
        Returns:
            List[Dict[str, Any]]: 所有决策记录
        """
        all_decisions = []
        for player_decisions in self.player_decisions.values():
            all_decisions.extend(player_decisions)
        
        # 按轮次排序
        return sorted(all_decisions, key=lambda x: (x.get("round_id", 0), x.get("decision_type", "")))
    
    def generate_decision_trends(self, player_name: Optional[str] = None) -> Dict[str, Any]:
        """生成决策趋势数据
        
        Args:
            player_name: 可选，指定玩家名称；如果未提供，则分析所有玩家
            
        Returns:
            Dict[str, Any]: 决策趋势数据
        """
        # 获取决策数据
        if player_name:
            if player_name not in self.player_decisions:
                return {}
            decisions = self.player_decisions[player_name]
        else:
            decisions = self.get_all_decisions()
        
        if not decisions:
            return {}
        
        # 按轮次统计
        bluff_rate_by_round = {}
        challenge_rate_by_round = {}
        confidence_by_round = {}
        key_factors_frequency = Counter()
        
        for decision in decisions:
            round_id = decision.get("round_id", 0)
            decision_type = decision.get("decision_type", "")
            
            # 统计关键因素
            factors = decision.get("decision_components", {}).get("key_factors", [])
            key_factors_frequency.update(factors)
            
            # 统计信心水平
            confidence = decision.get("decision_components", {}).get("confidence", 0.5)
            if round_id in confidence_by_round:
                confidence_by_round[round_id].append(confidence)
            else:
                confidence_by_round[round_id] = [confidence]
            
            # 统计虚张声势率
            if decision_type == "play":
                is_bluffing = decision.get("is_bluffing", False)
                if round_id in bluff_rate_by_round:
                    bluff_rate_by_round[round_id]["total"] += 1
                    if is_bluffing:
                        bluff_rate_by_round[round_id]["bluff"] += 1
                else:
                    bluff_rate_by_round[round_id] = {"total": 1, "bluff": 1 if is_bluffing else 0}
            
            # 统计质疑率
            elif decision_type == "challenge":
                was_challenged = decision.get("was_challenged", False)
                if round_id in challenge_rate_by_round:
                    challenge_rate_by_round[round_id]["total"] += 1
                    if was_challenged:
                        challenge_rate_by_round[round_id]["challenge"] += 1
                else:
                    challenge_rate_by_round[round_id] = {"total": 1, "challenge": 1 if was_challenged else 0}
        
        # 计算每轮虚张声势率
        bluff_rate = {}
        for round_id, counts in bluff_rate_by_round.items():
            if counts["total"] > 0:
                bluff_rate[round_id] = counts["bluff"] / counts["total"]
        
        # 计算每轮质疑率
        challenge_rate = {}
        for round_id, counts in challenge_rate_by_round.items():
            if counts["total"] > 0:
                challenge_rate[round_id] = counts["challenge"] / counts["total"]
        
        # 计算每轮平均信心
        confidence_avg = {}
        for round_id, confidences in confidence_by_round.items():
            if confidences:
                confidence_avg[round_id] = sum(confidences) / len(confidences)
        
        return {
            "bluff_rate_by_round": bluff_rate,
            "challenge_rate_by_round": challenge_rate,
            "confidence_by_round": confidence_avg,
            "key_factors_frequency": dict(key_factors_frequency)
        }
    
    def analyze_player_decision_patterns(self) -> Dict[str, Dict[str, Any]]:
        """分析玩家决策模式
        
        Returns:
            Dict[str, Dict[str, Any]]: 每个玩家的决策模式分析
        """
        results = {}
        
        for player_name, decisions in self.player_decisions.items():
            if not decisions:
                continue
            
            # 统计虚张声势相关指标
            play_decisions = [d for d in decisions if d.get("decision_type") == "play"]
            total_plays = len(play_decisions)
            
            if total_plays == 0:
                continue
            
            bluffs = [d for d in play_decisions if d.get("is_bluffing", False)]
            pure_bluffs = [d for d in bluffs if d.get("bluff_type") == "纯虚张声势"]
            semi_bluffs = [d for d in bluffs if d.get("bluff_type") == "半虚张声势"]
            
            bluff_rate = len(bluffs) / total_plays if total_plays > 0 else 0
            pure_bluff_rate = len(pure_bluffs) / total_plays if total_plays > 0 else 0
            semi_bluff_rate = len(semi_bluffs) / total_plays if total_plays > 0 else 0
            
            # 统计质疑相关指标
            challenge_decisions = [d for d in decisions if d.get("decision_type") == "challenge"]
            total_challenges = len(challenge_decisions)
            
            challenges = [d for d in challenge_decisions if d.get("was_challenged", False)]
            successful_challenges = [d for d in challenges if d.get("challenge_success", False)]
            
            challenge_rate = len(challenges) / total_challenges if total_challenges > 0 else 0
            challenge_success_rate = len(successful_challenges) / len(challenges) if len(challenges) > 0 else 0
            
            # 计算决策复杂度（基于推理步骤的数量）
            decision_complexity = 0
            reasoning_complexity_trend = []
            
            for decision in decisions:
                steps = decision.get("decision_components", {}).get("reasoning_steps", [])
                complexity = len(steps) / 5  # 标准化复杂度
                decision_complexity += complexity
                
                # 记录复杂度趋势
                reasoning_complexity_trend.append((decision.get("round_id", 0), complexity))
            
            avg_decision_complexity = decision_complexity / len(decisions) if decisions else 0
            
            # 计算平均信心水平
            total_confidence = sum(d.get("decision_components", {}).get("confidence", 0.5) for d in decisions)
            avg_confidence = total_confidence / len(decisions) if decisions else 0.5
            
            # 整理结果
            results[player_name] = {
                "bluff_rate": bluff_rate,
                "pure_bluff_rate": pure_bluff_rate,
                "semi_bluff_rate": semi_bluff_rate,
                "challenge_rate": challenge_rate,
                "challenge_success_rate": challenge_success_rate,
                "avg_confidence": avg_confidence,
                "decision_complexity": avg_decision_complexity,
                "reasoning_complexity_trend": sorted(reasoning_complexity_trend)
            }
        
        return results
    
    def extract_emotion_indicators(self) -> Dict[str, List[Dict[str, Any]]]:
        """提取情绪指标
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: 每个玩家的情绪指标
        """
        results = {}
        
        # 情绪词典
        emotion_keywords = {
            "紧张": ["紧张", "焦虑", "不安", "担忧", "忧虑"],
            "自信": ["自信", "确信", "笃定", "有把握", "确定"],
            "犹豫": ["犹豫", "迟疑", "踌躇", "不确定", "摇摆"],
            "冒险": ["冒险", "大胆", "勇敢", "激进", "豪赌"],
            "保守": ["保守", "谨慎", "小心", "稳妥", "安全"],
            "愤怒": ["愤怒", "恼火", "生气", "恼怒", "不满"],
            "沮丧": ["沮丧", "失望", "挫折", "低落", "丧气"],
            "兴奋": ["兴奋", "激动", "高兴", "欣喜", "欢呼"],
            "中性": ["平静", "淡定", "冷静", "理性", "思考"]
        }
        
        for player_name, decisions in self.player_decisions.items():
            player_emotions = []
            
            for decision in decisions:
                reasoning_text = decision.get("raw_reasoning", "")
                emotion_counts = {emotion: 0 for emotion in emotion_keywords}
                
                # 统计情绪词出现次数
                for emotion, keywords in emotion_keywords.items():
                    for keyword in keywords:
                        emotion_counts[emotion] += reasoning_text.count(keyword)
                
                # 确定主导情绪
                dominant_emotion = "中性"
                max_count = 0
                for emotion, count in emotion_counts.items():
                    if count > max_count:
                        max_count = count
                        dominant_emotion = emotion
                
                # 获取信心水平
                confidence = decision.get("decision_components", {}).get("confidence", 0.5)
                
                player_emotions.append({
                    "round_id": decision.get("round_id", 0),
                    "decision_type": decision.get("decision_type", ""),
                    "dominant_emotion": dominant_emotion,
                    "emotion_counts": emotion_counts,
                    "confidence": confidence
                })
            
            results[player_name] = player_emotions
        
        return results
    
    def export_decision_data_to_csv(self, output_file: str = "decision_analysis_data.csv") -> None:
        """将决策数据导出为CSV格式
        
        Args:
            output_file: 输出文件名
        """
        all_decisions = self.get_all_decisions()
        
        if not all_decisions:
            print("没有决策数据可导出")
            return
        
        # CSV标题行
        headers = [
            "player_name", "decision_type", "round_id", "target_card", 
            "played_cards", "remaining_cards", "is_bluffing", "bluff_type",
            "challenged_player", "was_challenged", "challenge_success",
            "observed_player", "confidence"
        ]
        
        # 将数据写入CSV
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for decision in all_decisions:
                row = [
                    decision.get("player_name", ""),
                    decision.get("decision_type", ""),
                    decision.get("round_id", ""),
                    decision.get("target_card", ""),
                    ",".join(decision.get("played_cards", [])),
                    ",".join(decision.get("remaining_cards", [])),
                    decision.get("is_bluffing", ""),
                    decision.get("bluff_type", ""),
                    decision.get("challenged_player", ""),
                    decision.get("was_challenged", ""),
                    decision.get("challenge_success", ""),
                    decision.get("observed_player", ""),
                    decision.get("decision_components", {}).get("confidence", "")
                ]
                writer.writerow(row)
        
        print(f"已导出决策数据到: {output_path}")
    
    def export_player_decision_data_to_csv(self, player_name: str, output_file: str, decisions=None) -> None:
        """将指定玩家的决策数据导出为CSV格式
        
        Args:
            player_name: 玩家名称
            output_file: 输出文件名
            decisions: 可选，指定的决策数据；如果为None则使用玩家的所有决策
        """
        if decisions is None:
            # 获取指定玩家的决策
            if player_name not in self.player_decisions:
                print(f"玩家 {player_name} 没有决策数据")
                return
            decisions = self.player_decisions[player_name]
        
        if not decisions:
            print(f"玩家 {player_name} 没有决策数据")
            return
        
        # 确保玩家目录存在
        player_dir = os.path.join(self.output_dir, player_name)
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)
        
        # 创建完整的输出路径
        output_path = os.path.join(player_dir, output_file)
        
        # CSV标题行
        headers = [
            "player_name", "decision_type", "round_id", "target_card", 
            "played_cards", "remaining_cards", "is_bluffing", "bluff_type",
            "challenged_player", "was_challenged", "challenge_success",
            "observed_player", "confidence"
        ]
        
        # 将数据写入CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for decision in decisions:
                row = [
                    decision.get("player_name", ""),
                    decision.get("decision_type", ""),
                    decision.get("round_id", ""),
                    decision.get("target_card", ""),
                    ",".join(decision.get("played_cards", [])),
                    ",".join(decision.get("remaining_cards", [])),
                    decision.get("is_bluffing", ""),
                    decision.get("bluff_type", ""),
                    decision.get("challenged_player", ""),
                    decision.get("was_challenged", ""),
                    decision.get("challenge_success", ""),
                    decision.get("observed_player", ""),
                    decision.get("decision_components", {}).get("confidence", "")
                ]
                writer.writerow(row)
        
        print(f"已导出玩家 {player_name} 的决策数据到 {output_path}") 