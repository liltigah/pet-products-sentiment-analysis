# sentiment_analysis_single.py
"""
情感分析单个文件版本
用于快速测试和运行
"""
import duckdb
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

# ==================== 配置 ====================
class Config:
    # DuckDB文件路径（根据实际情况修改）
    DB_PATH = "data/amazon_pets.duckdb"  # 修改为你的实际路径
    
    # 表名配置
    TABLE_NAMES = {
        'comments': 'pets_comment_cleaned',
        'sentiment_results': 'pets_comment_sentiment'
    }
    
    # 情感阈值
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

# ==================== 文本预处理 ====================
class TextPreprocessor:
    """文本预处理类"""
    
    @staticmethod
    def clean_text(text):
        """基础文本清洗"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除特殊字符，保留字母、数字、中文和基本标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?]', ' ', text)
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def normalize_text(text, max_length=500):
        """文本标准化"""
        text = TextPreprocessor.clean_text(text)
        # 截断过长的文本
        if len(text) > max_length:
            text = text[:max_length] + "..."
        return text

# ==================== 情感分析器 ====================
class RuleBasedSentimentAnalyzer:
    """基于VADER规则的情感分析器"""
    
    def __init__(self, positive_threshold=0.05, negative_threshold=-0.05):
        self.analyzer = SentimentIntensityAnalyzer()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
    
    def analyze_single(self, text):
        """分析单条文本"""
        if not text or len(text.strip()) < 3:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'details': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            }
        
        try:
            scores = self.analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound >= self.positive_threshold:
                sentiment = 'positive'
            elif compound <= self.negative_threshold:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': compound,
                'confidence': abs(compound),  # 使用绝对值作为置信度
                'details': scores
            }
        except Exception as e:
            print(f"分析文本出错: {e}, 文本: {text[:50]}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'details': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            }
    
    def analyze_batch(self, texts, batch_size=1000):
        """批量分析文本"""
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.analyze_single(text)
                results.append(result)
            
            if i % 1000 == 0 and i > 0:
                print(f"已处理 {min(i+batch_size, total)}/{total} 条文本")
        
        return pd.DataFrame(results)

# ==================== DuckDB数据库工具 ====================
class DuckDBManager:
    """DuckDB数据库管理器"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接到DuckDB数据库"""
        try:
            if not Path(self.db_path).exists():
                raise FileNotFoundError(f"DuckDB文件不存在: {self.db_path}")
            
            self.conn = duckdb.connect(database=self.db_path)
            print(f"成功连接到数据库: {self.db_path}")
            return True
        except Exception as e:
            print(f"连接数据库失败: {e}")
            return False
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def get_table_info(self):
        """获取数据库中的表信息"""
        if not self.conn:
            print("未连接到数据库")
            return []
        
        try:
            tables = self.conn.execute("SHOW TABLES;").fetchall()
            table_list = [t[0] for t in tables]
            print(f"数据库中的表: {table_list}")
            return table_list
        except Exception as e:
            print(f"获取表信息失败: {e}")
            return []
    
    def get_table_preview(self, table_name, limit=5):
        """预览表数据"""
        if not self.conn:
            return None
        
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            df = self.conn.execute(query).fetchdf()
            print(f"\n表 '{table_name}' 预览 ({len(df)} 行):")
            print(df.head())
            return df
        except Exception as e:
            print(f"预览表 {table_name} 失败: {e}")
            return None
    
    def load_comments(self, table_name, limit=None, sample=False):
        """加载评论数据"""
        if not self.conn:
            return pd.DataFrame()
        
        try:
            if sample:
                # 随机抽样
                query = f"""
                    SELECT * FROM {table_name} 
                    USING SAMPLE 1000 ROWS
                """
            elif limit:
                query = f"SELECT * FROM {table_name} LIMIT {limit}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            print(f"执行查询: {query}")
            df = self.conn.execute(query).fetchdf()
            print(f"成功加载 {len(df)} 条评论")
            return df
        except Exception as e:
            print(f"加载评论数据失败: {e}")
            return pd.DataFrame()
    
    def save_sentiment_results(self, df, table_name):
        """保存情感分析结果"""
        if not self.conn or df.empty:
            return False
        
        try:
            # 创建表（如果不存在）
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    asin VARCHAR,
                    clean_text TEXT,
                    sentiment VARCHAR(20),
                    score DOUBLE,
                    confidence DOUBLE,
                    analysis_timestamp TIMESTAMP,
                    analysis_method VARCHAR(50)
                )
            """)
            
            # 清空表（如果已存在数据）
            self.conn.execute(f"DELETE FROM {table_name}")
            
            # 分批次插入数据
            batch_size = 1000
            total = len(df)
            
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size]
                
                # 将DataFrame注册为临时表
                temp_table = f"temp_batch_{i}"
                self.conn.register(temp_table, batch)
                
                # 插入数据
                insert_query = f"""
                    INSERT INTO {table_name} 
                    SELECT asin, clean_text, sentiment, score, 
                           confidence, analysis_timestamp, analysis_method
                    FROM {temp_table}
                """
                self.conn.execute(insert_query)
                
                # 删除临时表
                self.conn.unregister(temp_table)
                
                if i % 5000 == 0 and i > 0:
                    print(f"已保存 {min(i+batch_size, total)}/{total} 条记录")
            
            # 验证保存的数据
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"成功保存 {count} 条情感分析结果到表 {table_name}")
            
            return True
        except Exception as e:
            print(f"保存结果失败: {e}")
            return False
    
    def query_sentiment_stats(self, table_name):
        """查询情感分析统计"""
        if not self.conn:
            return None
        
        try:
            query = f"""
                SELECT 
                    sentiment,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage,
                    AVG(score) as avg_score,
                    AVG(confidence) as avg_confidence
                FROM {table_name}
                GROUP BY sentiment
                ORDER BY 
                    CASE sentiment
                        WHEN 'positive' THEN 1
                        WHEN 'neutral' THEN 2
                        WHEN 'negative' THEN 3
                        ELSE 4
                    END
            """
            return self.conn.execute(query).fetchdf()
        except Exception as e:
            print(f"查询统计失败: {e}")
            return None

# ==================== 可视化工具 ====================
class Visualizer:
    """优化版可视化工具类"""
    
    @staticmethod
    def plot_sentiment_distribution(df, save_path=None, title="亚马逊宠物用品评论情感分析"):
        """
        绘制专业级情感分布图
        兼容不同Matplotlib版本的接口
        
        参数:
            df: DataFrame，包含'sentiment'和可选的'score'列
            save_path: 保存路径，如为None则不保存
            title: 图表主标题
        """
        if df.empty or 'sentiment' not in df.columns:
            print("没有情感数据可可视化")
            return
        
        # 导入必要的库
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.patches import Patch
        
        # 设置专业图表样式
        plt.style.use('seaborn-v0_8-whitegrid')
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass  # 如果中文字体设置失败，继续使用默认字体
        
        # 准备数据
        sentiment_counts = df['sentiment'].value_counts()
        total = sentiment_counts.sum()
        
        # 计算百分比
        percentages = (sentiment_counts / total * 100).round(1)
        
        # 确保情感顺序一致
        sentiment_order = ['positive', 'negative', 'neutral']
        # 只保留存在的情感类别
        sentiment_counts = sentiment_counts.reindex([s for s in sentiment_order if s in sentiment_counts.index])
        percentages = percentages.reindex([s for s in sentiment_order if s in percentages.index])
        
        if sentiment_counts.empty:
            print("错误: 情感数据为空")
            return
        
        # 根据你的图片数据设置颜色
        colors = {
            'positive': '#4CAF50',  # Material Green
            'negative': '#F44336',  # Material Red
            'neutral': '#9E9E9E'    # Material Grey
        }
        
        # 确保颜色列表与数据顺序匹配
        color_list = [colors.get(s, '#2196F3') for s in sentiment_counts.index]
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=100)
        
        # 1. 环形饼图 (优化版)
        ax1 = axes[0]
        
        try:
            # 不同Matplotlib版本兼容处理
            pie_result = ax1.pie(
                sentiment_counts,
                colors=color_list,
                startangle=90,
                wedgeprops=dict(width=0.3, edgecolor='white', linewidth=2),  # 环形效果
                pctdistance=0.85
            )
            
            # 兼容不同版本的返回值
            if len(pie_result) >= 2:
                wedges, texts = pie_result[:2]
            else:
                wedges = pie_result[0]
                texts = []
                
        except Exception as e:
            print(f"绘制饼图时出错: {e}")
            # 简化版本
            ax1.pie(sentiment_counts, colors=color_list, startangle=90)
            wedges = []
            texts = []
        
        # 美化标签
        for i, (count, pct) in enumerate(zip(sentiment_counts, percentages)):
            if i < len(wedges):
                wedge = wedges[i]
                # 计算标签位置
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = 0.7 * np.cos(np.deg2rad(angle))
                y = 0.7 * np.sin(np.deg2rad(angle))
                
                # 添加情感标签和数值
                sentiment_label = sentiment_counts.index[i].title()
                ax1.text(x, y, 
                        f"{sentiment_label}\n{count}\n({pct}%)", 
                        ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor='white', 
                                 edgecolor='lightgray',
                                 alpha=0.9))
        
        # 添加中心总数
        ax1.text(0, 0, f'总计\n{total:,}', 
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", 
                         facecolor='white', 
                         edgecolor='lightgray'))
        
        ax1.set_title('(a) 情感分布比例图', fontsize=14, fontweight='bold', pad=20)
        ax1.axis('equal')
        
        # 2. 分组柱状图 (优化版)
        ax2 = axes[1]
        
        x_positions = range(len(sentiment_counts))
        bars = ax2.bar(x_positions, sentiment_counts.values,
                      color=color_list,
                      edgecolor='black', linewidth=1.5,
                      width=0.6)
        
        # 添加数据标签
        for i, (bar, count, pct) in enumerate(zip(bars, sentiment_counts, percentages)):
            height = bar.get_height()
            # 在柱顶显示数量和百分比
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(sentiment_counts)*0.02,
                    f'{count:,}\n({pct}%)',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor='white', 
                             edgecolor='lightgray'))
            
            # 在柱体内部显示情感标签
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    sentiment_counts.index[i].title(),
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white',
                    rotation=0)
        
        ax2.set_title('(b) 情感数量统计图', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('情感类别', fontsize=12, fontweight='bold')
        ax2.set_ylabel('评论数量', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels([s.title() for s in sentiment_counts.index], 
                           fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(sentiment_counts) * 1.15)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 3. 情感分数散点图 (优化版) 或 情感强度图
        ax3 = axes[2]
        
        if 'score' in df.columns and len(df) > 0:
            try:
                # 创建颜色映射
                sentiment_colors = df['sentiment'].map(colors)
                # 对于NaN值使用默认颜色
                sentiment_colors = sentiment_colors.fillna('#9E9E9E')
                
                # 创建散点图
                scatter = ax3.scatter(range(len(df)), df['score'].fillna(0), 
                                    c=sentiment_colors, 
                                    alpha=0.7, 
                                    s=40,
                                    edgecolors='black',
                                    linewidths=0.5)
                
                # 添加水平参考线
                ax3.axhline(y=0.05, color='green', linestyle=':', alpha=0.5, linewidth=1)
                ax3.axhline(y=-0.05, color='red', linestyle=':', alpha=0.5, linewidth=1)
                ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1.5)
                
                ax3.set_title('(c) 情感分数分布图', fontsize=14, fontweight='bold', pad=20)
                ax3.set_xlabel('评论序号', fontsize=12, fontweight='bold')
                ax3.set_ylabel('情感分数', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.2, linestyle='--')
                
                # 添加图例
                legend_elements = [
                    Patch(facecolor='green', alpha=0.3, label='积极 (≥0.05)'),
                    Patch(facecolor='gray', alpha=0.3, label='中性 (-0.05~0.05)'),
                    Patch(facecolor='red', alpha=0.3, label='消极 (≤-0.05)')
                ]
                
                ax3.legend(handles=legend_elements, loc='upper right', 
                          fontsize=9, framealpha=0.9)
                
                # 添加关键统计信息
                scores = df['score'].dropna()
                if len(scores) > 0:
                    stats_text = (
                        f'分数统计:\n'
                        f'• 平均值: {scores.mean():.3f}\n'
                        f'• 中位数: {scores.median():.3f}\n'
                        f'• 标准差: {scores.std():.3f}\n'
                        f'• 样本数: {len(scores):,}'
                    )
                    
                    ax3.text(0.02, 0.98, stats_text,
                            transform=ax3.transAxes,
                            verticalalignment='top',
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.5", 
                                     facecolor='lightyellow', 
                                     alpha=0.8))
                
            except Exception as e:
                print(f"绘制分数分布图时出错: {e}")
                ax3.text(0.5, 0.5, '情感分数数据异常',
                        ha='center', va='center',
                        fontsize=12, transform=ax3.transAxes)
        else:
            # 如果没有分数数据，显示情感强度分布
            if 'sentiment' in df.columns:
                # 计算情感强度分布
                sentiment_intensity = sentiment_counts
                bars_ax3 = ax3.bar(range(len(sentiment_intensity)), sentiment_intensity.values,
                                  color=color_list,
                                  edgecolor='black', linewidth=1.5,
                                  width=0.6)
                
                ax3.set_title('(c) 情感强度分布', fontsize=14, fontweight='bold', pad=20)
                ax3.set_xlabel('情感类别', fontsize=12, fontweight='bold')
                ax3.set_ylabel('评论数量', fontsize=12, fontweight='bold')
                ax3.set_xticks(range(len(sentiment_intensity)))
                ax3.set_xticklabels([s.title() for s in sentiment_intensity.index], 
                                   fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
            else:
                ax3.text(0.5, 0.5, '无情感分数数据',
                        ha='center', va='center',
                        fontsize=12, transform=ax3.transAxes)
                ax3.set_title('(c) 情感分数分布图', fontsize=14, fontweight='bold', pad=20)
        
        # 添加主标题
        plt.suptitle(title, fontsize=18, fontweight='bold', y=1)
        
        # 添加脚注
        fig.text(0.5, 0.01, 
                f'数据来源: Amazon Review 2023 宠物用品数据集 | 样本数量: {total:,}',
                ha='center', fontsize=9, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"图表已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表时出错: {e}")
        
        plt.show()
        
        return fig, axes
    
    @staticmethod
    def plot_score_distribution(df, save_path=None, title="情感分数分布分析"):
        """
        绘制专业级情感分数分布组合图
        根据图片信息优化：左侧直方图 + 右侧箱线图
        
        参数:
            df: DataFrame，包含'score'和'sentiment'列
            save_path: 保存路径
            title: 图表标题
        """
        if df.empty or 'score' not in df.columns:
            print("没有分数数据可可视化")
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        # ========== 中文显示优化 ==========
        # 设置中文字体，解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置专业图表样式
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'SimHei'  # 确保使用中文字体
        })
        
        # 准备数据
        scores = df['score'].dropna()
        
        if len(scores) == 0:
            print("没有有效的分数数据")
            return
        
        # 创建组合图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # ========== 左侧：直方图 (根据图片信息) ==========
        # 图片中显示蓝色柱状图，有绿色和黄色虚线
        n_bins = 30
        n, bins, patches = ax1.hist(scores, bins=n_bins, alpha=0.7, 
                                   edgecolor='black', linewidth=0.8,
                                   color='#2196F3',  # Material Blue
                                   density=True,  # 图片中Y轴是Proportion
                                   label='分数分布')
        
        # 计算关键统计量
        mean_score = scores.mean()
        median_score = scores.median()
        std_score = scores.std()
        
        # 添加统计线（根据图片中的绿色和黄色虚线）
        # 绿色虚线 - 均值
        ax1.axvline(mean_score, color='green', linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'均值: {mean_score:.3f}')
        
        # 黄色虚线 - 中位数
        ax1.axvline(median_score, color='orange', linestyle='--', 
                   linewidth=2, alpha=0.8, label=f'中位数: {median_score:.3f}')
        
        # 添加±1标准差线
        ax1.axvline(mean_score + std_score, color='red', linestyle=':', 
                   linewidth=1.5, alpha=0.6, label='+1 标准差')
        ax1.axvline(mean_score - std_score, color='red', linestyle=':', 
                   linewidth=1.5, alpha=0.6, label='-1 标准差')
        
        # 设置标题和标签（中文优化）
        ax1.set_title('(a) 情感分数直方图', fontsize=13, fontweight='bold', pad=12)
        ax1.set_xlabel('情感评分', fontsize=11, fontweight='bold')
        ax1.set_ylabel('比例', fontsize=11, fontweight='bold')
        
        # 设置X轴刻度（根据图片中的刻度）
        x_min, x_max = scores.min(), scores.max()
        x_range = x_max - x_min
        
        # 设置合理的刻度范围
        x_ticks = np.linspace(-1.0, 1.0, 9)  # 从-1到1，分成8个区间
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f'{x:.2f}' for x in x_ticks])
        
        # 设置Y轴刻度
        y_max = n.max() * 1.2
        y_ticks = np.linspace(0, y_max, 6)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        
        # 添加网格
        ax1.grid(True, alpha=0.3, linestyle='-')
        
        # 添加图例（中文优化）
        ax1.legend(loc='upper right', framealpha=0.9, prop={'family': 'SimHei'})
        
        # 添加统计信息文本框（中文优化）
        stats_text = (
            f'统计信息:\n'
            f'• 样本量 N = {len(scores):,}\n'
            f'• 均值 Mean = {mean_score:.3f}\n'
            f'• 标准差 Std = {std_score:.3f}\n'
            f'• 极差 Range = [{scores.min():.3f}, {scores.max():.3f}]\n'
            f'• 偏度 Skewness = {scores.skew():.3f}\n'
            f'• 峰度 Kurtosis = {scores.kurtosis():.3f}'
        )
        
        ax1.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                fontsize=9,
                fontfamily='SimHei',  # 确保文本框内中文显示正确
                bbox=dict(boxstyle="round,pad=0.4", 
                         facecolor='lightyellow', 
                         edgecolor='gray',
                         alpha=0.9))
        
        # ========== 右侧：分组箱线图 (根据图片信息) ==========
        # 图片中显示Positive(绿色), Neutral(灰色), Negative(红色)的箱线图
        
        if 'sentiment' in df.columns:
            # 准备分组数据
            sentiment_data = []
            sentiment_labels = []
            box_colors = []
            
            # 情感类别顺序（中文优化）
            sentiments = ['积极', '中性', '消极']  # 改为中文标签
            sentiment_map = {
                'positive': '积极',
                'neutral': '中性', 
                'negative': '消极'
            }
            color_map = {
                '积极': '#4CAF50',  # Material Green
                '中性': '#9E9E9E',   # Material Grey
                '消极': '#F44336'   # Material Red
            }
            
            # 处理情感数据，映射到中文
            for eng_sentiment, chn_sentiment in sentiment_map.items():
                if eng_sentiment in df['sentiment'].str.lower().values:
                    sentiment_scores = df[df['sentiment'].str.lower() == eng_sentiment]['score'].dropna()
                    if len(sentiment_scores) > 0:
                        sentiment_data.append(sentiment_scores)
                        sentiment_labels.append(chn_sentiment)
                        box_colors.append(color_map[chn_sentiment])
            
            if sentiment_data:
                # 创建箱线图
                box_plot = ax2.boxplot(sentiment_data, 
                                      labels=sentiment_labels,
                                      patch_artist=True,
                                      widths=0.6,
                                      medianprops=dict(color='black', linewidth=2),
                                      whiskerprops=dict(color='black', linewidth=1.5),
                                      capprops=dict(color='black', linewidth=1.5),
                                      flierprops=dict(marker='o', 
                                                     markerfacecolor='black',
                                                     markersize=4,
                                                     alpha=0.5))
                
                # 设置箱体颜色
                for i, box in enumerate(box_plot['boxes']):
                    box.set_facecolor(box_colors[i])
                    box.set_alpha(0.7)
                    box.set_edgecolor('black')
                    box.set_linewidth(1.5)
                
                # 添加平均值点
                for i, data in enumerate(sentiment_data, 1):
                    mean_val = np.mean(data)
                    ax2.plot(i, mean_val, 'D',  # 菱形标记
                            color='white',
                            markersize=8,
                            markeredgecolor='black',
                            markeredgewidth=1,
                            label=f'均值 ({sentiment_labels[i-1]})' if i == 1 else "")
                
                # 设置标题和标签（中文优化）
                ax2.set_title('(b) 情感类别的箱线图', fontsize=13, fontweight='bold', pad=12)
                ax2.set_xlabel('情感类别', fontsize=11, fontweight='bold')
                ax2.set_ylabel('情感评分', fontsize=11, fontweight='bold')
                
                # 添加水平参考线
                ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
                ax2.axhline(y=0.05, color='green', linestyle=':', alpha=0.5, linewidth=1)
                ax2.axhline(y=-0.05, color='red', linestyle=':', alpha=0.5, linewidth=1)
                
                # 设置Y轴范围
                all_scores = np.concatenate(sentiment_data)
                y_min, y_max = all_scores.min(), all_scores.max()
                y_range = y_max - y_min
                ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                # 设置网格
                ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                # 添加图例（中文优化）
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#4CAF50', alpha=0.7, label='积极'),
                    Patch(facecolor='#9E9E9E', alpha=0.7, label='中性'),
                    Patch(facecolor='#F44336', alpha=0.7, label='消极'),
                    plt.Line2D([0], [0], marker='D', color='w', 
                              markerfacecolor='white', markeredgecolor='black',
                              markersize=8, label='均值')
                ]
                
                ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9, prop={'family': 'SimHei'})
                
                # 添加统计信息表格（中文优化）
                stats_data = []
                for i, (label, data) in enumerate(zip(sentiment_labels, sentiment_data)):
                    stats_data.append([
                        label,
                        f'{len(data):,}',
                        f'{np.mean(data):.3f}',
                        f'{np.std(data):.3f}',
                        f'{np.median(data):.3f}',
                        f'[{np.min(data):.3f}, {np.max(data):.3f}]'
                    ])
                
                # 创建表格（中文表头）
                col_labels = ['类别', '样本量', '均值', '标准差', '中位数', '范围']
                
                # 调整表格位置
                table = ax2.table(cellText=stats_data,
                                 colLabels=col_labels,
                                 cellLoc='center',
                                 colWidths=[0.12, 0.1, 0.12, 0.12, 0.12, 0.2],
                                 bbox=[0.02, -0.4, 0.96, 0.3],
                                 edges='closed')
                
                # 美化表格（中文优化）
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # 标题行
                        cell.set_text_props(fontweight='bold', color='white', fontfamily='SimHei')
                        cell.set_facecolor('#2196F3')
                    else:
                        cell.set_facecolor('#F5F5F5')
                        cell.set_text_props(fontfamily='SimHei')  # 表格内容也使用中文字体
                    cell.set_edgecolor('gray')
                    cell.set_linewidth(0.5)
                
            else:
                # 没有情感分类数据（中文优化）
                ax2.text(0.5, 0.5, '无情感分类数据',
                        ha='center', va='center',
                        fontsize=12, transform=ax2.transAxes, fontfamily='SimHei')
                ax2.set_title('(b) 情感类别的箱线图', fontsize=13, fontweight='bold')
        else:
            # 没有情感列（中文优化）
            ax2.text(0.5, 0.5, '无情感数据可用',
                    ha='center', va='center',
                    fontsize=12, transform=ax2.transAxes, fontfamily='SimHei')
            ax2.set_title('(b) 情感类别的箱线图', fontsize=13, fontweight='bold')
        
        # 添加主标题（中文优化）
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # 添加脚注（中文优化）
        fig.text(0.5, 0.01, 
                f'数据来源: Amazon宠物用品评论2023 | '
                f'分析时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
                ha='center', fontsize=9, style='italic', fontfamily='SimHei',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"图表已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表时出错: {e}")
        
        plt.show()

        return fig, (ax1, ax2)
# ==================== 主分析器 ====================
class SentimentAnalysisPipeline:
    """情感分析流水线"""
    
    def __init__(self, db_path, method='rule_based'):
        self.db = DuckDBManager(db_path)
        self.method = method
        
        if method == 'rule_based':
            self.analyzer = RuleBasedSentimentAnalyzer(
                positive_threshold=Config.POSITIVE_THRESHOLD,
                negative_threshold=Config.NEGATIVE_THRESHOLD
            )
        else:
            raise ValueError(f"不支持的分析方法: {method}")
        
        self.preprocessor = TextPreprocessor()
    
    def run(self, limit=None, sample=False, visualize=True, save_results=True):
        """运行情感分析"""
        print("=" * 60)
        print("开始情感分析流水线")
        print(f"数据库: {self.db.db_path}")
        print(f"分析方法: {self.method}")
        print("=" * 60)
        
        # 1. 连接数据库
        if not self.db.connect():
            return None
        
        try:
            # 2. 获取表信息
            tables = self.db.get_table_info()
            comments_table = Config.TABLE_NAMES['comments']
            
            if comments_table not in tables:
                print(f"错误: 评论表 '{comments_table}' 不存在")
                print(f"可用的表: {tables}")
                return None
            
            # 3. 预览数据
            self.db.get_table_preview(comments_table, limit=3)
            
            # 4. 加载评论数据
            print("\n" + "=" * 60)
            print("加载评论数据...")
            comments_df = self.db.load_comments(comments_table, limit=limit, sample=sample)
            
            if comments_df.empty:
                print("错误: 没有加载到评论数据")
                return None
            
            print(f"成功加载 {len(comments_df)} 条评论")
            
            # 5. 检查必要的列
            if 'clean_text' not in comments_df.columns:
                print("警告: 没有找到 'clean_text' 列，使用原始文本")
                text_column = 'review_body' if 'review_body' in comments_df.columns else 'text'
            else:
                text_column = 'clean_text'
            
            # 6. 情感分析
            print("\n" + "=" * 60)
            print("开始情感分析...")
            
            # 提取文本
            texts = comments_df[text_column].fillna('').tolist()
            
            # 预处理文本
            print("预处理文本...")
            texts = [self.preprocessor.normalize_text(t) for t in texts]
            
            # 分析情感
            print("分析情感...")
            sentiment_results = self.analyzer.analyze_batch(texts)
            
            # 7. 合并结果
            print("合并结果...")
            result_df = comments_df.copy()
            
            # 添加情感分析结果
            result_df['sentiment'] = sentiment_results['sentiment']
            result_df['score'] = sentiment_results['score']
            result_df['confidence'] = sentiment_results['confidence']
            result_df['analysis_timestamp'] = datetime.now()
            result_df['analysis_method'] = self.method
            
            # 8. 显示统计信息
            print("\n" + "=" * 60)
            print("情感分析结果统计:")
            print("=" * 60)
            
            total = len(result_df)
            positive = len(result_df[result_df['sentiment'] == 'positive'])
            negative = len(result_df[result_df['sentiment'] == 'negative'])
            neutral = len(result_df[result_df['sentiment'] == 'neutral'])
            
            print(f"总评论数: {total}")
            print(f"正面评论: {positive} ({positive/total*100:.1f}%)")
            print(f"负面评论: {negative} ({negative/total*100:.1f}%)")
            print(f"中性评论: {neutral} ({neutral/total*100:.1f}%)")
            print(f"平均分数: {result_df['score'].mean():.3f}")
            print(f"平均置信度: {result_df['confidence'].mean():.3f}")
            
            # 9. 保存结果
            if save_results:
                print("\n" + "=" * 60)
                print("保存结果到数据库...")
                success = self.db.save_sentiment_results(
                    result_df, 
                    Config.TABLE_NAMES['sentiment_results']
                )
                
                if success:
                    # 查询并显示保存的统计
                    stats = self.db.query_sentiment_stats(Config.TABLE_NAMES['sentiment_results'])
                    if stats is not None:
                        print("\n数据库中的情感统计:")
                        print(stats)
            
            # 10. 可视化
            if visualize and not result_df.empty:
                print("\n" + "=" * 60)
                print("生成可视化图表...")
                Visualizer.plot_sentiment_distribution(result_df)
                
                if 'score' in result_df.columns:
                    Visualizer.plot_score_distribution(result_df)
            
            print("\n" + "=" * 60)
            print("情感分析完成!")
            print("=" * 60)
            
            return result_df
            
        finally:
            # 确保关闭数据库连接
            self.db.disconnect()

# ==================== 主函数 ====================
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Amazon宠物用品评论情感分析')
    parser.add_argument('--db-path', default=Config.DB_PATH, 
                       help='DuckDB文件路径 (默认: data/amazon_pets.duckdb)')
    parser.add_argument('--method', choices=['rule_based'], default='rule_based',
                       help='分析方法 (目前只支持rule_based)')
    parser.add_argument('--limit', type=int, default=None, 
                       help='限制分析数量 (用于测试)')
    parser.add_argument('--sample', action='store_true', 
                       help='随机抽样1000条评论 (用于测试)')
    parser.add_argument('--no-visualize', action='store_true', 
                       help='不显示可视化图表')
    parser.add_argument('--no-save', action='store_true', 
                       help='不保存结果到数据库')
    
    args = parser.parse_args()
    
    # 检查数据库文件是否存在
    db_path = args.db_path
    if not Path(db_path).exists():
        print(f"错误: DuckDB文件不存在: {db_path}")
        print("请检查文件路径，或使用 --db-path 参数指定正确路径")
        return
    
    # 创建并运行分析流水线
    pipeline = SentimentAnalysisPipeline(
        db_path=db_path,
        method=args.method
    )
    
    result = pipeline.run(
        limit=args.limit,
        sample=args.sample,
        visualize=not args.no_visualize,
        save_results=not args.no_save
    )
    
    if result is not None and not args.no_save:
        # 提供一些查询示例
        print("\n查询示例:")
        print("1. 查看前5条分析结果:")
        print(result[['asin', 'clean_text' if 'clean_text' in result.columns else 'review_body', 
                     'sentiment', 'score']].head())
        
        print("\n2. 情感分布:")
        print(result['sentiment'].value_counts())
        
        # 保存结果到CSV文件
        csv_file = 'sentiment_results.csv'
        result.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n详细结果已保存到: {csv_file}")

# ==================== 快速测试函数 ====================
def quick_test():
    """快速测试函数"""
    db_path = Config.DB_PATH
    
    if not Path(db_path).exists():
        print(f"测试失败: DuckDB文件不存在: {db_path}")
        print("请先确保数据库文件存在")
        return
    
    print("=" * 60)
    print("快速测试情感分析")
    print("=" * 60)
    
    # 测试数据库连接
    db = DuckDBManager(db_path)
    if db.connect():
        tables = db.get_table_info()
        if tables:
            print(f"\n数据库包含以下表: {tables}")
            
            # 测试加载少量数据
            for table in tables:
                if 'comment' in table.lower():
                    df = db.load_comments(table, limit=3)
                    if not df.empty:
                        print(f"\n表 '{table}' 包含列: {list(df.columns)}")
                        break
        db.disconnect()
    
    # 测试分析少量数据
    print("\n" + "=" * 60)
    print("测试情感分析 (分析5条数据)...")
    print("=" * 60)
    
    pipeline = SentimentAnalysisPipeline(db_path)
    result = pipeline.run(limit=5, visualize=False, save_results=False)
    
    if result is not None:
        print("\n测试成功!")
        print(f"分析了 {len(result)} 条评论")
        print("\n前3条结果:")
        for i, row in result.head(3).iterrows():
            text_preview = row.get('clean_text', row.get('review_body', ''))[:50]
            print(f"{i+1}. [{row['sentiment']}] {text_preview}... (分数: {row['score']:.3f})")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    import sys
    
    # 检查是否安装了vaderSentiment
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("错误: 需要安装 vaderSentiment 库")
        print("请运行: pip install vaderSentiment")
        sys.exit(1)
    
    # 检查命令行参数
    if len(sys.argv) == 1:
        # 没有参数，显示帮助
        print("使用方法:")
        print("  python sentiment_analysis.py --db-path 路径/到/你的数据库.duckdb")
        print("  python sentiment_analysis.py --limit 1000  (分析1000条)")
        print("  python sentiment_analysis.py --sample  (随机抽样1000条)")
        print("  python sentiment_analysis.py --no-visualize  (不显示图表)")
        print("\n快速测试:")
        print("  python sentiment_analysis.py test")
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        # 运行快速测试
        quick_test()
    else:
        # 运行主程序
        main()