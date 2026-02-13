#!/usr/bin/env python3
"""
生成Excel表格用于人工标注情感分析结果
功能：从DuckDB中筛选100条评论，生成包含以下列的Excel表格：
- 评论ID
- 评论原文
- 人工标注情感标签
- 模型预测情感标签
- 是否一致
"""

import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime

class Config:
    """配置类"""
    # DuckDB文件路径
    DB_PATH = "data/amazon_pets.duckdb"
    
    # 表名配置
    TABLE_NAMES = {
        'comments': 'pets_comment_cleaned',
        'sentiment_results': 'pets_comment_sentiment'
    }
    
    # 输出Excel文件路径
    OUTPUT_EXCEL = f'annotation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

class ExcelGenerator:
    """Excel表格生成器"""
    
    def __init__(self, db_path):
        """初始化
        
        Args:
            db_path: DuckDB文件路径
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接到DuckDB数据库
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if not Path(self.db_path).exists():
                print(f"错误: DuckDB文件不存在: {self.db_path}")
                return False
            
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
        """获取数据库表信息
        
        Returns:
            list: 表名列表
        """
        try:
            tables = self.conn.execute("SHOW TABLES;").fetchall()
            table_list = [t[0] for t in tables]
            print(f"数据库中的表: {table_list}")
            return table_list
        except Exception as e:
            print(f"获取表信息失败: {e}")
            return []
    
    def check_tables(self):
        """检查必要的表是否存在
        
        Returns:
            bool: 表是否存在
        """
        tables = self.get_table_info()
        comments_table = Config.TABLE_NAMES['comments']
        sentiment_table = Config.TABLE_NAMES['sentiment_results']
        
        if comments_table not in tables:
            print(f"错误: 评论表 '{comments_table}' 不存在")
            return False
        
        if sentiment_table not in tables:
            print(f"错误: 情感分析结果表 '{sentiment_table}' 不存在")
            print("请先运行 sentiment_analysis.py 生成情感分析结果")
            return False
        
        return True
    
    def sample_comments(self, sample_size=100):
        """从数据库中随机抽样评论
        
        Args:
            sample_size: 抽样数量
            
        Returns:
            pd.DataFrame: 抽样的评论数据
        """
        try:
            comments_table = Config.TABLE_NAMES['comments']
            sentiment_table = Config.TABLE_NAMES['sentiment_results']
            
            # 构建查询，确保每条评论都有对应的情感分析结果
            query = f"""
                SELECT 
                    c.id as comment_id,
                    c.text as original_comment,
                    s.sentiment as model_predicted_sentiment
                FROM {comments_table} c
                JOIN {sentiment_table} s ON c.asin = s.asin AND c.clean_text = s.clean_text
                WHERE c.text IS NOT NULL AND c.text != ''
                USING SAMPLE {sample_size} ROWS
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """
            
            print(f"执行抽样查询，获取 {sample_size} 条评论...")
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                print("警告: 没有获取到足够的评论数据")
                # 如果没有足够的数据，尝试不使用JOIN
                fallback_query = f"""
                    SELECT 
                        id as comment_id,
                        text as original_comment
                    FROM {comments_table}
                    WHERE text IS NOT NULL AND text != ''
                    USING SAMPLE {sample_size} ROWS
                    ORDER BY RANDOM()
                    LIMIT {sample_size}
                """
                df = self.conn.execute(fallback_query).fetchdf()
                df['model_predicted_sentiment'] = 'unknown'
            
            print(f"成功获取 {len(df)} 条评论")
            return df
        except Exception as e:
            print(f"抽样评论失败: {e}")
            return pd.DataFrame()
    
    def generate_excel(self, df, output_file):
        """生成Excel表格
        
        Args:
            df: 评论数据
            output_file: 输出文件路径
        """
        try:
            # 添加必要的列
            df['人工标注情感标签'] = ''  # 留空，等待人工标注
            df['是否一致'] = ''  # 留空，等待人工标注后计算
            
            # 调整列顺序
            columns_order = ['comment_id', 'original_comment', 'model_predicted_sentiment', '人工标注情感标签', '是否一致']
            df = df[columns_order]
            
            # 重命名列
            df = df.rename(columns={
                'comment_id': '评论ID',
                'original_comment': '评论原文',
                'model_predicted_sentiment': '模型预测情感标签'
            })
            
            # 生成Excel文件
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='情感标注', index=False)
                
                # 获取工作表
                worksheet = writer.sheets['情感标注']
                
                # 调整列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    # 设置列宽（留一些余量）
                    adjusted_width = min(max_length + 2, 100)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            
            print(f"成功生成Excel文件: {output_file}")
            print(f"\n文件包含 {len(df)} 条评论，需要人工标注")
            print("标注说明:")
            print("1. 人工标注情感标签: 填写 positive/negative/neutral")
            print("2. 是否一致: 标注完成后，比较人工标注与模型预测是否一致，填写 是/否")
            
        except Exception as e:
            print(f"生成Excel文件失败: {e}")
    
    def run(self):
        """运行生成Excel的完整流程"""
        print("=" * 60)
        print("开始生成情感标注Excel表格")
        print(f"数据库: {self.db_path}")
        print(f"输出文件: {Config.OUTPUT_EXCEL}")
        print("=" * 60)
        
        # 1. 连接数据库
        if not self.connect():
            return False
        
        try:
            # 2. 检查必要的表
            if not self.check_tables():
                return False
            
            # 3. 抽样评论
            df = self.sample_comments(sample_size=100)
            
            if df.empty:
                print("错误: 没有获取到评论数据")
                return False
            
            # 4. 生成Excel
            self.generate_excel(df, Config.OUTPUT_EXCEL)
            
            print("\n" + "=" * 60)
            print("Excel表格生成完成!")
            print(f"请打开文件: {Config.OUTPUT_EXCEL} 进行人工标注")
            print("=" * 60)
            
            return True
            
        finally:
            # 确保关闭数据库连接
            self.disconnect()

def main():
    """主函数"""
    # 检查是否安装了必要的库
    try:
        import openpyxl
    except ImportError:
        print("错误: 需要安装 openpyxl 库")
        print("请运行: pip install openpyxl")
        return
    
    # 创建并运行Excel生成器
    generator = ExcelGenerator(Config.DB_PATH)
    generator.run()

if __name__ == "__main__":
    main()
