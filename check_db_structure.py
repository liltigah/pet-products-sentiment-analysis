#!/usr/bin/env python3
"""
检查数据库表结构
"""
import duckdb
import pandas as pd

# 连接数据库
db_path = "data/amazon_pets.duckdb"
db = duckdb.connect(database=db_path)

print("=" * 60)
print("检查数据库结构")
print(f"数据库: {db_path}")
print("=" * 60)

# 显示所有表
tables = db.execute("SHOW TABLES;").fetchall()
table_list = [t[0] for t in tables]
print(f"数据库中的表: {table_list}")

# 显示每个表的列
for table in table_list:
    print(f"\n表 '{table}' 的列:")
    try:
        # 获取表的前1行数据，以了解列结构
        df = db.execute(f"SELECT * FROM {table} LIMIT 1;").fetchdf()
        if not df.empty:
            columns = list(df.columns)
            print(f"列数: {len(columns)}")
            print(f"列名: {columns}")
            
            # 显示前几列的样例数据
            print("\n样例数据:")
            print(df.head())
        else:
            print("表为空")
    except Exception as e:
        print(f"获取表结构失败: {e}")

# 关闭数据库连接
db.close()

print("\n" + "=" * 60)
print("数据库结构检查完成")
print("=" * 60)
