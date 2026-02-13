import duckdb
import os
import time


def import_data_to_duckdb(db_path, meta_file, comment_file):
    """
    将 Amazon Review 和 Meta 数据导入 DuckDB (修正版：解决类型推断错误)
    """
    # 1. 连接数据库

    con = duckdb.connect(database=db_path, read_only=False)

    print(f"🔗 已连接数据库: {db_path}")
    print("-" * 40)

    # ---------------------------------------------------------
    # 任务 A: 导入元数据 (Meta) - 已修复报错
    # ---------------------------------------------------------
    if os.path.exists(meta_file):
        print(f"📦 正在导入商品元数据: {meta_file} ...")
        print("   (正在全量扫描以推断数据类型，请稍候...)")
        start_time = time.time()

        # 【关键修改】：添加 sample_size=-1
        # 这会强制 DuckDB 扫描整个文件来确定列类型，避免因后面出现的特殊字符导致报错
        try:
            con.execute(f"""
                 CREATE OR REPLACE TABLE pets_meta AS 
                 SELECT * FROM read_json_auto('{meta_file}', sample_size=-1)
             """)

            row_count = con.execute("SELECT COUNT(*) FROM pets_meta").fetchone()[0]
            print(f"✅ 元数据表 (pets_meta) 导入完成！")
            print(f"   耗时: {time.time() - start_time:.2f} 秒")
            print(f"   总行数: {row_count}")
        except Exception as e:
            print(f"❌ 导入 Meta 失败: {e}")
    else:
        print(f"⚠️ 跳过: 未找到文件 {meta_file}")

    print("-" * 40)

    # ---------------------------------------------------------
    # 任务 B: 导入评论数据 (Data) - 同样建议加上 sample_size=-1
    # ---------------------------------------------------------
    if os.path.exists(comment_file):
        print(f"📦 正在导入评论数据: {comment_file} ...")
        print("   (正在全量扫描以推断数据类型，请稍候...)")
        start_time = time.time()

        try:
            # 【关键修改】：添加 sample_size=-1
            # 评论数据也可能包含奇怪的格式，加上这个参数更保险
            con.execute(f"""
                   CREATE OR REPLACE TABLE pets_comment AS 
                   SELECT * FROM read_json_auto('{comment_file}', sample_size=-1)
               """)

            row_count = con.execute("SELECT COUNT(*) FROM pets_comment").fetchone()[0]
            print(f"✅ 评论表 (pets_comment) 导入完成！")
            print(f"   耗时: {time.time() - start_time:.2f} 秒")
            print(f"   总行数: {row_count}")
        except Exception as e:
            print(f"❌ 导入 Data 失败: {e}")
    else:
        print(f"⚠️ 跳过: 未找到文件 {comment_file}")

    print("=" * 40)

    # ---------------------------------------------------------
    # 任务 C: 验证与关联查询演示
    # ---------------------------------------------------------
    print("🔎 数据验证：尝试关联两张表 (pets_comment + pets_meta)")

    try:
        # 注意：如果导入成功，某些包含 "—" 的列现在变成了 VARCHAR
        # 在 SQL 中关联通常不受影响，但在计算数值时可能需要 try_cast
        sample_query = """
        SELECT 
            r.rating,
            r.title AS review_title,
            m.title AS product_name,
            m.main_category
        FROM pets_comment r
        JOIN pets_meta m ON r.parent_asin = m.parent_asin
        LIMIT 5;
        """

        result = con.execute(sample_query).df()
        print(result)

    except Exception as e:
        print(f"查询演示失败 (可能是字段名不匹配或表未成功创建): {e}")

    # 关闭连接
    con.close()
    print("\n🎉 所有操作完成。数据库文件已保存。")


if __name__ == "__main__":
    # ================= 配置区域 =================
    DB_NAME = "amazon_pets.duckdb"

    # 保持你原来的路径不变
    FILE_META = "meta_Pet_Supplies.jsonl/meta_Pet_Supplies.jsonl"
    FILE_COMMENT = "Pet_Supplies.jsonl/Pet_Supplies.jsonl"
    # ===========================================

    import_data_to_duckdb(DB_NAME, FILE_META, FILE_COMMENT)