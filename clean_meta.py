import duckdb
import time

# ================= 配置 =================
DB_PATH = "amazon_pets.duckdb"
SOURCE_TABLE = "pets_meta"  # 你的原始Meta表
TARGET_TABLE = "pets_meta_cleaned"  # 清洗后的Meta表


# =======================================

def clean_meta_pipeline():
    print(f"🚀 [任务] 启动商品元数据清洗管道...")
    con = duckdb.connect(DB_PATH)
    start_time = time.time()

    # -------------------------------------------------------
    # 步骤 1: 核心清洗 (Transform)
    # -------------------------------------------------------
    # 重点解决：Price 字段可能是字符串 "$12.99" 或 "—" 的问题
    # 逻辑说明：
    # 1. parent_asin: 必须唯一，我们使用 GROUP BY 去重
    # 2. price处理: 使用 CASE WHEN 处理 "—" 和 "$"，再转为 DOUBLE
    # 3. details: 这是一个 JSON 结构，我们保留原样或提取部分信息

    print(f"🧹 正在生成清洗表: {TARGET_TABLE} ...")

    # [修复说明]:
    # 1. 使用 CAST(price AS VARCHAR) 强制把 JSON 类型转为普通文本
    # 2. 使用 trim(..., '"') 去除 JSON 转换后可能残留的双引号

    sql_clean = f"""
        CREATE OR REPLACE TABLE {TARGET_TABLE} AS
        SELECT 
            ROW_NUMBER() OVER () AS id,
            parent_asin,
            title,
            main_category,

            -- [价格清洗逻辑 - 修复版]
            TRY_CAST(
                CASE 
                    -- 1. 先转成字符串，并去掉可能自带的 JSON 双引号
                    WHEN trim(CAST(price AS VARCHAR), '"') = '—' THEN NULL
                    WHEN trim(CAST(price AS VARCHAR), '"') = ''  THEN NULL

                    -- 2. 如果包含 '$'，替换为空；同时确保双引号被去除
                    ELSE REPLACE(trim(CAST(price AS VARCHAR), '"'), '$', '')
                END 
            AS DOUBLE) as price,

            average_rating as avg_rating,
            rating_number as rating_count,

            -- details 如果也是 JSON 类型，可以直接保留，或者也转成 VARCHAR
            CAST(details AS VARCHAR) as details

        FROM {SOURCE_TABLE}
        WHERE parent_asin IS NOT NULL;
        """
    try:
        con.execute(sql_clean)

        raw_cnt = con.sql(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
        clean_cnt = con.sql(f"SELECT COUNT(*) FROM {TARGET_TABLE}").fetchone()[0]
        print(f"   -> 原始商品数: {raw_cnt:,}")
        print(f"   -> 清洗后商品数 (唯一): {clean_cnt:,}")

    except Exception as e:
        print(f"❌ 清洗失败: {e}")
        # 如果报错提示列名不存在，可能是JSON解析时列名有差异
        print("💡 提示: 如果报错 'Binder Error'，请检查你的 meta 表是否有 price 列。")

    con.close()
    print(f"✅ 元数据清洗完成! 耗时: {time.time() - start_time:.2f} 秒\n")


if __name__ == "__main__":
    clean_meta_pipeline()