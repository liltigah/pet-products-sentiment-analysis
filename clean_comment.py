import duckdb
import time

# ================= 配置 =================
DB_PATH = "amazon_pets.duckdb"
SOURCE_TABLE = "pets_comment"

# 输出的三张表
TABLE_DUPLICATES = "pets_comment_duplicates"
TABLE_INVALID = "pets_comment_invalid"
TABLE_CLEANED = "pets_comment_cleaned"


# =======================================

def sieve_clean_pipeline():
    print(f"🚀 [任务] 启动“物理筛选/删除”模式数据清洗管道...")
    con = duckdb.connect(DB_PATH)
    start_time = time.time()

    # 1. 获取原始总账
    total_raw = con.sql(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]
    print(f"📊 原始总记录: {total_raw:,} 条")
    print("-" * 50)

    # -------------------------------------------------------
    # 步骤 0: 创建中间工作台 (Staged Data)
    # -------------------------------------------------------
    # 我们创建一个实体表 staged_comment，因为我们要对它进行 DELETE 操作
    print(f"🔨 [步骤 0] 构建中间工作台 (计算行号)...")

    # 为了让 DELETE 更快，我们不仅计算 rn，还把清洗后的文本 clean_text 算出来存好
    con.execute(f"""
    CREATE OR REPLACE TABLE staged_comment AS
    SELECT 
        -- [新增] 全局自增主键 (从 1 开始)
        -- OVER() 里面不加条件表示对全表生成序号
        ROW_NUMBER() OVER () AS id,
        *,
        -- [新增] 时间戳清洗：将毫秒级整数转为可读的时间格式
        -- 结果示例：2023-02-04 12:30:45
        epoch_ms(timestamp) AS comment_time,
        -- 预先计算清洗后的文本
        trim(regexp_replace(text, '<[^>]+>', ' ', 'g')) AS clean_text,
        -- 计算行号 (用于判断重复)
        ROW_NUMBER() OVER(
            PARTITION BY parent_asin, user_id, timestamp, rating, title, helpful_vote, verified_purchase, trim(regexp_replace(text, '<[^>]+>', ' ', 'g'))
            ORDER BY timestamp
        ) as rn
    FROM {SOURCE_TABLE}
    """)

    # 创建备份表，用于回溯。
    con.execute(f"""
    CREATE OR REPLACE TABLE staged_comment_bak AS
    SELECT * from staged_comment
    """)

    # -------------------------------------------------------
    # 步骤 1: 处理重复数据 (Duplicates)
    # -------------------------------------------------------
    print(f"🔪 [步骤 1] 正在提取重复数据...")

    # 1.1 先把重复的存到目标表
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_DUPLICATES} AS
        SELECT * 
        FROM staged_comment 
        WHERE rn > 1
    """)
    count_duplicates = con.sql(f"SELECT COUNT(*) FROM {TABLE_DUPLICATES}").fetchone()[0]
    print(f"   -> 已提取重复: {count_duplicates:,} 条")

    # 1.2 【关键操作】从工作台中物理删除重复数据
    print(f"   -> 正在从工作台中删除重复数据...")
    con.execute("DELETE FROM staged_comment WHERE rn > 1")

    # 验证：现在 staged_comment 里应该全是 rn=1 的唯一数据了

    # -------------------------------------------------------
    # 步骤 2: 处理废弃数据 (Invalid)
    # -------------------------------------------------------
    print(f"🔪 [步骤 2] 正在提取废弃数据...")

    # 定义废弃条件 (直接复用)
    condition_invalid = """
        (clean_text IS NULL OR clean_text = '') OR 
        rating IS NULL OR 
        verified_purchase IS DISTINCT FROM true OR 
        length(clean_text) <= 5
    """

    # 2.1 先把废弃的存到目标表
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_INVALID} AS
        SELECT 
            * ,
            CASE 
                WHEN clean_text IS NULL OR clean_text = '' THEN '缺失评论内容'
                    WHEN rating IS NULL THEN '缺失评分'
                WHEN verified_purchase IS DISTINCT FROM true THEN '非真实购买'
                WHEN length(clean_text) <= 5 THEN '内容过短'
                ELSE '其他'
            END AS rejection_reason
        FROM staged_comment 
        WHERE {condition_invalid}
    """)
    count_invalid = con.sql(f"SELECT COUNT(*) FROM {TABLE_INVALID}").fetchone()[0]
    print(f"   -> 已提取废弃: {count_invalid:,} 条")

    # 2.2 【关键操作】从工作台中物理删除废弃数据
    print(f"   -> 正在从工作台中删除废弃数据...")
    con.execute(f"DELETE FROM staged_comment WHERE {condition_invalid}")

    # -------------------------------------------------------
    # 步骤 3: 剩余即有效 (Remaining is Clean)
    # -------------------------------------------------------
    print(f"🧼 [步骤 3] 收割剩余的干净数据...")

    # 3.1 此时 staged_comment 里剩下的，就是这就通过了前两轮筛选的幸存者
    # 我们不需要再写 WHERE 条件了！
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_CLEANED} AS
        SELECT * FROM staged_comment
    """)
    count_cleaned = con.sql(f"SELECT COUNT(*) FROM {TABLE_CLEANED}").fetchone()[0]
    print(f"   -> 最终入库: {count_cleaned:,} 条")

    # -------------------------------------------------------
    # 步骤 4: 终极验证
    # -------------------------------------------------------
    print("-" * 40)
    print("🧮 最终数字校验 (物理删除验证法):")

    sum_parts = count_cleaned + count_duplicates + count_invalid
    diff = total_raw - sum_parts

    print(f"   1. 重复剔除 (Duplicates)  : {count_duplicates:>12,}")
    print(f" + 2. 质量剔除 (Invalid)     : {count_invalid:>12,}")
    print(f" + 3. 最终有效 (Cleaned)     : {count_cleaned:>12,}")
    print(f" = 三表之和                  : {sum_parts:>12,}")
    print(f"   原始总数                  : {total_raw:>12,}")
    print(f"   ---------------------------------------")

    if diff == 0:
        print(f"✅ 完美匹配！逻辑绝对闭环。")
    else:
        print(f"❌ 警告：仍有差额 {diff} 条！")

    # 清理工作台 (用完了就删掉)
    con.execute("DROP TABLE IF EXISTS staged_comment")

    con.close()
    print(f"\n🎉 流程结束! 耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    sieve_clean_pipeline()