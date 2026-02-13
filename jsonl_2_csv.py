import pandas as pd
import os
import time


def convert_jsonl_to_csv(jsonl_file, csv_file, chunk_size=10000):
    """
    将 JSONL 文件分块转换为 CSV 文件，以避免内存溢出。

    Args:
        jsonl_file (str): 输入的 jsonl 文件路径
        csv_file (str): 输出的 csv 文件路径
        chunk_size (int): 每次处理的行数
    """
    print(f"开始转换: {jsonl_file} -> {csv_file}")
    start_time = time.time()

    # 检查文件是否存在
    if not os.path.exists(jsonl_file):
        print(f"错误: 找不到文件 {jsonl_file}")
        return

    try:
        # 使用 chunksize 分块读取数据
        # lines=True 表示这是 jsonl 格式（每一行是一个 json 对象）
        with pd.read_json(jsonl_file, lines=True, chunksize=chunk_size) as reader:
            for i, chunk in enumerate(reader):
                # 如果是第一块，写入模式为 'w' (覆盖)，并保留表头
                # 如果是后续块，写入模式为 'a' (追加)，并不再写入表头
                mode = 'w' if i == 0 else 'a'
                header = True if i == 0 else False

                # 将数据写入 CSV，index=False 表示不保存 pandas 的索引列
                chunk.to_csv(csv_file, mode=mode, header=header, index=False, encoding='utf-8-sig')

                print(f"  - 已处理第 {i + 1} 块 (每块 {chunk_size} 条数据)...")

        end_time = time.time()
        print(f"转换完成！耗时: {end_time - start_time:.2f} 秒")
        print(f"输出文件: {csv_file}\n" + "-" * 30)

    except ValueError as e:
        print(f"读取错误: {e}")
        print("提示: 请检查文件格式是否正确，或者文件是否损坏。")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请在这里替换为你下载的实际文件名
    # 通常 Amazon 数据集包含一个 Review 文件和一个 Meta (Data) 文件

    # 1. 设置 Review 数据的文件名
    review_jsonl = "Pet_Supplies.jsonl/Pet_Supplies.jsonl"  # 修改这里
    review_csv = "Pet_Supplies_Reviews.csv"

    # 2. 设置 Meta 数据的文件名
    meta_jsonl = "meta_Pet_Supplies.jsonl/meta_Pet_Supplies.jsonl"  # 修改这里
    meta_csv = "Pet_Supplies_Meta.csv"
    # ===========================================

    # 执行转换
    # 如果你下载的文件名不同，请确保上面配置区域已修改

    # 转换评论数据
    if os.path.exists(review_jsonl):
        convert_jsonl_to_csv(review_jsonl, review_csv)
    else:
        print(f"跳过: 未找到 {review_jsonl}，请检查文件名。")

    # 转换元数据
    if os.path.exists(meta_jsonl):
        convert_jsonl_to_csv(meta_jsonl, meta_csv)
    else:
        print(f"跳过: 未找到 {meta_jsonl}，请检查文件名。")