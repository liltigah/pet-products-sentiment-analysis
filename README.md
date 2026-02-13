# pet-products-sentiment-analysis基于电商平台宠物用品用户评论的数据分析
## 介绍
- 这是一个数据分析小组作业，聚焦宠物用品评论，旨在通过情感分析与评分关联分析，揭示用户情感倾向与评分的一致性，识别“高分低情感”或“低分高情感”等异常评论，以检测潜在问题（如评分欺诈或产品误解），从而为商家优化产品和服务提供数据支持。
## 数据来源
- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
## 数据清洗
- 经过清洗后的数据文件（`amazon_pets.duckdb`，约42.4 GB）因体积过大未上传
- 仓库中的 `sample.duckdb` 包含100条示例数据供测试

## 示例数据库创建情况
### 数据库信息
- 文件路径 ： sample.duckdb
- 总数据量 ：210 条数据
- 包含表数 ：8 个表（与原始数据库结构一致）
### 各表数据分布
| 表名 | 数据量 | 说明 |
|------|--------|------|
| `pets_comment` | 30 条 | 原始评论数据 |
| `pets_comment_cleaned` | 30 条 | 清洗后的评论数据 |
| `pets_comment_duplicates` | 30 条 | 重复评论数据 |
| `pets_comment_invalid` | 30 条 | 无效评论数据 |
| `pets_comment_sentiment` | 30 条 | 情感分析结果 |
| `pets_meta` | 15 条 | 产品元数据 |
| `pets_meta_cleaned` | 15 条 | 清洗后的产品元数据 |
| `staged_comment_bak` | 30 条 | 评论备份数据 |
