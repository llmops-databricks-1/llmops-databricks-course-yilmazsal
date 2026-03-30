# Databricks notebook source
# from azure.ai.documentintelligence.models import ParagraphRole
import os

import pandas as pd
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from loguru import logger
from pyspark.sql import SparkSession

from llmops_databricks.config import ProjectConfig, get_env
from llmops_databricks.document_chunker import chunk_analyze_result

# COMMAND ----------
# create Spark session
spark = SparkSession.builder.getOrCreate()

# load config
env = get_env(spark)
cfg = ProjectConfig.from_yaml(config_path="../project_config.yml", env=env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "policy_docs"

# Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------

# Block 3 — Analyze PDF with Azure Document Intelligence

endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]

pdf_path = "../data/IN_HM Group Customer Privacy Notice.pdf"

di_client = DocumentIntelligenceClient(
    endpoint=endpoint, credential=AzureKeyCredential(key)
)

with open(pdf_path, "rb") as f:
    poller = di_client.begin_analyze_document("prebuilt-layout", body=f)

result = poller.result()

paragraph_count = len(result.paragraphs or [])
table_count = len(result.tables or [])
logger.info(
    f"Document analysis complete: {paragraph_count} paragraphs, {table_count} tables"
)

# COMMAND ----------

# Block 4 — Chunk the document
source_file = "IN_HM Group Customer Privacy Notice.pdf"
chunks = chunk_analyze_result(result, source_file=source_file)

logger.info(f"Produced {len(chunks)} chunks")
for c in chunks:
    logger.debug(
        f"  [{c.chunk_index:02d}] heading={c.section_heading!r:40s} "
        f"chars={c.char_count:4d}  pages={c.page_numbers}"
    )

# COMMAND ----------

# Block 5 — Write chunks to Delta table
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.{TABLE_NAME} (
  chunk_id        STRING     NOT NULL,
  source_file     STRING     NOT NULL,
  document_title  STRING,
  section_heading STRING,
  page_numbers    ARRAY<INT>,
  content         STRING     NOT NULL,
  chunk_index     INT        NOT NULL,
  char_count      INT,
  paragraph_count INT,
  ingestion_ts    TIMESTAMP
)
USING DELTA
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")
logger.info(f"Table {CATALOG}.{SCHEMA}.{TABLE_NAME} ready")

rows = [c.model_dump() for c in chunks]
pdf_df = pd.DataFrame(rows)
pdf_df["ingestion_ts"] = pd.Timestamp.utcnow()
pdf_df["page_numbers"] = pdf_df["page_numbers"].apply(list)

spark_df = spark.createDataFrame(pdf_df)
spark_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
)
logger.info(f"Wrote {len(chunks)} chunks to {CATALOG}.{SCHEMA}.{TABLE_NAME}")

# COMMAND ----------

# Block 6 — Validation
agg = spark.sql(f"""
SELECT
  COUNT(*)                        AS total_chunks,
  COUNT(DISTINCT section_heading) AS distinct_sections,
  MIN(char_count)                 AS min_chars,
  MAX(char_count)                 AS max_chars,
  ROUND(AVG(char_count), 1)       AS avg_chars,
  SUM(CASE WHEN content IS NULL OR content = '' THEN 1 ELSE 0 END) AS empty_content
FROM {CATALOG}.{SCHEMA}.{TABLE_NAME}
""")
agg.show()

logger.info("Spot-check: first 3 chunks (200-char preview)")
spark.sql(f"""
SELECT chunk_index, section_heading, page_numbers, SUBSTRING(content, 1, 200) AS content_preview
FROM {CATALOG}.{SCHEMA}.{TABLE_NAME}
ORDER BY chunk_index
LIMIT 3
""").show(truncate=False)

stats = agg.collect()[0]
assert stats["empty_content"] == 0, (
    f"Found {stats['empty_content']} chunk(s) with empty content"
)
assert stats["max_chars"] <= 3000, (
    f"Largest chunk ({stats['max_chars']} chars) exceeds 3000-char limit"
)
logger.info(
    f"Validation passed: {stats['total_chunks']} chunks, "
    f"{stats['distinct_sections']} sections, "
    f"max_chars={stats['max_chars']}"
)
