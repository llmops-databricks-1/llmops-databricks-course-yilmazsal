# Databricks notebook source
# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from llmops_databricks.config import ProjectConfig, get_env
from llmops_databricks.data_processor import DataProcessor

# COMMAND ----------
# spark = DatabricksSession.builder.getOrCreate()
spark = SparkSession.builder.getOrCreate()
logger.info("Using Databricks Connect Spark Session")

env = get_env(spark)
cfg = ProjectConfig.from_yaml(config_path="../project_config.yml", env=env)

logger.info(cfg)

# COMMAND ----------
processor = DataProcessor(spark=spark, config=cfg)

# COMMAND ----------
# start = processor._get_range_start()
# end = processor.end
# pdf_dir = processor.pdf_dir
# pdf_local = f"data/arxiv_files/{end}"
# client = arxiv.Client()
# search = arxiv.Search(
#     query = f"cat:cs.AI AND submittedDate:[{start} TO {end}]"
# )
# papers = client.results(search)


# for paper in papers:
#     paper_id = paper.get_short_id()
#     print(paper_id)
#     print(pdf_dir)
#     try:
#         paper.download_pdf(
#             #dirpath=str(pdf_dir), filename=f"{paper_id}.pdf"
#             dirpath=f"{pdf_local}", filename=f"{paper_id}.pdf"
#         )
#     except Exception as e:
#         logger.warning(
#             f"Paper {paper_id} failed:{str(e)}"
#         )
# COMMAND ----------
# processor.download_and_store_papers()
# COMMAND ----------

# processor.parse_pdfs_with_ai()

# COMMAND ----------
processor.process_and_save()


# COMMAND ----------
# import arxiv
# import time
# start = time.strftime(
#                 "%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 3))
# end = time.strftime("%Y%m%d%H%M", time.gmtime(time.time()))
# client = arxiv.Client()
# search = arxiv.Search(
#     query = f"cat:cs.AI AND submittedDate:[{start} TO {end}]"
# )
# papers = client.results(search)
#
# COMMAND ----------
# spark.sql(f"DROP TABLE {cfg.catalog}.{cfg.schema}.arxiv_papers")
# # COMMAND ----------

# # COMMAND ----------
# dbutils.fs.rm(
#     "/Volumes/mlops_dev/yilmazfs/arxiv_files",
#     True  # recursive delete
# )
# COMMAND ----------
