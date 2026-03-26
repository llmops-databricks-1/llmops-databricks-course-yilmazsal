from loguru import logger
from pyspark.sql import SparkSession
import arxiv
from pathlib import Path
import time
import os

logger.info("Using Databricks Connect Spark Session")


def download_papers():
    end = time.strftime("%Y%m%d%H%M", time.gmtime(time.time()))
    end_date = time.strftime("%Y%m%d", time.gmtime(time.time()))
    start = time.strftime(
                "%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 3))
    pdf_local = f"data/arxiv_files/{end_date}"
    os.makedirs(pdf_local, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query = f"cat:cs.AI AND submittedDate:[{start} TO {end_date}]"
    )
    papers = client.results(search)


    for paper in papers:
        paper_id = paper.get_short_id()
        try:
            paper.download_pdf(
                #dirpath=str(pdf_dir), filename=f"{paper_id}.pdf"
                dirpath=f"{pdf_local}/", filename=f"{paper_id}.pdf"
            )
        except Exception as e:
            logger.warning(
                f"Paper {paper_id} failed:{str(e)}"
            )   

if __name__ == "__main__":
    download_papers()
 