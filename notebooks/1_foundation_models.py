# Databricks notebook source
# COMMAND ----------
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from loguru import logger



w = WorkspaceClient()
#
# COMMAND ----------
endpoints = w.serving_endpoints.list()
logger.info("Available Foundation Model Endpoints:")
logger.info("-"*80)

for endpoint in endpoints:
    if endpoint.name and "databricks" in endpoint.name:
        logger.info(f"Name: {endpoint.name}")
        logger.info(f"State: {endpoint.state}")
        logger.info("-"*80)
        
        
# COMMAND ----------

import subprocess, json, openai
host =  w.config.host

result = subprocess.run(
    [r"C:\Users\pnl0ux31\AppData\Local\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe\databricks.exe", "auth", "token", "--host", "https://adb-4443256865345152.12.azuredatabricks.net/"],
    capture_output=True, text=True
)

token = json.loads(result.stdout)["access_token"]
client = openai.OpenAI(
    api_key=token,
    base_url=f"{host.rstrip('/')}/serving-endpoints"
)


model_name = "databricks-llama-4-maverick"
response = client.chat.completions.create(
    model = model_name,
    messages = [{"role": "system", "content" :"You are a helpful AI assistant"},
                {"role": "user", "content" : "Explain LLMOps in 3 sentences"}
                ],
    max_tokens = 300, 
    temperature=0.7
)


logger.info("Response")
logger.info(response.choices[0].message.content)
logger.info(f"Tokens used: {response.usage.total_tokens}")
logger.info(f"Input tokens{response.usage.prompt_tokens}")
logger.info(f"Tokens outpout: {response.usage.completion_tokens}")
# 
# COMMAND ----------
import arxiv

search = arxiv.Search(
    query="machine learning",
    max_results=5,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in search.results():
    print(result.title)
    print(result.entry_id)
    print(result.published)
    print("-" * 50)
# COMMAND ----------
