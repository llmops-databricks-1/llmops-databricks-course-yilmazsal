import yaml
<<<<<<< HEAD
<<<<<<< HEAD
=======
from databricks.connect import DatabricksSession
<<<<<<< HEAD
from databricks.sdk.runtime import DBUtils
>>>>>>> d4334e6 (fixed formatting)
=======
from pyspark.dbutils import DBUtils
>>>>>>> b710dad (fixed format)
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
=======
from databricks.connect import DatabricksSession
<<<<<<< HEAD
>>>>>>> 53e86ae (created chunks, vector search endpoint, index table  and search functions)
=======
from pydantic import BaseModel, Field
>>>>>>> d486143 (fixed formatting)


class ProjectConfig(BaseModel):
    """Load project configuration from YAML."""

    catalog: str = Field(
        ..., description="Name of the catalog to use for this environment"
    )
    schema: str = Field(..., description="Name of the schema associated with the catalog")
    volume: str = Field(..., description="Name of the Volume associated with the catalog")
    llm_endpoints: str = Field(..., description="Endpoint identifier for the LLM service")
    embedding_endpoint: str = Field(..., description="Endpoint for embedding generation")
    vector_search_endpoint: str = Field(
        ..., description="Endpoint for vector search service"
    )

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load and parse configuraton settings from YAML file.
        :param config_path: Path to the yaml configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initilized with parsed configuration
        """
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        available_envs = list(config_dict.keys())
        if env not in available_envs:
            raise ValueError(
                f"Invalid environment: {env}. Availabel environments: {available_envs}"
            )

        env_config = config_dict[env]
        return cls(**env_config)


def get_env(spark: DatabricksSession) -> str:
    """Get current environment from dbutils widget.
    Returns:
        Environment name (dev, acc, dev)
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"
