import mlflow
from agent import AGENT

# Create the agent object, and specify it as the agent object to use when
# loading the agent back for inference via mlflow.models.set_model()
mlflow.langchain.autolog()
mlflow.models.set_model(AGENT)

