import mlflow
from agent import AGENT

# ChatAgentを直接設定することで、Agent Framework互換の標準署名が自動適用される
# input_exampleを提供すれば、MLflowが自動的にChatCompletionResponse形式の署名を推論する
mlflow.models.set_model(AGENT)


