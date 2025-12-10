import mlflow
from mlflow.pyfunc import PythonModel
from agent import AGENT


class AgentModel(PythonModel):
    """Wrapper to expose AGENT (ChatAgent) via models-from-code."""

    def predict(self, context, model_input):
        # 期待される入力は {"messages": [...]} の辞書
        return AGENT.predict(model_input["messages"])


# models-from-code で登録
mlflow.models.set_model(AgentModel())


