import mlflow
import uuid
from mlflow.pyfunc import PythonModel
from mlflow.types.agent import ChatAgentMessage
from agent import AGENT


class AgentModel(PythonModel):
    """Wrapper to expose AGENT (ChatAgent) via models-from-code."""

    def predict(self, context, model_input):
        # model_inputは {"messages": [{"role": "user", "content": "..."}]} の形式
        # AGENT.predict()は list[ChatAgentMessage] を期待するため変換が必要
        if isinstance(model_input, dict) and "messages" in model_input:
            messages = model_input["messages"]
        else:
            # DataFrameの場合など
            messages = model_input["messages"] if hasattr(model_input, "__getitem__") else model_input
        
        # 辞書のリストをChatAgentMessageオブジェクトのリストに変換
        chat_messages = [
            ChatAgentMessage(
                id=msg.get("id", str(uuid.uuid4())),
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            )
            if isinstance(msg, dict)
            else msg  # 既にChatAgentMessageオブジェクトの場合
            for msg in messages
        ]
        
        return AGENT.predict(chat_messages)

mlflow.models.set_model(AgentModel())


