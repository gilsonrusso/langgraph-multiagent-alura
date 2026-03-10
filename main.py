import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from rich import print

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t["name"] in self.tools:
                print(f"\n bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(
                ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
            )
        print("Back to the model!")
        return {"messages": results}


prompt = """Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

# Usando a integração oficial do LangChain para o Gemini
# (O LangChain espera objetos compatíveis com a interface BaseChatModel,
# que implementam métodos como .bind_tools() e .invoke() com HumanMessage).
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    # Você não precisa passar a API Key se a variável GOOGLE_API_KEY estiver no .env
)

agent = Agent(model=model, tools=[tool], system=prompt)

mermaid_code = agent.graph.get_graph().draw_mermaid()
print(mermaid_code)

from IPython.display import display, Image

try:
    image_data = agent.graph.get_graph().draw_mermaid_png()
    display(Image(data=image_data))
except Exception as e:
    print(f"Error: {e}")

messages = [HumanMessage(content="Como esta o tempo hoje em Manaus?")]

print("Iniciando o agente...")

final_result_state = None

for s in agent.graph.stream({"messages": messages}):
    print(s)
    print("\n\n-------------------------")
    final_result_state = s

print("\n\nFinal result:")
if (
    final_result_state
    and "llm" in final_result_state
    and final_result_state["llm"]["messages"]
):  # verifica se existe uma chave 'llm' no dicionário final_result_state e se a chave 'messages' existe no dicionário 'llm'
    print(
        final_result_state["llm"]["messages"][-1].content
    )  # acessa o último elemento da lista 'messages' no dicionário 'llm' e imprime o conteúdo
else:
    print("No final result state")
