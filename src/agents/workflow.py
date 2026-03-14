import logging
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next_step: str

class WorkflowManager:
    """
    Manages Agentic Workflows using LangGraph.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        self.workflow = StateGraph(AgentState)
        self._setup_graph()

    def _setup_graph(self):
        self.workflow.add_node("planner", self.plan_node)
        self.workflow.add_node("executor", self.execute_node)
        
        self.workflow.set_entry_point("planner")
        self.workflow.add_edge("planner", "executor")
        self.workflow.add_edge("executor", END)
        
        self.app = self.workflow.compile()

    async def plan_node(self, state: AgentState):
        logger.info("Planning next steps...")
        # Mock logic for planning
        return {"next_step": "execute"}

    async def execute_node(self, state: AgentState):
        logger.info("Executing plan...")
        # Mock logic for execution
        last_message = state['messages'][-1].content
        response = await self.llm.ainvoke(f"Process this request: {last_message}")
        return {"messages": [response]}

    async def run_workflow(self, query: str):
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next_step": ""
        }
        async for output in self.app.astream(initial_state):
            for key, value in output.items():
                logger.info(f"Node '{key}' completed.")
                yield value
