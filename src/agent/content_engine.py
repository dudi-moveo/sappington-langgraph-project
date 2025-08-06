from langchain_core.messages import AIMessage
from typing import Annotated
from langgraph.graph.message import add_messages
from typing import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal, List
from langchain.chat_models import init_chat_model


llm = init_chat_model(model="openai:gpt-4o")



class ContentEngineState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    chosen_engine: str
    info_complete: bool
    content_item: Optional[Literal["CaseStudy", "eBook", "OnePager", "Blog", "Campaign"]]

class ContentItemInput(BaseModel):
    content_item: Literal["CaseStudy", "eBook", "OnePager", "Blog", "Campaign", "Not Sure"]


WELCOME_MESSAGE = """Welcome to Content Engine

We can can create a number of different content items, including:

One-pager
Customer case study
eBook
Blog, article
Campaign bill of materials

Which type of content do we want to create?

"""

NOT_SURE_MESSAGE = """
I'm not sure what you mean. Please try again.
"""

def ContentEngineWelcomeMessage(state: ContentEngineState):
    content_item = state.get("content_item", False)
    if content_item == "Not Sure":
        return {}
    else:
        new_message = AIMessage(content=WELCOME_MESSAGE)
        return {"messages": new_message}

def ContentEngineNode(state: ContentEngineState):
    thread_history = '\n'.join([f"{msg.type}: {msg.content}" for msg in state["messages"][:-1]]) # without the current message
    current_message = '\n'.join([f"{msg.type}: {msg.content}" for msg in state["messages"][-1:]]) # just the current message

    prompt = """
       Understand the user's request and choose the type of content they want to create.
        
        Thread history: {thread_history}
        
        Current message: {current_message}
        """
    
    llm_with_content_item_schema = llm.with_structured_output(schema=ContentItemInput)
    llm_response = llm_with_content_item_schema.invoke(prompt.format(thread_history=thread_history, current_message=current_message))

    if llm_response.content_item == "Not Sure":
        new_message = AIMessage(content=f"I'm not sure what you mean. Please try again.")
    else:
        new_message = AIMessage(content=f"Great, we'll create a {llm_response.content_item}.")

    return {"messages": new_message}


subgraph_builder = StateGraph(ContentEngineState)

subgraph_builder.add_node("ContentEngineWelcomeMessage", ContentEngineWelcomeMessage)
subgraph_builder.add_node("ContentEngineNode", ContentEngineNode)

subgraph_builder.add_edge(START, "ContentEngineWelcomeMessage")
subgraph_builder.add_edge("ContentEngineWelcomeMessage", "ContentEngineNode")
subgraph_builder.add_edge("ContentEngineNode", END)

ContentEngineSubgraph = subgraph_builder.compile()