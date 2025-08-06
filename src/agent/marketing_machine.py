from typing_extensions import Annotated, Optional, Literal, TypedDict
import logging
import os
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel
from langgraph.types import interrupt, Command
from agent.messaging_framework import MessagingEngineSubgraph
from agent.competitive_analysis import CompetitiveAnalysisSubgraph
from agent.marketing_research import MarketingResearchSubgraph
from agent.content_engine import ContentEngineSubgraph
from agent.PROMPTS import QuestionAnsweringPrompt, QuestionAnsweringContext, ImplicitRoutingContext, ClassifyInputPrompt

# --------------------
# Logger Setup
# --------------------
logger = logging.getLogger(__name__)

# Get log level from environment variable, default to INFO
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)

# --------------------
# LLM Definition
# --------------------
llm = init_chat_model(model="openai:gpt-4o")
logger.info("Initialized LLM with model: openai:gpt-4o")



# --------------------
# State Definition 
# --------------------
class RouterState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    routing_type: Optional[Literal["button", "explicit", "implicit", "product_question", "irrelevant", "greeting"]]
    engine_suggestion: Optional[str]
    confirmation_given: Optional[bool]
    confirmation_message_for_implicit: Optional[str]
    chosen_engine: Optional[Literal["MessagingEngine", "CompetitiveAnalysis", "MarketingResearch", "ContentEngine"]]


engine_list = ["MessagingEngine", "CompetitiveAnalysis", "MarketingResearch", "ContentEngine"]

# --------------------
# Structured Output Definition
# --------------------

class ApprovalResponse(BaseModel):
    approved: bool
    follow_up_message: Optional[str] = None


class RoutingOutput(BaseModel):
    routing_type: Literal["button", "explicit", "implicit", "product_question", "irrelevant", "greeting"]
    engine_suggestion: Optional[Literal["MessagingEngine", "CompetitiveAnalysis", "MarketingResearch", "ContentEngine"]]
    greeting_message: Optional[str]
    confirmation_message_for_implicit: Optional[str]


# --------------------
# Node Implementations
# --------------------
def classify_input(state: RouterState) -> RouterState:
    """Classify user input and determine routing strategy."""
    logger.info("Starting input classification")
    logger.debug("Current state messages count: %d", len(state["messages"]))
    
    # Get conversation context (exclude the current message)
    if len(state["messages"]) > 1:
        # Get the last 2-3 messages before the current one
        context_messages = state["messages"][-3:-1] if len(state["messages"]) >= 3 else state["messages"][:-1]
        context_text = "\n".join([f"{msg.type}: {msg.content}" for msg in context_messages])
        context_section = f"Previous conversation context:\n{context_text}\n"
        logger.debug("Using conversation context from %d previous messages", len(context_messages))
    else:
        context_section = ""
        logger.debug("No previous conversation context available")

    prompt_with_user_message = ClassifyInputPrompt.format(
        engine_list=engine_list,
        user_message=state["messages"][-1].content,
        examples=ImplicitRoutingContext,
        context_section=context_section
    )
    logger.debug("Generated classification prompt")
    logger.debug("User message: %s", state["messages"][-1].content)
    
    llm_with_schema = llm.with_structured_output(schema=RoutingOutput)
    llm_response = llm_with_schema.invoke(prompt_with_user_message)
    logger.debug("LLM classification response: %s", llm_response)
    
    if llm_response.routing_type == "greeting":
        logger.info("Classified as greeting, responding with greeting message")
        return {**state, "routing_type": llm_response.routing_type, "messages": AIMessage(content=llm_response.greeting_message) }
    else:
        logger.info("Classified as: %s, suggested engine: %s", llm_response.routing_type, llm_response.engine_suggestion)
        return {**state, "routing_type": llm_response.routing_type, "engine_suggestion": llm_response.engine_suggestion, "confirmation_message_for_implicit": llm_response.confirmation_message_for_implicit}

def direct_router(state: RouterState) -> RouterState:
    """Route directly to the suggested engine."""
    engine = state.get("engine_suggestion", "UnknownEngine")
    logger.info("Routing directly to engine: %s", engine)
    
    if engine == "UnknownEngine":
        logger.warning("No engine suggestion found in state, using UnknownEngine")
    
    return state

def ask_for_confirmation(state: RouterState) -> RouterState:
    """Ask user for confirmation before routing to suggested engine."""
    engine = state.get("engine_suggestion", "some engine")
    user_confirmation_message = state.get("confirmation_message_for_implicit", f"Just to confirm, do you want to proceed with {engine}?")
    logger.info("Requesting user confirmation for engine: %s", engine)
    logger.debug("Confirmation message: %s", user_confirmation_message)
    
    # Get user response
    user_response = interrupt({
        "question": user_confirmation_message,
    })
    logger.info("Received user response for confirmation")
    logger.debug("User response: %s", user_response)
    
    # Use LLM to process the approval with structured output
    last_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]
    context_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in last_messages])
    
    approval_prompt = f"""
    Analyze the user's response to determine if they approved the routing suggestion.
    
    Previous conversation context:
    {context_messages}
    
    User's response: "{user_response}"
    
    Return:
    - approved: true if the user said yes/agreed/approved, false otherwise
    - follow_up_message: if not approved, provide a helpful message asking what they'd like to do instead
    """
    
    logger.debug("Processing approval with LLM")
    llm_with_approval_schema = llm.with_structured_output(schema=ApprovalResponse)
    approval_result = llm_with_approval_schema.invoke(approval_prompt)
    logger.debug("Approval analysis result: %s", approval_result)
    
    if approval_result.approved:
        logger.info("User approved routing to %s", engine)
        # Add user's approval to the conversation history
        updated_state = {
            **state, 
            "messages": state["messages"] + [HumanMessage(content=user_response)]
        }
        return Command(goto="DirectRouter", update=updated_state)
    else:
        logger.info("User did not approve routing, providing follow-up")
        follow_up = approval_result.follow_up_message or "I understand you'd like to do something different. What would you like to accomplish today?"
        logger.debug("Follow-up message: %s", follow_up)
        return {
            **state, 
            "messages": state["messages"] + [
                HumanMessage(content=user_response),
                AIMessage(content=follow_up)
            ]
        }

def answer_system_question(state: RouterState) -> RouterState:
    """Answer questions about the system using predefined context."""
    user_question = state["messages"][-1].content
    logger.info("Answering system question")
    logger.debug("User question: %s", user_question)
    
    qna_prompt = QuestionAnsweringPrompt.format(context=QuestionAnsweringContext, question=user_question)
    logger.debug("Generated QA prompt")
    
    llm_response = llm.invoke(qna_prompt)
    logger.debug("QA LLM response: %s", llm_response.content)
    logger.info("Successfully generated answer for system question")
    
    state["messages"].append(AIMessage(content=llm_response.content))
    return state

def reject_irrelevant(state: RouterState) -> RouterState:
    """Reject irrelevant requests with a helpful message."""
    logger.info("Rejecting irrelevant request")
    logger.debug("Last message was classified as irrelevant: %s", state["messages"][-1].content)
    
    rejection_message = "Sorry, I can't help with that. Please tell me what you want to do today, click a button to get started, or ask a relevant question to get started."
    state["messages"].append(AIMessage(content=rejection_message))
    return state

def route_directly_based_on_prior_choice(state: RouterState) -> RouterState:
    """Route directly to the suggested engine."""
    chosen_engine = state.get("chosen_engine", "UnknownEngine")
    if chosen_engine == "ContentEngine":
        return Command(goto="ContentEngine", update=state)
    elif chosen_engine == "MessagingEngine":
        return Command(goto="MessagingEngine", update=state)
    elif chosen_engine == "CompetitiveAnalysis":
        return Command(goto="CompetitiveAnalysis", update=state)
    elif chosen_engine == "MarketingResearch":
        return Command(goto="MarketingResearch", update=state)


# --------------------
# Graph Definition
# --------------------
builder = StateGraph(RouterState)
builder.add_node("ClassifyInput", classify_input)
builder.add_node("DirectRouter", direct_router)
builder.add_node("UserConfirmation", ask_for_confirmation)
builder.add_node("AnswerSystemQuestion", answer_system_question)
builder.add_node("RejectIrrelevant", reject_irrelevant)
builder.add_node("RouteDirectlyBasedOnPriorChoice", route_directly_based_on_prior_choice)

# --------------------
# Subgraphs
# --------------------
logger.info("Setting up subgraphs")
builder.add_node("MessagingEngine", MessagingEngineSubgraph)
builder.add_node("CompetitiveAnalysis", CompetitiveAnalysisSubgraph)
builder.add_node("MarketingResearch", MarketingResearchSubgraph)
builder.add_node("ContentEngine", ContentEngineSubgraph)

# Edges
logger.info("Configuring graph edges and entry point")
builder.set_entry_point("RouteDirectlyBasedOnPriorChoice")
builder.add_edge("RouteDirectlyBasedOnPriorChoice", "ClassifyInput")

builder.add_conditional_edges(
    "ClassifyInput",
    lambda state: state["routing_type"],
    {
        "button": "DirectRouter",
        "explicit": "DirectRouter",
        "implicit": "UserConfirmation",
        "product_question": "AnswerSystemQuestion",
        "irrelevant": "RejectIrrelevant",
        "greeting": END
    },
)

builder.add_conditional_edges(
    "DirectRouter",
    lambda state: state["engine_suggestion"],
    {
        "MessagingEngine": "MessagingEngine",
        "CompetitiveAnalysis": "CompetitiveAnalysis",
        "MarketingResearch": "MarketingResearch",
        "ContentEngine": "ContentEngine"
    }
)

builder.add_edge("AnswerSystemQuestion", END)
builder.add_edge("RejectIrrelevant", END)
builder.add_edge("DirectRouter", END)
builder.add_edge("UserConfirmation", END)

graph = builder.compile()
logger.info("Marketing machine graph compiled successfully")
