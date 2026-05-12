import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from retriever import retrieve

load_dotenv()

# ─── State ────────────────────────────────────────────────
# This is the memory that flows through every node
class AgentState(TypedDict):
    question: str           # Original user question
    search_query: str       # Query sent to ChromaDB
    retrieved_chunks: List  # Chunks returned from ChromaDB
    answer: str             # Final answer
    iterations: int         # How many times we've searched

# ─── LLM Setup ────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# ─── Node 1: THINK ────────────────────────────────────────
def think_node(state: AgentState) -> AgentState:
    """
    Decide what to search for based on the question
    """
    print(f"\n[THINK] Deciding search query...")

    prompt = f"""You are a research assistant. 
Given this question, generate the best search query to find relevant information.
Return ONLY the search query, nothing else.

Question: {state['question']}
Search query:"""

    response = llm.invoke([HumanMessage(content=prompt)])
    search_query = response.content.strip()

    print(f"[THINK] Search query: {search_query}")

    return {
        **state,
        "search_query": search_query,
        "iterations": state.get("iterations", 0) + 1
    }

# ─── Node 2: RETRIEVE ─────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    """
    Search ChromaDB with the generated query
    """
    print(f"\n[RETRIEVE] Searching ChromaDB...")

    chunks = retrieve(state["search_query"], k=3)

    # Deduplicate chunks by content
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk.page_content not in seen:
            seen.add(chunk.page_content)
            unique_chunks.append(chunk)

    print(f"[RETRIEVE] Found {len(unique_chunks)} unique chunks")

    return {
        **state,
        "retrieved_chunks": unique_chunks
    }

# ─── Node 3: ANSWER ───────────────────────────────────────
def answer_node(state: AgentState) -> AgentState:
    """
    Generate final answer using retrieved chunks
    """
    print(f"\n[ANSWER] Generating answer...")

    # Format chunks into context
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in state["retrieved_chunks"]
    ])

    prompt = f"""You are an expert on NVIDIA GPU architecture.
Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which page your answer comes from.

Context:
{context}

Question: {state['question']}

Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)])

    print(f"[ANSWER] Answer generated")

    return {
        **state,
        "answer": response.content.strip()
    }

# ─── Edge: Should we search again? ───────────────────────
def should_continue(state: AgentState) -> str:
    """
    Decide whether to search again or give final answer
    """
    # Stop if we've searched 3 times (prevent infinite loop)
    if state.get("iterations", 0) >= 3:
        return "answer"

    # Stop if we have enough chunks
    if len(state.get("retrieved_chunks", [])) >= 2:
        return "answer"

    # Otherwise search again
    return "think"

# ─── Build the Graph ──────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("think", think_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    # Add edges
    graph.set_entry_point("think")
    graph.add_edge("think", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        should_continue,
        {
            "think": "think",
            "answer": "answer"
        }
    )
    graph.add_edge("answer", END)

    return graph.compile()

# ─── Main ─────────────────────────────────────────────────
def main():
    print("=== Agentic RAG System ===\n")

    agent = build_agent()

    # Test questions
    questions = [
        "What is the memory bandwidth of H100?",
        "How does H100 compare to A100 in terms of performance?",
        "What is the Transformer Engine in H100?"
    ]

    for question in questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)

        result = agent.invoke({
            "question": question,
            "search_query": "",
            "retrieved_chunks": [],
            "answer": "",
            "iterations": 0
        })

        print(f"\nFinal Answer:\n{result['answer']}")

if __name__ == "__main__":
    main()