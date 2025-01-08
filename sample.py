from typing import List, Dict, Tuple, Optional
from langgraph.graph import StateGraph, Graph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Define the state of our agent
class AgentState(BaseModel):
    problem: str
    current_step: int = 0
    steps: List[str] = Field(default_factory=list)
    solution: Optional[str] = None
    is_complete: bool = False
    needs_revision: bool = False
    
# Initialize our LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define the nodes (components) of our graph
def break_down_problem(state: AgentState) -> AgentState:
    """Break down the math problem into steps."""
    response = llm.invoke(
        [HumanMessage(content=f"""
        Break down this math problem into clear steps:
        {state.problem}
        
        Format your response as a list of steps only.
        """)]
    )
    steps = response.content.strip().split('\n')
    return state.model_copy(update={"steps": steps})

def solve_step(state: AgentState) -> AgentState:
    """Solve the current step of the problem."""
    current_step = state.steps[state.current_step]
    previous_work = "\n".join(state.steps[:state.current_step])
    
    response = llm.invoke(
        [HumanMessage(content=f"""
        Previous work:
        {previous_work}
        
        Current step to solve:
        {current_step}
        
        Show your work and provide the solution for this step.
        """)]
    )
    
    state.steps[state.current_step] = f"{current_step}\nSolution: {response.content}"
    return state.model_copy(update={
        "current_step": state.current_step + 1,
        "is_complete": state.current_step + 1 >= len(state.steps)
    })

def verify_solution(state: AgentState) -> AgentState:
    """Verify the complete solution for correctness."""
    full_solution = "\n".join(state.steps)
    response = llm.invoke(
        [HumanMessage(content=f"""
        Original problem:
        {state.problem}
        
        Complete solution:
        {full_solution}
        
        Verify if this solution is correct. If there are any errors, explain them.
        Respond with either "CORRECT" or "NEEDS_REVISION: <explanation>"
        """)]
    )
    
    needs_revision = not response.content.startswith("CORRECT")
    return state.model_copy(update={
        "needs_revision": needs_revision,
        "solution": full_solution if not needs_revision else None
    })

def revise_solution(state: AgentState) -> AgentState:
    """Revise the solution if errors were found."""
    response = llm.invoke(
        [HumanMessage(content=f"""
        Original problem:
        {state.problem}
        
        Current solution with errors:
        {state.steps}
        
        Please provide a corrected solution approach.
        """)]
    )
    
    new_steps = response.content.strip().split('\n')
    return state.model_copy(update={
        "steps": new_steps,
        "current_step": 0,
        "is_complete": False,
        "needs_revision": False
    })

# Define edges (conditional logic)
def should_continue(state: AgentState) -> str:
    """Determine the next step in the solution process."""
    if state.is_complete and not state.needs_revision:
        return "end"
    elif state.is_complete and state.needs_revision:
        return "revise"
    else:
        return "solve"

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("break_down", break_down_problem)
workflow.add_node("solve_step", solve_step)
workflow.add_node("verify", verify_solution)
workflow.add_node("revise", revise_solution)

# Add edges
workflow.add_edge("break_down", "solve_step")
workflow.add_conditional_edges(
    "solve_step",
    should_continue,
    {
        "solve": "solve_step",
        "end": "verify",
        "revise": "revise"
    }
)
workflow.add_edge("verify", "revise")
workflow.add_edge("revise", "solve_step")

# Set entry point
workflow.set_entry_point("break_down")

# Compile the graph
app = workflow.compile()

# Example usage
def solve_math_problem(problem: str) -> str:
    initial_state = AgentState(problem=problem)
    final_state = app.invoke(initial_state)
    return final_state.solution

# Example problem
if __name__ == "__main__":
    problem = """
    A train leaves Boston at 9:00 AM traveling at 60 mph. 
    Another train leaves New York at 10:00 AM traveling at 75 mph towards Boston. 
    If the cities are 215 miles apart, at what time will the trains meet?
    """
    
    solution = solve_math_problem(problem)
    print(f"Solution:\n{solution}")
