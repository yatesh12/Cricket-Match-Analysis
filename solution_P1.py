
# ========== TODO - YOUR CODE HERE ==========
#
# LLM setup
#
# Instantiate a ChatBedrockConverse LLM and expose it as `llm` for the
# rest of the file. Downstream nodes call `llm.invoke(...)` and
# `llm.bind_tools(...)`, so the variable must be named `llm`.
# ============================================

llm = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=500
)


# ========== GIVEN - DO NOT MODIFY ==========
# State schema — downstream nodes read these exact field names
# ============================================

class State(TypedDict):
    claim: str
    claim_type: str
    tool_used: str
    evidence: dict
    confidence: float
    human_feedback: Optional[str]
    final_report: str


# ========== GIVEN - DO NOT MODIFY ==========
# Predefined evidence datasets — tools look up by exact claim text.
# ============================================

CLIMATE_DATA = {
    "India renewable energy capacity grew by 30% in 2024": [
        "IEA Report (2024): Renewable energy capacity increased by 28% year-over-year.",
        "Bloomberg Energy Outlook: India ranked third globally in new solar installations.",
        "Government Data: Installed capacity reached 192 GW of renewables by late 2024."
    ],
    "Ocean temperatures hit record highs in 2024": [
        "NOAA Data: 2024 was the warmest year for ocean surface temperatures since 1880.",
        "Nature Climate Journal: Average ocean temperature anomalies exceeded 1.2°C globally.",
        "UN Climate Division: Marine heatwaves affected 48% of world's oceans in 2024."
    ]
}

GENERAL_FACT_DATA = {
    "AI replaced 40% of human jobs in 2024": [
        "OECD Study: Automation affected about 11% of roles globally by 2024.",
        "McKinsey Report: AI adoption created new roles, offsetting large-scale job losses.",
        "Tech Review: AI shifted, not replaced, major segments of the workforce."
    ],
    "Global inflation rate dropped in 2024": [
        "IMF Report: Global inflation fell from 8.1% in 2023 to 6.3% in 2024.",
        "World Bank Data: Several economies saw moderation in inflation post-2023.",
        "Financial Times: Central banks eased interest rates in response to declining inflation."
    ]
}


# ========== TODO - YOUR CODE HERE ==========
# Tools
#
# Define two @tool-decorated functions:
# climate_data_tool(query: str) — for environmental claims
# general_fact_tool(query: str) — for economic/technological claims
#
# Each should look up `query` in the corresponding dataset above and
# return {"evidence": [list of evidence strings]}.
# If the query isn't found, return a placeholder list explaining that.
# ============================================

@tool
def climate_data_tool(query: str):
    for key, values in CLIMATE_DATA.items():
        if query == key:
            return {"evidence": values}

    return {"evidence": ["Query not found in climate dataset."]}


@tool
def general_fact_tool(query: str):
    for key, values in GENERAL_FACT_DATA.items():
        if query == key:
            return {"evidence": values}

    return {"evidence": ["Query not found in general fact dataset."]}


# ========== TODO - YOUR CODE HERE ==========
# Tool registration
#
# Bind the tools to the LLM so decide_and_invoke_tool can route to them
# by name. You'll need three things available below:
#   tools       : a list of tool functions
#   tools_by_name : a name -> tool lookup dict
#   llm_with_tools : the LLM with tools bound
# ============================================

tools = [climate_data_tool, general_fact_tool]

tools_by_name = {
    tool.name: tool for tool in tools
}

llm_with_tools = llm.bind_tools(tools)


# ========== TODO - YOUR CODE HERE ==========
# Node 1: classify_claim
#
# Use `llm` to classify state['claim'] into exactly one of:
#   [Environmental, Economic, Technological]
# Return {"claim_type": "<the single word>"}
# ============================================

def classify_claim(state: State):

    prompt = f"""
    Classify the following claim into exactly one category:

    Environmental
    Economic
    Technological

    Return only the category name.

    Claim: {state['claim']}
    """

    response = llm.invoke(prompt)

    res = response.content.strip()

    if "Environmental" in res:
        claim_type = "Environmental"
    elif "Economic" in res:
        claim_type = "Economic"
    else:
        claim_type = "Technological"

    print("Classified Claim:", claim_type)

    return {"claim_type": claim_type}


# ========== TODO - YOUR CODE HERE ==========
# Node 2: decide_and_invoke_tool
#
# Use `llm_with_tools` to pick and invoke exactly ONE tool:
#   Environmental       -> climate_data_tool
#   Economic or Technological -> general_fact_tool
#
# Pass the original claim text as the query argument.
# Return {"tool_used": "<tool name>", "evidence": <tool output dict>}
#
# HINT: after invoking the LLM, check response.tool_calls.
# Handle the case where the LLM didn't call a tool.
# ============================================

def decide_and_invoke_tool(state: State):

    prompt = f"""
    You need to use the proper tool for further process.

    If claim type = Environmental -> invoke climate_data_tool
    If claim type = Economic or Technological -> invoke general_fact_tool

    Given claim: {state['claim']}
    Claim type: {state['claim_type']}

    Do not give any extra text. Only invoke the correct tool.
    """

    response = llm_with_tools.invoke(prompt)

    if not response.tool_calls:
        if state["claim_type"] == "Environmental":
            tool_used = "climate_data_tool"
        else:
            tool_used = "general_fact_tool"

        tool_output = tools_by_name[tool_used].invoke({
            "query": state["claim"]
        })

        return {
            "tool_used": tool_used,
            "evidence": tool_output
        }

    tool_call = response.tool_calls[0]

    tool_used = tool_call["name"]

    print("Tool invoked:", tool_used)

    tool_output = tools_by_name[tool_used].invoke(
        tool_call["args"]
    )

    print("Tool output:", tool_output)

    return {
        "tool_used": tool_used,
        "evidence": tool_output
    }


# ========== TODO - YOUR CODE HERE ==========
# Node 3: verify_confidence
#
# Use `llm` to score how well the evidence supports the claim.
# Return {"confidence": <float between 0.0 and 1.0>}
#
# Guidance in the prompt: prefer low confidence when evidence and claim
# don't strongly align. This is what triggers the human review branch.
# ============================================

def verify_confidence(state: State):

    prompt = f"""
    Evaluate how strongly the evidence supports the claim.

    Claim: {state['claim']}

    Evidence: {state['evidence']}

    Give a confidence score between 0.0 and 1.0.

    If evidence does not strongly support the claim,
    give a low confidence score.

    Return only a numeric value.
    """

    response = llm.invoke(prompt)

    try:
        confidence = float(response.content.strip())
    except (ValueError, TypeError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    print("Confidence:", confidence)

    return {"confidence": confidence}


# ========== TODO - YOUR CODE HERE ==========
# Node 4: human_review
#
# Pause execution using interrupt() with a payload containing
# the claim, evidence, and confidence. Return {"human_feedback": <value>}.
# ============================================

def human_review(state: State):

    feedback = interrupt({
        "claim": state["claim"],
        "evidence": state["evidence"],
        "confidence": state["confidence"]
    })

    return {
        "human_feedback": feedback
    }


# ========== TODO - YOUR CODE HERE ==========
# Node 5: generate_final_report
#
# Use `llm` to produce a formatted verification report.
# Return {"final_report": <the text>}
# The report must include Claim Type, Tool Invoked, Confidence, Decision,
# Human Feedback (or "Not Required"), and a two-line Final Report.
# ============================================

def generate_final_report(state: State):

    if state.get("human_feedback"):
        decision = "Human Reviewed"
        human_feedback = state["human_feedback"]
    else:
        decision = (
            "Verified"
            if state["confidence"] >= 0.7
            else "Low Confidence"
        )
        human_feedback = "Not Required"

    prompt = f"""
    Generate a formatted claim verification report.

    Claim: {state['claim']}
    Claim Type: {state['claim_type']}
    Tool Invoked: {state['tool_used']}
    Evidence: {state['evidence']}
    Confidence: {state['confidence']}
    Decision: {decision}
    Human Feedback: {human_feedback}

    Include these sections:

    Claim Type:
    Tool Invoked:
    Confidence:
    Decision:
    Human Feedback:
    Final Report:

    Final Report must contain exactly two lines.
    """

    response = llm.invoke(prompt)

    final_report = response.content.strip()

    print("Final Report:", final_report)

    return {
        "final_report": final_report
    }


# ========== TODO - YOUR CODE HERE ==========
# Graph builder
#
# Wire five nodes into a StateGraph as per the instructions in the QP.
#
# Node names in the graph (must match exactly, since run_test references them):
#   "classify_claim", "retrieve_evidence", "verify_confidence",
#   "human_review", "finalize_report"
#
# Compile with checkpointer.
# Expose the compiled graph via a function called `build_graph()`
# that returns it (main() calls build_graph()).
# ============================================

def build_graph():

    builder = StateGraph(State)

    builder.add_node("classify_claim", classify_claim)

    builder.add_node("retrieve_evidence", decide_and_invoke_tool)

    builder.add_node("verify_confidence", verify_confidence)

    builder.add_node("human_review", human_review)

    builder.add_node("finalize_report", generate_final_report)

    builder.add_edge(START, "classify_claim")

    builder.add_edge("classify_claim", "retrieve_evidence")

    builder.add_edge("retrieve_evidence", "verify_confidence")

    builder.add_conditional_edges(
        "verify_confidence",
        lambda state: (
            "human_review"
            if state["confidence"] < 0.7
            else "finalize_report"
        )
    )

    builder.add_edge("human_review", "finalize_report")

    builder.add_edge("finalize_report", END)

    memory = MemorySaver()

    app = builder.compile(checkpointer=memory)

    return app


# ========== TODO - YOUR CODE HERE ==========
# Graph display
#
# Implement a `display_graph(app)` function that prints the compiled
# graph structure (ASCII view is enough). main() will call it.
# ============================================

def display_graph(app):

    print(app.get_graph().draw_ascii())


# ========== PARTIALLY GIVEN - COMPLETE THE TODO ==========
# Test
# ============================================

def run_test(app, claim, thread_id="test_thread"):
    config = {'configurable': {'thread_id': thread_id}}

    result = app.invoke({'claim': claim}, config=config)
    print("\nInitial Output:\n", result)

    # ========== TODO - YOUR CODE HERE ==========
    # Implement the interrupt handling: prompt for human feedback and
    # resume the graph with Command(resume=<feedback>).
    # ==========================================

    if result.get("__interrupt__"):

        interrupt_data = result["__interrupt__"][0].value

        print("\nHuman Review Required")
        print("Claim:", interrupt_data["claim"])
        print("Evidence:", interrupt_data["evidence"])
        print("Confidence:", interrupt_data["confidence"])

        feedback = input("\nEnter human feedback: ")

        result = app.invoke(
            Command(resume=feedback),
            config=config
        )

        print("\nResumed Output:\n", result)


def main():
    app = build_graph()
    display_graph(app)

    print("\n========== TEST CASE 1: Technological (HIL expected) ==========")
    run_test(app, "AI replaced 40% of human jobs in 2024", "claim_test_1")

    print("\n========== TEST CASE 2: Environmental (Auto-Verified) ==========")
    run_test(app, "Ocean temperatures hit record highs in 2024", "claim_test_2")


if __name__ == "__main__":
    main()
