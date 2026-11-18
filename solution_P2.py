
# ========== TODO - YOUR CODE HERE ==========
#
# LLM setup
#
# Instantiate a ChatBedrockConverse LLM and expose it as `llm` for the
# rest of the file. Downstream nodes call `llm.invoke(...)`, so the
# variable must be named `llm`.
# ============================================

llm = ChatBedrockConverse(
    model_id="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=500
)


# ========== GIVEN - DO NOT MODIFY ==========
# Target language for the pipeline
# ============================================

TARGET_LANGUAGE = "Hindi"  # change to "French" if preferred


# ========== GIVEN - DO NOT MODIFY ==========
# State schema — downstream nodes read these exact field names
# ============================================

class DocState(TypedDict):
    source_text: str
    translated: str
    summary: str
    refined: str


# ========== TODO - YOUR CODE HERE ==========
# Node 1: translate_text
#
# Use `llm` to translate state['source_text'] from English into
# TARGET_LANGUAGE using a formal tone.
# Return {"translated": <the translation>}
# ============================================

def translate_text(state: DocState):

    prompt = f"""
    Translate the following English text into {TARGET_LANGUAGE}.

    Use a formal tone.
    Preserve all facts, names, numbers and entities.
    Do not summarize.

    Text:
    {state['source_text']}

    Return only the translated text.
    """

    response = llm.invoke(prompt)

    translated = response.content.strip()

    print("Translated Text:", translated)

    return {
        "translated": translated
    }


# ========== TODO - YOUR CODE HERE ==========
# Node 2: summarize_text
#
# Use `llm` to produce a concise summary of state['translated'].
# Keep the summary in TARGET_LANGUAGE.
# Return {"summary": <the summary>}
# ============================================

def summarize_text(state: DocState):

    prompt = f"""
    Create a concise summary of the following translated text.

    Language: {TARGET_LANGUAGE}

    Instructions:
    - Keep the summary in {TARGET_LANGUAGE}.
    - Preserve the important facts.
    - Preserve names and numbers.
    - Avoid unnecessary details.
    - Do not translate back to English.

    Translated Text:
    {state['translated']}

    Return only the summary.
    """

    response = llm.invoke(prompt)

    summary = response.content.strip()

    print("Summary:", summary)

    return {
        "summary": summary
    }


# ========== TODO - YOUR CODE HERE ==========
# Node 3: refine_summary
#
# Use `llm` to refine state['summary'] for readability and public-release tone.
# Keep the refined text in TARGET_LANGUAGE. Do NOT add, remove, or change any
# facts (names, numbers, entities). Never translate back to English.
# Return {"refined": <the refined text>}
# ============================================

def refine_summary(state: DocState):

    prompt = f"""
    Refine the following summary for readability and public-release tone.

    Language: {TARGET_LANGUAGE}

    Instructions:
    - Improve readability and clarity.
    - Use professional language.
    - Keep the output in {TARGET_LANGUAGE}.
    - Do not add any new information.
    - Do not remove any facts.
    - Do not change names, numbers or entities.
    - Never translate back to English.

    Summary:
    {state['summary']}

    Return only the refined text.
    """

    response = llm.invoke(prompt)

    refined = response.content.strip()

    print("Refined Summary:", refined)

    return {
        "refined": refined
    }


# ========== TODO - YOUR CODE HERE ==========
# Graph builder
#
# Wire three nodes into a StateGraph:
#   START -> translate_text -> summarize_text -> refine_summary -> END
#
# Node names in the graph (must match exactly):
#   "translate_text", "summarize_text", "refine_summary"
#
# Compile with an InMemorySaver checkpointer so state is persisted after
# every node (this is what makes time travel possible).
# Expose the compiled graph via a function called `build_graph()`
# that returns it (main() calls build_graph()).
# ============================================

def build_graph():

    builder = StateGraph(DocState)

    builder.add_node("translate_text", translate_text)

    builder.add_node("summarize_text", summarize_text)

    builder.add_node("refine_summary", refine_summary)

    builder.add_edge(START, "translate_text")

    builder.add_edge("translate_text", "summarize_text")

    builder.add_edge("summarize_text", "refine_summary")

    builder.add_edge("refine_summary", END)

    memory = InMemorySaver()

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
# Test harness
#
# Runs the full pipeline, then walks through the time-travel demo:
# inspect checkpoints, roll back to Version 1 and replay to regenerate
# Versions 2 and 3 from that point.
# ============================================

def run_test(app, source_text, thread_id="test_thread"):
    config = {'configurable': {'thread_id': thread_id}}

    # 1. Run the full pipeline (V1 -> V2 -> V3)
    final_state = app.invoke(
        {"source_text": source_text},
        config=config
    )

    print("\nFinal State after full pipeline:")
    print(final_state)

    # ========== TODO - YOUR CODE HERE ==========
    #
    # 2. Show all saved checkpoints. Each snapshot's `.next` field tells us
    #    which node is about to run at that point:
    #       next == ('translate_text',)  -> nothing done yet
    #       next == ('summarize_text',)  -> V1 done (translated)
    #       next == ('refine_summary',)  -> V2 done (summary)
    #       next == ()                   -> V3 done (refined)
    #
    # Fetch the state history for this thread and print each checkpoint's
    # `.next` value along with its checkpoint_id. Store the history in a
    # variable called `history` — you'll need it in the next step.
    # ==========================================

    history = list(app.get_state_history(config))

    print("\n========== CHECKPOINT HISTORY ==========")

    for snapshot in history:
        print("Next:", snapshot.next)
        print(
            "Checkpoint ID:",
            snapshot.config["configurable"]["checkpoint_id"]
        )

    # ========== TODO - YOUR CODE HERE ==========
    #
    # 3. Time travel: pick the checkpoint that is about to run summarize_text.
    #    That's V1 by definition — translation is done, summary hasn't started.
    #
    # Store the selected snapshot in a variable called `v1`, then print
    # its checkpoint_id and v1.values["translated"].
    # ==========================================

    v1 = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("summarize_text",)
    )

    print("\n========== VERSION 1 ==========")

    print(
        "Checkpoint ID:",
        v1.config["configurable"]["checkpoint_id"]
    )

    print("Translated Text:", v1.values["translated"])

    # ========== TODO - YOUR CODE HERE ==========
    #
    # 4. Replay from V1 to regenerate V2 and V3.
    #
    # HINT: passing None as the input to app.invoke tells LangGraph to
    # "continue from this checkpoint". Use v1.config as the config.
    # Print the regenerated summary and refined text.
    # ==========================================

    regenerated = app.invoke(
        None,
        config=v1.config
    )

    print("\n========== REGENERATED VERSIONS ==========")

    print("Regenerated Summary:", regenerated["summary"])

    print("Regenerated Refined Text:", regenerated["refined"])


# ========== GIVEN - DO NOT MODIFY ==========
# Entry point
# ============================================

def main():
    app = build_graph()
    display_graph(app)

    print("\n========== TEST CASE 1 ==========")

    source = (
        "The city council approved a new policy to expand public transport. "
        "The plan adds 50 electric buses and three new metro lines over the next two years, "
        "aiming to cut traffic congestion and reduce carbon emissions across the region."
    )

    run_test(app, source, "doc_test_1")


if __name__ == "__main__":
    main()
