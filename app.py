import sys, io, os, traceback
from typing import TypedDict, List, Optional
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# --- LLM ---
llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",  # verify this model id is valid for your key
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0,
)

# --- State ---
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

def extract_text(content) -> str:
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return str(content[0]) if content else ""
    return str(content)

# --- Tools ---
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return stdout or the error trace."""
    clean_code = code.replace("```python", "").replace("```", "").strip()
    old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        exec(clean_code, {}, {})
        result = sys.stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
    return result.strip() or "Success (no terminal output)"

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 specific test scenarios "
        f"for this task: '{task_description}'. Include edge cases. Numbered list."
    )
    response = llm_flash.invoke(prompt)
    return extract_text(response.content)

# --- Graph nodes ---
def developer_node(state: CrewState):
    task = state["messages"][-1].content
    response = llm_flash.invoke(
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    return {"code": extract_text(response.content)}

def tester_node(state: CrewState):
    task = state["messages"][-1].content
    test_cases = generate_test_cases.invoke(task)
    execution_result = run_python_code.invoke({"code": state["code"]})
    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{test_cases}"
    )
    return {"report": report}

def manager_node(state: CrewState):
    return {"next_step": "exit", "report": state["report"]}

# --- Graph construction ---
workflow = StateGraph(CrewState)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

rt_app = workflow.compile()

# --- FastAPI app ---
app = FastAPI(title="LangGraph Crew Workflow API")

# Keep the raw LangServe route too, for programmatic / /docs access
add_routes(app, rt_app, path="/crew", playground_type="default")


# --- Simple JSON endpoint: plain {"task": "..."} in, report out ---
class TaskInput(BaseModel):
    task: str = Field(..., description="The coding task to solve")

@app.post("/run-task")
async def run_task(payload: TaskInput):
    result = rt_app.invoke({
        "messages": [HumanMessage(content=payload.task)],
        "next_step": None,
        "code": None,
        "report": None,
    })
    return {
        "task": payload.task,
        "code": result.get("code", ""),
        "report": result.get("report", ""),
    }


# --- Homepage: opens directly at the root URL, simple form ---
HOMEPAGE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LangGraph Dev Crew</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    label { display: block; margin-top: 16px; font-weight: 600; font-size: 0.9rem; }
    textarea {
      width: 100%; padding: 8px; margin-top: 6px; box-sizing: border-box;
      border: 1px solid #ccc; border-radius: 6px; font-size: 0.95rem; min-height: 90px;
      font-family: inherit;
    }
    button {
      margin-top: 20px; padding: 10px 20px; border: none; border-radius: 6px;
      background: #4f46e5; color: white; font-size: 0.95rem; cursor: pointer;
    }
    button:disabled { background: #a5a5a5; cursor: not-allowed; }
    #status { margin-top: 16px; font-size: 0.9rem; color: #555; }
    pre {
      margin-top: 16px; background: #f5f5f7; padding: 14px; border-radius: 8px;
      white-space: pre-wrap; word-wrap: break-word; font-size: 0.85rem;
    }
    h3 { margin-top: 20px; font-size: 0.95rem; }
  </style>
</head>
<body>
  <h1>🤖 LangGraph Dev Crew</h1>
  <p>Describe a coding task. A "developer" node writes the code, then a "tester" node executes it and generates test scenarios.</p>

  <form id="taskForm">
    <label for="task">Coding task</label>
    <textarea id="task" name="task" placeholder="e.g. Write a function that reverses a string" required></textarea>
    <button type="submit" id="submitBtn">Run Dev Crew</button>
  </form>

  <div id="status"></div>
  <div id="resultBox" style="display:none;">
    <h3>Generated code</h3>
    <pre id="codeOut"></pre>
    <h3>Report</h3>
    <pre id="reportOut"></pre>
  </div>

  <script>
    const form = document.getElementById("taskForm");
    const statusEl = document.getElementById("status");
    const resultBox = document.getElementById("resultBox");
    const codeOut = document.getElementById("codeOut");
    const reportOut = document.getElementById("reportOut");
    const submitBtn = document.getElementById("submitBtn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      resultBox.style.display = "none";
      submitBtn.disabled = true;
      statusEl.textContent = "Running... this can take 10-30 seconds.";

      try {
        const res = await fetch("/run-task", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task: document.getElementById("task").value }),
        });
        const data = await res.json();
        if (!res.ok) {
          statusEl.textContent = "Error: " + (data.detail || res.statusText);
        } else {
          statusEl.textContent = "Done.";
          resultBox.style.display = "block";
          codeOut.textContent = data.code || "(no code returned)";
          reportOut.textContent = data.report || "(no report returned)";
        }
      } catch (err) {
        statusEl.textContent = "Request failed: " + err;
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return HOMEPAGE_HTML


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
