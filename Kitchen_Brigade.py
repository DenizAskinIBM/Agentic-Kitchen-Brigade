# placeholders for retrieval globals
embedder = None
index = None
texts = None
import os
import pickle
import faiss  # ensure faiss is imported
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from ibm_watsonx_ai import APIClient, Credentials
from tqdm.auto import tqdm
import numpy as np

# capture all LLM interactions
llm_log = []

# ------------------------------------------------
# 0. Load credentials & config from .env
# ------------------------------------------------
load_dotenv(dotenv_path=".env")  # <-- loads all entries in .env into os.environ
print("[🔑] Loaded environment variables")

# ------------------------------------------------
# 1. LLM Wrapper: OpenAI or watsonx
# ------------------------------------------------
class LLMWrapper:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model = model_name

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in environment")
            self.client = OpenAI(api_key=api_key)
        elif provider == "watsonx":
            url = os.getenv("WATSONX_URL", "")
            apikey = os.getenv("WATSONX_APIKEY", "")
            if not url or not apikey:
                raise ValueError("Missing WATSONX_URL or WATSONX_APIKEY in environment")
            creds = Credentials(url=url, token=apikey)
            self.client = APIClient(credentials=creds)
        else:
            raise ValueError("Provider must be 'openai' or 'watsonx'")

    def generate(self, prompt: str) -> str:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content.strip()
        else:  # watsonx
            resp = self.client.generations.create(
                model=self.model,
                input=prompt
            )
            return resp.generations[0].text

# ------------------------------------------------
# 2. Prepare RAG: Load dataset & build FAISS index
# ------------------------------------------------
# dataset = load_dataset("formido/recipes", split="train")
# # Combine instruction and output fields into text corpus
# texts = [
#     f"{example['instruction']} {example['output']}"
#     for example in dataset
#     if 'instruction' in example and 'output' in example
# ]
#
# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
#
# d = embeddings.shape[1]
# index = faiss.IndexFlatIP(d)
# index.add(embeddings)

def retrieve(query: str, k: int = 5):
    global embedder, index, texts
    print("[🔍] Retrieving documents for query:", query)
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, I = index.search(q_emb, k)
    return [texts[i] for i in I[0]]

# ------------------------------------------------
# 3. Define kitchen brigade agents
# ------------------------------------------------
kitchen_roles = [
    "chef_de_cuisine", "sous_chef", "saucier", "chef_de_partie",
    "cuisinier", "commis", "apprenti", "plongeur",
    "rotisseur", "grillardin", "poissonnier", "entremetier",
    "garde_manger", "tournant", "patissier"
]

def make_agents(llm: LLMWrapper):
    def agent_func(role):
        def func(task):
            prompt = f"[{role}] {task}"
            stage = f"Execute: {role}"
            response = llm.generate(prompt)
            llm_log.append((stage, prompt, response))
            return response
        return func
    return {
        role: agent_func(role)
        for role in kitchen_roles
    }

# ------------------------------------------------
# 4. Build LangGraph workflow
# ------------------------------------------------
def build_workflow(llm: LLMWrapper, dish: str):
    graph = StateGraph(dict)  # use dict as state schema

    def retriever(state):
        state["docs"] = retrieve(state["command"])
        return state

    def planner(state):
        print("[📝] Generating plan from documents")
        docs = "\n".join(state["docs"])
        prompt = (
            f"You are the Chef de cuisine tasked to prepare {dish}. Based on these recipes:\n"
            f"{docs}\n\n"
            "Plan step-by-step subtasks and assign each to a kitchen role in the format 'Role: Task description'."
        )
        stage = "Planning"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        state["plan"] = response
        return state

    def executor(state):
        print("[⚙️] Executing subtasks")
        agents = make_agents(llm)
        subtasks = [
            line.strip()
            for line in state["plan"].split("\n")
            if line.strip()
        ]
        results = []
        for task in subtasks:
            if ":" in task:
                role, desc = task.split(":", 1)
                key = role.strip().lower().replace(" ", "_")
                # The agent function logs its own LLM call
                result = agents.get(key, lambda _: f"Unknown role: {role}")(desc.strip())
                results.append(f"{role}: {result}")
        state["results"] = results
        return state

    def aggregator(state):
        print("[📦] Aggregating results into final recipe")
        joined = "\n".join(state["results"])
        prompt = (
            "Combine these executed subtasks into a final recipe with clear steps:\n"
            f"{joined}"
        )
        stage = "Aggregation"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        state["final_recipe"] = response
        return state

    graph.add_node("retriever", retriever)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("aggregator", aggregator)

    # Define entrypoint to ensure graph has a START node
    graph.add_edge(START, "retriever")

    graph.add_edge("retriever", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "aggregator")
    return graph

# ------------------------------------------------
# 5. Main entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    dish = input("Please enter the dish you want to prepare: ")
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL", "gpt-4o")
    llm = LLMWrapper(provider, model_name)

    EMBEDDINGS_FILE = "embeddings.npy"
    INDEX_FILE = "faiss.index"

    # instantiate embedder once
    embedder_local = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(INDEX_FILE):
        print("[💾] Loading embeddings and FAISS index from disk")
        embeddings = np.load(EMBEDDINGS_FILE)
        index = faiss.read_index(INDEX_FILE)
        # Load texts from dataset for retrieval
        dataset = load_dataset("formido/recipes", split="train[:1000]")
        texts = [
            f"{example['instruction']} {example['output']}"
            for example in dataset
            if 'instruction' in example and 'output' in example
        ]
    else:
        print("[📚] Loading recipe dataset...")
        dataset = load_dataset("formido/recipes", split="train[:1000]")
        texts = [
            f"{example['instruction']} {example['output']}"
            for example in dataset
            if 'instruction' in example and 'output' in example
        ]
        print(f"[📚] Loaded subset of {len(texts)} recipes")
        print("[⚙️] Encoding embeddings with progress:")
        embeddings_list = []
        for text in tqdm(texts, desc="Embedding recipes"):
            embeddings_list.append(
                embedder_local.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            )
        embeddings = np.vstack(embeddings_list)
        # Save embeddings
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"[💾] Saved embeddings to {EMBEDDINGS_FILE}")
        # Build and save FAISS index
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, INDEX_FILE)
        print(f"[💾] Saved FAISS index to {INDEX_FILE}")

    # expose to retrieve()
    globals()['embedder'] = embedder_local
    globals()['index'] = index
    globals()['texts'] = texts

    workflow = build_workflow(llm, dish)

    initial_state = {"command": dish}
    final_state = workflow.compile().invoke(initial_state)

    plan_text = final_state.get("plan", "")
    execution_results = final_state.get("results", [])
    final_recipe = final_state.get("final_recipe", "")

    print("[✅] Final recipe generated, running judges...")

    print("\n=== Final Recipe ===\n")
    print(final_recipe)

    # ------------------------------------------------
    # 6. Judges Evaluation
    # ------------------------------------------------
    # Planning Judge: evaluate quality of routing/assignment
    planning_prompt = (
        "As the Planning Judge, evaluate the quality of the routing/assignment of subtasks to specific kitchen roles. "
        "Discuss if each assignment makes sense given the role's responsibilities.\n\n"
        f"Plan Provided:\n{plan_text}"
    )
    print("[👩‍⚖️] Planning Judge Evaluation")
    stage = "Planning Judge"
    prompt = planning_prompt
    planning_feedback = llm.generate(prompt)
    llm_log.append((stage, prompt, planning_feedback))

    # Execution Judge: evaluate efficiency & effectiveness
    exec_prompt = (
        "As the Execution Judge, evaluate the quality of execution of the subtasks and the final recipe. "
        "Focus on efficiency (minimal number of steps) and effectiveness (completion and correctness).\n\n"
        f"Subtasks Results:\n{chr(10).join(execution_results)}\n\n"
        f"Final Recipe:\n{final_recipe}"
    )
    print("[👨‍⚖️] Execution Judge Evaluation")
    stage = "Execution Judge"
    prompt = exec_prompt
    execution_feedback = llm.generate(prompt)
    llm_log.append((stage, prompt, execution_feedback))

    print("\n=== Judge Results ===\n")
    print("-- Planning Judge --\n", planning_feedback)
    print("\n-- Execution Judge --\n", execution_feedback)

    print("\n=== LLM Interactions Report ===\n")
    for stage, prompt, response in llm_log:
        print(f"--- {stage} ---")
        print("Prompt:")
        print(prompt)
        print("\nResponse:")
        print(response)
        print("\n")