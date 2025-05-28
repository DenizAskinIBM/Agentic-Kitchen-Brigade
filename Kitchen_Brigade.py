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
from typing_extensions import TypedDict

class KitchenState(TypedDict, total=False):
    docs: list[str]
    command: str
    plan: str
    chef_de_cuisine: str
    sous_chef: str
    saucier: str
    chef_de_partie: str
    cuisinier: str
    commis: str
    apprenti: str
    plongeur: str
    rotisseur: str
    grillardin: str
    poissonnier: str
    entremetier: str
    garde_manger: str
    tournant: str
    patissier: str
    final_recipe: str

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

# ------------------------------------------------
# 3b. Define kitchen brigade agent descriptions
# ------------------------------------------------
role_descriptions = {
    "chef_de_cuisine": (
        "Chef de cuisine (Head Chef): Oversees all kitchen operations, creates menus, "
        "and ensures the quality of dishes."
    ),
    "sous_chef": (
        "Sous-chef (Deputy Chef): Second-in-command, manages staff, handles inventories, "
        "and steps in for the head chef when needed."
    ),
    "saucier": (
        "Saucier (Sauce Chef): Prepares sauces, stews, and hot hors d'oeuvres, "
        "ensuring flavors are perfected."
    ),
    "chef_de_partie": (
        "Chef de partie (Station Chef): Responsible for a specific station in the kitchen, "
        "such as grill, pastry, or fish."
    ),
    "cuisinier": (
        "Cuisinier (Line Cook): Executes individual dishes on the line, "
        "following recipes and timing to coordinate service."
    ),
    "commis": (
        "Commis (Junior Cook): Assists station chefs, performs prep work, "
        "and learns station operations."
    ),
    "apprenti": (
        "Apprenti (Apprentice): Beginner cook learning the fundamentals of kitchen work, "
        "assisting commis and chefs."
    ),
    "plongeur": (
        "Plongeur (Dishwasher): Handles cleaning of dishes, utensils, and kitchen equipment, "
        "maintaining sanitation."
    ),
    "rotisseur": (
        "Rôtisseur (Roast Chef): Manages roasted dishes and grilling, "
        "ensuring proper cooking of meats."
    ),
    "grillardin": (
        "Grillardin (Grill Chef): Specializes in grilling meats and fishes, "
        "maintaining grill stations."
    ),
    "poissonnier": (
        "Poissonnier (Fish Chef): Prepares fish and seafood dishes, "
        "overseeing fish station."
    ),
    "entremetier": (
        "Entremetier (Vegetable Chef): Prepares vegetables, soups, pastas, "
        "and egg dishes."
    ),
    "garde_manger": (
        "Garde-manger (Pantry Chef): Handles cold dishes, salads, "
        "pates, and charcuterie."
    ),
    "tournant": (
        "Tournant (Swing Cook): Floats between stations as needed, "
        "filling in for absent station chefs."
    ),
    "patissier": (
        "Pâtissier (Pastry Chef): Prepares pastries, desserts, "
        "breads, and other baked goods."
    ),
}

# Backup original descriptions for good scenarios
original_role_descriptions = role_descriptions.copy()
# Define bad descriptions for testing
bad_role_descriptions = { key: "Generic task performer." for key in role_descriptions }

def make_agents(llm: LLMWrapper):
    """
    Create a mapping of brigade roles to agent functions.
    Each agent returns a string describing the action taken.
    """
    agents = {}
    for key, description in role_descriptions.items():
        def make_func(role_key, desc):
            def func(task: str) -> dict:
                # Agent performs the task and returns its output under its role name
                return {
                    role_key: f"[{desc}] I have completed the task: '{task}'."
                }
            # Attach description metadata to the function for router use
            func.description = desc
            return func
        agents[key] = make_func(key, description)
    return agents

# ------------------------------------------------
# 4. Build LangGraph workflow
# ------------------------------------------------
def build_workflow(llm: LLMWrapper, dish: str):
    graph = StateGraph(KitchenState)  # use KitchenState as state schema

    def retriever(state):
        state["docs"] = retrieve(state["command"])
        return state

    def planner(state):
        print("[📝] Generating plan from documents")
        docs = "\n".join(state["docs"])
        prompt = (
            f"You are the Chef de cuisine tasked to prepare {dish}. Based on these recipes:\n"
            f"{docs}\n\n"
            "Plan step-by-step subtasks as a plain list (one per line, no role assignments)."
        )
        stage = "Planning"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        state["plan"] = response
        return state

    def router(state):
        print("[🔀] Routing tasks to agents")
        subtasks = [line.strip() for line in state["plan"].split("\n") if line.strip()]
        routing: dict[str, list[str]] = {}
        agents = make_agents(llm)
        for task in subtasks:
            assigned = None
            for role_key, agent_func in agents.items():
                # use the description attached to the function
                keywords = agent_func.description.lower().split()
                if any(word in task.lower() for word in keywords):
                    assigned = role_key
                    break
            if assigned is None:
                assigned = "chef_de_cuisine"
            routing.setdefault(assigned, []).append(task)
        state["routing"] = routing
        return state

    def executor(state):
        print("[⚙️] Executing routed tasks")
        agents = make_agents(llm)
        # For each role, run all its tasks through the corresponding agent
        for role_key, tasks in state.get("routing", {}).items():
            results = []
            for task in tasks:
                # agent returns {role_key: message}
                res_dict = agents[role_key](task)
                results.append(res_dict[role_key])
            # Store combined execution output under the role key
            state[role_key] = "\n".join(results)
        return state

    def aggregator(state):
        print("[📦] Aggregating results into final recipe")
        # Gather each role's output from state
        outputs = []
        for role in role_descriptions.keys():
            if role in state:
                outputs.append(state[role])
        joined = "\n".join(outputs)
        prompt = (
            "Combine these executed subtasks into a final recipe with clear steps:\n"
            f"{joined}"
        )
        stage = "Aggregation"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        # Preserve previous state and add final_recipe
        state["final_recipe"] = response
        return state

    graph.add_node("retriever", retriever)
    graph.add_node("planner", planner)
    graph.add_node("router", router)
    graph.add_node("executor", executor)
    graph.add_node("aggregator", aggregator)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "aggregator")
    return graph

# ------------------------------------------------
# 5. Main entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    scenarios = [
        {
            "label": "Simple Hamburger - Bad Descriptions",
            "command": "Make me a Hamburger",
            "description_set": "bad"
        },
        {
            "label": "Custom Hamburger - Bad Descriptions",
            "command": (
                "Make me a Hamburger with no lettuce and only one tomato, "
                "ketchup and make sure to simmer the pan you cook the Hamburger on "
                "for 5 minutes with butter"
            ),
            "description_set": "bad"
        },
        {
            "label": "Simple Hamburger - Good Descriptions",
            "command": "Make me a Hamburger",
            "description_set": "good"
        },
        {
            "label": "Custom Hamburger - Good Descriptions",
            "command": (
                "Make me a Hamburger with no lettuce and only one tomato, "
                "ketchup and make sure to simmer the pan you cook the Hamburger on "
                "for 5 minutes with butter"
            ),
            "description_set": "good"
        },
    ]

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

    for scenario in scenarios:
        label = scenario["label"]
        command = scenario["command"]
        desc_set = scenario.get("description_set")
        if desc_set == "bad":
            role_descriptions.clear()
            role_descriptions.update(bad_role_descriptions)
        elif desc_set == "good":
            role_descriptions.clear()
            role_descriptions.update(original_role_descriptions)
        print(f"\n=== Scenario: {label} ===\n")
        workflow = build_workflow(llm, command)
        initial_state = {"command": command}
        final_state = workflow.compile().invoke(initial_state)

        plan_text = final_state.get("plan", "")
        # Gather execution outputs from all agent keys
        execution_results = [
            value for key, value in final_state.items()
            if key not in {"docs", "command", "plan", "routing", "final_recipe"}
        ]
        final_recipe = final_state.get("final_recipe", "")

        print("=== Plan ===\n")
        print(plan_text)
        print("\n=== Execution Results ===\n")
        print("\n".join(execution_results))
        print("\n=== Final Recipe ===\n")
        print(final_recipe)

        # Judges Evaluation
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