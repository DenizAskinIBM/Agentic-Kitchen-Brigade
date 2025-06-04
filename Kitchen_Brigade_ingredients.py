import argparse
import collections
import faiss  # ensure faiss is imported
import json
import os
import pickle
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from ibm_watsonx_ai import APIClient, Credentials
from tqdm.auto import tqdm
import numpy as np
from typing_extensions import TypedDict

# placeholders for retrieval globals
embedder = None
index = None
texts = None

class KitchenState(TypedDict, total=False):
    docs: list[str]
    command: str
    recipe: str
    final_recipe: str
    plan: str
    routing: dict
    staff: dict
    results: dict

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


def make_agents(llm: LLMWrapper):
    """
    Create a mapping of brigade roles to agent functions.
    Each agent returns a string describing the action taken.
    """
    def make_func(staff_member: str):
        return lambda task: {"result": f"[{staff_member}] I have completed the task: {task}"} 

    agents = {}

    for role, (count, description) in kitchen_roles.items():
        for i in range(1, count + 1):
            staff_member = f"{role} {i}"
            agents[staff_member] = make_func(staff_member)

    return agents

# ------------------------------------------------
# 4. Build LangGraph workflow
# ------------------------------------------------
def build_workflow(llm: LLMWrapper, dish: str):
    graph = StateGraph(KitchenState)  # use KitchenState as state schema

    def retriever(state):
        state["docs"] = retrieve(state["command"])
        return state

    def recipe_creator(state):
        print("[📝] Reviewing recipe...")
        if args.recipe:
            print(f"  [📝] Retrieving recipe from '{args.recipe}'")
            with open(args.recipe, "r") as f:
                response = f.read()

        else:
            print(f"  [📝] Sythesizing recipe from document store")
            docs = "\n".join(state["docs"])
            prompt = (
                f"You are the Chef de cuisine tasked to prepare {dish}. Create a recipe for the dish based on these recipes:\n"
                f"{docs}\n\n"
            )
            stage = "Creating"
            response = llm.generate(prompt)
            llm_log.append((stage, prompt, response))

            print(f"  [📝] Saving generated recipe")
            with open(os.path.join(args.output_directory, args.generated_recipe), "w") as f:
                print(f"{response}", file=f)

        state["recipe"] = response
        return state

    def planner(state):
        print("[📝] Generating execution plan")
        prompt = (
            f"You are the Chef de cuisine tasked to prepare {dish} based on this recipe:\n\n"
            f"{state['recipe']}\n\n"
            "The following ingredients and utensils are available to make the dish:\n\n"
            f"{ingredients}\n\n"
            "Only these ingredients and utensils are available.\n"
            "Plan a time-step sequence of tasks to prepare the dish beginning at Time 0, denoted T0, and incrementing by one for each step until completion.\n"
            "Within each time step list the tasks to be completed by each team member in the form 'Team member: Task' such that each of the tasks within the step can be completed independently.\n"
            "Combine multiple tasks within a step completed by the same team member into a single task.\n"
            f"The list below lists the roles in the team and, in brackets, the number of members in the role.\n"
            "Team members are identified by their role name combined with a unique number within the role.\n"
            "For example, 'Sous Chef [2]' would indicate there are two Sous Chefs in the team; 'Sous Chef 1', and 'Sous Chef 2'. Here's the list:\n\n"
            f"{os.linesep.join(f'{k} [{v[0]}]' for (k, v) in kitchen_roles.items())}\n\n"
            "No other roles or team members are available.\n"
            "All tasks must be assigned to a specific team member.\n" 
            "Groupings such as 'All team members' are not allowed.\n"
            "If it's unclear which role should perform a task the task can be assigned by the team member 'Nonce 1'."
        )
        stage = "Planning"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        state["plan"] = response

        print("  [📝] Saving execution plan")
        with open(os.path.join(args.output_directory, args.execution_plan), "w") as f:
            print(f"{response}", file=f)

        return state

    def router(state):
        print("[🔀] Routing tasks to agents")
        routing = collections.defaultdict(list)
        state["staff"] = make_agents(llm)

        print("  [🔀] Meet the team...")
        print(f"{os.linesep.join(f'    {member}' for member in state['staff'].keys())}")

        for task in state["plan"].split("\n"):
            task = task.strip()

            if len(task) > 2:
                parts  = task[2:].split(':')

                if len(parts) == 2:
                    (team_member, subtask) = parts
                    team_member = team_member.replace('*', '')
                    team_member = team_member.strip()

                    if team_member in state["staff"]:
                        subtask = subtask.replace('*', '')
                        subtask = subtask.strip()
                        routing[team_member].append(subtask)

        state["routing"] = routing
        return state

    def executor(state):
        print("[⚙️] Executing routed tasks")
        state["results"] = {}

        # For each role, run all its tasks through the corresponding agent
        for team_member, tasks in state.get("routing", {}).items():
            results = []

            for task in tasks:
                # agent returns {role_key: message}
                res_dict = (state["staff"][team_member])(task)
                results.append(res_dict["result"])

            # Store combined execution output under the role key
            state["results"][team_member] = "\n".join(results)

        return state

    def aggregator(state):
        print("[📦] Aggregating results into final recipe")
        # Gather each role's output from state
        outputs = []

        for activity in state["routing"].values():
            outputs.append(os.linesep.join(activity))

        joined = "\n".join(outputs)
        prompt = (
            "Combine these executed subtasks into a final recipe with clear steps:\n"
            f"{joined}"
        )
        stage = "Aggregation"
        response = llm.generate(prompt)
        llm_log.append((stage, prompt, response))
        state["final_recipe"] = response

        print("  [📦] Saving final recipe")
        with open(os.path.join(args.output_directory, args.final_recipe), "w") as f:
            print(f"{response}", file=f)

        return state

    graph.add_node("retriever", retriever)
    graph.add_node("recipe_creator", recipe_creator)
    graph.add_node("planner", planner)
    graph.add_node("router", router)
    graph.add_node("executor", executor)
    graph.add_node("aggregator", aggregator)

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "recipe_creator")
    graph.add_edge("recipe_creator", "planner")
    graph.add_edge("planner", "router")
    graph.add_edge("router", "executor")
    graph.add_edge("executor", "aggregator")
    return graph

# ------------------------------------------------
# 5. Main entrypoint
# ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Kitchen_Brigade_ingredients")
    parser.add_argument("--dish", "-d", 
                        required=True,
                        help="The name of the dish to prepare. Used to create a recipe if none is provided.")
    parser.add_argument("--crew", "-c", 
                        required=True,
                        help="File defining the available roles and number of team members in each role.")
    parser.add_argument("--ingredients", "-i",
                        required=True,
                        help="Text file listing the available ingredients and utensils.")
    parser.add_argument("--recipe", "-r", 
                        required=False,
                        help="Text file containing the recipe to prepare. Will be generated via RAG if not supplied.")
    parser.add_argument("--provider", "-p", 
                        default="openai",
                        help="Name of the planning and judging model provider. One of: openai (default), or watsonx")
    parser.add_argument("--model", "-m", 
                        default="gpt-4o",
                        help="Name of the model to use of planning and judging. Defaults to 'gpt-4o'")
    group = parser.add_argument_group("Output files")
    group.add_argument("--output-directory", '-o', 
                       default=".",
                       help="Path to directory for output files.")
    group.add_argument("--generated-recipe", "--gr", 
                       default="generated-recipe.txt",
                       help="Output file for the recipe generated by RAG if no --recipe is provided. Default: generated-recipe.txt")
    group.add_argument("--final-recipe", "--fr", 
                       default="final-recipe.txt",
                       help="Output file for the final recipe after planning. Default: final-recipe.txt")
    group.add_argument("--execution-plan", "--ep", 
                       default="execution-plan.txt",
                       help="Output file for the generated preparation plan. Default: execution-plan.txt")
    group.add_argument("--plan-feedback", "--pf", 
                       default="plan-feedback.txt",
                       help="File for the output of the Planning Judge. Default: plan-feedback.txt")
    group.add_argument("--execution-feedback", "--ef", 
                       default="execution-feedback.txt",
                       help="File for the output of the Execution Judge. Default: execution-feedback.txt")
    args = parser.parse_args()

    print(f"Invocation command line: {' '.join(sys.argv)}")

    # ------------------------------------------------
    # Load the kitchen brigade, i.e. the team roles and the number of each role
    # ------------------------------------------------
    with open(args.crew, 'r') as f:
        kitchen_roles = json.load(f)

    kitchen_roles["Nonce"] = [1,
        (
            "Doesn't do anything. Don't assign any tasks to this team member because "
            "they will not get done."
        )
    ]

    # ------------------------------------------------
    # Load the available ingredients
    # ------------------------------------------------
    with open(args.ingredients, 'r') as f:
        ingredients = f.read()

    # ------------------------------------------------
    # Recipe / request scenarios
    # ------------------------------------------------
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

    # Backup original descriptions for good scenarios
    original_role_descriptions = kitchen_roles.copy()
    # Define bad descriptions for testing
    bad_role_descriptions = { key: "Generic task performer." for key in kitchen_roles }

    EMBEDDINGS_FILE = "embeddings.npy"
    INDEX_FILE = "faiss.index"

    llm = LLMWrapper(args.provider, args.model)
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

    """
    for scenario in scenarios:
        label = scenario["label"]
        command = scenario["command"]
        desc_set = scenario.get("description_set")

    """

    """
        if desc_set == "bad":
            role_descriptions.clear()
            role_descriptions.update(bad_role_descriptions)
        elif desc_set == "good":
            role_descriptions.clear()
            role_descriptions.update(original_role_descriptions)

        print(f"\n=== Scenario: {label} ===\n")
        workflow = build_workflow(llm, command)
        initial_state = {"command": command}
    """    
    workflow = build_workflow(llm, args.dish)
    initial_state = {"command": args.dish}
    final_state = workflow.compile().invoke(initial_state)
    # Gather execution outputs from all agent keys

    final_recipe = final_state.get("final_recipe", "")
    print("\n=== Final Recipe ===\n")
    print(final_recipe)

    plan_text = final_state.get("plan", "")
    print("=== Plan ===\n")
    print(plan_text)

    execution_results = [ v for v in final_state["results"].values() ]
    print("\n=== Execution Results ===\n")
    print("\n".join(execution_results))
    
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

    print("  [👩‍⚖️] Saving planning evaluation")
    with open(os.path.join(args.output_directory, args.plan_feedback), "w") as f:
        print(f"{planning_feedback}", file=f)

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

    print("  [👩‍⚖️] Saving execution evaluation")
    with open(os.path.join(args.output_directory, args.execution_feedback), "w") as f:
        print(f"{execution_feedback}", file=f)

    print("\n=== Judge Results ===\n")
    print("-- Planning Judge --\n", planning_feedback)
    print("\n-- Execution Judge --\n", execution_feedback)


