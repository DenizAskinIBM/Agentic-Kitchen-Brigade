Overview

This project implements a kitchen brigade–style agentic workflow using LangGraph, with support for either OpenAI’s gpt-4o or IBM’s watsonx LLMs. It performs Retrieval-Augmented Generation (RAG) over a public recipe corpus, plans cooking subtasks assigned to specific brigade roles (e.g., sous-chef, saucier), executes those subtasks via dedicated LLM agents, and finally aggregates the results into a complete recipe. After cooking, two separate judge agents evaluate (1) the quality of the planning assignments and (2) the efficiency and effectiveness of the execution, providing structured feedback.

⸻

Features
	•	Configurable LLM backend: Switch between OpenAI and Watsonx models via .env  ￼
	•	RAG with open-source recipes: Uses the formido/recipes dataset from Hugging Face  ￼
	•	Embeddings & FAISS search: Leverages all-MiniLM-L6-v2 sentence embeddings for retrieval  ￼, and indexes them with FAISS for fast similarity lookup  ￼
	•	Agentic “brigade” architecture: Models each kitchen role (chef de cuisine, commis, pâtissier, etc.) as an LLM agent node in a LangGraph workflow  ￼
	•	Four-stage workflow:
	1.	Retriever: Fetch relevant recipe snippets
	2.	Planner: Generate and assign subtasks to roles
	3.	Executor: Dispatch each subtask to the appropriate agent
	4.	Aggregator: Combine execution results into a final recipe
	•	Automated judges: Two post-hoc evaluations—Planning Judge and Execution Judge—provide qualitative feedback on the workflow

⸻

Installation
	1.	Clone the repository

git clone https://github.com/your-org/kitchen-brigade-workflow.git
cd kitchen-brigade-workflow


	2.	Create & activate a Python virtual environment

python3 -m venv venv
source venv/bin/activate


	3.	Install dependencies

pip install -r requirements.txt


	4.	Install core libraries

pip install langgraph                  # LangGraph orchestration  [oai_citation:5‡PyPI](https://pypi.org/project/langgraph/?utm_source=chatgpt.com)
pip install python-dotenv              # Load .env credentials  [oai_citation:6‡PyPI](https://pypi.org/project/python-dotenv/?utm_source=chatgpt.com)
pip install faiss-cpu                  # FAISS vector store  [oai_citation:7‡PyPI](https://pypi.org/project/faiss/?utm_source=chatgpt.com)
pip install sentence-transformers      # Embeddings (all-MiniLM)  [oai_citation:8‡Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com)
pip install datasets                   # Hugging Face Datasets  
pip install openai ibm-watsonx         # LLM clients  



⸻

Configuration
	1.	Create a .env file in the project root:

OPENAI_API_KEY=your_openai_key_here
WATSONX_URL=https://your-watsonx-instance
WATSONX_APIKEY=your_watsonx_key_here
LLM_PROVIDER=openai        # or "watsonx"
LLM_MODEL=gpt-4o           # or your Watsonx model


	2.	Secure your .env by adding it to .gitignore.

⸻

Usage

Run the workflow with a single command:

python kitchen_brigade.py

	•	Input: Modify the initial_state["command"] variable to request any dish (e.g., "Recipe for mushroom risotto").
	•	Output:
	1.	The Final Recipe printed to stdout.
	2.	Judge Results with Planning and Execution feedback.

⸻

Code Structure
	•	kitchen_brigade.py: Main entry point
	•	LLMWrapper: Abstracts OpenAI/Watsonx calls
	•	RAG Setup: Loads recipes, builds embeddings & FAISS index
	•	Agent Definitions: Maps kitchen roles to LLM agents
	•	LangGraph Workflow: Defines nodes (retriever, planner, executor, aggregator) and edges
	•	Judges: Two additional LLM prompts that evaluate planning and execution quality

⸻

Extending & Customizing
	•	Add new roles by updating the kitchen_roles list.
	•	Swap datasets by changing the HF load path in the RAG setup.
	•	Adjust retrieval size via retrieve(query, k=…).
	•	Refine prompts in the planner, executor, and judge nodes for domain-specific cooking styles.

⸻

Contributing
	1.	Fork the repo
	2.	Create a branch (git checkout -b feature/your-feature)
	3.	Commit changes (git commit -m 'Add some feature')
	4.	Push to branch (git push origin feature/your-feature)
	5.	Open a Pull Request

Please follow our Code of Conduct and Contributing Guide.

⸻

License

This project is licensed under the MIT License. See LICENSE for details.