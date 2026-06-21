from pathlib import Path

DIRS = [
    "configs",

    "data/raw/fastapi",
    "data/processed",
    "data/indexes",

    "experiments/questions",
    "experiments/runs",

    "scripts",

    "src/doc_agent/corpus",
    "src/doc_agent/retrieval",
    "src/doc_agent/rag",
    "src/doc_agent/agent",
    "src/doc_agent/evaluation",
    "src/doc_agent/utils",
]

INIT_FILES = [
    "src/doc_agent/__init__.py",
    "src/doc_agent/corpus/__init__.py",
    "src/doc_agent/retrieval/__init__.py",
    "src/doc_agent/rag/__init__.py",
    "src/doc_agent/agent/__init__.py",
    "src/doc_agent/evaluation/__init__.py",
    "src/doc_agent/utils/__init__.py",
]

for directory in DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)

for file_path in INIT_FILES:
    Path(file_path).touch(exist_ok=True)

print("Project structure created successfully.")