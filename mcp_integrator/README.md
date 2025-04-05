# MCP Integrator

A collection of tools for discovering, indexing, and integrating MCP (Model, Component, or Protocol) servers from registries. The system helps users find the best MCPs for their needs and recommends optimal combinations for complex tasks.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Components](#components)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Project Roadmap](#project-roadmap)
- [Performance Optimization Components](#performance-optimization-components)
- [Troubleshooting](#troubleshooting)

## Overview

MCP Integrator provides tools to:

1. **Find MCPs**: Search for specific MCPs using natural language queries
2. **Recommend MCP Stacks**: Analyze complex tasks and suggest optimal MCP combinations
3. **Integrate MCPs**: Interface with multiple MCP registries and integrate MCPs with other systems

### Package Structure

```
mcp_integrator/
  ├── cli/                  # Command-line interfaces
  │   ├── mcp_finder_cli.py # CLI for finding MCPs
  │   └── stack_recommender_cli.py # CLI for stack recommendation
  ├── core/                 # Core finder functionality
  │   └── finder.py         # MCP finder implementation
  ├── task/                 # Task analysis and decomposition
  │   ├── analyzer.py       # Task analysis implementation
  │   └── decomposer.py     # Smart task decomposition
  ├── query/                # Query generation and processing
  │   ├── generator.py      # Query generation
  │   ├── clarifier.py      # Query clarification
  │   ├── cache.py          # Query caching
  │   └── models.py         # Query data models
  ├── solution/             # Solution recommendation and explanation
  │   ├── recommender.py    # Stack recommendation implementation
  │   └── explainer.py      # Solution explanation generation
  └── utils/                # Utility functions and interfaces
      ├── context.py        # Cross-subtask learning
      └── interface.py      # Interface for external integration (TBD/Review)
```

## Installation

1. Clone the repository
2. Install the package:
   ```bash
   pip install -e .
   ```
   This will install the package in development mode, allowing you to make changes to the code without reinstalling.

3. Set up environment variables in a `.env` file:
   ```
   SMITHERY_API_TOKEN=your_smithery_api_token
   OPENAI_API_KEY=your_openai_api_key
   ```

## Components

This package provides both command-line tools and importable Python modules.

**Important:** When running command-line tools, always execute them from the *parent directory* (`Hydra2` in this case) using `python -m <module_path>`. This ensures Python correctly finds the installed package and resolves internal imports.

### MCP Finder CLI
(`mcp_integrator.cli.mcp_finder_cli`)

A command-line tool for searching and discovering MCPs in the Smithery Registry.

```bash
# Navigate to the parent directory (e.g., C:\projects\Hydra2)
cd /path/to/parent/directory

# Basic usage (replace YOUR_TOKEN)
python -m mcp_integrator.cli.mcp_finder_cli --query "your search query" --api-token YOUR_TOKEN

# Interactive mode with GPT-4o-powered assistant
python -m mcp_integrator.cli.mcp_finder_cli --gpt4o
```

### MCP Stack Recommender CLI
(`mcp_integrator.cli.stack_recommender_cli`)

A command-line tool that analyzes complex tasks, breaks them down into subtasks, and recommends optimal MCP combinations.

```bash
# Navigate to the parent directory (e.g., C:\projects\Hydra2)
cd /path/to/parent/directory

# Run in interactive mode
python -m mcp_integrator.cli.stack_recommender_cli --interactive

# Run with a specific task description
python -m mcp_integrator.cli.stack_recommender_cli --task "Describe your complex task here"

# Use through the MCP finder (will invoke the stack recommender)
python -m mcp_integrator.cli.mcp_finder_cli --stack-recommender
```

#### Key Features

- **Task Decomposition**: Uses AI to break down complex tasks into manageable subtasks.
- **Multi-Query Generation**: Creates specialized queries for each subtask.
- **Intelligent Recommendations**: Suggests the best MCP combinations.
- **Solution Explanations**: Provides setup guides and integration instructions.

## Usage

There are two primary ways to use the `mcp_integrator` package:

1.  **Command-Line Interface (CLI):** For interactive use or quick tasks directly from your terminal. See the [Components](#components) section for examples.
2.  **Programmatic Import:** For integrating the functionality into other Python scripts or applications.

### Programmatic Usage Examples

#### Basic MCP Search

```python
# Import the finder function
from mcp_integrator.core.finder import find_mcps

# Ensure SMITHERY_API_TOKEN is set in your environment or pass it directly
api_token = os.getenv("SMITHERY_API_TOKEN") # or "YOUR_TOKEN"

# Search for MCPs
results = find_mcps("text processing", api_token_to_use=api_token)

if results:
    for mcp in results:
        print(f"- {mcp.get('name', 'N/A')}: {mcp.get('description', 'N/A')}")
else:
    print("No MCPs found.")
```

#### Stack Recommendation

```python
import os
from mcp_integrator.solution.recommender import MCPStackRecommender
from mcp_integrator.core.finder import find_mcps # Need the finder function

# Ensure required environment variables are set (OPENAI_API_KEY, SMITHERY_API_TOKEN)
if not os.getenv("OPENAI_API_KEY") or not os.getenv("SMITHERY_API_TOKEN"):
    print("Error: OPENAI_API_KEY and SMITHERY_API_TOKEN environment variables are required.")
else:
    # Initialize recommender, passing the finder function
    recommender = MCPStackRecommender(
        mcp_finder_func=find_mcps, # Pass the actual function
        enable_caching=True,
        enable_cross_learning=True,
        enable_smart_decomposition=True
    )

    # Define a simple clarification callback for interactive scripts
    def get_clarification(question):
        print(f"\n> Clarification needed: {question}")
        return input("> Your answer: ")

    # Analyze a high-level task
    task = "Build a system to search, summarize, and analyze legal documents"
    print(f"Analyzing task: {task}...")

    results = recommender.analyze_task(
        task,
        clarification_callback=get_clarification # Use None if no interaction desired
    )

    # Display results (includes explanation)
    print("\n--- Recommendation Results ---")
    recommender.display_results(results)
```

#### Task Analysis and Decomposition

```python
from mcp_integrator.task.analyzer import TaskDecomposer
from mcp_integrator.task.decomposer import SmartDecomposer

# Initialize decomposers
task_decomposer = TaskDecomposer()
smart_decomposer = SmartDecomposer()

task = "Build a system to search, summarize and analyze legal documents"

# Decompose the task
print(f"Decomposing task: {task}...")
subtasks = task_decomposer.decompose_task(task)
print("Initial Subtasks:", subtasks)

# Use smart decomposer for potential optimization (merging, etc.)
optimized_subtasks = smart_decomposer.optimize_decomposition(subtasks)
print("Optimized Subtasks:", optimized_subtasks)
```

#### Query Generation and Clarification

```python
from mcp_integrator.query.generator import QueryGenerator
from mcp_integrator.query.clarifier import QueryClarifier

# Assume 'optimized_subtasks' from previous example
subtasks_to_query = optimized_subtasks # or the initial subtasks

# Generate queries
generator = QueryGenerator()
queries = generator.generate_queries(subtasks_to_query)
print("Generated Queries:", queries)

# Optionally clarify queries interactively
clarifier = QueryClarifier()
def clarify_from_user(question):
    print(f"\n> Clarification needed: {question}")
    return input("> Your answer: ")

print("\nAttempting query clarification (if needed)...")
clarified_queries = clarifier.clarify_queries(
    queries,
    clarification_callback=clarify_from_user
)
print("Clarified Queries:", clarified_queries)
```

#### Cross-Subtask Learning (Global Context)

```python
from mcp_integrator.utils.context import GlobalContext

# Initialize global context (requires OPENAI_API_KEY for embeddings)
context = GlobalContext(
    relevance_threshold=0.7,
    enable_cross_learning=True
)

# Add a clarification learned from a previous interaction
context.add_clarification(
    subtask={"name": "document_search", "description": "Search for documents"},
    query="document search",
    clarification_text="User specified they need *patent* documents specifically.",
    refined_query="patent document search"
)

# Later, when processing a *similar* subtask, apply relevant clarifications
def apply_learned_terms(subtask, query, clarification):
    # Example refinement: Append key terms from the learned clarification
    key_terms = " ".join(clarification.key_terms)
    print(f"Applying terms '{key_terms}' from learned clarification to query: '{query}'")
    return f"{query} focusing on {key_terms}"

similar_subtask = {"name": "patent_analysis", "description": "Analyze patent documents"}
original_query = "patent analysis"

refined_query = context.apply_relevant_clarifications(
    subtask=similar_subtask,
    query=original_query,
    refine_func=apply_learned_terms # Your function to apply the clarification
)

print(f"\nOriginal Query for similar task: {original_query}")
print(f"Refined Query after applying context: {refined_query}")
```

## Development

### Contributing

Contributions are welcome! Please check the tasks.md file for current work and planned improvements.

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=.
```

Current test coverage is focused on key components. Future work will expand coverage to all components, aiming for at least 80% code coverage.

## Project Roadmap

See tasks.md for the detailed project roadmap, which includes:

1. Code reorganization (✅ Completed)
2. Create/Review `utils/interface.py`
3. Comprehensive testing improvements
4. Documentation enhancements
5. Performance optimization
6. User experience improvements

## Performance Optimization Components

The system includes several performance optimization components:

### Query Cache

Prevents redundant API calls by caching query results and detecting semantically similar queries.

### Global Context

Enables cross-subtask learning by storing user clarifications and applying them to related subtasks.

### Smart Decomposer

Optimizes task decomposition by detecting similar subtasks, merging redundant ones, and adjusting complexity.

## Troubleshooting

### UTF-16 Encoding Issues in __init__.py Files

If you encounter errors like this when trying to import the package:

```
SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

or

```
SyntaxError: source code string cannot contain null bytes
```

This is likely due to the `__init__.py` files being saved with UTF-16 encoding instead of UTF-8. Here's how to fix it:

1. Identify all the problematic files:
   ```powershell
   Get-ChildItem -Recurse -Path mcp_integrator -Filter "__init__.py" | ForEach-Object {
       $filePath = $_.FullName
       $content = Get-Content $filePath -Encoding Byte -TotalCount 4
       if ($content[0] -eq 0xFF -and $content[1] -eq 0xFE) {
           Write-Output "UTF-16 BOM found in: $filePath"
       }
   }
   ```

2. Fix the encoding by replacing the content with a simple UTF-8 encoded file:
   ```powershell
   foreach ($dir in @('', 'cli', 'core', 'query', 'solution', 'task', 'utils')) {
       $path = if ($dir -eq '') {
           "mcp_integrator\__init__.py"
       } else {
           "mcp_integrator\$dir\__init__.py"
       }
       Set-Content -Path $path -Value "# Package initialization" -Encoding UTF8 -Force
   }
   ```

3. Reinstall the package:
   ```powershell
   pip install -e .
   ```

4. Test the import:
   ```powershell
   python -c "import mcp_integrator; print('Package imported successfully!')"
   ```

### Preventing the Issue in the Future

When creating new Python files:
- Always use UTF-8 encoding without BOM (Byte Order Mark)
- Use editors that default to UTF-8 encoding for Python files
- If using automated scripts to create files, ensure they use UTF-8 encoding

### Command Line Issues on Windows

If you encounter issues with command-line operators like `&&` in PowerShell, use separate commands or switch to Command Prompt where these operators are supported:

```bash
# In Command Prompt this works:
python -c "import mcp_integrator" && python -m mcp_integrator.cli.stack_recommender_cli --help

# In PowerShell, use separate commands:
python -c "import mcp_integrator"
python -m mcp_integrator.cli.stack_recommender_cli --help
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 