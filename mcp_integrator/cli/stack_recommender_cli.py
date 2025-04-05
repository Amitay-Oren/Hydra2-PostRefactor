#!/usr/bin/env python3
"""
Stack Recommender CLI Module

This module provides a command-line interface for the MCP Stack Recommender system,
allowing users to analyze high-level tasks and get recommendations for optimal MCP stacks.
"""

import sys
# print("--- sys.path inside stack_recommender_cli.py ---")
# print("\n".join(sys.path))
# print("-----------------------------------------------")

import argparse
import json
import os
from typing import Optional, Dict, Any, Callable

# Import the core functionality from solution module
try:
    from mcp_integrator.solution.recommender import MCPStackRecommender
except ImportError:
    print("Error: Could not import from mcp_integrator.solution.recommender module.")
    print("Please make sure the package is installed correctly.")
    sys.exit(1)

# Import the MCP finder functionality
try:
    from mcp_integrator.core.finder import find_mcps
except ImportError:
    print("Error: Could not import from mcp_integrator.core.finder module.")
    print("Please make sure the package is installed correctly.")
    sys.exit(1)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="MCP Stack Recommender")
    
    # Task input options
    parser.add_argument("--task", type=str, help="High-level task description")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--output", type=str, help="Optional file to save results as JSON")
    
    # Configuration options
    parser.add_argument("--max-mcps-per-stack", type=int, default=5, 
                       help="Maximum number of MCPs to include in a stack")
    parser.add_argument("--min-mcps-required", type=int, default=1,
                       help="Minimum number of subtasks that need MCPs for a valid stack")
    parser.add_argument("--max-mcps-per-query", type=int, default=10,
                       help="Maximum number of MCPs to process per query")
    parser.add_argument("--use-domain-context", action="store_true",
                       help="Use domain-specific query generation (not recommended)")
    parser.add_argument("--clarification-threshold", type=int, default=50,
                       help="Number of MCPs found that triggers clarification")
    parser.add_argument("--no-query-clarification", action="store_true",
                       help="Disable interactive query clarification")
    
    # Optimization options
    parser.add_argument("--no-caching", action="store_true",
                       help="Disable caching of MCP finder results")
    parser.add_argument("--no-cross-learning", action="store_true",
                       help="Disable cross-subtask learning")
    parser.add_argument("--no-smart-decomposition", action="store_true",
                       help="Disable smart task decomposition")
    parser.add_argument("--similarity-threshold", type=float, default=0.75,
                       help="Threshold for considering subtasks or queries similar (0.0-1.0)")
    parser.add_argument("--min-subtasks", type=int, default=3,
                       help="Minimum number of subtasks to aim for")
    parser.add_argument("--max-subtasks", type=int, default=8,
                       help="Maximum number of subtasks to allow")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Task description from command line or interactive input
    task_description = args.task
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required for the MCP Stack Recommender.")
        print("Please set it before running this command.")
        sys.exit(1)
    
    # Initialize the MCP Stack Recommender
    try:
        # Create the recommender with imported MCP finder
        recommender = MCPStackRecommender(
            mcp_finder_func=find_mcps,
            max_mcps_per_stack=args.max_mcps_per_stack,
            min_mcps_required=args.min_mcps_required,
            use_domain_context=args.use_domain_context,
            max_mcps_per_query=args.max_mcps_per_query,
            clarification_threshold=args.clarification_threshold,
            enable_query_clarification=not args.no_query_clarification,
            enable_caching=not args.no_caching,
            enable_cross_learning=not args.no_cross_learning,
            enable_smart_decomposition=not args.no_smart_decomposition,
            similarity_threshold=args.similarity_threshold,
            min_subtasks=args.min_subtasks,
            max_subtasks=args.max_subtasks
        )
    except Exception as e:
        print(f"Error initializing MCP Stack Recommender: {str(e)}")
        sys.exit(1)
    
    # Show optimization status
    if not args.no_caching:
        print("✅ Query caching is enabled (use --no-caching to disable)")
    if not args.no_cross_learning:
        print("✅ Cross-subtask learning is enabled (use --no-cross-learning to disable)")
    if not args.no_smart_decomposition:
        print("✅ Smart task decomposition is enabled (use --no-smart-decomposition to disable)")
    
    if not task_description or args.interactive:
        print("MCP Stack Recommender - Interactive Mode")
        print("---------------------------------------")
        task_description = input("Enter your high-level task: ")
    
    # Define clarification callback for interactive use
    def get_clarification(question):
        print(f"\nClarification needed: {question}")
        return input("Your answer: ")
    
    # Use clarification callback only if in interactive mode and clarification is enabled
    clarification_callback = None
    if args.interactive and not args.no_query_clarification:
        clarification_callback = get_clarification
        print("\nInteractive query clarification is enabled. You may be asked to provide")
        print("additional details to help find the most relevant MCPs.")
    elif not args.interactive and not args.no_query_clarification:
        print("\nWarning: Query clarification is enabled but not available in non-interactive mode.")
        print("Use --interactive to enable clarification or --no-query-clarification to disable it.")
    
    # Analyze task and display results
    results = recommender.analyze_task(
        task_description, 
        clarification_callback=clarification_callback
    )
    
    # Display results
    recommender.display_results(results)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    elif args.interactive:
        save_option = input("\nSave results to file? (y/n): ")
        if save_option.lower() == 'y':
            filename = input("Enter filename (default: results.json): ") or "results.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {filename}")


if __name__ == "__main__":
    main() 