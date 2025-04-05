#!/usr/bin/env python3
import requests
import argparse
import os
import sys
import json
from dotenv import load_dotenv
from typing import Optional, Tuple

# Load environment variables from .env file
load_dotenv()

# Import functionality from the core.finder module
try:
    from mcp_integrator.core.finder import find_mcps, get_all_servers, get_server_details, list_servers, get_headers
except ImportError:
    print("Error: Could not import from mcp_integrator.core.finder module.")
    print("Please make sure the package is installed correctly.")
    sys.exit(1)

api_token = os.getenv("SMITHERY_API_TOKEN")
if not api_token:
    print("Error: SMITHERY_API_TOKEN environment variable not set")
    sys.exit(1)

API_BASE_URL = "https://registry.smithery.ai/servers"

def gather_user_intent_and_query() -> tuple:
    """
    Uses the MCPAgent to gather user intentions and generate a refined query through
    a dynamic conversation using GPT-4o.
    
    Returns:
        tuple: (query_string, page_size, api_token) for the API request.
    """
    try:
        print("\n=== MCP Query Assistant (powered by GPT-4o) ===")
        print("This assistant will help you find the right MCP servers through a conversation.")
        print("It will ask questions to understand your needs and generate an optimized search query.")
        print("Type 'exit' at any time to quit the assistant.\n")
        
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("The assistant requires an OpenAI API key to function properly.")
            use_mock = input("Would you like to continue with a simplified experience? (y/n): ").lower() == 'y'
            if not use_mock:
                return None
            # If continuing without API key, fall back to command-line args
            return None
        
        # Import here to avoid circular imports
        from mcp_integrator.query.models import OpenAIService, MCPAgent
        
        # Initialize the OpenAI service and MCP Agent
        openai_service = OpenAIService(api_key=openai_api_key)
        agent = MCPAgent(openai_service=openai_service)
        
        # Gather user input through dynamic conversation
        # The agent's gather_user_input method now returns the query string directly
        final_response = agent.gather_user_input()
        
        # At this point, final_response should be a string representing the query
        if isinstance(final_response, str):
            # Make sure the query is visible
            print("\n" + "="*80)
            print(f"EXECUTING QUERY: {final_response}")
            print("="*80)
            
            # Use a default page size of 10, or 95 if the user mentioned wanting many results
            # This is a guess based on conversation context
            many_results = any(word in " ".join(str(msg) for msg in agent.conversation_history).lower() 
                              for word in ["many", "all", "lots", "maximum"])
            page_size = 95 if many_results else 10
            
            return final_response, page_size, api_token
        
        # This should not happen with the updated agent, but just in case
        print("Error: Invalid response from agent. Falling back to default query.")
        return "is:deployed", 10, api_token
        
    except Exception as e:
        print(f"Error gathering user intent: {str(e)}")
        print("Falling back to command-line arguments.")
        return None

def main():
    # Parse command-line arguments for backward compatibility and non-interactive use
    parser = argparse.ArgumentParser(description="Smithery MCP Finder Script")
    parser.add_argument("--api-token", type=str, default=os.environ.get("SMITHERY_API_TOKEN"),
                        help="Your Smithery API token. Can also be set via SMITHERY_API_TOKEN environment variable.")
    parser.add_argument("--query", type=str, default="is:deployed",
                        help="Search query for MCP servers. Supports semantic search. Example: 'owner:smithery-ai is:deployed'.")
    parser.add_argument("--pageSize", type=int, default=10, help="Number of servers per page to retrieve.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode with user intent gathering.")
    parser.add_argument("--gpt4o", action="store_true", help="Use GPT-4o for dynamic conversation-based query generation.")
    parser.add_argument("--stack-recommender", action="store_true", 
                       help="Use the MCP Stack Recommender to analyze complex tasks and recommend optimal MCP stacks.")
    args = parser.parse_args()

    if not args.api_token:
        print("API token is required. Provide it via --api-token argument or SMITHERY_API_TOKEN environment variable.")
        sys.exit(1)

    # Check if we should run the stack recommender
    if args.stack_recommender:
        try:
            # Import the MCP Stack Recommender
            try:
                from mcp_integrator.solution.recommender import MCPStackRecommender
            except ImportError:
                print("Error: Could not import the MCP Stack Recommender.")
                print("Please ensure the package is installed correctly.")
                sys.exit(1)
            
            print("Initializing MCP Stack Recommender...")
            # Initialize the recommender with our MCP finder function from the imported module
            recommender = MCPStackRecommender(
                mcp_finder_func=lambda query: find_mcps(query, args.api_token)
            )
            
            # Ask for task in interactive mode
            print("MCP Stack Recommender - Interactive Mode")
            print("---------------------------------------")
            task_description = input("Enter your high-level task: ")
            
            # Define clarification callback for interactive use
            def get_clarification(question):
                print(f"\nClarification needed: {question}")
                return input("Your answer: ")
            
            # Analyze task and display results
            results = recommender.analyze_task(task_description, get_clarification)
            recommender.display_results(results)
            
            # Optionally save results to file
            save_option = input("\nSave results to file? (y/n): ")
            if save_option.lower() == 'y':
                filename = input("Enter filename (default: results.json): ") or "results.json"
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {filename}")
            
            return
        except Exception as e:
            print(f"Error running MCP Stack Recommender: {str(e)}")
            print("Falling back to regular MCP Finder mode.")
    
    # Determine whether to use interactive mode
    use_gpt4o = args.gpt4o or len(sys.argv) == 1  # Default to GPT-4o if no args provided
    interactive_mode = args.interactive or use_gpt4o  # Interactive mode if either flag is set
    
    # Gather user intent if in interactive mode
    if interactive_mode:
        if use_gpt4o:
            result = gather_user_intent_and_query()
            if not result:
                print("GPT-4o mode failed or was aborted. Falling back to basic interactive mode.")
                from mcp_integrator.query.models import MCPAgent as LegacyMCPAgent
                # Create a legacy agent instance
                legacy_agent = LegacyMCPAgent()
                legacy_query_input = legacy_agent.gather_user_input()
                legacy_query = legacy_query_input.generate_query_string()
                result = (legacy_query, legacy_query_input.parameters.page_size, args.api_token)
        else:
            from mcp_integrator.query.models import MCPAgent as LegacyMCPAgent
            # Create a legacy agent instance
            legacy_agent = LegacyMCPAgent()
            legacy_query_input = legacy_agent.gather_user_input()
            legacy_query = legacy_query_input.generate_query_string()
            result = (legacy_query, legacy_query_input.parameters.page_size, args.api_token)
        
        if result:
            query, pageSize, token = result
        else:
            # If user intent gathering failed, use command-line args
            query = args.query
            pageSize = args.pageSize
            token = args.api_token
    else:
        # Use command-line args
        query = args.query
        pageSize = args.pageSize
        token = args.api_token
    
    # Execute the query
    servers = get_all_servers(token, query, pageSize)
    
    # Show results
    print(f"\nFound {len(servers)} server(s) matching query: '{query}'")
    
    # Score and sort results
    for server in servers:
        if "score" not in server:
            # Use the imported score_server from core.finder
            import mcp_integrator.core.finder as finder
            server["score"] = finder.score_server(server)
    
    # Sort by score
    sorted_servers = sorted(servers, key=lambda x: x.get("score", 0), reverse=True)
    
    # Display results
    for i, server in enumerate(sorted_servers):
        print(f"\n{i+1}. {server.get('displayName', 'Unnamed')} - {server.get('qualifiedName', 'No ID')}")
        if "description" in server and server["description"]:
            print(f"   Description: {server['description']}")
        print(f"   Score: {server.get('score', 0)}")
        print(f"   Deployed: {'Yes' if server.get('isDeployed', False) else 'No'}")
        
    # Option to view details for a specific server
    if sorted_servers:
        while True:
            selection = input("\nEnter number to view details (or 'q' to quit): ")
            if selection.lower() == 'q':
                break
            
            try:
                index = int(selection) - 1
                if 0 <= index < len(sorted_servers):
                    selected_server = sorted_servers[index]
                    name = selected_server.get("qualifiedName")
                    details = get_server_details(token, name)
                    
                    print("\nServer Details:")
                    print(json.dumps(details, indent=2))
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")


if __name__ == "__main__":
    main() 