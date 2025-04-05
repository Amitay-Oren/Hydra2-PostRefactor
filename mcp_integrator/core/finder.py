#!/usr/bin/env python3
"""
MCP Finder Module

This module provides functions for searching and retrieving MCPs from the Smithery Registry.
Designed to be imported by other modules like the MCP Stack Recommender.
"""

import os
import requests
import sys
import json
from dotenv import load_dotenv
import importlib.util

# Load environment variables from .env file
load_dotenv()

# Define a simple scoring function
def score_server(server):
    """
    Simple scoring function for MCP servers.
    
    Args:
        server: Server dictionary with metadata
        
    Returns:
        Floating point score between 0.0 and 1.0
    """
    score = 0.5  # Default score
    
    # Deployed servers get higher score
    if server.get("isDeployed", False):
        score += 0.3
        
    # Consider usage count if available
    usage_count = server.get("usageCount", 0)
    if usage_count > 0:
        score += min(0.2, usage_count / 100)  # Cap at 0.2
    
    return score

# Default API base URL
API_BASE_URL = "https://registry.smithery.ai/servers"

def get_headers(api_token: str) -> dict:
    """
    Get the headers for API requests with authentication.
    
    Args:
        api_token: The Smithery API token
        
    Returns:
        Dictionary of headers
    """
    return {"Authorization": f"Bearer {api_token}"}

def list_servers(api_token: str, query: str = "", page: int = 1, pageSize: int = 10) -> dict:
    """
    Calls the Smithery Registry API to retrieve a single page of MCP servers.
    
    Args:
        api_token: The Smithery API token
        query: Search query string
        page: Page number
        pageSize: Number of results per page
        
    Returns:
        Dictionary with servers and pagination info
    """
    headers = get_headers(api_token)
    params = {"q": query, "page": page, "pageSize": pageSize}
    response = requests.get(API_BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error retrieving servers:", response.status_code, response.text)
        return {"servers": [], "pagination": {"totalPages": 0}}

def get_all_servers(api_token: str, query: str = "", pageSize: int = 10) -> list:
    """
    Iterates through pages to retrieve all MCP servers matching the query.
    
    Args:
        api_token: The Smithery API token
        query: Search query string
        pageSize: Number of results per page
        
    Returns:
        List of server dictionaries
    """
    all_servers = []
    page = 1
    while True:
        data = list_servers(api_token, query, page, pageSize)
        servers = data.get("servers", [])
        if not servers:
            break
        all_servers.extend(servers)
        pagination = data.get("pagination", {})
        if page >= pagination.get("totalPages", 1):
            break
        page += 1
    return all_servers

def get_server_details(api_token: str, qualifiedName: str) -> dict:
    """
    Retrieves detailed information about a specific MCP server using its qualified name.
    
    Args:
        api_token: The Smithery API token
        qualifiedName: Qualified name of the server
        
    Returns:
        Dictionary with server details
    """
    headers = get_headers(api_token)
    url = f"{API_BASE_URL}/{qualifiedName}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error retrieving server details for", qualifiedName, response.text)
        return {}

def find_mcps(query: str, api_token_to_use: str = None) -> list:
    """
    Search for MCPs matching the query and format them for the stack recommender.
    
    Args:
        query: Search query string
        api_token_to_use: API token to use for authentication (optional)
        
    Returns:
        List of MCP dictionaries with name, description, and score fields
    """
    token = api_token_to_use or os.environ.get("SMITHERY_API_TOKEN")
    if not token:
        print("Error: No API token available for MCP search")
        return []
        
    try:
        print(f"Searching for MCPs with query: {query}")
        servers = get_all_servers(token, query, 20)  # Get up to 20 matching servers
        
        if not servers:
            print(f"No MCPs found for query: {query}")
            return []
            
        # Score and format servers
        formatted_mcps = []
        for server in servers:
            # Score server if not already scored
            if "score" not in server:
                server["score"] = score_server(server)
                
            # Format for stack recommender
            formatted_mcps.append({
                "name": server.get("displayName") or server.get("qualifiedName", "Unknown MCP"),
                "description": server.get("description", "No description available"),
                "qualifiedName": server.get("qualifiedName"),
                "score": server.get("score", 0.5),
                "isDeployed": server.get("isDeployed", False),
                "homepage": server.get("homepage", ""),
                "api_schema": server.get("apiSpecification", {})
            })
        
        # Sort by score
        formatted_mcps.sort(key=lambda x: x.get("score", 0), reverse=True)
        return formatted_mcps
        
    except Exception as e:
        print(f"Error in find_mcps: {str(e)}")
        return []

# This allows the module to be run standalone for testing
if __name__ == "__main__":
    api_token = os.environ.get("SMITHERY_API_TOKEN")
    if not api_token:
        print("Error: SMITHERY_API_TOKEN environment variable not set")
        sys.exit(1)
        
    test_query = "text processing"
    print(f"\nTesting find_mcps with query: {test_query}")
    results = find_mcps(test_query, api_token)
    
    print(f"\nFound {len(results)} results:")
    for i, mcp in enumerate(results[:5]):  # Show top 5
        print(f"{i+1}. {mcp['name']} (Score: {mcp['score']:.2f})")
        print(f"   {mcp['description'][:100]}...") 