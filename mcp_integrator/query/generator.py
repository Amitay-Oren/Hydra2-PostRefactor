#!/usr/bin/env python3
"""
Query Generator Module

This module provides functionality to generate effective search queries
from task descriptions for finding relevant MCPs.
"""

import os
from typing import Dict, Any, Optional

class QueryGenerator:
    """
    Generates optimized search queries from task descriptions or subtasks.
    
    This class is responsible for turning task descriptions or subtasks into
    effective search queries that can be used to find relevant MCPs.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                use_domain_context: bool = False):
        """
        Initialize the QueryGenerator.
        
        Args:
            api_key: OpenAI API key (optional, can use env var).
            use_domain_context: Whether to include domain-specific context.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
        self.use_domain_context = use_domain_context
    
    def generate_query(self, subtask: Dict[str, Any]) -> str:
        """
        Generate an optimized search query for a subtask.
        
        Args:
            subtask: Subtask dictionary with 'name' and 'description'
            
        Returns:
            Optimized search query string
        """
        subtask_name = subtask.get('name', '')
        subtask_desc = subtask.get('description', '')
        requirements = subtask.get('requirements', [])
        
        # In a real implementation, this would use an LLM to generate an optimized query
        # For now, we'll use a simplified approach
        
        import openai
        
        system_content = """
        You are a search query optimization expert who specializes in creating effective queries for finding relevant tools.
        
        Your task is to create a search query that will find the most relevant MCP (Machine-Callable Package) tools for a given subtask.
        
        MCPs are API tools that can be called by AI assistants to perform specific tasks, like:
        - GitHub operations (creating PRs, commenting on issues, etc.)
        - Database operations (querying, updating records, etc.)
        - File operations (reading, writing, searching)
        - Web operations (searching, scraping, fetching)
        - Data analysis and visualization
        - Many other API-accessible functions
        
        For the subtask you're given, create a search query that:
        1. Focuses on the core functional need
        2. Uses precise technical terms
        3. Includes any specific technologies mentioned
        4. Avoids overly broad terms
        5. Is concise (10-15 words maximum)
        6. Includes key action verbs
        7. Prioritizes terms that would appear in MCP descriptions
        
        Output only the optimized search query string, nothing else.
        """
        
        if self.use_domain_context:
            # Add domain-specific context to improve the query
            system_content += """
            Additionally, consider the domain context of the subtask.
            If the subtask is clearly in a specific technical domain (web development, data science, etc.),
            include 1-2 domain-specific technical terms to improve results.
            """
        
        user_content = f"""
        Subtask name: {subtask_name}
        Subtask description: {subtask_desc}
        """
        
        if requirements:
            user_content += f"Requirements: {', '.join(requirements)}"
        
        if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
            )
            query = response.choices[0].message.content.strip()
            
        else:  # For older OpenAI Python versions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
            )
            query = response.choices[0].message.content.strip()
        
        print(f"Generated query for '{subtask_name}': {query}")
        return query
    
    def optimize_query(self, query: str, results_count: int = 0) -> str:
        """
        Optimize an existing query based on the number of results.
        
        Args:
            query: Original query string
            results_count: Number of results from the original query
            
        Returns:
            Optimized query string
        """
        # If the query is returning too many results, make it more specific
        if results_count > 50:
            import openai
            
            system_content = """
            You are a search query optimization expert who specializes in refining queries to be more specific.
            
            The current query is returning too many results. Make it more specific and focused by:
            1. Adding more precise technical terms
            2. Including additional constraints
            3. Using more specific action verbs
            4. Focusing on the most unique aspects of the task
            
            Output only the optimized search query string, nothing else.
            """
            
            user_content = f"""
            Original query: {query}
            Number of results: {results_count}
            
            Create a more specific version of this query to reduce the number of results.
            """
            
            if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                )
                optimized_query = response.choices[0].message.content.strip()
                
            else:  # For older OpenAI Python versions
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                )
                optimized_query = response.choices[0].message.content.strip()
            
            print(f"Optimized query: {optimized_query}")
            return optimized_query
            
        # If the query is returning too few results, make it more general
        elif results_count < 2:
            import openai
            
            system_content = """
            You are a search query optimization expert who specializes in refining queries to be more general.
            
            The current query is returning too few results. Make it more general by:
            1. Removing overly specific terms
            2. Using more common technical terminology
            3. Focusing on the core functional need rather than specific implementation
            4. Using broader category terms
            
            Output only the optimized search query string, nothing else.
            """
            
            user_content = f"""
            Original query: {query}
            Number of results: {results_count}
            
            Create a more general version of this query to increase the number of results.
            """
            
            if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                )
                optimized_query = response.choices[0].message.content.strip()
                
            else:  # For older OpenAI Python versions
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.2,
                )
                optimized_query = response.choices[0].message.content.strip()
            
            print(f"Optimized query: {optimized_query}")
            return optimized_query
            
        # If the query is returning a reasonable number of results, keep it as is
        return query 