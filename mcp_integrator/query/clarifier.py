#!/usr/bin/env python3
"""
Query Clarifier Module

This module provides functionality to interactively refine search queries
when they return too many or irrelevant results.
"""

import os
from typing import List, Dict, Any, Optional, Callable

class QueryClarifier:
    """
    Interactive query refinement for ambiguous or broad queries.
    
    This class provides functionality to analyze search results and
    generate clarifying questions to help refine the search query.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                clarification_threshold: int = 50):
        """
        Initialize the QueryClarifier.
        
        Args:
            api_key: OpenAI API key (optional, can use env var).
            clarification_threshold: Results count that triggers clarification.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.clarification_threshold = clarification_threshold
    
    def _analyze_results(self, subtask: Dict[str, Any], query: str, 
                        results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze search results to identify different categories/clusters.
        
        Args:
            subtask: The subtask dictionary
            query: The original query
            results: List of MCP result dictionaries
            
        Returns:
            Analysis of result clusters and possible clarification dimensions
        """
        # In a real implementation, this would analyze the result set using an LLM
        # For now, we'll use a simplified approach
        
        import openai
        
        # Extract just the names and descriptions for analysis
        # (to avoid overwhelming the context window)
        simplified_results = []
        for i, result in enumerate(results[:20]):  # Limit to first 20
            simplified_results.append({
                "name": result.get('name', f"MCP {i}"),
                "description": result.get('description', '')[:200]  # Truncate long descriptions
            })
        
        system_content = """
        You are a search results analysis expert. Your goal is to help refine an ambiguous query.
        
        Given a task description, search query, and a list of results, analyze the results to identify:
        1. The main categories/clusters of results
        2. The key dimensions that differentiate these clusters
        3. Suggestions for how to narrow down the search
        
        Format your response as a JSON object with the following fields:
        - "clusters": Array of identified result clusters (3-5 clusters), each with a "name" and "description"
        - "dimensions": Array of dimensions for clarification (2-4 dimensions), each with a "name" and "possible_values" array
        - "clarification_question": A single question to ask the user to clarify their needs
        - "default_recommendation": Your best guess at which cluster/dimension is most relevant
        
        Keep your analysis focused and concise.
        """
        
        user_content = f"""
        Subtask: {subtask.get('name', 'Unknown')} - {subtask.get('description', 'No description')}
        Original query: {query}
        Number of results: {len(results)}
        
        Top results:
        {simplified_results}
        
        Analyze these results and identify the main clusters and dimensions for clarification.
        """
        
        if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            import json
            analysis = json.loads(response.choices[0].message.content)
            
        else:  # For older OpenAI Python versions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
            )
            
            import json
            analysis = json.loads(response.choices[0].message.content)
        
        return analysis
    
    def _refine_query(self, subtask: Dict[str, Any], original_query: str, 
                     clarification: str) -> str:
        """
        Generate a refined query based on user clarification.
        
        Args:
            subtask: The subtask dictionary
            original_query: The original query string
            clarification: User's clarification response
            
        Returns:
            Refined query string
        """
        import openai
        
        system_content = """
        You are a search query refinement expert. Your task is to refine a search query based on
        the original query and the user's clarification.
        
        Generate an improved search query that:
        1. Maintains the core intent of the original query
        2. Incorporates the specific details from the user's clarification
        3. Is more precise and targeted
        4. Remains concise and focused (10-15 words maximum)
        
        Output only the refined query string, nothing else.
        """
        
        user_content = f"""
        Subtask: {subtask.get('name', 'Unknown')} - {subtask.get('description', 'No description')}
        Original query: {original_query}
        
        User clarification: {clarification}
        
        Generate a refined search query based on this clarification.
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
            refined_query = response.choices[0].message.content.strip()
            
        else:  # For older OpenAI Python versions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
            )
            refined_query = response.choices[0].message.content.strip()
        
        return refined_query
    
    def clarify_query(self, subtask: Dict[str, Any], original_query: str,
                     results: List[Dict[str, Any]], 
                     clarification_callback: Callable[[str, Dict[str, Any]], str]) -> str:
        """
        Interactively clarify a query by analyzing results and asking the user for clarification.
        
        Args:
            subtask: The subtask dictionary
            original_query: The original query string
            results: List of MCP result dictionaries
            clarification_callback: Function to get user clarification
            
        Returns:
            Refined query string
        """
        # Only clarify if we have too many results
        if len(results) <= self.clarification_threshold:
            return original_query
        
        # Analyze results to identify clusters and dimensions
        analysis = self._analyze_results(subtask, original_query, results)
        
        # Get clarification from the user using the provided callback
        question = analysis.get('clarification_question', 
                             "Can you clarify your specific requirements?")
                             
        # Create a user-friendly message with options
        display_message = f"{question}\n\n"
        
        # Add clusters information
        if 'clusters' in analysis:
            display_message += "Result categories found:\n"
            for i, cluster in enumerate(analysis['clusters'], 1):
                display_message += f"{i}. {cluster.get('name', f'Category {i}')}:"
                display_message += f" {cluster.get('description', 'No description')[:100]}\n"
        
        # Add dimensions information
        if 'dimensions' in analysis:
            display_message += "\nPossible clarification dimensions:\n"
            for i, dimension in enumerate(analysis['dimensions'], 1):
                display_message += f"{i}. {dimension.get('name', f'Dimension {i}')}: "
                values = dimension.get('possible_values', [])
                display_message += f"{', '.join(values)}\n"
        
        # Add default recommendation
        if 'default_recommendation' in analysis:
            display_message += f"\nRecommendation: {analysis['default_recommendation']}"
        
        # Get user clarification
        user_clarification = clarification_callback(display_message, analysis)
        
        # If user provided clarification, refine the query
        if user_clarification:
            refined_query = self._refine_query(
                subtask, original_query, user_clarification
            )
            print(f"Refined query: {refined_query}")
            return refined_query
        
        # If no clarification, return the original query
        return original_query 