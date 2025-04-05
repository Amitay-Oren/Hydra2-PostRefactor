#!/usr/bin/env python3
"""
Task Analyzer Module

This module provides functionality to decompose high-level tasks into subtasks
for more effective MCP recommendation.
"""

import os
from typing import List, Dict, Any, Optional

class TaskDecomposer:
    """
    Decomposes a high-level task description into multiple subtasks.
    
    This class uses an LLM to analyze a task description and break it down
    into smaller, more specific subtasks that can be mapped to MCPs.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the TaskDecomposer.
        
        Args:
            api_key: OpenAI API key (optional, can use env var).
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
        # Initialize any required components here
        # For now, we'll use a simple implementation
    
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a high-level task into subtasks.
        
        Args:
            task_description: Description of the high-level task
            
        Returns:
            List of subtask dictionaries, each with 'name' and 'description'
        """
        # In a real implementation, this would call an LLM to analyze the task
        # For this prototype, we'll use a simplified approach
        
        print(f"Decomposing task: {task_description}")
        
        # Analyze the task using OpenAI GPT
        # This is a placeholder for the actual implementation
        # In a real implementation, this would call the OpenAI API
        
        # For now, return a simple decomposition
        # The actual implementation would use the OpenAI API to decompose the task
        
        # Example implementation placeholder:
        import openai
        
        if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """
                    You are a task decomposition expert. Your job is to break down high-level tasks 
                    into specific, actionable subtasks that can be mapped to technical solutions.
                    
                    For each task, provide:
                    1. A clear, concise name for the subtask
                    2. A detailed description explaining what needs to be done
                    3. A difficulty rating (1-5)
                    4. Any technical requirements or constraints
                    
                    Format your response as a valid JSON array of objects, each with the following fields:
                    - name: String with a concise subtask name
                    - description: String with detailed explanation
                    - difficulty: Integer from 1-5
                    - requirements: Array of strings listing technical requirements
                    
                    Ensure your decomposition:
                    - Covers all aspects of the original task
                    - Breaks the task into 3-8 logically separated subtasks
                    - Makes each subtask specific enough to map to technical solutions
                    - Preserves any constraints mentioned in the original task
                    """},
                    {"role": "user", "content": f"Decompose the following task into subtasks: {task_description}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            # Parse the response as JSON
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Return subtasks from the result
            if "subtasks" in result:
                return result["subtasks"]
            return result  # Assume the whole response is the subtasks array
            
        else:  # For older OpenAI Python versions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """
                    You are a task decomposition expert. Your job is to break down high-level tasks 
                    into specific, actionable subtasks that can be mapped to technical solutions.
                    
                    For each task, provide:
                    1. A clear, concise name for the subtask
                    2. A detailed description explaining what needs to be done
                    3. A difficulty rating (1-5)
                    4. Any technical requirements or constraints
                    
                    Format your response as a valid JSON array of objects, each with the following fields:
                    - name: String with a concise subtask name
                    - description: String with detailed explanation
                    - difficulty: Integer from 1-5
                    - requirements: Array of strings listing technical requirements
                    
                    Ensure your decomposition:
                    - Covers all aspects of the original task
                    - Breaks the task into 3-8 logically separated subtasks
                    - Makes each subtask specific enough to map to technical solutions
                    - Preserves any constraints mentioned in the original task
                    """},
                    {"role": "user", "content": f"Decompose the following task into subtasks: {task_description}"}
                ],
                temperature=0.2,
            )
            
            # Parse the response as JSON
            import json
            result = json.loads(response.choices[0].message.content)
            
            # Return subtasks from the result
            if "subtasks" in result:
                return result["subtasks"]
            return result  # Assume the whole response is the subtasks array
    
    def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a task without fully decomposing it.
        
        Args:
            task_description: Description of the high-level task
            
        Returns:
            Dictionary with complexity metrics
        """
        # Placeholder implementation
        return {
            "complexity_score": len(task_description) / 100,
            "estimated_subtasks": max(3, min(8, len(task_description) // 200)),
            "requires_expert_knowledge": len(task_description) > 500
        } 