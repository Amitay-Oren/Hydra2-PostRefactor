#!/usr/bin/env python3
"""
Smart Task Decomposer Module

This module provides advanced task decomposition capabilities with features like
subtask merging, similarity detection, and optimized subtask granularity.
"""

import os
from typing import List, Dict, Any, Optional
import json

class SmartDecomposer:
    """
    Advanced task decomposer with optimization features.
    
    This class extends basic task decomposition with intelligent features:
    - Controls subtask count within specified range
    - Merges similar subtasks to avoid redundancy
    - Splits overly broad subtasks into more specific ones
    - Optimizes subtask granularity for better MCP mapping
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                similarity_threshold: float = 0.75,
                min_subtasks: int = 3,
                max_subtasks: int = 8,
                enable_merging: bool = True):
        """
        Initialize the SmartDecomposer.
        
        Args:
            api_key: OpenAI API key (optional, can use env var).
            similarity_threshold: Threshold for considering subtasks similar.
            min_subtasks: Minimum number of subtasks to aim for.
            max_subtasks: Maximum number of subtasks to allow.
            enable_merging: Whether to enable similar subtask merging.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.similarity_threshold = similarity_threshold
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.enable_merging = enable_merging
    
    def _calculate_similarity(self, subtask1: Dict[str, Any], subtask2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two subtasks.
        
        Args:
            subtask1: First subtask dictionary
            subtask2: Second subtask dictionary
            
        Returns:
            Similarity score between 0 and 1
        """
        # In a real implementation, this would use embeddings or a semantic similarity model
        # For now, we'll use a simple keyword-based similarity
        
        # Extract name and description
        name1 = subtask1.get('name', '').lower()
        desc1 = subtask1.get('description', '').lower()
        
        name2 = subtask2.get('name', '').lower()
        desc2 = subtask2.get('description', '').lower()
        
        # Combine name and description
        text1 = f"{name1} {desc1}"
        text2 = f"{name2} {desc2}"
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        common_words = words1.intersection(words2)
        all_words = words1.union(words2)
        
        if not all_words:
            return 0.0
            
        return len(common_words) / len(all_words)
    
    def _merge_similar_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge subtasks that are too similar.
        
        Args:
            subtasks: List of subtask dictionaries
            
        Returns:
            List of merged subtask dictionaries
        """
        if not self.enable_merging or len(subtasks) <= self.min_subtasks:
            return subtasks
            
        # Find pairs of similar subtasks
        merged_subtasks = subtasks.copy()
        i = 0
        
        while i < len(merged_subtasks) - 1 and len(merged_subtasks) > self.min_subtasks:
            j = i + 1
            while j < len(merged_subtasks) and len(merged_subtasks) > self.min_subtasks:
                similarity = self._calculate_similarity(merged_subtasks[i], merged_subtasks[j])
                
                if similarity >= self.similarity_threshold:
                    # Merge the subtasks
                    merged_subtask = {
                        "name": f"{merged_subtasks[i]['name']} & {merged_subtasks[j]['name']}",
                        "description": f"Combined task: {merged_subtasks[i]['description']} Additionally: {merged_subtasks[j]['description']}",
                        "difficulty": max(
                            merged_subtasks[i].get('difficulty', 3), 
                            merged_subtasks[j].get('difficulty', 3)
                        ),
                        "requirements": list(set(
                            merged_subtasks[i].get('requirements', []) + 
                            merged_subtasks[j].get('requirements', [])
                        ))
                    }
                    
                    # Replace i with merged subtask and remove j
                    merged_subtasks[i] = merged_subtask
                    merged_subtasks.pop(j)
                    
                    # Don't increment j since we removed an element
                else:
                    j += 1
            
            i += 1
            
        return merged_subtasks
        
    def _refine_subtask_count(self, task_description: str, initial_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine the number of subtasks to be within the specified range.
        
        Args:
            task_description: Original task description
            initial_subtasks: Initial list of subtask dictionaries
            
        Returns:
            Refined list of subtask dictionaries
        """
        # If we have too few subtasks, split some
        if len(initial_subtasks) < self.min_subtasks:
            return self._split_subtasks(task_description, initial_subtasks)
            
        # If we have too many subtasks, merge some
        if len(initial_subtasks) > self.max_subtasks:
            return self._merge_similar_subtasks(initial_subtasks)
            
        return initial_subtasks
        
    def _split_subtasks(self, task_description: str, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split broader subtasks into more specific ones.
        
        Args:
            task_description: Original task description
            subtasks: List of subtask dictionaries
            
        Returns:
            List with some subtasks split into multiple
        """
        if len(subtasks) >= self.min_subtasks:
            return subtasks
            
        import openai
        
        # Find the broadest subtask based on description length
        subtasks_by_length = sorted(
            subtasks, 
            key=lambda s: len(s.get('description', '')), 
            reverse=True
        )
        
        # Split the broadest subtasks until we have enough
        split_subtasks = subtasks.copy()
        
        for broad_subtask in subtasks_by_length:
            if len(split_subtasks) >= self.min_subtasks:
                break
                
            # Use LLM to split this subtask
            if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": """
                        You are a task decomposition expert. Your job is to split a subtask into
                        2-3 more specific subtasks that together accomplish the same goal.
                        
                        Format your response as a valid JSON array of objects, each with:
                        - name: String with a concise subtask name
                        - description: String with detailed explanation
                        - difficulty: Integer from 1-5
                        - requirements: Array of strings listing technical requirements
                        
                        Make sure the split subtasks:
                        - Together cover all aspects of the original subtask
                        - Are clearly distinct from each other
                        - Are more specific and actionable
                        """},
                        {"role": "user", "content": f"""
                        Original high-level task: {task_description}
                        
                        Split this subtask into 2-3 more specific subtasks:
                        Name: {broad_subtask.get('name')}
                        Description: {broad_subtask.get('description')}
                        """}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                
                # Get the split subtasks
                if "subtasks" in result:
                    new_subtasks = result["subtasks"]
                else:
                    new_subtasks = result  # Assume the result is the subtasks array
                
            else:  # For older OpenAI Python versions
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """
                        You are a task decomposition expert. Your job is to split a subtask into
                        2-3 more specific subtasks that together accomplish the same goal.
                        
                        Format your response as a valid JSON array of objects, each with:
                        - name: String with a concise subtask name
                        - description: String with detailed explanation
                        - difficulty: Integer from 1-5
                        - requirements: Array of strings listing technical requirements
                        
                        Make sure the split subtasks:
                        - Together cover all aspects of the original subtask
                        - Are clearly distinct from each other
                        - Are more specific and actionable
                        """},
                        {"role": "user", "content": f"""
                        Original high-level task: {task_description}
                        
                        Split this subtask into 2-3 more specific subtasks:
                        Name: {broad_subtask.get('name')}
                        Description: {broad_subtask.get('description')}
                        """}
                    ],
                    temperature=0.2,
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                
                # Get the split subtasks
                if "subtasks" in result:
                    new_subtasks = result["subtasks"]
                else:
                    new_subtasks = result  # Assume the result is the subtasks array
            
            # Replace the broad subtask with the new split subtasks
            split_subtasks.remove(broad_subtask)
            split_subtasks.extend(new_subtasks)
            
        return split_subtasks
    
    def decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Decompose a high-level task into an optimized set of subtasks.
        
        Args:
            task_description: Description of the high-level task
            
        Returns:
            List of subtask dictionaries, each with 'name' and 'description'
        """
        print(f"Smart decomposing task: {task_description}")
        
        # Initial decomposition
        import openai
        
        if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""
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
                    - Breaks the task into {self.min_subtasks}-{self.max_subtasks} logically separated subtasks
                    - Makes each subtask specific enough to map to technical solutions
                    - Preserves any constraints mentioned in the original task
                    """},
                    {"role": "user", "content": f"Decompose the following task into subtasks: {task_description}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Get the initial subtasks
            if "subtasks" in result:
                initial_subtasks = result["subtasks"]
            else:
                initial_subtasks = result  # Assume the result is the subtasks array
                
        else:  # For older OpenAI Python versions
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"""
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
                    - Breaks the task into {self.min_subtasks}-{self.max_subtasks} logically separated subtasks
                    - Makes each subtask specific enough to map to technical solutions
                    - Preserves any constraints mentioned in the original task
                    """},
                    {"role": "user", "content": f"Decompose the following task into subtasks: {task_description}"}
                ],
                temperature=0.2,
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Get the initial subtasks
            if "subtasks" in result:
                initial_subtasks = result["subtasks"]
            else:
                initial_subtasks = result  # Assume the result is the subtasks array
        
        # Refine the subtasks
        optimized_subtasks = self._refine_subtask_count(task_description, initial_subtasks)
        
        # If enabled, merge similar subtasks
        if self.enable_merging:
            optimized_subtasks = self._merge_similar_subtasks(optimized_subtasks)
            
        print(f"Task decomposed into {len(optimized_subtasks)} subtasks")
        return optimized_subtasks 