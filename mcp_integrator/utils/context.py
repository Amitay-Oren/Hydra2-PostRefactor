#!/usr/bin/env python3
"""
Global Context Module for MCP Stack Recommender

Enables cross-subtask learning by storing user clarifications and applying them to related subtasks.
Analyzes semantic similarity between subtasks to share relevant clarifications.
"""

import os
import re
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Set
from collections import defaultdict

class Clarification:
    """Represents a user clarification for a specific query."""
    
    def __init__(self, 
                 subtask: Dict[str, Any],
                 original_query: str,
                 clarification_text: str,
                 refined_query: str):
        """
        Initialize a clarification.
        
        Args:
            subtask: Dictionary with subtask information (name, description, capabilities)
            original_query: Original query that needed clarification
            clarification_text: User's clarification text
            refined_query: Query after applying the clarification
        """
        self.subtask = subtask
        self.original_query = original_query
        self.clarification_text = clarification_text
        self.refined_query = refined_query
        self.timestamp = time.time()
        self.key_terms = self._extract_key_terms()
        self.shared_count = 0  # Count how many times this clarification was shared
    
    def _extract_key_terms(self) -> Set[str]:
        """
        Extract key terms from the clarification text.
        
        Returns:
            Set of key terms extracted from the clarification
        """
        # Simple extraction - split by space and filter stopwords
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'need', 'want', 'looking', 'for', 'to', 'from', 'with', 'would', 'like',
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'can', 'could', 'should'
        }
        
        # Get all words, convert to lowercase, remove punctuation
        words = re.findall(r'\b\w+\b', self.clarification_text.lower())
        key_terms = {word for word in words if word not in stopwords and len(word) > 2}
        
        # Also extract terms from the refined query that weren't in the original
        original_words = set(re.findall(r'\b\w+\b', self.original_query.lower()))
        refined_words = set(re.findall(r'\b\w+\b', self.refined_query.lower()))
        
        # Add new terms that appeared in the refined query
        new_terms = {word for word in refined_words if word not in original_words and word not in stopwords and len(word) > 2}
        key_terms.update(new_terms)
        
        return key_terms


class GlobalContext:
    """
    Maintains a global context of user clarifications across subtasks to enable cross-subtask learning.
    
    This class stores clarifications provided by users and applies them to semantically similar
    subtasks, reducing redundant questioning and improving query quality across related tasks.
    """
    
    def __init__(self, 
                 relevance_threshold: float = 0.7,
                 enable_cross_learning: bool = True,
                 api_key: Optional[str] = None):
        """
        Initialize the global context.
        
        Args:
            relevance_threshold: Threshold for considering clarifications relevant (0.0-1.0)
            enable_cross_learning: Whether to enable cross-subtask learning
            api_key: OpenAI API key for embeddings (optional, can use env var)
        """
        self.clarifications: List[Clarification] = []
        self.subtask_embeddings: Dict[str, List[float]] = {}
        self.relevance_threshold = relevance_threshold
        self.enable_cross_learning = enable_cross_learning
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Statistics
        self.clarification_count = 0
        self.shared_application_count = 0
        
        # Initialize OpenAI client if cross-learning is enabled
        self.embeddings_available = False
        if enable_cross_learning:
            try:
                import openai
                if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
                    self.openai_client = openai.OpenAI(api_key=self.api_key)
                    self.embeddings_available = True
                else:  # For older OpenAI Python versions
                    self.openai = openai
                    self.embeddings_available = True
            except ImportError:
                print("Warning: OpenAI library not found. Using text similarity for cross-learning.")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}. Using text similarity for cross-learning.")
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a text string using OpenAI API.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            Embedding vector or empty list if failed
        """
        if not self.embeddings_available:
            return []
            
        try:
            if hasattr(self, 'openai_client'):  # OpenAI Python v1.0.0+
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            else:  # For older OpenAI Python versions
                response = self.openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Warning: Failed to get embedding: {e}")
            return []
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        # Cosine similarity calculation
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        return dot_product / (magnitude1 * magnitude2)
    
    def _calculate_term_similarity(self, subtask1: Dict[str, Any], subtask2: Dict[str, Any]) -> float:
        """
        Calculate similarity between subtasks based on text similarity.
        
        Args:
            subtask1: First subtask dictionary
            subtask2: Second subtask dictionary
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Combine name and description
        text1 = f"{subtask1.get('name', '')} {subtask1.get('description', '')}"
        text2 = f"{subtask2.get('name', '')} {subtask2.get('description', '')}"
        
        # Get all words, convert to lowercase, remove punctuation
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _get_subtask_key(self, subtask: Dict[str, Any]) -> str:
        """
        Get a unique key for a subtask.
        
        Args:
            subtask: Subtask dictionary
            
        Returns:
            Unique key string for the subtask
        """
        return subtask.get('name', 'unknown')
    
    def _get_subtask_embedding(self, subtask: Dict[str, Any]) -> List[float]:
        """
        Get or create embedding for a subtask.
        
        Args:
            subtask: Subtask dictionary
            
        Returns:
            Embedding vector for the subtask
        """
        subtask_key = self._get_subtask_key(subtask)
        
        # Return cached embedding if available
        if subtask_key in self.subtask_embeddings:
            return self.subtask_embeddings[subtask_key]
        
        # Create embedding if embeddings are available
        if self.embeddings_available:
            text = f"{subtask.get('name', '')} {subtask.get('description', '')}"
            embedding = self._get_embedding(text)
            self.subtask_embeddings[subtask_key] = embedding
            return embedding
        
        return []
    
    def calculate_subtask_similarity(self, subtask1: Dict[str, Any], subtask2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two subtasks.
        
        Args:
            subtask1: First subtask dictionary
            subtask2: Second subtask dictionary
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Don't compare to itself
        if self._get_subtask_key(subtask1) == self._get_subtask_key(subtask2):
            return 1.0
        
        # Try embedding similarity first
        if self.embeddings_available:
            embedding1 = self._get_subtask_embedding(subtask1)
            embedding2 = self._get_subtask_embedding(subtask2)
            
            similarity = self._calculate_similarity(embedding1, embedding2)
            if similarity > 0:
                return similarity
        
        # Fallback to term similarity
        return self._calculate_term_similarity(subtask1, subtask2)
    
    def add_clarification(self, 
                         subtask: Dict[str, Any],
                         query: str,
                         clarification_text: str,
                         refined_query: str = None) -> None:
        """
        Add a user clarification to the global context.
        
        Args:
            subtask: Dictionary with subtask information
            query: Original query that needed clarification
            clarification_text: User's clarification text
            refined_query: Query after applying the clarification. If None, uses original query.
        """
        if refined_query is None:
            refined_query = query
            
        clarification = Clarification(
            subtask=subtask,
            original_query=query,
            clarification_text=clarification_text,
            refined_query=refined_query
        )
        
        self.clarifications.append(clarification)
        self.clarification_count += 1
    
    def get_relevant_clarifications(self, subtask: Dict[str, Any]) -> List[Clarification]:
        """
        Get clarifications relevant to a subtask.
        
        Args:
            subtask: Dictionary with subtask information
            
        Returns:
            List of relevant Clarification objects
        """
        if not self.enable_cross_learning or not self.clarifications:
            return []
        
        relevant_clarifications = []
        subtask_key = self._get_subtask_key(subtask)
        
        for clarification in self.clarifications:
            # Skip clarifications for the same subtask
            if self._get_subtask_key(clarification.subtask) == subtask_key:
                continue
            
            # Calculate similarity
            similarity = self.calculate_subtask_similarity(subtask, clarification.subtask)
            
            # Add if above threshold
            if similarity >= self.relevance_threshold:
                relevant_clarifications.append(clarification)
        
        return relevant_clarifications
    
    def apply_relevant_clarifications(self, 
                                     subtask: Dict[str, Any],
                                     query: str,
                                     refine_func: Callable[[Dict[str, Any], str, Clarification], str]) -> str:
        """
        Apply relevant clarifications to a query.
        
        Args:
            subtask: Dictionary with subtask information
            query: Query to refine
            refine_func: Function that takes (subtask, query, clarification) and returns refined query
            
        Returns:
            Refined query with relevant clarifications applied
        """
        if not self.enable_cross_learning:
            return query
        
        relevant_clarifications = self.get_relevant_clarifications(subtask)
        if not relevant_clarifications:
            return query
        
        refined_query = query
        for clarification in relevant_clarifications:
            refined_query = refine_func(subtask, refined_query, clarification)
            clarification.shared_count += 1
            self.shared_application_count += 1
        
        return refined_query
    
    def get_common_vocabulary(self, min_occurrences: int = 2) -> Set[str]:
        """
        Get common vocabulary terms from clarifications.
        
        Args:
            min_occurrences: Minimum number of occurrences for a term to be included
            
        Returns:
            Set of common vocabulary terms
        """
        if not self.clarifications:
            return set()
        
        # Count term occurrences
        term_counts = defaultdict(int)
        for clarification in self.clarifications:
            for term in clarification.key_terms:
                term_counts[term] += 1
        
        # Return terms that occur at least min_occurrences times
        return {term for term, count in term_counts.items() if count >= min_occurrences}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get context statistics.
        
        Returns:
            Dictionary with context statistics
        """
        return {
            'clarification_count': self.clarification_count,
            'shared_application_count': self.shared_application_count,
            'embedding_similarity_available': self.embeddings_available,
            'cross_learning_enabled': self.enable_cross_learning,
            'relevance_threshold': self.relevance_threshold
        }


# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    # Example usage
    context = GlobalContext()
    
    # Create example subtasks
    search_subtask = {
        "name": "document_search",
        "description": "Find relevant legal documents based on keywords"
    }
    
    summarize_subtask = {
        "name": "document_summarization",
        "description": "Create concise summaries of legal documents"
    }
    
    unrelated_subtask = {
        "name": "voice_recording",
        "description": "Record voice notes about the case"
    }
    
    # Add a clarification
    context.add_clarification(
        subtask=search_subtask,
        query="document search",
        clarification_text="I need to find legal cases related to intellectual property",
        refined_query="intellectual property legal cases search"
    )
    
    # Define a refinement function
    def refine_query(subtask, query, clarification):
        # Simple example that appends key terms from the clarification
        key_terms = " ".join(clarification.key_terms)
        return f"{query} relating to {key_terms}"
    
    # Apply to a related subtask
    refined_query = context.apply_relevant_clarifications(
        subtask=summarize_subtask,
        query="document summarization",
        refine_func=refine_query
    )
    
    print(f"Original query: document summarization")
    print(f"Refined query: {refined_query}")
    
    # Apply to an unrelated subtask (shouldn't change)
    unrelated_refined = context.apply_relevant_clarifications(
        subtask=unrelated_subtask,
        query="voice recording",
        refine_func=refine_query
    )
    
    print(f"Unrelated original: voice recording")
    print(f"Unrelated refined: {unrelated_refined}")
    
    # Print stats
    print(f"Stats: {context.get_statistics()}") 