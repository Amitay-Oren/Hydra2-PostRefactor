#!/usr/bin/env python3
"""
Query Cache Module

This module provides caching functionality for MCP search queries
to improve performance and reduce redundant API calls.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class QueryCache:
    """
    Cache for MCP search queries and results.
    
    This class provides functionality to store and retrieve search results
    for queries, including exact and similar matches based on semantic similarity.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None,
                cache_ttl: int = 3600,
                cache_dir: Optional[str] = None,
                similarity_threshold: float = 0.75,
                enable_similarity_check: bool = True):
        """
        Initialize the QueryCache.
        
        Args:
            api_key: OpenAI API key for similarity checks (optional, can use env var).
            cache_ttl: Time-to-live for cache entries in seconds (default: 1 hour).
            cache_dir: Directory to store cache files (default: ~/.mcp_integrator/cache).
            similarity_threshold: Threshold for considering queries similar (0.0-1.0).
            enable_similarity_check: Whether to enable similar query matching.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold
        self.enable_similarity_check = enable_similarity_check
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".mcp_integrator" / "cache"
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize in-memory cache
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """
        Load cache from disk.
        
        Returns:
            Dictionary of cached queries and results
        """
        cache = {}
        
        # Load cache from cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cache_entry = json.load(f)
                    
                    # Extract query and check cache expiry
                    query = cache_entry.get("query", "")
                    timestamp = cache_entry.get("timestamp", 0)
                    
                    if time.time() - timestamp <= self.cache_ttl:
                        # Cache is still valid
                        cache[query] = cache_entry
                    else:
                        # Cache has expired, delete the file
                        cache_file.unlink()
            except Exception as e:
                print(f"Error loading cache file {cache_file}: {e}")
        
        return cache
    
    def _save_to_cache(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Save query results to cache.
        
        Args:
            query: The search query
            results: List of MCP results
        """
        cache_entry = {
            "query": query,
            "results": results,
            "timestamp": time.time(),
            "embedding": None  # Will be populated if similarity check is enabled
        }
        
        # Generate embedding for similarity checks if enabled
        if self.enable_similarity_check and self.api_key:
            cache_entry["embedding"] = self._generate_embedding(query)
        
        # Save to in-memory cache
        self.cache[query] = cache_entry
        
        # Save to disk
        cache_file = self.cache_dir / f"{hash(query)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(cache_entry, f)
        except Exception as e:
            print(f"Error saving to cache file {cache_file}: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            Embedding vector or None if error
        """
        try:
            import openai
            
            if hasattr(openai, 'OpenAI'):  # OpenAI Python v1.0.0+
                client = openai.OpenAI(api_key=self.api_key)
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            else:  # For older OpenAI Python versions
                response = openai.Embedding.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
            
        import numpy as np
        
        # Calculate cosine similarity
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _find_similar_query(self, query: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Find a similar query in the cache.
        
        Args:
            query: The search query
            
        Returns:
            Tuple of (similar query string, results) or (None, []) if no match
        """
        if not self.enable_similarity_check or not self.api_key:
            return None, []
            
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return None, []
            
        # Find the most similar cached query
        most_similar_query = None
        highest_similarity = 0.0
        
        for cached_query, cache_entry in self.cache.items():
            cached_embedding = cache_entry.get("embedding")
            if not cached_embedding:
                continue
                
            similarity = self._calculate_similarity(query_embedding, cached_embedding)
            
            if similarity > highest_similarity and similarity >= self.similarity_threshold:
                most_similar_query = cached_query
                highest_similarity = similarity
        
        if most_similar_query:
            return most_similar_query, self.cache[most_similar_query].get("results", [])
            
        return None, []
    
    def add(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Add a query and its results to the cache.
        
        Args:
            query: The search query
            results: List of MCP results
        """
        self._save_to_cache(query, results)
    
    def get(self, query: str) -> List[Dict[str, Any]]:
        """
        Get results for a query from the cache.
        
        Args:
            query: The search query
            
        Returns:
            List of MCP results or empty list if not in cache
        """
        # Check for exact match
        if query in self.cache:
            cache_entry = self.cache[query]
            
            # Check if cache has expired
            if time.time() - cache_entry.get("timestamp", 0) <= self.cache_ttl:
                return cache_entry.get("results", [])
            
            # Remove expired entry
            del self.cache[query]
        
        # Check for similar queries
        similar_query, results = self._find_similar_query(query)
        if similar_query:
            print(f"Using cached results for similar query: {similar_query}")
            return results
            
        return []
    
    def clear(self) -> None:
        """
        Clear the cache.
        """
        self.cache = {}
        
        # Delete all cache files
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Error deleting cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self.cache),
            "cache_dir": str(self.cache_dir),
            "cache_ttl": self.cache_ttl,
            "similarity_threshold": self.similarity_threshold,
            "similarity_enabled": self.enable_similarity_check
        } 