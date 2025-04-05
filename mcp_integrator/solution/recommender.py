#!/usr/bin/env python3
"""
MCP Stack Recommender Module

This module integrates all components of the MCP Stack Recommendation system,
providing a unified interface to analyze complex tasks and recommend optimal
MCP stacks.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
import sys

# Import component modules
from mcp_integrator.task.analyzer import TaskDecomposer
from mcp_integrator.query.generator import QueryGenerator
from mcp_integrator.solution.explainer import SolutionExplainer
from mcp_integrator.query.clarifier import QueryClarifier
from mcp_integrator.query.cache import QueryCache
from mcp_integrator.utils.context import GlobalContext
from mcp_integrator.task.decomposer import SmartDecomposer

class MCPStackRecommender:
    """
    Main class that coordinates the MCP stack recommendation process.
    
    This class integrates all components of the system and provides a unified
    interface for analyzing complex tasks and recommending optimal MCP stacks.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                mcp_finder_func: Optional[Callable] = None,
                max_mcps_per_stack: int = 5,
                min_mcps_required: int = 1,
                use_domain_context: bool = False,
                max_mcps_per_query: int = 10,
                clarification_threshold: int = 50,
                enable_query_clarification: bool = True,
                enable_caching: bool = True,
                enable_cross_learning: bool = True,
                enable_smart_decomposition: bool = True,
                similarity_threshold: float = 0.75,
                min_subtasks: int = 3,
                max_subtasks: int = 8):
        """
        Initialize the MCP Stack Recommender.
        
        Args:
            api_key: OpenAI API key for GPT-4o (optional, can use env var).
            mcp_finder_func: Function to query MCPs with a given query string.
            max_mcps_per_stack: Maximum number of MCPs to include in a stack.
            min_mcps_required: Minimum number of subtasks that need MCPs for a valid stack.
            use_domain_context: Whether to use domain-specific query generation (not recommended).
            max_mcps_per_query: Maximum number of MCPs to process per query (limits results when too many are found).
            clarification_threshold: Number of MCPs found that triggers clarification.
            enable_query_clarification: Whether to enable interactive query clarification.
            enable_caching: Whether to enable caching of MCP finder results.
            enable_cross_learning: Whether to enable cross-subtask learning.
            enable_smart_decomposition: Whether to enable smart task decomposition.
            similarity_threshold: Threshold for similarity detection.
            min_subtasks: Minimum number of subtasks to aim for.
            max_subtasks: Maximum number of subtasks to allow.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Store the MCP finder function
        self.mcp_finder = mcp_finder_func
        
        # Configuration parameters
        self.max_mcps_per_stack = max_mcps_per_stack
        self.min_mcps_required = min_mcps_required
        self.use_domain_context = use_domain_context
        self.max_mcps_per_query = max_mcps_per_query
        self.clarification_threshold = clarification_threshold
        self.enable_query_clarification = enable_query_clarification
        
        # Performance optimization settings
        self.enable_caching = enable_caching
        self.enable_cross_learning = enable_cross_learning
        self.enable_smart_decomposition = enable_smart_decomposition
        self.similarity_threshold = similarity_threshold
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        
        # Initialize components
        self.task_decomposer = TaskDecomposer(api_key=self.api_key)
        self.query_generator = QueryGenerator(
            api_key=self.api_key,
            use_domain_context=use_domain_context
        )
        self.solution_explainer = SolutionExplainer(api_key=self.api_key)
        
        # Initialize the query clarifier if enabled
        if enable_query_clarification:
            self.query_clarifier = QueryClarifier(
                clarification_threshold=clarification_threshold,
                api_key=self.api_key
            )
        else:
            self.query_clarifier = None
        
        # Initialize optimization components
        if enable_caching:
            self.query_cache = QueryCache(
                similarity_threshold=similarity_threshold,
                cache_ttl=3600,  # 1 hour cache time
                enable_similarity_check=True,
                api_key=self.api_key
            )
        else:
            self.query_cache = None
        
        if enable_cross_learning:
            self.global_context = GlobalContext(
                relevance_threshold=similarity_threshold,
                enable_cross_learning=True,
                api_key=self.api_key
            )
        else:
            self.global_context = None
        
        if enable_smart_decomposition:
            self.smart_decomposer = SmartDecomposer(
                similarity_threshold=similarity_threshold,
                min_subtasks=min_subtasks,
                max_subtasks=max_subtasks,
                enable_merging=True,
                api_key=self.api_key
            )
        else:
            self.smart_decomposer = None
        
        if not self.mcp_finder:
            raise ValueError("MCP finder function is required. You must provide a function to search for MCPs.")
    
    def _search_mcps_for_subtask(self, subtask: Dict[str, Any], query: str, 
                              clarification_callback: Optional[Callable] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search for MCPs for a specific subtask, with clarification if needed.
        
        Args:
            subtask: The subtask dictionary
            query: The query to search for
            clarification_callback: Function to get user clarification if needed
            
        Returns:
            Tuple of (list of MCPs, search statistics)
        """
        if not self.mcp_finder:
            raise ValueError("MCP finder function is required but not provided")
        
        subtask_name = subtask.get('name', 'unknown')
        search_stats = {
            "original_query": query,
            "clarification_requested": False,
            "mcp_count": 0,
            "cache_hit": False
        }
        
        # Check cache first if enabled
        if self.query_cache and self.enable_caching:
            cached_result = self.query_cache.get(query)
            if cached_result:
                search_stats["cache_hit"] = True
                search_stats["mcp_count"] = len(cached_result)
                return cached_result, search_stats
        
        # Apply cross-subtask learning if enabled (before initial search)
        if self.enable_cross_learning and self.global_context:
            # Define a refine function that applies clarifications
            def cross_task_refine(subtask, query, clarification):
                # Extract key terms from the clarification to enhance our query
                clarification_terms = " ".join(clarification.key_terms)
                enhanced_query = f"{query} {clarification_terms}"
                search_stats["cross_learning_applied"] = True
                search_stats["applied_from_subtask"] = clarification.subtask.get('name', 'unknown')
                return enhanced_query
            
            # Apply relevant clarifications from other subtasks
            original_query = query
            refined_query = self.global_context.apply_relevant_clarifications(
                subtask, query, cross_task_refine
            )
            
            # If query was refined by cross-learning, use it
            if refined_query != original_query:
                query = refined_query
                search_stats["refined_query"] = refined_query
        
        # Execute initial search with current query
        print(f"🔍 Searching for MCPs for subtask: {subtask_name}")
        print(f"   Query: {query}")
        
        mcps = self.mcp_finder(query)
        search_stats["initial_mcps_found"] = len(mcps)
        
        # Limit results if needed
        if len(mcps) > self.max_mcps_per_query:
            print(f"   Found {len(mcps)} MCPs, limiting to top {self.max_mcps_per_query}")
            mcps = mcps[:self.max_mcps_per_query]
        
        # If query clarification is enabled and we have too many results,
        # and a clarification callback is provided
        if (self.enable_query_clarification and 
            self.query_clarifier and
            clarification_callback and
            len(mcps) > self.clarification_threshold):
            
            print(f"   Found {len(mcps)} MCPs, clarifying query...")
            refined_query = self.query_clarifier.clarify_query(
                subtask, query, mcps, clarification_callback
            )
            
            if refined_query != query:
                search_stats["clarification_requested"] = True
                search_stats["clarified_query"] = refined_query
                
                # Execute search with refined query
                print(f"   Refined query: {refined_query}")
                refined_mcps = self.mcp_finder(refined_query)
                search_stats["clarified_mcps_found"] = len(refined_mcps)
                
                # Store the clarification for cross-learning if enabled
                if self.enable_cross_learning and self.global_context:
                    self.global_context.add_clarification(subtask, query, refined_query)
                
                # Limit results if still too many
                if len(refined_mcps) > self.max_mcps_per_query:
                    print(f"   Found {len(refined_mcps)} MCPs after clarification, limiting to top {self.max_mcps_per_query}")
                    refined_mcps = refined_mcps[:self.max_mcps_per_query]
                
                # Cache the result if enabled
                if self.query_cache and self.enable_caching:
                    self.query_cache.add(refined_query, refined_mcps)
                
                search_stats["mcp_count"] = len(refined_mcps)
                return refined_mcps, search_stats
        
        # Cache the result if enabled
        if self.query_cache and self.enable_caching:
            self.query_cache.add(query, mcps)
        
        search_stats["mcp_count"] = len(mcps)
        return mcps, search_stats
    
    def analyze_task(self, task_description: str, 
                    clarification_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze a high-level task and recommend optimal MCP stacks.
        
        Args:
            task_description: Description of the high-level task
            clarification_callback: Optional function to get user clarification if needed
            
        Returns:
            Dict with analysis results, including recommended MCP stacks
        """
        start_time = time.time()
        
        # Step 1: Decompose the task into subtasks
        print("\n🔍 ANALYZING TASK...")
        if self.enable_smart_decomposition and self.smart_decomposer:
            subtasks = self.smart_decomposer.decompose_task(task_description)
        else:
            subtasks = self.task_decomposer.decompose_task(task_description)
        
        decomposition_time = time.time() - start_time
        print(f"✅ Task decomposed into {len(subtasks)} subtasks [{decomposition_time:.1f}s]")
        
        # Step 2: Generate queries for each subtask
        subtask_queries = {}
        for subtask in subtasks:
            subtask_name = subtask.get('name', 'unknown')
            query = self.query_generator.generate_query(subtask)
            subtask_queries[subtask_name] = query
        
        query_generation_time = time.time() - start_time - decomposition_time
        print(f"✅ Generated queries for all subtasks [{query_generation_time:.1f}s]")
        
        # Step 3: Search for MCPs for each subtask
        subtask_mcps = {}
        mcp_search_stats = {}
        
        for subtask in subtasks:
            subtask_name = subtask.get('name', 'unknown')
            query = subtask_queries[subtask_name]
            
            mcps, stats = self._search_mcps_for_subtask(
                subtask, query, clarification_callback
            )
            
            subtask_mcps[subtask_name] = mcps
            mcp_search_stats[subtask_name] = stats
        
        mcp_search_time = time.time() - start_time - decomposition_time - query_generation_time
        print(f"✅ Found MCPs for all subtasks [{mcp_search_time:.1f}s]")
        
        # Step 4: Generate MCP stacks
        mcp_stacks = []
        
        if hasattr(self, 'stack_recommender'):
            # Use the imported stack recommender if available
            mcp_stacks = self.stack_recommender.generate_stacks(
                subtasks, subtask_mcps, 
                max_mcps_per_stack=self.max_mcps_per_stack,
                min_mcps_required=self.min_mcps_required
            )
        else:
            # Simplified stack generation if stack_recommender.py wasn't imported
            # Create a single stack with the highest-scored MCP for each subtask
            stack = {}
            for subtask_name, mcps in subtask_mcps.items():
                if mcps:  # If we found any MCPs for this subtask
                    stack[subtask_name] = [mcps[0]]  # Take the highest-scored MCP
            
            if len(stack) >= self.min_mcps_required:
                mcp_stacks.append({
                    "mcps": stack,
                    "score": sum(mcp[0]["score"] for mcp in stack.values()) / len(stack),
                    "coverage": len(stack) / len(subtasks)
                })
        
        stack_generation_time = time.time() - start_time - decomposition_time - query_generation_time - mcp_search_time
        print(f"✅ Generated {len(mcp_stacks)} MCP stacks [{stack_generation_time:.1f}s]")
        
        # Step 5: Generate explanation for the best stack
        explanation = None
        if mcp_stacks and self.solution_explainer:
            best_stack = mcp_stacks[0]
            explanation = self.solution_explainer.explain_solution(
                task_description, subtasks, best_stack
            )
        
        explanation_time = time.time() - start_time - decomposition_time - query_generation_time - mcp_search_time - stack_generation_time
        if explanation:
            print(f"✅ Generated solution explanation [{explanation_time:.1f}s]")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Cross-learning stats
        cross_learning_stats = None
        if self.enable_cross_learning and self.global_context:
            cross_learning_stats = self.global_context.get_statistics()
        
        # Compile results
        results = {
            "task_description": task_description,
            "subtasks": subtasks,
            "subtask_queries": subtask_queries,
            "subtask_mcps": subtask_mcps,
            "mcp_stacks": mcp_stacks,
            "explanation": explanation,
            "mcp_search_stats": mcp_search_stats,
            "cross_learning_stats": cross_learning_stats,
            "processing_time": {
                "decomposition": decomposition_time,
                "query_generation": query_generation_time,
                "mcp_search": mcp_search_time,
                "stack_generation": stack_generation_time,
                "explanation": explanation_time,
                "total": total_time
            }
        }
        
        return results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display the analysis results to the console.
        
        Args:
            results: The analysis results from analyze_task
        """
        task_description = results.get('task_description', '')
        subtasks = results.get('subtasks', [])
        subtask_mcps = results.get('subtask_mcps', {})
        mcp_stacks = results.get('mcp_stacks', [])
        explanation = results.get('explanation')
        
        print("\n" + "=" * 80)
        print(f"TASK ANALYSIS RESULTS: {task_description[:50]}{'...' if len(task_description) > 50 else ''}")
        print("=" * 80)
        
        # Display subtasks
        print(f"\n🔍 DECOMPOSED INTO {len(subtasks)} SUBTASKS:")
        for i, subtask in enumerate(subtasks, 1):
            subtask_name = subtask.get('name', f'Subtask {i}')
            subtask_desc = subtask.get('description', 'No description')
            mcp_count = len(subtask_mcps.get(subtask_name, []))
            print(f"  {i}. {subtask_name} - {subtask_desc[:80]}{'...' if len(subtask_desc) > 80 else ''}")
            print(f"     Found {mcp_count} relevant MCPs")
        
        # Display recommended stacks
        if mcp_stacks:
            best_stack = mcp_stacks[0]
            coverage = best_stack.get('coverage', 0) * 100
            stack_mcps = best_stack.get('mcps', {})
            
            print(f"\n🏆 RECOMMENDED MCP STACK (Covers {coverage:.0f}% of subtasks):")
            
            # Display MCPs in the stack for each subtask
            for subtask_name, mcps in stack_mcps.items():
                if mcps:  # Ensure we have MCPs for this subtask
                    mcp = mcps[0]  # In most cases, we'll have 1 MCP per subtask
                    print(f"\n  • For {subtask_name}:")
                    print(f"    {mcp['name']} (Score: {mcp['score']:.2f})")
                    print(f"    {mcp['description'][:120]}{'...' if len(mcp['description']) > 120 else ''}")
            
            # Display subtasks with no recommended MCPs
            uncovered_subtasks = [s.get('name') for s in subtasks if s.get('name') not in stack_mcps]
            if uncovered_subtasks:
                print("\n  ⚠️ Subtasks with no suitable MCPs found:")
                for subtask_name in uncovered_subtasks:
                    print(f"    • {subtask_name}")
        else:
            print("\n⚠️ NO COMPLETE MCP STACKS FOUND")
            print("  Try adjusting search parameters or providing more details about the task.")
        
        # Display explanation if available
        if explanation:
            print("\n📝 SOLUTION EXPLANATION:")
            
            # Summary/Overview
            if 'summary' in explanation:
                print(f"\n  OVERVIEW: {explanation['summary']}")
            elif 'overview' in explanation:
                print(f"\n  OVERVIEW: {explanation['overview']}")
            
            # Rationale/Tool mapping
            if 'tool_mapping' in explanation:
                print(f"\n  TOOL MAPPING: {explanation['tool_mapping']}")
            elif 'rationale' in explanation:
                print(f"\n  RATIONALE: {explanation['rationale']}")
                
            # Setup Guide
            if 'setup_guide' in explanation:
                print("\n  SETUP GUIDE:")
                print(f"  {explanation['setup_guide']}")
            
            # Alternatives
            if 'alternatives' in explanation and explanation['alternatives']:
                print("\n  ALTERNATIVES:")
                if isinstance(explanation['alternatives'], list):
                    for alt in explanation['alternatives']:
                        print(f"  • {alt}")
                else:
                    print(f"  {explanation['alternatives']}")
            
            # Integration
            if 'integration' in explanation:
                print("\n  INTEGRATION:")
                integration = explanation['integration']
                if 'steps' in integration and integration['steps']:
                    print("\n  STEPS:")
                    for i, step in enumerate(integration['steps'], 1):
                        print(f"  {i}. {step}")
                
                if 'example_code' in integration and integration['example_code']:
                    print("\n  EXAMPLE CODE:")
                    print(f"  {integration['example_code']}")
        
        # Cross-learning statistics
        if 'cross_learning_stats' in results and results['cross_learning_stats']:
            stats = results['cross_learning_stats']
            
            if stats.get('shared_application_count', 0) > 0:
                print("\n🔄 CROSS-SUBTASK LEARNING:")
                print(f"  • Applied user insights across {stats.get('shared_application_count', 0)} subtasks")
                
                if 'clarification_count' in stats:
                    print(f"  • Collected {stats.get('clarification_count', 0)} user clarifications")
                
                # If we have detailed information about which subtasks benefited
                if 'mcp_search_stats' in results:
                    benefited_subtasks = []
                    for subtask, substats in results['mcp_search_stats'].items():
                        if substats.get('cross_learning_applied', False):
                            benefited_subtasks.append(subtask)
                    
                    if benefited_subtasks:
                        print(f"  • Subtasks that benefited from shared insights: {', '.join(benefited_subtasks)}")
        
        # Processing time information
        if 'processing_time' in results:
            timing = results['processing_time']
            print(f"\n⏱️ TOTAL PROCESSING TIME: {timing.get('total', 0):.1f}s")
        
        print("\n" + "=" * 80) 