#!/usr/bin/env python3
"""
MCP Query Models Module

This module defines Pydantic models for capturing user intentions and validating
input for MCP queries. It provides a structured approach for gathering necessary
information from users to refine queries to MCP registries.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
import os
import json
import requests
from pydantic import BaseModel, Field, validator
import sys


class IntentType(str, Enum):
    """Enumeration of possible user intent types for MCP queries."""
    GENERAL_DISCOVERY = "general_discovery"
    SPECIFIC_CAPABILITY = "specific_capability"
    INTEGRATION = "integration"
    EVALUATION = "evaluation"
    COMPARISON = "comparison"


class UserIntent(BaseModel):
    """
    Pydantic model capturing user intent for MCP queries.
    This helps in refining the search query for more relevant results.
    """
    intent_type: IntentType = Field(
        ...,
        description="The type of intention behind the MCP query."
    )
    description: str = Field(
        ...,
        description="A detailed description of what the user is looking for.",
        min_length=10
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Keywords related to the desired MCPs.",
        min_items=0
    )
    desired_capabilities: List[str] = Field(
        default_factory=list,
        description="Specific capabilities that the MCP should have.",
        min_items=0
    )
    
    @validator('description')
    def description_min_length(cls, v):
        if len(v) < 10:
            raise ValueError('Description must be at least 10 characters long')
        return v


class QueryParameters(BaseModel):
    """
    Pydantic model for MCP query parameters.
    This includes technical parameters for the query execution.
    """
    api_token: Optional[str] = Field(
        None,
        description="API token for the MCP registry. If not provided, will use environment variable."
    )
    page_size: int = Field(
        default=10,
        description="Number of results per page to retrieve.",
        ge=1,
        le=100
    )
    deployed_only: bool = Field(
        default=True,
        description="Whether to only include deployed MCPs in the search results."
    )
    min_score: Optional[float] = Field(
        None,
        description="Minimum score threshold for including MCPs in results.",
        ge=0.0,
        le=100.0
    )


class MCPQueryInput(BaseModel):
    """
    Comprehensive Pydantic model for gathering all necessary information
    for an MCP query. Includes both user intent and technical parameters.
    """
    user_intent: UserIntent = Field(
        ...,
        description="Information about the user's intentions for the MCP query."
    )
    parameters: QueryParameters = Field(
        default_factory=QueryParameters,
        description="Technical parameters for the query execution."
    )
    
    def generate_query_string(self) -> str:
        """
        Generates a formatted query string for the MCP registry based on user intent.
        
        Returns:
            str: The formatted query string.
        """
        query_parts = []
        
        # Add deployment filter if needed
        if self.parameters.deployed_only:
            query_parts.append("is:deployed")
        
        # Add capability filters
        for capability in self.user_intent.desired_capabilities:
            query_parts.append(f"capability:{capability}")
        
        # Add keywords from user intent
        for keyword in self.user_intent.keywords:
            query_parts.append(keyword)
        
        # Add description terms, filtered to most relevant
        description_terms = [
            term for term in self.user_intent.description.split()
            if len(term) > 3 and term.lower() not in ['the', 'and', 'for', 'with']
        ]
        query_parts.extend(description_terms[:5])  # Limit to top 5 terms
        
        return " ".join(query_parts)


class OpenAIService:
    """
    Service class to interact with OpenAI's GPT-4o API.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key: The OpenAI API key. If None, will try to load from environment variable.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide directly.")
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        Send a chat request to the OpenAI API.
        
        Args:
            messages: List of message objects with role and content.
            temperature: Controls randomness in the response (0.0 to 1.0).
            
        Returns:
            Dict with the API response.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        return response.json()
    
    def extract_structure(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extract structured information from a conversation using GPT-4o.
        
        Args:
            conversation_history: List of conversation messages.
            
        Returns:
            Structured information extracted from the conversation.
        """
        # Add instructions for extracting structured information
        messages = conversation_history.copy()
        messages.append({
            "role": "system",
            "content": """
            Based on the conversation, extract structured information for an MCP query.
            Return a JSON object with the following structure:
            {
                "intent_type": "general_discovery", "specific_capability", "integration", "evaluation", or "comparison",
                "description": "A description of what the user is looking for",
                "keywords": ["list", "of", "keywords"],
                "desired_capabilities": ["list", "of", "capabilities"],
                "deployed_only": true/false,
                "page_size": number
            }
            
            Make educated guesses for any missing information based on the conversation context.
            """
        })
        
        try:
            response = self.chat(messages, temperature=0.3)
            content = response['choices'][0]['message']['content']
            
            # Try to parse JSON from the response
            # First, find the JSON block if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting structure: {str(e)}")
            # Return a default structure if extraction fails
            return {
                "intent_type": "general_discovery",
                "description": "Discovering MCP servers",
                "keywords": [],
                "desired_capabilities": [],
                "deployed_only": True,
                "page_size": 10
            }
    
    def generate_refined_query(self, query_input: MCPQueryInput) -> str:
        """
        Uses GPT-4o to generate a refined and optimized query string.
        
        Args:
            query_input: The populated MCPQueryInput model.
            
        Returns:
            str: The refined query string.
        """
        messages = [
            {
                "role": "system",
                "content": """
                You are a search query optimization expert. Your task is to create the most effective
                search query for an MCP registry based on the user's intent and parameters.
                
                MCP registry queries support the following syntax:
                - is:deployed - Filter to only deployed MCPs
                - capability:X - Filter to MCPs with capability X
                - owner:X - Filter to MCPs owned by X
                - keywords - General keywords for semantic search
                """
            },
            {
                "role": "user",
                "content": f"""
                Please optimize this search query:
                
                User Intent: {query_input.user_intent.intent_type.value}
                Description: {query_input.user_intent.description}
                Keywords: {', '.join(query_input.user_intent.keywords) if query_input.user_intent.keywords else 'None'}
                Desired Capabilities: {', '.join(query_input.user_intent.desired_capabilities) if query_input.user_intent.desired_capabilities else 'None'}
                Deployed Only: {query_input.parameters.deployed_only}
                
                Return only the optimized query string with no additional explanation.
                """
            }
        ]
        
        try:
            response = self.chat(messages, temperature=0.3)
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating refined query: {str(e)}")
            # Fallback to basic query generation
            return query_input.generate_query_string()


class MCPAgent:
    """
    Agent class that manages the conversation with the user to gather information
    for MCP queries.
    """
    
    def __init__(self, openai_service=None):
        """
        Initialize the MCP Agent.
        
        Args:
            openai_service: Optional OpenAIService instance. If None, will create a new one.
        """
        self.conversation_history = []
        self.current_state = "initial"
        
        # Initialize OpenAI service
        if openai_service:
            self.openai_service = openai_service
        else:
            try:
                self.openai_service = OpenAIService()
            except ValueError:
                # If OpenAI API key is not available, set service to None
                self.openai_service = None
                print("Warning: No OpenAI API key available. Running in manual mode.")
        
        # System message to guide the conversation
        self.system_message = {
            "role": "system",
            "content": """
            You are an MCP Query Assistant. Your job is to help users find the right MCP servers by having
            a natural conversation to understand their needs.
            
            Ask relevant questions to understand:
            1. What the user is trying to accomplish
            2. Any specific capabilities they need
            3. Any preferences for deployment status
            
            Be conversational and helpful. After understanding their needs, let them know you'll generate
            a query for them. Ask one question at a time.
            """
        }
        
        # Add system message to conversation history
        self.conversation_history.append(self.system_message)
    
    def process_user_input(self, user_message: str) -> str:
        """
        Process user input and update the conversation.
        
        Args:
            user_message: The user's message.
            
        Returns:
            str: The agent's response.
        """
        # Check for exit command
        if user_message.lower() in ["exit", "quit", "bye"]:
            self.current_state = "finalizing"
            return "Generating final query based on our conversation..."
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # If we're using the OpenAI service, generate a response with it
        if self.openai_service:
            agent_response = self._generate_response()
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            
            # If we have enough information, move to finalizing
            if len(self.conversation_history) >= 7:  # At least 3 exchanges (6 messages) plus system
                # Check if the last assistant message is asking a question
                last_msg = agent_response.strip()
                if not (last_msg.endswith("?") or "?" in last_msg.split(".")[-1]):
                    self.current_state = "finalizing"
                    return agent_response + "\n\nGenerating final query based on our conversation..."
            
            return agent_response
        else:
            # Manual mode without OpenAI
            if self.current_state == "initial":
                self.current_state = "gathering_intent"
                return "What are you trying to accomplish with the MCP you're looking for?"
            elif self.current_state == "gathering_intent":
                self.current_state = "gathering_capabilities"
                return "Are there any specific capabilities you need from the MCP?"
            elif self.current_state == "gathering_capabilities":
                self.current_state = "gathering_deployment"
                return "Do you want to see only deployed MCPs, or all MCPs including those in development?"
            else:
                self.current_state = "finalizing"
                return "Got it. Generating a query based on our conversation..."
    
    def _generate_response(self) -> str:
        """
        Generate a response using the OpenAI API based on the conversation history.
        
        Returns:
            str: The generated response.
        """
        response = self.openai_service.chat(self.conversation_history)
        return response['choices'][0]['message']['content']
    
    def _finalize_query(self) -> str:
        """
        Finalize and generate the optimized query string.
        
        Returns:
            str: The optimized query string.
        """
        # Extract structured information from conversation
        if self.openai_service:
            try:
                extracted_data = self.openai_service.extract_structure(self.conversation_history)
                
                # Create UserIntent object
                intent = UserIntent(
                    intent_type=extracted_data.get("intent_type", "general_discovery"),
                    description=extracted_data.get("description", "Discovering MCP servers"),
                    keywords=extracted_data.get("keywords", []),
                    desired_capabilities=extracted_data.get("desired_capabilities", [])
                )
                
                # Create QueryParameters object
                params = QueryParameters(
                    deployed_only=extracted_data.get("deployed_only", True),
                    page_size=extracted_data.get("page_size", 10)
                )
                
                # Create MCPQueryInput object
                query_input = MCPQueryInput(user_intent=intent, parameters=params)
                
                # Generate refined query using GPT-4o
                return self.refine_query_with_llm(query_input)
            except Exception as e:
                print(f"Error finalizing query: {str(e)}")
                # Return a basic query as fallback
                return "is:deployed"
        else:
            # Manual mode - simple query generation
            # Extract keywords from the conversation
            all_user_content = " ".join([msg["content"] for msg in self.conversation_history if msg["role"] == "user"])
            important_words = [word for word in all_user_content.split() if len(word) > 3 and word.lower() not in ["the", "and", "for", "with"]]
            
            # Basic query with deployed filter and up to 5 keywords
            query_parts = ["is:deployed"] + important_words[:5]
            return " ".join(query_parts)
    
    def gather_user_input(self) -> str:
        """
        Interactive function to gather user input through a conversation.
        
        Returns:
            str: The finalized query string.
        """
        print("Hello! I'm here to help you find the right MCP servers.")
        print("Tell me what you're looking for, and I'll generate an optimized search query.")
        print("(Type 'exit' at any time to finish the conversation and generate the query.)")
        
        # Initial prompt
        if self.openai_service:
            # Generate first prompt using OpenAI
            response = self._generate_response()
            self.conversation_history.append({"role": "assistant", "content": response})
            print(f"\nAssistant: {response}")
        else:
            # Manual prompt
            response = "What are you trying to accomplish with the MCP you're looking for?"
            self.conversation_history.append({"role": "assistant", "content": response})
            print(f"\nAssistant: {response}")
        
        # Conversation loop
        while self.current_state != "finalizing":
            user_input = input("\nYou: ")
            response = self.process_user_input(user_input)
            print(f"\nAssistant: {response}")
        
        # Generate the final query
        final_query = self._finalize_query()
        print(f"\nGenerated Query: {final_query}")
        
        return final_query
    
    def refine_query_with_llm(self, query_input: MCPQueryInput) -> str:
        """
        Refines the query using the OpenAI service.
        
        Args:
            query_input: The populated MCPQueryInput model.
            
        Returns:
            str: The refined query string.
        """
        if self.openai_service:
            return self.openai_service.generate_refined_query(query_input)
        else:
            # Fallback to basic query generation
            return query_input.generate_query_string() 