#!/usr/bin/env python3
"""
Solution Explainer for MCP Stack Recommendation

Generates clear explanations of how recommended MCP stacks can be integrated
and visualizes the workflow between components.
"""

import os
import json
import requests
import re
from typing import List, Dict, Any, Optional

class SolutionExplainer:
    """Explains MCP stack recommendations to users."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the SolutionExplainer with an OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4o"
    
    def _call_openai(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """Make an API call to OpenAI."""
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
        response.raise_for_status()
        return response.json()
    
    def _extract_json_safely(self, content: str) -> Dict[str, Any]:
        """
        Extract JSON from a string more safely, handling common issues.
        
        Args:
            content: String that may contain JSON.
            
        Returns:
            Extracted JSON as a dictionary.
        """
        # Try standard JSON extraction methods first
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                # Try to find JSON-like content with regex
                json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
                match = re.search(json_pattern, content)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)
                else:
                    # If no pattern found, try loading the whole content
                    return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON extraction failed: {str(e)}. Attempting fallback extraction...")
            
            # Fallback: Try to clean up and fix common JSON issues
            try:
                # Fix unescaped quotes by replacing \\"
                content_cleaned = content.replace('\\n', ' ')
                
                # Fix unescaped quotes in JSON
                content_cleaned = re.sub(r'(?<!\\\\)\"(?=.*\":\\s*)', r'\\\\\"', content_cleaned)
                content_cleaned = re.sub(r'(?<!\\\\)\"(?!,|\\s*}|\\s*]|\\s*:)', r'\\\\\"', content_cleaned)
                
                # Add escape sequences for special characters
                for char in ['\\b', '\\f', '\\n', '\\r', '\\t']:
                    content_cleaned = content_cleaned.replace(char, '\\\\' + char)
                
                # Look for content that looks like JSON objects
                pattern = r'\\{.*?\\}'
                matches = re.findall(pattern, content_cleaned, re.DOTALL)
                
                if matches:
                    # Try each potential JSON object
                    for match in matches:
                        try:
                            return json.loads(match)
                        except:
                            continue
                            
                # Try a more aggressive approach - extract just the structure and rebuild
                try:
                    # Extract key-value pairs using regex
                    pattern = r'\"([^"]+)\"\\s*:\\s*(\"([^\"\\\\]*(\\\\.[^\"\\\\]*)*)\"|(\\{.*?\\})|(\\[.*?\\])|([^,} \\]]+))'
                    matches = re.findall(pattern, content, re.DOTALL)
                    
                    if matches:
                        result = {}
                        for match in matches:
                            key = match[0]
                            value = match[1]
                            
                            # Try to convert value to appropriate type
                            try:
                                if value.startswith('\"') and value.endswith('\"'):
                                    # String value
                                    result[key] = value[1:-1]
                                elif value.startswith('{') and value.endswith('}'):
                                    # Object value
                                    result[key] = json.loads(value)
                                elif value.startswith('[') and value.endswith(']'):
                                    # Array value
                                    result[key] = json.loads(value)
                                else:
                                    # Number, boolean, or null
                                    if value.lower() == 'true':
                                        result[key] = True
                                    elif value.lower() == 'false':
                                        result[key] = False
                                    elif value.lower() == 'null':
                                        result[key] = None
                                    else:
                                        try:
                                            result[key] = int(value)
                                        except:
                                            try:
                                                result[key] = float(value)
                                            except:
                                                result[key] = value
                            except:
                                # If parsing fails, just use the raw value
                                result[key] = value
                                
                        return result
                except:
                    pass
            except:
                pass
            
            # If all extraction methods fail, return a basic fallback
            return self._generate_fallback_explanation()
    
    def _generate_fallback_explanation(self) -> Dict[str, Any]:
        """
        Generate a fallback explanation when JSON parsing fails.
        
        Returns:
            Basic explanation dictionary.
        """
        return {
            "summary": "The recommended MCP stack provides specialized components for each part of your task.",
            "tool_mapping": "Use each MCP for its corresponding subtask as shown in the results.",
            "benefits": "This combination offers a balanced approach with specialized tools for each step.",
            "alternatives": "Consider exploring other combinations if specific requirements change.",
            "setup_guide": "To get started, follow the documentation for each MCP. Most require API key setup and basic configuration.",
            "integration": {
                "steps": ["Configure each MCP separately", "Pass data between components as specified in their documentation", "Handle errors appropriately"],
                "example_code": "# Basic integration example\\n# See detailed documentation for each MCP"
            }
        }
    
    def generate_explanation(self, 
                            stack: List[Dict[str, Any]],
                            subtasks: List[Dict[str, Any]],
                            evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a clear explanation of how the MCP stack should be integrated.
        
        Args:
            stack: List of MCP dictionaries in the recommended stack.
            subtasks: List of subtask dictionaries.
            evaluation: Evaluation metrics for the stack.
            
        Returns:
            Dictionary with explanation sections.
        """
        # Create simplified representations for the prompt
        stack_summary = []
        for mcp in stack:
            stack_summary.append({
                "name": mcp.get("name", "Unknown"),
                "description": mcp.get("description", "No description"),
                "subtask": mcp.get("subtask", "Unknown")
            })
        
        subtasks_summary = []
        for subtask in subtasks:
            subtasks_summary.append({
                "name": subtask.get("name", "Unknown"),
                "description": subtask.get("description", "No description"),
                "capabilities": subtask.get("capabilities", []),
                "inputs": subtask.get("inputs", []),
                "outputs": subtask.get("outputs", [])
            })
        
        messages = [
            {
                "role": "system",
                "content": """
                You are a practical solution architect explaining how to use a stack of MCPs (Models, Components, or Protocols).
                
                Create a clear, concise explanation that directly maps tools to specific subtasks and provides actionable guidance.
                
                Your explanation should be returned as a valid JSON object with these sections:
                {
                  "summary": "Brief, direct overview of the solution (1-2 sentences)",
                  "tool_mapping": "Direct mapping of tools to subtasks in format: 'Use [Tool X] for [subtask], [Tool Y] for [subtask]'",
                  "benefits": "Clear explanation of why this combination works well (1-3 sentences)",
                  "alternatives": "Brief mention of alternative combinations if available",
                  "setup_guide": "Practical steps to get started with this stack",
                  "integration": {
                    "steps": ["Step 1", "Step 2", "Step 3"],
                    "example_code": "Basic pseudocode showing how components connect"
                  }
                }
                
                Keep your explanations short and practical. Focus on what users need to get started quickly.
                Avoid technical jargon unless necessary. Make your tool mapping very explicit.
                Structure your JSON response carefully to avoid parsing errors.
                """
            },
            {
                "role": "user",
                "content": f"""
                Explain how to use this MCP stack:
                
                Stack Components:
                {json.dumps(stack_summary, indent=2)}
                
                Subtasks:
                {json.dumps(subtasks_summary, indent=2)}
                
                Evaluation:
                {json.dumps(evaluation, indent=2)}
                """
            }
        ]
        
        try:
            response = self._call_openai(messages)
            content = response['choices'][0]['message']['content']
            
            # Extract JSON from response using the safer method
            explanation = self._extract_json_safely(content)
            
            # Ensure all required fields are present
            required_fields = ["summary", "tool_mapping", "benefits", "alternatives", "setup_guide", "integration"]
            for field in required_fields:
                if field not in explanation:
                    explanation[field] = self._generate_fallback_explanation()[field]
            
            # Ensure integration field is properly structured
            if not isinstance(explanation.get("integration"), dict):
                explanation["integration"] = self._generate_fallback_explanation()["integration"]
            
            # Ensure integration has steps and example_code
            integration = explanation["integration"]
            if "steps" not in integration or not isinstance(integration["steps"], list):
                integration["steps"] = ["Configure each MCP", "Connect components according to data flow", "Handle errors appropriately"]
            if "example_code" not in integration:
                integration["example_code"] = "# Basic integration example\\n# See documentation for each MCP"
                
            return explanation
        except Exception as e:
            print(f"Warning: Explanation generation failed: {str(e)}")
            # Return basic explanation on failure
            return self._generate_fallback_explanation()
    
    def generate_code_examples(self, stack: List[Dict[str, Any]], subtasks: List[Dict[str, Any]]) -> str:
        """
        Generate pseudocode examples for integrating the MCP stack.
        
        Args:
            stack: List of MCP dictionaries in the recommended stack.
            subtasks: List of subtask dictionaries.
            
        Returns:
            String with pseudocode examples.
        """
        # Create simplified representations for the prompt
        stack_summary = []
        for mcp in stack:
            stack_summary.append({
                "name": mcp.get("name", "Unknown"),
                "description": mcp.get("description", "No description"),
                "subtask": mcp.get("subtask", "Unknown")
            })
        
        subtasks_summary = []
        for subtask in subtasks:
            subtasks_summary.append({
                "name": subtask.get("name", "Unknown"),
                "description": subtask.get("description", "No description"),
                "capabilities": subtask.get("capabilities", []),
                "inputs": subtask.get("inputs", []),
                "outputs": subtask.get("outputs", [])
            })
        
        messages = [
            {
                "role": "system",
                "content": """
                You are an expert programmer creating integration examples for MCP (Model, Component, or Protocol) stacks.
                Create pseudocode examples that show how to connect and use the MCPs in a workflow.
                
                Focus on:
                1. How data flows between components
                2. API interactions
                3. Error handling considerations
                4. Authentication and setup
                
                Use Python-like pseudocode that is easy to understand but detailed enough to be useful.
                """
            },
            {
                "role": "user",
                "content": f"""
                Create pseudocode examples for integrating this MCP stack:
                
                Stack Components:
                {json.dumps(stack_summary, indent=2)}
                
                Subtasks and their relationships:
                {json.dumps(subtasks_summary, indent=2)}
                """
            }
        ]
        
        try:
            response = self._call_openai(messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Warning: Code example generation failed: {str(e)}")
            # Simple fallback pseudocode
            fallback_code = """
# Basic integration pseudocode
import mcp_library  # Hypothetical library for MCP integration

# Initialize the MCP components
components = {}
for mcp_info in stack:
    mcp = mcp_library.load_mcp(mcp_info['name'])
    components[mcp_info['subtask']] = mcp
    
# Set up authentication for each component
for subtask, mcp in components.items():
    mcp.authenticate(api_key='YOUR_API_KEY_HERE')
    
# Process data through the workflow
def process_workflow(input_data):
    results = {'input': input_data}
    
    # Execute each component in sequence based on subtask dependencies
    for subtask_name, subtask_info in subtasks.items():
        if subtask_name in components:
            current_mcp = components[subtask_name]
            
            # Get inputs from previous subtasks
            subtask_inputs = {}
            for input_name in subtask_info.get('inputs', []):
                if input_name in results:
                    subtask_inputs[input_name] = results[input_name]
            
            # Process with current MCP
            try:
                result = current_mcp.process(subtask_inputs)
                results[subtask_name] = result
            except Exception as e:
                print(f"Error in {subtask_name}: {str(e)}")
                # Implement fallback or recovery strategy
    
    return results

# Example usage
final_result = process_workflow({"query": "user input here"})
"""
            return fallback_code


if __name__ == "__main__":
    # Sample data for testing
    sample_stack = [
        {
            "name": "LegalDocFinder",
            "description": "Search engine specialized for legal documents with case law integration",
            "subtask": "document_search"
        },
        {
            "name": "LegalSummarizer",
            "description": "Tool for creating concise summaries of legal documents and contracts",
            "subtask": "document_summarization"
        },
        {
            "name": "VoiceGenerator",
            "description": "High-quality text-to-speech system with natural voice options",
            "subtask": "audio_conversion"
        }
    ]
    
    sample_subtasks = [
        {
            "name": "document_search",
            "description": "Find relevant legal documents based on search criteria",
            "capabilities": ["search", "legal_domain_knowledge"],
            "inputs": [],
            "outputs": ["document_summarization"]
        },
        {
            "name": "document_summarization",
            "description": "Create concise summaries of legal documents",
            "capabilities": ["text_summarization", "legal_domain_knowledge"],
            "inputs": ["document_search"],
            "outputs": ["audio_conversion"]
        },
        {
            "name": "audio_conversion",
            "description": "Convert text summaries to spoken audio",
            "capabilities": ["speech_synthesis"],
            "inputs": ["document_summarization"],
            "outputs": []
        }
    ]
    
    sample_evaluation = {
        "score": 0.85,
        "coverage": 0.9,
        "efficiency": 0.8,
        "workflow_compatibility": 0.85,
        "compatibility_score": 0.9,
        "combined_score": 0.865,
        "notes": "Strong stack with specialized components for each task.",
        "missing_capabilities": []
    }
    
    # Test the explainer
    explainer = SolutionExplainer()
    explanation = explainer.generate_explanation(
        sample_stack, sample_subtasks, sample_evaluation
    )
    
    # Display the explanation
    print("=== SOLUTION EXPLANATION ===\\n")
    print("OVERVIEW:")
    print(explanation.get("overview", ""))
    
    print("\\nWORKFLOW:")
    print(explanation.get("workflow", ""))
    
    print("\\nINTEGRATION TIPS:")
    print(explanation.get("integration_tips", ""))
    
    print("\\nDIAGRAM:")
    print(explanation.get("diagram", ""))
    
    # Generate code examples
    print("\\n=== CODE EXAMPLES ===\\n")
    code_examples = explainer.generate_code_examples(sample_stack, sample_subtasks)
    print(code_examples) 