import google.generativeai as genai
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import re
import random

# Load environment variables
load_dotenv("config.env")

class LLMModule:
    def __init__(self, model_name: str = None):
        """
        Initialize the LLM module with Gemini API
        
        Args:
            model_name: Gemini model name (optional, defaults to env variable)
        """
        print("Initializing Gemini LLM module")
        start_time = time.time()
        
        try:
            # Get API key from environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "your_gemini_api_key_here":
                raise ValueError("Please set your Gemini API key in config.env file")
            
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Set up the model
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
            self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
            self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))
            
            print(f"Using Gemini model: {self.model_name}")
            print(f"Model configured successfully in {time.time() - start_time:.2f} seconds")
            
            # Generation config
            self.generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.95,
                "top_k": 0,
            }
            
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise
    
    def generate_answer(self, question: str, contexts: List[str], max_length: int = 1024) -> str:
        """
        Generate an answer to a question based on the provided contexts using Gemini API
        
        Args:
            question: The question to answer
            contexts: List of context passages from documents
            max_length: Maximum output length (overridden by env variable)
            
        Returns:
            Generated answer
        """
        if not contexts:
            return "No context information available to answer this question."
            
        print(f"Generating answer for question: '{question}'")
        print(f"Using {len(contexts)} context passages")
        
        # Combine context passages into a single context with appropriate handling
        combined_context = "\n\n".join(contexts[:5])  # Use up to 5 context chunks
        
        # Create an improved prompt
        prompt = f"""You are an intelligent assistant tasked with answering questions based mostly on the provided context information.
Your goal is to be accurate, comprehensive, and helpful.

CONTEXT INFORMATION:
{combined_context}

IMPORTANT INSTRUCTIONS:
1. Format your answer in a clear, readable way.

QUESTION: {question}

ANSWER:"""
        
        print(f"Prompt created with {len(combined_context)} characters of context")
        
        try:
            # Generate content with Gemini
            print("Sending request to Gemini API...")
            start_time = time.time()
            
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
            
            # Add retry logic for rate limit errors
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = model.generate_content(prompt)
                    generation_time = time.time() - start_time
                    print(f"Response received in {generation_time:.2f} seconds")
                    
                    # Extract the answer text
                    answer = response.text.strip()
                    print(f"Answer generated: {answer[:100]}...")
                    return answer
                
                except Exception as api_error:
                    if "429" in str(api_error) and retry_count < max_retries:
                        # Rate limit error, wait and retry
                        retry_count += 1
                        wait_time = 5 * retry_count  # Increasing backoff
                        print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        # Either not a rate limit error or we've exhausted retries
                        raise
            
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple extraction-based answer when API fails
            try:
                print("API call failed. Using fallback local extraction method...")
                return self._extract_answer_locally(question, contexts)
            except Exception as fallback_error:
                print(f"Fallback method also failed: {fallback_error}")
                return f"Unable to generate answer. API error: {str(e)}"
    
    def _extract_answer_locally(self, question: str, contexts: List[str]) -> str:
        """
        Simple fallback method that extracts sentences from the context that might answer the question.
        Used when the API call fails.
        """
        if not contexts:
            return "No context information available to answer this question."
            
        # Join all contexts
        all_text = " ".join(contexts)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', all_text)
        
        # Find keywords from the question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}  # Filter out short words
        
        # Score sentences by keyword matches
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            matches = len(question_words.intersection(sentence_words))
            if matches > 0:
                scored_sentences.append((sentence, matches))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if not scored_sentences:
            return "Could not find relevant information in the provided documents."
            
        # Take top 3 sentences or fewer if less available
        top_sentences = [s[0] for s in scored_sentences[:min(3, len(scored_sentences))]]
        
        # Join sentences and return
        return " ".join(top_sentences)
    
    def filter_relevant_contexts(self, question: str, contexts: List[Dict[str, Any]], threshold: float = 0.5) -> List[str]:
        """
        Filter contexts to keep only the most relevant ones
        
        Args:
            question: The question being asked
            contexts: List of context dictionaries from the vector DB
            threshold: Relevance threshold
            
        Returns:
            List of relevant context strings
        """
        print(f"Filtering contexts for question: '{question}'")
        print(f"Number of contexts before filtering: {len(contexts)}")
        
        # Get all documents
        documents = [context["document"] for context in contexts]
        
        # Use a smart filtering approach:
        # 1. Sort by vector similarity (which ChromaDB already did for us)
        # 2. Limit to reasonable number to avoid context overflow
        max_contexts = min(5, len(documents))
        filtered_contexts = documents[:max_contexts]
        
        print(f"Returning {len(filtered_contexts)} most relevant contexts")
        return filtered_contexts 