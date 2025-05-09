"""
Simple test script to verify if the LLM module is working
"""
from llm_module import LLMModule
import time

def main():
    print("Starting LLM test script...")
    
    # Sample context and question
    context = """
    FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+
    based on standard Python type hints. It was created by Sebastián Ramírez and is designed to make
    it easy to build robust and standards-compliant APIs quickly. FastAPI is inspired by and compatible
    with previous Python web frameworks like Flask and Django, but offers improved performance and
    features. Key features include automatic data validation, serialization, interactive documentation,
    and asynchronous support.
    """
    
    question = "What is FastAPI and who created it?"
    
    # Initialize the LLM 
    print("Initializing LLM module...")
    start_time = time.time()
    llm = LLMModule(model_name="google/flan-t5-large")
    print(f"LLM initialized in {time.time() - start_time:.2f} seconds")
    
    # Generate an answer
    print("\nGenerating answer to test question...")
    start_time = time.time()
    answer = llm.generate_answer(question, [context])
    print(f"Answer generated in {time.time() - start_time:.2f} seconds")
    
    print("\n" + "=" * 50)
    print("QUESTION:")
    print(question)
    print("\nCONTEXT:")
    print(context)
    print("\nANSWER:")
    print(answer)
    print("=" * 50)
    
    print("\nLLM test completed.")

if __name__ == "__main__":
    main() 