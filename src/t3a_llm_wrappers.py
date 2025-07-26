#!/usr/bin/env python3
"""
Proper LLM Wrapper usage examples for T3A agent
Based on Android World's concrete wrapper implementations
"""

from android_world.agents import infer
from enhanced_t3a_agent import EnhancedT3A

def create_openai_t3a_agent(env, prompt_type: str = "original", model: str = "gpt-4o-mini"):
    """Create T3A agent with OpenAI wrapper
    
    Args:
        env: Android World environment
        prompt_type: Reasoning approach ("original", "chain_of_thought", "self_reflection", "combined")
        model: OpenAI model name
    
    Returns:
        EnhancedT3A agent instance
    """
    
    # Use concrete Gpt4Wrapper class (not abstract LlmWrapper)
    llm_wrapper = infer.Gpt4Wrapper(model)
    
    agent = EnhancedT3A(
        env=env,
        llm=llm_wrapper,
        prompt_type=prompt_type,
        name=f"T3A_{prompt_type}_{model}",
        enable_summarization=True,
        max_history_length=15
    )
    
    return agent

def create_anthropic_t3a_agent(env, prompt_type: str = "original", model: str = "claude-3-sonnet-20240229"):
    """Create T3A agent with Anthropic wrapper
    
    Args:
        env: Android World environment  
        prompt_type: Reasoning approach
        model: Anthropic model name
    
    Returns:
        EnhancedT3A agent instance
    """
    
    # Use concrete ClaudeWrapper class
    llm_wrapper = infer.ClaudeWrapper(model)
    
    agent = EnhancedT3A(
        env=env,
        llm=llm_wrapper,
        prompt_type=prompt_type,
        name=f"T3A_{prompt_type}_{model.split('-')[0]}",
        enable_summarization=True,
        max_history_length=15
    )
    
    return agent

def get_available_models():
    """Get available models for each provider"""
    return {
        "openai": [
            "gpt-4o-mini",      # Cost-effective, good performance
            "gpt-4o",           # Latest GPT-4 model
            "gpt-4-turbo",      # Fast GPT-4 variant
            "gpt-4",            # Original GPT-4
            "gpt-3.5-turbo"     # Faster, cheaper option
        ],
        "anthropic": [
            "claude-3-sonnet-20240229",   # Balanced performance
            "claude-3-opus-20240229",     # Highest capability
            "claude-3-haiku-20240307"     # Fastest, cheapest
        ]
    }

def demonstrate_wrapper_usage():
    """Demonstrate correct LLM wrapper usage"""
    
    print("🔧 CORRECT T3A LLM WRAPPER USAGE")
    print("=" * 50)
    
    print("✅ CORRECT - Use concrete wrapper classes:")
    print("   llm = infer.Gpt4Wrapper('gpt-4o-mini')")
    print("   llm = infer.ClaudeWrapper('claude-3-sonnet-20240229')")
    print()
    
    print("❌ INCORRECT - Don't use abstract class:")
    print("   llm = infer.LlmWrapper()  # This will fail!")
    print("   llm = infer.LlmWrapper(provider='openai')  # Also fails!")
    print()
    
    print("📋 Available Models:")
    models = get_available_models()
    
    for provider, model_list in models.items():
        print(f"\n{provider.upper()}:")
        for model in model_list:
            print(f"  • {model}")
    
    print("\n💡 Usage Examples:")
    print("""
# Basic usage
from android_world.agents import infer
from enhanced_t3a_agent import EnhancedT3A

# OpenAI GPT-4o-mini (recommended for cost-effectiveness)
llm = infer.Gpt4Wrapper('gpt-4o-mini')
agent = EnhancedT3A(env, llm, prompt_type='chain_of_thought')

# Anthropic Claude Sonnet (balanced performance)
llm = infer.ClaudeWrapper('claude-3-sonnet-20240229')
agent = EnhancedT3A(env, llm, prompt_type='self_reflection')

# High-performance setup
llm = infer.Gpt4Wrapper('gpt-4o')
agent = EnhancedT3A(env, llm, prompt_type='combined')
""")

def create_t3a_agent_factory(env):
    """Factory function to create T3A agents with different configurations"""
    
    def create_agent(provider: str = "openai", 
                    model: str = None, 
                    prompt_type: str = "chain_of_thought"):
        """
        Create T3A agent with specified configuration
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (None for default)
            prompt_type: Reasoning approach
        
        Returns:
            Configured EnhancedT3A agent
        """
        
        if provider == "openai":
            model = model or "gpt-4o-mini"
            llm = infer.Gpt4Wrapper(model)
        elif provider == "anthropic":
            model = model or "claude-3-sonnet-20240229"
            llm = infer.ClaudeWrapper(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return EnhancedT3A(
            env=env,
            llm=llm,
            prompt_type=prompt_type,
            name=f"T3A_{prompt_type}_{provider}",
            enable_summarization=True,
            max_history_length=15
        )
    
    return create_agent

# Example usage in evaluation
def example_evaluation_setup(android_framework):
    """Example of setting up T3A evaluation with proper wrappers"""
    
    from t3a_evaluation_framework import T3AEvaluationFramework
    
    # Create evaluation framework
    evaluator = T3AEvaluationFramework(android_framework)
    
    # Override the create_llm_wrapper method to use correct implementations
    def create_proper_llm_wrapper(provider: str = "openai", model: str = None):
        if provider == "openai":
            model = model or "gpt-4o-mini"
            return infer.Gpt4Wrapper(model)
        elif provider == "anthropic":
            model = model or "claude-3-sonnet-20240229"
            return infer.ClaudeWrapper(model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Replace the method
    evaluator.create_llm_wrapper = create_proper_llm_wrapper
    
    return evaluator

if __name__ == "__main__":
    demonstrate_wrapper_usage()