"""
Utility module for setting up IBM Watsonx LLM and embedding models for LangGraph programming.

This module provides convenient functions to create and configure Watsonx models
with proper authentication and parameter settings.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from ibm_watsonx_ai import APIClient, Credentials
from langchain_ibm import ChatWatsonx, WatsonxEmbeddings
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


def load_environment() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict containing environment variables
    
    Raises:
        ValueError: If required environment variables are missing
    """
    load_dotenv()
    
    required_vars = ['WATSONX_APIKEY', 'WATSONX_URL', 'PROJ_ID']
    env_vars = {}
    
    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            raise ValueError(f"Missing required environment variable: {var}")
        env_vars[var] = value
    
    # Optional variables
    env_vars['WATSONX_USERNAME'] = os.environ.get('WATSONX_USERNAME')
    
    return env_vars


def create_watsonx_llm(
    model_id: str = "ibm/granite-3-2-8b-instruct",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    custom_params: Optional[Dict[str, Any]] = None
) -> ChatWatsonx:
    """
    Create and configure a ChatWatsonx instance using APIClient.
    
    Args:
        model_id: The model identifier to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty factor
        custom_params: Additional custom parameters to override defaults
    
    Returns:
        Configured ChatWatsonx instance
    
    Raises:
        ValueError: If environment variables are not properly configured
    """
    env_vars = load_environment()
    
    # Default parameters
    params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_P: top_p,
        GenParams.TOP_K: top_k,
        GenParams.REPETITION_PENALTY: repetition_penalty,
    }
    
    # Override with custom parameters if provided
    if custom_params:
        params.update(custom_params)
    
    # Create credentials and client
    credentials = Credentials(
        url=env_vars['WATSONX_URL'],
        api_key=env_vars['WATSONX_APIKEY']
    )
    
    client = APIClient(
        credentials, 
        api_key=env_vars['WATSONX_APIKEY'], 
        project_id=env_vars['PROJ_ID']
    )
    
    # Create ChatWatsonx instance
    llm = ChatWatsonx(
        model_id=model_id,
        watsonx_client=client,
        params=params
    )
    
    return llm


def create_watsonx_embeddings(
    model_id: str = "intfloat/multilingual-e5-large",
    truncate_input_tokens: int = 512,
    return_input_text: bool = True,
    return_input_tokens: bool = False,
    custom_params: Optional[Dict[str, Any]] = None
) -> WatsonxEmbeddings:
    """
    Create and configure a WatsonxEmbeddings instance.
    
    Args:
        model_id: The embedding model identifier to use
        truncate_input_tokens: Maximum input tokens to process
        return_input_text: Whether to return input text with embeddings
        return_input_tokens: Whether to return tokenization info
        custom_params: Additional custom parameters to override defaults
    
    Returns:
        Configured WatsonxEmbeddings instance
    
    Raises:
        ValueError: If environment variables are not properly configured
    """
    env_vars = load_environment()
    
    # Default embedding parameters
    embed_params = {
        "truncate_input_tokens": truncate_input_tokens,
        "return_options": {
            "input_text": return_input_text,
            "input_tokens": return_input_tokens
        }
    }
    
    # Override with custom parameters if provided
    if custom_params:
        embed_params.update(custom_params)
    
    embeddings = WatsonxEmbeddings(
        model_id=model_id,
        url=env_vars['WATSONX_URL'],
        project_id=env_vars['PROJ_ID'],
        params=embed_params,
        username=env_vars.get('WATSONX_USERNAME', 'apikey')  # Use 'apikey' as default username
    )
    
    return embeddings


def create_models_for_langgraph(
    llm_model_id: str = "ibm/granite-3-2-8b-instruct",
    embedding_model_id: str = "intfloat/multilingual-e5-large",
    llm_params: Optional[Dict[str, Any]] = None,
    embedding_params: Optional[Dict[str, Any]] = None
) -> tuple[ChatWatsonx, WatsonxEmbeddings]:
    """
    Create both LLM and embedding models configured for LangGraph usage.
    
    Args:
        llm_model_id: The LLM model identifier
        embedding_model_id: The embedding model identifier
        llm_params: Custom parameters for LLM
        embedding_params: Custom parameters for embeddings
    
    Returns:
        Tuple of (ChatWatsonx, WatsonxEmbeddings) instances
    
    Raises:
        ValueError: If environment variables are not properly configured
    """
    # Create LLM with custom parameters
    llm_config = llm_params or {}
    llm = create_watsonx_llm(
        model_id=llm_model_id,
        custom_params=llm_config.get('params'),
        **{k: v for k, v in llm_config.items() if k != 'params'}
    )
    
    # Create embeddings with custom parameters
    embedding_config = embedding_params or {}
    embeddings = create_watsonx_embeddings(
        model_id=embedding_model_id,
        custom_params=embedding_config.get('params'),
        **{k: v for k, v in embedding_config.items() if k != 'params'}
    )
    
    return llm, embeddings


# Predefined model configurations for common use cases
MODEL_CONFIGS = {
    "default": {
        "llm_model": "ibm/granite-3-2-8b-instruct",
        "embedding_model": "intfloat/multilingual-e5-large"
    },
    "fast": {
        "llm_model": "ibm/granite-3-2b-instruct",
        "embedding_model": "ibm/granite-embedding-107m-multilingual"
    },
    "english_optimized": {
        "llm_model": "ibm/granite-3-2-8b-instruct",
        "embedding_model": "ibm/slate-125m-english-rtrvr"
    },
    "lightweight": {
        "llm_model": "meta-llama/llama-3-2-1b-instruct",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "llama4": {
        "llm_model": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "embedding_model": "intfloat/multilingual-e5-large"
    }
}


def create_models_by_config(config_name: str = "default") -> tuple[ChatWatsonx, WatsonxEmbeddings]:
    """
    Create models using predefined configurations.
    
    Args:
        config_name: Name of the configuration to use ("default", "fast", "english_optimized", "lightweight")
    
    Returns:
        Tuple of (ChatWatsonx, WatsonxEmbeddings) instances
    
    Raises:
        ValueError: If configuration name is not found
    """
    if config_name not in MODEL_CONFIGS:
        available_configs = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown configuration '{config_name}'. Available: {available_configs}")
    
    config = MODEL_CONFIGS[config_name]
    return create_models_for_langgraph(
        llm_model_id=config["llm_model"],
        embedding_model_id=config["embedding_model"]
    )


def test_models(llm: ChatWatsonx, embeddings: WatsonxEmbeddings) -> Dict[str, Any]:
    """
    Test the created models with simple examples.
    
    Args:
        llm: ChatWatsonx instance to test
        embeddings: WatsonxEmbeddings instance to test
    
    Returns:
        Dictionary containing test results
    """
    results = {}
    
    try:
        # Test LLM - ChatWatsonx uses different API
        test_prompt = "What is artificial intelligence?"
        from langchain_core.messages import HumanMessage
        llm_response = llm.invoke([HumanMessage(content=test_prompt)])
        response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        results["llm_test"] = {
            "success": True,
            "prompt": test_prompt,
            "response_length": len(response_content),
            "response_preview": response_content[:100] + "..." if len(response_content) > 100 else response_content
        }
    except Exception as e:
        results["llm_test"] = {
            "success": False,
            "error": str(e)
        }
    
    try:
        # Test embeddings
        test_text = "LangChain is a framework for building AI applications"
        embedding_vector = embeddings.embed_query(test_text)
        results["embedding_test"] = {
            "success": True,
            "text": test_text,
            "vector_dimension": len(embedding_vector),
            "vector_preview": embedding_vector[:5]
        }
    except Exception as e:
        results["embedding_test"] = {
            "success": False,
            "error": str(e)
        }
    
    return results


if __name__ == "__main__":
    """
    Example usage and testing
    """
    try:
        print("Creating Watsonx models...")
        
        # Create models using default configuration
        llm, embeddings = create_models_by_config("default")
        
        print("Models created successfully!")
        print(f"LLM Model: {llm.model_id}")
        print(f"Embedding Model: {embeddings.model_id}")
        
        # Test the models
        print("\nTesting models...")
        test_results = test_models(llm, embeddings)
        
        print("\nTest Results:")
        for test_name, result in test_results.items():
            print(f"\n{test_name.upper()}:")
            if result["success"]:
                print("Success")
                for key, value in result.items():
                    if key != "success":
                        print(f"  {key}: {value}")
            else:
                print("Failed")
                print(f"  Error: {result['error']}")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your .env file is properly configured with valid credentials.")
