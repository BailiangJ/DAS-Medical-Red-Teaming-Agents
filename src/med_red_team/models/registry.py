"""
Centralized registry of all supported models and their configurations.

This file defines the MODEL_CONFIGS dictionary that maps model_id strings
to their implementation classes and default kwargs.
"""

from med_red_team.models.interfaces import InfrastructureConfig


OPENAI_MODEL = "med_red_team.models.api_based.openai_model:GPTModel"
OPENAI_O_MODEL = "med_red_team.models.api_based.openai_model:OGPTModel"
ANTHROPIC_MODEL = "med_red_team.models.api_based.anthropic_model:ClaudeModel"
GEMINI_MODEL = "med_red_team.models.api_based.google_model:GeminiModel"
DEEPSEEK_MODEL = "med_red_team.models.api_based.deepseek_model:DeepSeekModel"
OPENSOURCE_MODEL = "med_red_team.models.opensource.base:OpenSourceModel"


# ==============================================================================
# Model Registry
# ==============================================================================
# Maps model_id → {"class": ModelClass, "kwargs": default_kwargs}

MODEL_CONFIGS = {
    # -------------------------------------------------------------------------
    # OpenAI API Models
    # -------------------------------------------------------------------------
    "gpt-5": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "gpt-4o": {
        "class": OPENAI_MODEL,
        "kwargs": {}
    },
    "gpt-4.1": {
        "class": OPENAI_MODEL,
        "kwargs": {}
    },
    "gpt-4.1-mini": {
        "class": OPENAI_MODEL,
        "kwargs": {}
    },
    "o1": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "o1-mini": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "o1-pro": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "o3": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "o3-mini": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    "o4-mini": {
        "class": OPENAI_O_MODEL,
        "kwargs": {}
    },
    
    # -------------------------------------------------------------------------
    # Anthropic Models
    # -------------------------------------------------------------------------
    "claude-sonnet-4-5-20250929": {
        "class": ANTHROPIC_MODEL,
        "kwargs": {}
    },
    "claude-opus-4-5-20251101": {
        "class": ANTHROPIC_MODEL,
        "kwargs": {}
    },
    "claude-sonnet-4-20250514": {
        "class": ANTHROPIC_MODEL,
        "kwargs": {}
    },
    "claude-3-7-sonnet-20250219": {
        "class": ANTHROPIC_MODEL,
        "kwargs": {}
    },
    
    # -------------------------------------------------------------------------
    # Google Gemini Models
    # -------------------------------------------------------------------------
    "gemini-2.5-flash": {
        "class": GEMINI_MODEL,
        "kwargs": {}
    },
    "gemini-2.5-pro": {
        "class": GEMINI_MODEL,
        "kwargs": {}
    },
    "gemini-3-pro-preview": {
        "class": GEMINI_MODEL,
        "kwargs": {}
    },
    # -------------------------------------------------------------------------
    # DeepSeek Models
    # -------------------------------------------------------------------------
    "deepseek-chat": {
        "class": DEEPSEEK_MODEL,
        "kwargs": {}
    },
    "deepseek-reasoner": {
        "class": DEEPSEEK_MODEL,
        "kwargs": {}
    },
    # -------------------------------------------------------------------------
    # Open-Source Models - Qwen Family
    # -------------------------------------------------------------------------
    "qwen": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "Qwen/Qwen2.5-72B-Instruct",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
    "qwen3": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "Qwen/Qwen3-32B",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
    "qwq": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "Qwen/QwQ-32B",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
    
    # -------------------------------------------------------------------------
    # Open-Source Models - Llama Family
    # -------------------------------------------------------------------------
    "llama-3.3": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "meta-llama/Llama-3.3-70B-Instruct",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
                tensor_parallel_size=2,
            )
        }
    },
    "llama-4": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
                tensor_parallel_size=4,
                gpu_memory_utilization=0.92,
                quantization="fp8",
            )
        }
    },

    # -------------------------------------------------------------------------
    # Open-Source Models - Gemma Family
    # -------------------------------------------------------------------------
    "gemma":{
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "google/gemma-3-27b-it",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
    "medgemma":{
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "google/medgemma-27b-text-it",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
    
    # -------------------------------------------------------------------------
    # Open-Source Models - Medical Models
    # -------------------------------------------------------------------------
    "huatuo": {
        "class": OPENSOURCE_MODEL,
        "kwargs": {
            "hf_model_path": "FreedomIntelligence/HuatuoGPT-o1-70B",
            "infra_config": InfrastructureConfig(
                max_model_len=8192,
                max_num_seqs=2,
            )
        }
    },
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def list_api_models():
    """List all API-based models without importing provider SDKs."""
    return [
        model_id for model_id, config in MODEL_CONFIGS.items()
        if config["class"] != OPENSOURCE_MODEL
    ]


def list_opensource_models():
    """List all open-source (vLLM) models without importing vLLM."""
    return [
        model_id for model_id, config in MODEL_CONFIGS.items()
        if config["class"] == OPENSOURCE_MODEL
    ]
