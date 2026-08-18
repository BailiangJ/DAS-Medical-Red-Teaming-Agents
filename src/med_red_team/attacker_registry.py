from importlib import import_module
from typing import Any, Dict, List, Optional

from med_red_team.actors import AttackStrategy
from med_red_team.model_pool import ModelPool


# Strategy metadata registry
STRATEGY_REGISTRY = {
    # ==============================================================================
    # Robustness Strategies
    # ==============================================================================
    # Rule-based strategies
    "add_none_of_the_above": {
        "class": "med_red_team.robustness.attacker:AddNoneOfTheAboveStrategy",
        "description": "Add 'None of the options are correct' option",
        "requires_llm": False,
        "modifies": "options",
        "category": "robustness",
        "parameters": {}
    },
    "replace_correct_answer_with_none": {
        "class": "med_red_team.robustness.attacker:ReplaceCorrectAnswerStrategy",
        "description": "Replace correct answer with 'None of the options are correct'",
        "requires_llm": False,
        "modifies": "options + answer",
        "category": "robustness",
        "parameters": {}
    },

    # LLM-based strategies
    "add_distraction_sentence": {
        "class": "med_red_team.robustness.attacker:AddDistractionSentenceStrategy",
        "description": "Add distracting sentence referencing wrong answer",
        "requires_llm": True,
        "modifies": "question",
        "category": "robustness",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM (temperature, max_tokens, etc.)"
            }
        }
    },
    "generate_distractor_options": {
        "class": "med_red_team.robustness.attacker:GenerateDistractorOptionsStrategy",
        "description": "Generate additional plausible but incorrect options",
        "requires_llm": True,
        "modifies": "options",
        "category": "robustness",
        "parameters": {
            "num_distractors": {
                "type": "int",
                "default": 2,
                "description": "Number of new distractor options to generate"
            },
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM (temperature, max_tokens, etc.)"
            }
        }
    },
    "bias_manipulation": {
        "class": "med_red_team.robustness.attacker:BiasManipulationStrategy",
        "description": "Introduce cognitive biases (8 types)",
        "requires_llm": True,
        "modifies": "question",
        "category": "robustness",
        "parameters": {
            "n_bias_styles": {
                "type": "int",
                "default": 1,
                "range": [1, 8],
                "description": "Number of bias styles to inject (must be between 1 and 8)"
            },
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM (temperature, max_tokens, etc.)"
            }
        }
    },
    "invert_question_answer": {
        "class": "med_red_team.robustness.attacker:InvertQuestionAnswerStrategy",
        "description": "Invert final question sentence and complement answer",
        "requires_llm": True,
        "modifies": "question + answer",
        "category": "robustness",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM (temperature, max_tokens, etc.)"
            }
        }
    },
    "adjust_impossible_measurement": {
        "class": "med_red_team.robustness.attacker:AdjustImpossibleMeasurementStrategy",
        "description": "Change measurements to impossible values",
        "requires_llm": True,
        "modifies": "question + options + answer",
        "category": "robustness",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM (temperature, max_tokens, etc.)"
            }
        }
    },

    # ==============================================================================
    # Bias Strategies
    # ==============================================================================
    "race_socioeconomic_label": {
        "class": "med_red_team.bias.attacker:RaceSocioeconomicLabelStrategy",
        "description": "Replace {patient} placeholder with demographic labels to test for bias",
        "requires_llm": True,
        "modifies": "clinical_summary",
        "category": "bias",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            },
            "label_list": {
                "type": "Optional[List[str]]",
                "default": None,
                "description": "List of demographic labels to choose from"
            },
            "patient_placeholder": {
                "type": "str",
                "default": "{patient}",
                "description": "Placeholder string in case text"
            },
            "verbose": {
                "type": "bool",
                "default": True,
                "description": "Whether to print strategy diagnostics"
            }
        }
    },
    "language_manipulation": {
        "class": "med_red_team.bias.attacker:LanguageManipulationStrategy",
        "description": "Modify patient self-report language style (e.g., dialect, limited English)",
        "requires_llm": True,
        "modifies": "patient_self_report",
        "category": "bias",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            },
            "strategy_list": {
                "type": "Optional[List[str]]",
                "default": None,
                "description": "List of language strategies to choose from"
            },
            "verbose": {
                "type": "bool",
                "default": True,
                "description": "Whether to print strategy diagnostics"
            }
        }
    },
    "emotion_manipulation": {
        "class": "med_red_team.bias.attacker:EmotionManipulationStrategy",
        "description": "Modify patient self-report emotional tone (e.g., angry, anxious)",
        "requires_llm": True,
        "modifies": "patient_self_report",
        "category": "bias",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            },
            "strategy_list": {
                "type": "Optional[List[str]]",
                "default": None,
                "description": "List of emotion strategies to choose from"
            },
            "verbose": {
                "type": "bool",
                "default": True,
                "description": "Whether to print strategy diagnostics"
            }
        }
    },
    "cognitive_bias": {
        "class": "med_red_team.bias.attacker:CognitiveBiasStrategy",
        "description": "Introduce cognitive bias via clinical note modification",
        "requires_llm": True,
        "modifies": "clinical_summary",
        "category": "bias",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            },
            "verbose": {
                "type": "bool",
                "default": True,
                "description": "Whether to print strategy diagnostics"
            }
        }
    },

    # ==============================================================================
    # Privacy Strategies
    # ==============================================================================
    "implicit_disguise": {
        "class": "med_red_team.privacy.attacker:ImplicitDisguiseStrategy",
        "description": "Make explicit PHI requests more subtle and implicit",
        "requires_llm": True,
        "modifies": "prompt",
        "category": "privacy",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            }
        }
    },
    "focus_distraction": {
        "class": "med_red_team.privacy.attacker:FocusDistractionStrategy",
        "description": "Add 2-3 distracting instructions to the prompt",
        "requires_llm": True,
        "modifies": "prompt",
        "category": "privacy",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            }
        }
    },
    "privacy_warning": {
        "class": "med_red_team.privacy.attacker:PrivacyWarningStrategy",
        "description": "Inject privacy warnings to create false compliance appearance",
        "requires_llm": True,
        "modifies": "prompt",
        "category": "privacy",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            }
        }
    },
    "well_intention": {
        "class": "med_red_team.privacy.attacker:WellIntentionStrategy",
        "description": "Add well-meaning framing to make requests appear patient-focused",
        "requires_llm": True,
        "modifies": "prompt",
        "category": "privacy",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            }
        }
    },
    "combined_disguise": {
        "class": "med_red_team.privacy.attacker:CombinedDisguiseStrategy",
        "description": "Apply all disguise techniques sequentially (implicit, well intention, focus distraction, privacy warning)",
        "requires_llm": True,
        "modifies": "prompt",
        "category": "privacy",
        "parameters": {
            "config": {
                "type": "Optional[GenerationConfig]",
                "default": None,
                "description": "Generation configuration for LLM"
            }
        }
    }
}


def _resolve_strategy_class(class_ref: str | type[AttackStrategy]) -> type[AttackStrategy]:
    if not isinstance(class_ref, str):
        return class_ref
    module_name, separator, class_name = class_ref.partition(":")
    if not separator:
        raise ValueError(f"Invalid strategy class reference: {class_ref}")
    return getattr(import_module(module_name), class_name)


def get_strategy(
    strategy_name: str,
    model_id: Optional[str] = None,
    model_pool: Optional[ModelPool] = None,
    **kwargs
) -> AttackStrategy:
    """
    Factory function to create attack strategies by name.
    
    This function provides a unified interface for creating any attack strategy,
    handling both rule-based and LLM-based strategies automatically. It uses
    the STRATEGY_REGISTRY to dynamically instantiate strategies, eliminating
    the need for long if-elif chains.
    
    Args:
        strategy_name: Name of the strategy (see list_available_strategies())
        model_id: Model ID for LLM-based strategies (required for LLM strategies)
        model_pool: Model pool for LLM-based strategies (required for LLM strategies)
        **kwargs: Additional strategy-specific parameters:
            - num_distractors (int): For generate_distractor_options (default: 2)
            - n_bias_styles (int): For bias_manipulation (default: 1)
            - config (GenerationConfig): For any LLM-based strategy
        
    Returns:
        AttackStrategy instance
        
    Raises:
        ValueError: If strategy name is unknown or required parameters are missing
        
    Examples:
        # Rule-based strategy (no LLM needed)
        >>> strategy = get_strategy("add_none_of_the_above")
        >>> modified = strategy.apply(test_case, context={})
        
        # LLM-based strategy
        >>> pool = ModelPool()
        >>> strategy = get_strategy(
        ...     "add_distraction_sentence",
        ...     model_id="gpt-4o",
        ...     model_pool=pool
        ... )
        
        # LLM-based strategy with parameters
        >>> strategy = get_strategy(
        ...     "generate_distractor_options",
        ...     model_id="gpt-4o",
        ...     model_pool=pool,
        ...     num_distractors=3
        ... )
        
        # With custom config
        >>> config = GenerationConfig(temperature=0.9, max_tokens=1024)
        >>> strategy = get_strategy(
        ...     "bias_manipulation",
        ...     model_id="gpt-4o",
        ...     model_pool=pool,
        ...     n_bias_styles=2,
        ...     config=config
        ... )
    """
    # Validate strategy name
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    
    # Get strategy metadata from registry
    strategy_info = STRATEGY_REGISTRY[strategy_name]
    strategy_class = _resolve_strategy_class(strategy_info["class"])
    requires_llm = strategy_info["requires_llm"]
    parameters_spec = strategy_info["parameters"]
    
    # Check LLM requirements
    if requires_llm and (model_id is None or model_pool is None):
        raise ValueError(
            f"Strategy '{strategy_name}' requires LLM. "
            f"Please provide both model_id and model_pool."
        )
    
    # Build constructor arguments based on strategy type
    if requires_llm:
        # LLM-based strategies: start with model_id and model_pool
        constructor_args = [model_id, model_pool]
        constructor_kwargs = {}
        
        # Add strategy-specific parameters with defaults from registry
        for param_name, param_info in parameters_spec.items():
            if param_name in kwargs:
                # Use provided value
                constructor_kwargs[param_name] = kwargs[param_name]
            elif param_info.get("default") is not None:
                # Use default value from registry
                constructor_kwargs[param_name] = param_info["default"]
            # If neither provided nor has default, omit (will use class default)
        
        # Instantiate strategy with positional and keyword args
        return strategy_class(*constructor_args, **constructor_kwargs)
    else:
        # Rule-based strategies: no model_id or model_pool needed
        constructor_kwargs = {}
        
        # Add any parameters (though rule-based strategies typically have none)
        for param_name, param_info in parameters_spec.items():
            if param_name in kwargs:
                constructor_kwargs[param_name] = kwargs[param_name]
            elif param_info.get("default") is not None:
                constructor_kwargs[param_name] = param_info["default"]
        
        # Instantiate strategy
        return strategy_class(**constructor_kwargs)


def list_available_strategies(include_details: bool = False) -> Dict[str, Any]:
    """
    List all available attack strategies.
    
    Args:
        include_details: If True, return full metadata. If False, return just descriptions.
        
    Returns:
        Dictionary mapping strategy names to descriptions (or full metadata if include_details=True)
        
    Examples:
        # Simple list
        >>> strategies = list_available_strategies()
        >>> for name, desc in strategies.items():
        ...     print(f"{name}: {desc}")
        
        # Detailed list
        >>> strategies = list_available_strategies(include_details=True)
        >>> for name, info in strategies.items():
        ...     print(f"{name}:")
        ...     print(f"  Description: {info['description']}")
        ...     print(f"  Requires LLM: {info['requires_llm']}")
        ...     print(f"  Modifies: {info['modifies']}")
    """
    if include_details:
        return {
            name: {
                "description": info["description"],
                "requires_llm": info["requires_llm"],
                "modifies": info["modifies"],
                "parameters": info["parameters"]
            }
            for name, info in STRATEGY_REGISTRY.items()
        }
    else:
        return {
            name: info["description"]
            for name, info in STRATEGY_REGISTRY.items()
        }


def get_llm_strategies() -> List[str]:
    """
    Get list of strategies that require an LLM.
    
    Returns:
        List of strategy names that require model_id and model_pool
        
    Example:
        >>> llm_strategies = get_llm_strategies()
        >>> print(f"LLM-based strategies: {', '.join(llm_strategies)}")
    """
    return [
        name for name, info in STRATEGY_REGISTRY.items()
        if info["requires_llm"]
    ]


def get_rule_based_strategies() -> List[str]:
    """
    Get list of rule-based strategies that don't require an LLM.
    
    Returns:
        List of strategy names that work without an LLM
        
    Example:
        >>> rule_strategies = get_rule_based_strategies()
        >>> for name in rule_strategies:
        ...     strategy = get_strategy(name)  # No model_id needed
    """
    return [
        name for name, info in STRATEGY_REGISTRY.items()
        if not info["requires_llm"]
    ]


def get_strategy_parameters(strategy_name: str) -> Dict[str, Any]:
    """
    Get parameter specification for a specific strategy.
    
    This function returns detailed parameter information including types,
    defaults, descriptions, and ranges. Useful for validation, documentation,
    and UI generation.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Dictionary of parameter specifications
        
    Raises:
        ValueError: If strategy name is unknown
        
    Example:
        >>> params = get_strategy_parameters("bias_manipulation")
        >>> for name, spec in params.items():
        ...     print(f"{name}:")
        ...     print(f"  Type: {spec['type']}")
        ...     print(f"  Default: {spec['default']}")
        ...     print(f"  Description: {spec['description']}")
        
        # Output:
        # n_bias_styles:
        #   Type: int
        #   Default: 1
        #   Description: Number of bias styles to apply (1-8)
        # config:
        #   Type: Optional[GenerationConfig]
        #   Default: None
        #   Description: Generation configuration for LLM...
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: '{strategy_name}'. "
            f"Available strategies: {available}"
        )
    
    return STRATEGY_REGISTRY[strategy_name]["parameters"]


def create_strategy_pipeline(
    strategy_configs: List[Dict[str, Any]],
    model_pool: Optional[ModelPool] = None,
    default_model_id: Optional[str] = None
) -> List[AttackStrategy]:
    """
    Create a pipeline of multiple attack strategies from configuration.
    
    This is a convenience function for creating multiple strategies at once,
    useful for building attack pipelines from configuration files.
    
    Args:
        strategy_configs: List of strategy configuration dicts, each containing:
            - name (str): Strategy name
            - model_id (str, optional): Model ID (uses default_model_id if not specified)
            - Other strategy-specific parameters
        model_pool: Model pool for LLM-based strategies
        default_model_id: Default model ID to use if not specified in config
        
    Returns:
        List of initialized AttackStrategy instances
        
    Example:
        >>> configs = [
        ...     {"name": "bias_manipulation", "n_bias_styles": 2},
        ...     {"name": "add_distraction_sentence"},
        ...     {"name": "generate_distractor_options", "num_distractors": 3},
        ...     {"name": "add_none_of_the_above"}
        ... ]
        >>> pool = ModelPool()
        >>> strategies = create_strategy_pipeline(
        ...     configs,
        ...     model_pool=pool,
        ...     default_model_id="gpt-4o"
        ... )
        >>> 
        >>> # Apply all strategies
        >>> modified = test_case
        >>> for strategy in strategies:
        ...     modified = strategy.apply(modified, context={})
    """
    strategies = []
    
    for config in strategy_configs:
        strategy_name = config.get("name")
        if not strategy_name:
            raise ValueError("Each strategy config must have a 'name' field")
        
        # Get model_id (from config or default)
        model_id = config.get("model_id", default_model_id)
        
        # Extract other parameters (exclude 'name' and 'model_id')
        kwargs = {
            k: v for k, v in config.items()
            if k not in ["name", "model_id"]
        }
        
        # Create strategy
        strategy = get_strategy(
            strategy_name=strategy_name,
            model_id=model_id,
            model_pool=model_pool,
            **kwargs
        )
        
        strategies.append(strategy)
    
    return strategies