"""
Unified open-source model implementation using vLLM.

This module provides a single class that handles ALL open-source models
(Gemma, Qwen, Llama, HuatuoGPT, etc.) using vLLM for fast inference.
"""

from typing import Optional
import time
# import torch

from med_red_team.models.interfaces import (
    LLMInterface,
    ModelResponse,
    GenerationConfig,
    InfrastructureConfig,
    ModelGenerationError
)


class OpenSourceModel(LLMInterface):
    """
    Unified class for all open-source LLM models using vLLM.
    
    Supports any model with a HuggingFace-compatible chat template:
    - Gemma (Google)
    - Qwen, QwQ (Alibaba)
    - Llama (Meta)
    - HuatuoGPT (Medical)
    - Any other HuggingFace model
    
    Usage:
        model = OpenSourceModel(
            model_id="qwen-2.5-72b",
            hf_model_path="Qwen/Qwen2.5-72B-Instruct",
            infra_config=InfrastructureConfig(...)
        )
    """
    
    def __init__(
        self,
        model_id: str,
        hf_model_path: str,
        infra_config: Optional[InfrastructureConfig] = None
    ):
        """
        Initialize open-source model with vLLM.
        
        Args:
            model_id: Our internal model identifier (e.g., "qwen-2.5-72b")
            hf_model_path: HuggingFace model path (e.g., "Qwen/Qwen2.5-72B-Instruct")
            infra_config: Infrastructure settings (GPU memory, etc.)
        """
        super().__init__(model_id)
        
        self.hf_model_path = hf_model_path
        
        # Use provided config or defaults
        self.infra_config = infra_config or InfrastructureConfig()
        print(f"[OpenSourceModel] Initialized {model_id} with HF path {hf_model_path} and infra config: {self.infra_config}")
        
        # Enable TF32 for Ampere GPUs if requested
        # if self.infra_config.enable_tf32:
        #     torch.set_float32_matmul_precision('high')
        
        # Initialize vLLM components
        self.tokenizer = self._initialize_tokenizer()
        self.llm = self._initialize_vllm()
    
    def _initialize_tokenizer(self):
        """
        Initialize the HuggingFace tokenizer.
        
        Returns:
            HuggingFace tokenizer instance
        """
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.hf_model_path)
        except Exception as e:
            raise ModelGenerationError(
                f"Failed to load tokenizer for {self.hf_model_path}: {str(e)}"
            ) from e
    
    def _initialize_vllm(self):
        """
        Initialize the vLLM engine.

        Returns:
            vLLM LLM instance
        """
        try:
            from vllm import LLM

            # Build vLLM initialization kwargs
            vllm_kwargs = {
                "model": self.hf_model_path,
                "dtype": self.infra_config.dtype,
                "gpu_memory_utilization": self.infra_config.gpu_memory_utilization,
                "tensor_parallel_size": self.infra_config.tensor_parallel_size,
                "max_model_len": self.infra_config.max_model_len,
                "max_num_seqs": self.infra_config.max_num_seqs,
                "seed": self.infra_config.seed,
            }

            # Add quantization if specified
            if self.infra_config.quantization is not None:
                vllm_kwargs["quantization"] = self.infra_config.quantization

            return LLM(**vllm_kwargs)
        except Exception as e:
            raise ModelGenerationError(
                f"Failed to initialize vLLM for {self.hf_model_path}: {str(e)}"
            ) from e
    
    def _format_chat(
        self,
        user_prompt: str,
        system_prompt: str
    ) -> str:
        """
        Format messages using the model's chat template.
        
        All HuggingFace models with chat templates are supported automatically.
        
        Args:
            user_prompt: User message
            system_prompt: System instruction
            
        Returns:
            Formatted string ready for tokenization
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            raise ModelGenerationError(
                f"Failed to apply chat template for {self.model_id}: {str(e)}"
            ) from e
    
    def _build_sampling_params(self, config: GenerationConfig):
        """
        Convert GenerationConfig to vLLM SamplingParams.
        
        Args:
            config: Our unified generation config
            
        Returns:
            vLLM SamplingParams object
        """
        try:
            from vllm import SamplingParams
            
            return SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_tokens=config.max_tokens,
                repetition_penalty=config.repetition_penalty,
                stop=config.stop_sequences if config.stop_sequences else None,
            )
        except Exception as e:
            raise ModelGenerationError(
                f"Failed to build sampling params: {str(e)}"
            ) from e
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> ModelResponse:
        """
        Generate text using vLLM.
        
        This method:
        1. Formats chat with model-specific template
        2. Builds vLLM sampling parameters
        3. Generates via vLLM
        4. Parses CoT reasoning (auto)
        5. Returns structured response
        
        Args:
            user_prompt: User input
            system_prompt: System instruction
            config: Generation config
            
        Returns:
            ModelResponse with raw text, reasoning, and final answer
            
        Raises:
            ModelGenerationError: If generation fails
        """
        start_time = time.time()
        
        try:
            # Format chat
            formatted_prompt = self._format_chat(user_prompt, system_prompt)
            
            # Build sampling params
            sampling_params = self._build_sampling_params(config)
            
            # Generate
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            # Extract text from first (and only) output
            generated_text = outputs[0].outputs[0].text
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse CoT reasoning
            from med_red_team.models.utils.response_utils import parse_cot_response
            from med_red_team.models import MalformedResponseError

            # Try to parse, catching malformed response errors
            reasoning = ""
            final_answer = generated_text  # Fallback: use raw text as answer
            error_info = None

            try:
                reasoning, final_answer = parse_cot_response(generated_text, raise_on_malformed=True)
            except MalformedResponseError as e:
                # Store error info but don't crash - allow graceful handling
                reasoning = e.details.get("truncated_reasoning", "")
                final_answer = ""  # Empty since model never finished thinking
                error_info = {
                    "error_type": e.error_type,
                    "error_message": str(e),
                    "details": e.details
                }

            # Build metadata
            metadata = {
                "latency_ms": latency_ms,
                "finish_reason": outputs[0].outputs[0].finish_reason,
                "generated_tokens": len(outputs[0].outputs[0].token_ids),
            }

            # Add error info to metadata if present
            if error_info:
                metadata["parse_error"] = error_info
            
            return ModelResponse(
                raw_text=generated_text,
                reasoning=reasoning,
                final_answer=final_answer,
                model_id=self.model_id,
                metadata=metadata,
                generation_config=config
            )
            
        except Exception as e:
            raise ModelGenerationError(
                f"Failed to generate response from {self.model_id}: {str(e)}"
            ) from e