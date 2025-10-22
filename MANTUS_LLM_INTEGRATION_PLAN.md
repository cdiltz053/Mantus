# Mantus LLM Integration Plan: Initial Phase

## Executive Summary

This document outlines a detailed, step-by-step plan for integrating an open-source Large Language Model (LLM) into Mantus's architecture during the initial phase. The goal is to establish the foundational infrastructure and capabilities that will allow Mantus to function as an autonomous AI agent with LLM-driven reasoning and tool selection.

## Phase Overview

The initial integration phase consists of five sequential phases:

1. **Development Environment Setup**: Prepare the development environment and establish the foundation for LLM integration.
2. **LLM Wrapper Implementation**: Create the abstraction layer that enables Mantus to communicate with the LLM.
3. **Tool Selection Mechanism**: Implement the logic that allows the LLM to select and invoke appropriate tools.
4. **Integration Testing**: Validate that the LLM, tool selection, and agent loop work cohesively.
5. **Documentation and Repository Update**: Document the integration process and commit changes to the repository.

---

## Phase 1: Development Environment Setup

### Objectives

- Establish a clean, isolated development environment for Mantus.
- Clone or initialize the Mantus repository.
- Install necessary dependencies and frameworks.
- Configure the LLM backend (either local or API-based).

### Detailed Steps

#### 1.1 Environment Preparation

**Task**: Set up a Python virtual environment and install core dependencies.

**Steps**:
1. Create a Python 3.11+ virtual environment:
   ```bash
   python3.11 -m venv /path/to/mantus_env
   source /path/to/mantus_env/bin/activate
   ```

2. Upgrade pip, setuptools, and wheel:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. Install core dependencies from `requirements.txt`:
   ```bash
   cd /path/to/Mantus
   pip install -r requirements.txt
   ```

4. Install additional LLM-specific dependencies:
   ```bash
   pip install transformers torch huggingface-hub accelerate bitsandbytes peft
   ```

   - **transformers**: HuggingFace library for loading and using LLMs.
   - **torch**: PyTorch, the deep learning framework used by most modern LLMs.
   - **huggingface-hub**: Tools for downloading and managing models from HuggingFace.
   - **accelerate**: Utilities for distributed inference and optimization.
   - **bitsandbytes**: Quantization library for efficient model loading.
   - **peft**: Parameter-Efficient Fine-Tuning library for adapting models.

#### 1.2 LLM Backend Selection and Configuration

**Task**: Choose and configure the LLM backend for Mantus.

**Options**:

**Option A: Local LLM Deployment (Recommended for Development)**

If deploying a local LLM:

1. Select an open-source LLM from the recommendations in `open_source_llm_comparison.md`:
   - **Llama 3.1** (70B or 8B variant for resource constraints)
   - **Qwen3** (for multilingual support)
   - **DeepSeek-V3** (for reasoning-heavy tasks)

2. Download the model from HuggingFace:
   ```bash
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./models/llama-3.1-8b
   ```

3. Install a local serving framework:
   ```bash
   pip install vllm  # For high-performance inference
   # OR
   pip install ollama  # For simplified local deployment
   ```

4. Configure the model serving:
   - For vLLM:
     ```bash
     python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --port 8000
     ```
   - For Ollama:
     ```bash
     ollama pull llama2  # or another model
     ollama serve
     ```

**Option B: API-Based LLM (Recommended for Production)**

If using an API-based LLM:

1. Choose a provider:
   - **OpenAI** (GPT-4, GPT-3.5-turbo)
   - **Anthropic** (Claude)
   - **Google** (Gemini API)
   - **Together AI** (Open-source models via API)

2. Obtain API credentials and set environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export LLM_PROVIDER="openai"  # or "anthropic", "google", etc.
   ```

3. Install the appropriate SDK:
   ```bash
   pip install openai  # For OpenAI
   pip install anthropic  # For Anthropic
   pip install google-generativeai  # For Google
   ```

#### 1.3 Repository Structure Verification

**Task**: Ensure the Mantus repository has the necessary structure for LLM integration.

**Expected Structure**:
```
Mantus/
├── core/
│   ├── __init__.py
│   ├── main.py
│   ├── task_manager.py
│   ├── communication_manager.py
│   ├── context_manager.py
│   ├── llm_component.py  # To be implemented
│   └── tool_orchestrator.py  # To be implemented
├── tools/
│   ├── __init__.py
│   ├── system/
│   │   ├── shell_tool.py
│   │   └── file_tool.py
│   ├── web/
│   │   ├── search_tool.py
│   │   └── browser_tool.py
│   └── specialized/
│       └── schedule_tool.py
├── config/
│   ├── __init__.py
│   └── llm_config.py  # To be created
├── tests/
│   ├── __init__.py
│   └── test_llm_integration.py  # To be created
├── requirements.txt
├── setup.py
├── ARCHITECTURE.md
├── LLM_TRAINING_GUIDE.md
└── README.md
```

**Verification Steps**:
1. Check that `core/llm_component.py` exists (placeholder).
2. Create `config/llm_config.py` for LLM configuration.
3. Create `tests/test_llm_integration.py` for testing.

#### 1.4 Configuration File Creation

**Task**: Create a configuration file for LLM settings.

**File**: `config/llm_config.py`

```python
"""
LLM Configuration for Mantus
"""

import os
from typing import Optional, Dict, Any

class LLMConfig:
    """Configuration class for LLM settings."""
    
    # LLM Provider: "local", "openai", "anthropic", "google", "together"
    PROVIDER: str = os.getenv("LLM_PROVIDER", "local")
    
    # Model Selection
    if PROVIDER == "local":
        MODEL_NAME: str = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        MODEL_PATH: str = os.getenv("LLM_MODEL_PATH", "./models/llama-3.1-8b")
    elif PROVIDER == "openai":
        MODEL_NAME: str = os.getenv("LLM_MODEL", "gpt-4")
        API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    elif PROVIDER == "anthropic":
        MODEL_NAME: str = os.getenv("LLM_MODEL", "claude-3-opus")
        API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    elif PROVIDER == "google":
        MODEL_NAME: str = os.getenv("LLM_MODEL", "gemini-pro")
        API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    elif PROVIDER == "together":
        MODEL_NAME: str = os.getenv("LLM_MODEL", "meta-llama/Llama-3-70b-chat-hf")
        API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    
    # Inference Parameters
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("LLM_TOP_K", "50"))
    
    # Optimization
    USE_QUANTIZATION: bool = os.getenv("LLM_USE_QUANTIZATION", "false").lower() == "true"
    QUANTIZATION_BITS: int = int(os.getenv("LLM_QUANTIZATION_BITS", "8"))
    USE_FLASH_ATTENTION: bool = os.getenv("LLM_USE_FLASH_ATTENTION", "true").lower() == "true"
    
    # Device Configuration
    DEVICE: str = os.getenv("LLM_DEVICE", "auto")  # "auto", "cuda", "cpu"
    NUM_GPUS: int = int(os.getenv("LLM_NUM_GPUS", "1"))
    
    # Timeout and Retry
    REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
    MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "provider": cls.PROVIDER,
            "model_name": cls.MODEL_NAME,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "top_k": cls.TOP_K,
            "use_quantization": cls.USE_QUANTIZATION,
            "quantization_bits": cls.QUANTIZATION_BITS,
            "device": cls.DEVICE,
        }
```

#### 1.5 Environment Variables Setup

**Task**: Create a `.env` file with LLM configuration.

**File**: `.env`

```bash
# LLM Provider Configuration
LLM_PROVIDER=local  # Options: local, openai, anthropic, google, together
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct

# For Local Deployment
LLM_MODEL_PATH=./models/llama-3.1-8b

# For API-Based Deployment
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
TOGETHER_API_KEY=your-together-api-key

# Inference Parameters
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_TOP_K=50

# Optimization
LLM_USE_QUANTIZATION=false
LLM_QUANTIZATION_BITS=8
LLM_USE_FLASH_ATTENTION=true

# Device Configuration
LLM_DEVICE=auto
LLM_NUM_GPUS=1

# Timeout and Retry
LLM_REQUEST_TIMEOUT=300
LLM_MAX_RETRIES=3
```

#### 1.6 Dependency Installation and Verification

**Task**: Install all dependencies and verify the setup.

**Steps**:
1. Install all requirements:
   ```bash
   pip install -r requirements.txt
   pip install transformers torch huggingface-hub accelerate bitsandbytes peft vllm
   ```

2. Verify installations:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
   ```

3. Test LLM availability:
   ```bash
   # For local deployment
   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('Transformers ready')"
   
   # For API-based deployment
   python -c "import openai; print('OpenAI SDK ready')"
   ```

### Deliverables for Phase 1

- [ ] Python virtual environment created and activated.
- [ ] All dependencies installed and verified.
- [ ] LLM backend configured (local or API-based).
- [ ] `config/llm_config.py` created with proper configuration.
- [ ] `.env` file created with environment variables.
- [ ] Repository structure verified and ready for Phase 2.

---

## Phase 2: LLM Wrapper Implementation

### Objectives

- Create an abstraction layer (`LLMWrapper`) that encapsulates LLM communication.
- Implement methods for prompt construction, inference, and response parsing.
- Support multiple LLM providers (local, OpenAI, Anthropic, etc.).
- Implement error handling, retries, and logging.

### Detailed Steps

#### 2.1 LLM Wrapper Class Design

**File**: `core/llm_component.py`

The LLM Wrapper will be a flexible class that abstracts the details of communicating with different LLM providers. It will:

1. Load the LLM based on the configured provider.
2. Construct prompts for the agent loop.
3. Execute inference and parse responses.
4. Handle errors and retries.
5. Log interactions for debugging and analysis.

#### 2.2 Implementation of LLMWrapper

**File**: `core/llm_component.py`

```python
"""
LLM Component for Mantus
Provides abstraction for communicating with various LLM providers.
"""

import logging
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enumeration of supported LLM providers."""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TOGETHER = "together"


class LLMBase(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM."""
        pass
    
    @abstractmethod
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate text with tool selection capability."""
        pass


class LocalLLM(LLMBase):
    """Local LLM implementation using HuggingFace Transformers."""
    
    def __init__(self, model_name: str, device: str = "auto", use_quantization: bool = False):
        """
        Initialize the local LLM.
        
        Args:
            model_name: HuggingFace model identifier.
            device: Device to load the model on ("auto", "cuda", "cpu").
            use_quantization: Whether to use quantization for efficiency.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Loading local LLM: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with quantization if specified
            if use_quantization:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
            
            self.device = device
            logger.info(f"Local LLM loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using the local LLM."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate text with tool selection."""
        # Augment prompt with tool descriptions
        tool_descriptions = self._format_tools(tools)
        augmented_prompt = f"{prompt}\n\nAvailable tools:\n{tool_descriptions}\n\nRespond with your reasoning and selected tool (if any)."
        
        response = self.generate(augmented_prompt, **kwargs)
        
        # Parse tool selection from response
        tool_call = self._parse_tool_call(response, tools)
        
        return response, tool_call
    
    @staticmethod
    def _format_tools(tools: List[Dict[str, Any]]) -> str:
        """Format tools for inclusion in the prompt."""
        formatted = ""
        for i, tool in enumerate(tools, 1):
            formatted += f"{i}. {tool['name']}: {tool['description']}\n"
        return formatted
    
    @staticmethod
    def _parse_tool_call(response: str, tools: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse tool call from the LLM response."""
        # Simple heuristic: look for tool names in the response
        for tool in tools:
            if tool['name'].lower() in response.lower():
                return {
                    "tool_name": tool['name'],
                    "tool_id": tool.get('id'),
                    "arguments": {}  # To be enhanced with argument extraction
                }
        return None


class OpenAILLM(LLMBase):
    """OpenAI LLM implementation."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model_name: OpenAI model identifier.
            api_key: OpenAI API key.
        """
        try:
            import openai
            
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info(f"OpenAI LLM initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text via OpenAI: {e}")
            raise
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate text with tool selection using OpenAI's function calling."""
        try:
            # Convert tools to OpenAI format
            openai_tools = self._convert_tools_to_openai_format(tools)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                tools=openai_tools,
                tool_choice="auto",
                max_tokens=kwargs.get("max_tokens", 2048),
                temperature=kwargs.get("temperature", 0.7),
            )
            
            response_text = response.choices[0].message.content or ""
            tool_call = None
            
            if response.choices[0].message.tool_calls:
                tool_call_obj = response.choices[0].message.tool_calls[0]
                tool_call = {
                    "tool_name": tool_call_obj.function.name,
                    "arguments": json.loads(tool_call_obj.function.arguments)
                }
            
            return response_text, tool_call
            
        except Exception as e:
            logger.error(f"Error generating text with tools via OpenAI: {e}")
            raise
    
    @staticmethod
    def _convert_tools_to_openai_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tool definitions to OpenAI function calling format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool.get('parameters', {})
                }
            })
        return openai_tools


class LLMWrapper:
    """
    Main wrapper class for LLM interaction.
    Abstracts provider-specific details and provides a unified interface.
    """
    
    def __init__(self, provider: str = "local", **config):
        """
        Initialize the LLM Wrapper.
        
        Args:
            provider: LLM provider ("local", "openai", "anthropic", etc.).
            **config: Provider-specific configuration.
        """
        self.provider = LLMProvider(provider)
        self.config = config
        self.llm = self._initialize_llm()
        logger.info(f"LLMWrapper initialized with provider: {provider}")
    
    def _initialize_llm(self) -> LLMBase:
        """Initialize the appropriate LLM based on provider."""
        if self.provider == LLMProvider.LOCAL:
            return LocalLLM(
                model_name=self.config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct"),
                device=self.config.get("device", "auto"),
                use_quantization=self.config.get("use_quantization", False)
            )
        elif self.provider == LLMProvider.OPENAI:
            return OpenAILLM(
                model_name=self.config.get("model_name", "gpt-4"),
                api_key=self.config.get("api_key")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        """Generate text from the LLM."""
        try:
            response = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
            logger.debug(f"Generated response (length: {len(response)} chars)")
            return response
        except Exception as e:
            logger.error(f"Error in LLMWrapper.generate: {e}")
            raise
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate text with tool selection."""
        try:
            response, tool_call = self.llm.generate_with_tools(prompt, tools, **kwargs)
            logger.debug(f"Generated response with tool call: {tool_call}")
            return response, tool_call
        except Exception as e:
            logger.error(f"Error in LLMWrapper.generate_with_tools: {e}")
            raise
    
    def construct_agent_prompt(self, user_input: str, context: Dict[str, Any], available_tools: List[str]) -> str:
        """
        Construct a prompt for the agent loop.
        
        Args:
            user_input: The user's input/request.
            context: Current context (task, phase, history).
            available_tools: List of available tool names.
        
        Returns:
            Formatted prompt for the LLM.
        """
        prompt = f"""You are Mantus, an autonomous AI agent. Your role is to help the user accomplish their goals by reasoning, planning, and using available tools.

Current Task: {context.get('task', 'Not specified')}
Current Phase: {context.get('phase', 'Not specified')}

User Input: {user_input}

Available Tools: {', '.join(available_tools)}

Please analyze the user's input, reason about the best approach, and decide whether to use a tool or provide a direct response. If using a tool, specify which tool and any necessary arguments."""
        
        return prompt
```

### Deliverables for Phase 2

- [ ] `core/llm_component.py` implemented with LLMWrapper, LocalLLM, and OpenAILLM classes.
- [ ] Configuration system in place (`config/llm_config.py`).
- [ ] Logging configured for debugging and monitoring.
- [ ] Support for multiple LLM providers (local, OpenAI, extensible for others).
- [ ] Methods for prompt construction and inference.
- [ ] Error handling and retry logic implemented.

---

## Phase 3: Tool Selection Mechanism

### Objectives

- Implement the `ToolOrchestrator` class that manages tool invocation.
- Create tool wrapper classes for each tool category.
- Integrate the LLM's tool selection with the ToolOrchestrator.
- Enable the agent loop to execute selected tools and process results.

### Detailed Steps

#### 3.1 Tool Definition and Schema

**File**: `tools/tool_registry.py`

```python
"""
Tool Registry for Mantus
Maintains a registry of available tools and their schemas.
"""

from typing import Dict, Any, List, Callable
import logging

logger = logging.getLogger(__name__)


class ToolSchema:
    """Defines the schema for a tool."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable):
        """
        Initialize a tool schema.
        
        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON schema for tool parameters.
            handler: Callable that executes the tool.
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolSchema] = {}
    
    def register_tool(self, tool_schema: ToolSchema) -> None:
        """Register a tool."""
        self.tools[tool_schema.name] = tool_schema
        logger.info(f"Tool registered: {tool_schema.name}")
    
    def get_tool(self, name: str) -> ToolSchema:
        """Get a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.to_dict() for tool in self.tools.values()]
```

#### 3.2 ToolOrchestrator Implementation

**File**: `core/tool_orchestrator.py`

```python
"""
Tool Orchestrator for Mantus
Manages tool invocation and execution.
"""

import logging
from typing import Dict, Any, Optional, List
from tools.tool_registry import ToolRegistry, ToolSchema

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """Orchestrates tool selection and execution."""
    
    def __init__(self, tool_registry: ToolRegistry):
        """
        Initialize the ToolOrchestrator.
        
        Args:
            tool_registry: Registry of available tools.
        """
        self.tool_registry = tool_registry
        logger.info("ToolOrchestrator initialized")
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
        
        Returns:
            Result of tool execution.
        """
        try:
            tool_schema = self.tool_registry.get_tool(tool_name)
            logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
            
            result = tool_schema.handler(**arguments)
            logger.info(f"Tool execution successful: {tool_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Validate a tool call against its schema.
        
        Args:
            tool_name: Name of the tool.
            arguments: Arguments to validate.
        
        Returns:
            True if valid, False otherwise.
        """
        try:
            tool_schema = self.tool_registry.get_tool(tool_name)
            # Simple validation: check that all required parameters are present
            required_params = tool_schema.parameters.get("required", [])
            for param in required_params:
                if param not in arguments:
                    logger.warning(f"Missing required parameter: {param}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error validating tool call: {e}")
            return False
```

#### 3.3 Integration with Agent Loop

**File**: `core/main.py`

```python
"""
Main Agent Loop for Mantus
"""

import logging
from typing import Dict, Any, Optional
from core.llm_component import LLMWrapper
from core.tool_orchestrator import ToolOrchestrator
from core.context_manager import ContextManager
from core.communication_manager import CommunicationManager
from core.task_manager import TaskManager
from tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class MantusAgent:
    """Main Mantus Agent class."""
    
    def __init__(self, llm_config: Dict[str, Any], tool_registry: ToolRegistry):
        """
        Initialize the Mantus Agent.
        
        Args:
            llm_config: Configuration for the LLM.
            tool_registry: Registry of available tools.
        """
        self.llm = LLMWrapper(**llm_config)
        self.tool_orchestrator = ToolOrchestrator(tool_registry)
        self.context_manager = ContextManager()
        self.communication_manager = CommunicationManager()
        self.task_manager = TaskManager()
        self.tool_registry = tool_registry
        
        logger.info("Mantus Agent initialized")
    
    def agent_loop(self, user_input: str) -> str:
        """
        Main agent loop.
        
        Args:
            user_input: User's input/request.
        
        Returns:
            Agent's response to the user.
        """
        # Step 1: Analyze Context
        context = self.context_manager.get_current_context()
        logger.info(f"Current context: {context}")
        
        # Step 2: Think (LLM reasoning)
        available_tools = self.tool_registry.list_tools()
        prompt = self.llm.construct_agent_prompt(user_input, context, available_tools)
        
        logger.info("LLM thinking...")
        response, tool_call = self.llm.generate_with_tools(
            prompt,
            self.tool_registry.get_tool_schemas()
        )
        
        # Step 3: Select Tool (if needed)
        if tool_call:
            logger.info(f"Tool selected: {tool_call['tool_name']}")
            
            # Step 4: Execute Action
            tool_result = self.tool_orchestrator.execute_tool(
                tool_call['tool_name'],
                tool_call.get('arguments', {})
            )
            
            # Step 5: Receive Observation
            self.context_manager.add_observation(tool_call['tool_name'], tool_result)
            
            # Process result and generate final response
            final_response = self._process_tool_result(response, tool_result)
        else:
            final_response = response
        
        # Step 6: Deliver Outcome
        logger.info(f"Agent response: {final_response}")
        return final_response
    
    def _process_tool_result(self, llm_response: str, tool_result: Any) -> str:
        """Process tool result and generate final response."""
        # Simple processing: combine LLM response with tool result
        return f"{llm_response}\n\nTool Result:\n{tool_result}"
```

### Deliverables for Phase 3

- [ ] `tools/tool_registry.py` implemented with ToolSchema and ToolRegistry.
- [ ] `core/tool_orchestrator.py` implemented with ToolOrchestrator class.
- [ ] Integration of LLM with tool selection in the agent loop.
- [ ] Tool execution and result processing implemented.
- [ ] Error handling for tool execution.

---

## Phase 4: Integration Testing

### Objectives

- Test the LLM wrapper with different providers.
- Test tool selection and execution.
- Test the complete agent loop.
- Identify and fix issues.

### Detailed Steps

#### 4.1 Unit Tests

**File**: `tests/test_llm_integration.py`

```python
"""
Unit tests for LLM integration.
"""

import unittest
from core.llm_component import LLMWrapper, LocalLLM, OpenAILLM
from core.tool_orchestrator import ToolOrchestrator
from tools.tool_registry import ToolRegistry, ToolSchema


class TestLLMWrapper(unittest.TestCase):
    """Test cases for LLMWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        # This would be configured based on available providers
        pass
    
    def test_local_llm_initialization(self):
        """Test local LLM initialization."""
        # Test code here
        pass
    
    def test_openai_llm_initialization(self):
        """Test OpenAI LLM initialization."""
        # Test code here
        pass
    
    def test_generate_text(self):
        """Test text generation."""
        # Test code here
        pass
    
    def test_generate_with_tools(self):
        """Test text generation with tool selection."""
        # Test code here
        pass


class TestToolOrchestrator(unittest.TestCase):
    """Test cases for ToolOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tool_registry = ToolRegistry()
        
        # Register a simple test tool
        def test_tool_handler(x: int, y: int) -> int:
            return x + y
        
        test_tool = ToolSchema(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"}
                },
                "required": ["x", "y"]
            },
            handler=test_tool_handler
        )
        self.tool_registry.register_tool(test_tool)
        self.orchestrator = ToolOrchestrator(self.tool_registry)
    
    def test_execute_tool(self):
        """Test tool execution."""
        result = self.orchestrator.execute_tool("add", {"x": 2, "y": 3})
        self.assertEqual(result, 5)
    
    def test_validate_tool_call(self):
        """Test tool call validation."""
        is_valid = self.orchestrator.validate_tool_call("add", {"x": 2, "y": 3})
        self.assertTrue(is_valid)
    
    def test_invalid_tool_call(self):
        """Test invalid tool call."""
        is_valid = self.orchestrator.validate_tool_call("add", {"x": 2})
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
```

#### 4.2 Integration Tests

**File**: `tests/test_agent_integration.py`

```python
"""
Integration tests for the Mantus Agent.
"""

import unittest
from core.main import MantusAgent
from tools.tool_registry import ToolRegistry, ToolSchema


class TestMantusAgent(unittest.TestCase):
    """Test cases for the Mantus Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tool_registry = ToolRegistry()
        
        # Register test tools
        def search_tool_handler(query: str) -> str:
            return f"Search results for: {query}"
        
        search_tool = ToolSchema(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            },
            handler=search_tool_handler
        )
        self.tool_registry.register_tool(search_tool)
        
        # Initialize agent with local LLM
        llm_config = {
            "provider": "local",
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",
            "device": "cpu"  # Use CPU for testing
        }
        
        try:
            self.agent = MantusAgent(llm_config, self.tool_registry)
        except Exception as e:
            self.skipTest(f"Could not initialize agent: {e}")
    
    def test_agent_loop_without_tools(self):
        """Test agent loop without tool usage."""
        response = self.agent.agent_loop("What is 2 + 2?")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
    
    def test_agent_loop_with_tools(self):
        """Test agent loop with tool usage."""
        response = self.agent.agent_loop("Search for information about Python programming")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)


if __name__ == "__main__":
    unittest.main()
```

#### 4.3 Running Tests

**Steps**:
```bash
# Run unit tests
python -m pytest tests/test_llm_integration.py -v

# Run integration tests
python -m pytest tests/test_agent_integration.py -v

# Run all tests
python -m pytest tests/ -v
```

### Deliverables for Phase 4

- [ ] Unit tests for LLMWrapper implemented and passing.
- [ ] Unit tests for ToolOrchestrator implemented and passing.
- [ ] Integration tests for MantusAgent implemented and passing.
- [ ] All tests passing with local and API-based LLM providers.
- [ ] Issues identified and fixed.

---

## Phase 5: Documentation and Repository Update

### Objectives

- Document the LLM integration process.
- Update the repository with implementation details.
- Create a setup guide for users.
- Commit all changes to the repository.

### Detailed Steps

#### 5.1 Create Integration Documentation

**File**: `INTEGRATION_GUIDE.md`

This document should cover:
- Overview of the LLM integration.
- Supported LLM providers.
- Setup instructions for each provider.
- Configuration options.
- Usage examples.
- Troubleshooting guide.

#### 5.2 Update README

**File**: `README.md`

Update the README to include:
- Quick start guide for LLM integration.
- Links to detailed documentation.
- Examples of using Mantus with different LLM providers.

#### 5.3 Update Requirements

**File**: `requirements.txt`

Ensure all LLM-related dependencies are listed:
```
transformers>=4.30.0
torch>=2.0.0
huggingface-hub>=0.16.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
peft>=0.4.0
vllm>=0.1.0
openai>=1.0.0
```

#### 5.4 Commit to Repository

**Steps**:
```bash
cd Mantus
git add core/llm_component.py core/tool_orchestrator.py core/main.py
git add tools/tool_registry.py config/llm_config.py
git add tests/test_llm_integration.py tests/test_agent_integration.py
git add INTEGRATION_GUIDE.md README.md requirements.txt
git add .env.example
git commit -m "Implement LLM integration with tool selection and agent loop"
git push
```

### Deliverables for Phase 5

- [ ] `INTEGRATION_GUIDE.md` created with comprehensive documentation.
- [ ] `README.md` updated with LLM integration information.
- [ ] `requirements.txt` updated with all dependencies.
- [ ] `.env.example` created as a template for configuration.
- [ ] All changes committed and pushed to the repository.
- [ ] Repository is ready for Phase 2 (Enhanced Capabilities Integration).

---

## Success Criteria

The initial LLM integration phase is considered successful when:

1. **LLM Wrapper**: The LLMWrapper class successfully initializes and communicates with at least one LLM provider (local or API-based).

2. **Tool Selection**: The LLM can correctly identify and select appropriate tools based on user input and context.

3. **Tool Execution**: Selected tools are executed correctly, and results are processed and returned to the user.

4. **Agent Loop**: The complete agent loop (Analyze Context → Think → Select Tool → Execute Action → Receive Observation → Deliver Outcome) functions correctly.

5. **Testing**: All unit and integration tests pass without errors.

6. **Documentation**: Comprehensive documentation is available for users to set up and use Mantus with their chosen LLM provider.

7. **Repository**: All code is committed to the GitHub repository with clear commit messages and is ready for further development.

---

## Next Steps

After completing the initial LLM integration phase, the following phases should be pursued:

1. **Enhanced Capabilities Integration**: Integrate long-term memory systems (vector databases, knowledge graphs), proactive planning, and self-correction mechanisms as outlined in `MANUS_LIMITATIONS_AND_MANTUS_ENHANCEMENTS.md`.

2. **Fine-Tuning and RLHF**: Implement fine-tuning and Reinforcement Learning from Human Feedback (RLHF) to align Mantus's LLM with desired behaviors.

3. **Production Deployment**: Deploy Mantus in a production environment with proper monitoring, logging, and error handling.

4. **Continuous Improvement**: Collect user feedback and continuously improve Mantus's capabilities and performance.

---

## Conclusion

This detailed plan provides a comprehensive roadmap for integrating an open-source LLM into Mantus's architecture. By following these steps, you will establish the foundational infrastructure for Mantus to function as an autonomous AI agent with advanced reasoning and tool selection capabilities. The modular design ensures that each phase builds upon the previous one, allowing for iterative development and testing.

