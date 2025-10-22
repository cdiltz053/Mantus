# Mantus Building Manual: Phase 1-2
## Environment Setup, Infrastructure, and LLM Integration

**Author:** Manus AI  
**Version:** 1.0  
**Date:** October 22, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Phase 1: Development Environment Setup](#phase-1-development-environment-setup)
3. [Phase 2: LLM Integration and Core Intelligence](#phase-2-llm-integration-and-core-intelligence)
4. [References](#references)

---

## Introduction

Building Mantus from the ground up requires a systematic, production-grade approach. This manual covers the first two critical phases: establishing a robust development environment and integrating a state-of-the-art Large Language Model (LLM) as Mantus's core intelligence.

These phases form the foundation upon which all subsequent capabilities—tool orchestration, memory systems, planning mechanisms, self-correction, multimodal integration, and continuous learning—will be built. A well-executed Phase 1-2 ensures that Mantus's "neural heart" is properly configured, secured, and ready for advanced feature development.

### Key Principles

**Modularity:** Each component should be independently testable and replaceable.  
**Scalability:** Infrastructure must support growth from prototype to production deployment.  
**Observability:** Every operation should be logged, monitored, and debuggable.  
**Security:** API keys, credentials, and sensitive data must be handled with care.  
**Reproducibility:** All setup steps should be documented and automated where possible.

---

## Phase 1: Development Environment Setup

### 1.1 Prerequisites and System Requirements

Before beginning, ensure your system meets the following requirements:

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS or later |
| **Python** | 3.9 | 3.11 or later |
| **RAM** | 16 GB | 32 GB or more |
| **GPU** (Optional) | NVIDIA with CUDA 11.8+ | NVIDIA A100 or H100 |
| **Disk Space** | 50 GB | 200 GB+ |
| **Docker** | 20.10+ | Latest stable version |

### 1.2 Repository Initialization and Structure

Begin by cloning the Mantus repository and establishing the project structure:

```bash
# Clone the Mantus repository
gh repo clone cdiltz053/Mantus
cd Mantus

# Create necessary directories for Phase 1-2
mkdir -p config logs data/cache data/models
mkdir -p tests/unit tests/integration
mkdir -p docs/phase_1_2
```

The project structure should follow this organization:

```
Mantus/
├── core/                      # Core Mantus components
│   ├── main.py               # Main entry point
│   ├── llm_component.py       # LLM wrapper and integration
│   ├── task_manager.py        # Task orchestration
│   └── communication_manager.py
├── config/                    # Configuration files
│   ├── llm_config.yaml        # LLM configuration
│   ├── environment.example    # Environment variables template
│   └── logging_config.yaml    # Logging configuration
├── tests/                     # Test suites
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── .env.example              # Environment variables template
```

### 1.3 Python Virtual Environment Setup

Create an isolated Python environment to manage dependencies:

```bash
# Create a Python virtual environment
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate  # On Windows

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel
```

### 1.4 Dependency Installation

Install all required Python packages. Create a `requirements.txt` file with the following dependencies:

```
# Core LLM and AI libraries
openai==1.3.0
anthropic==0.7.0
transformers==4.35.0
torch==2.0.0
numpy==1.24.0
pandas==2.0.0

# Web and API interaction
requests==2.31.0
aiohttp==3.9.0
fastapi==0.104.0
uvicorn==0.24.0

# Data processing and storage
sqlalchemy==2.0.0
redis==5.0.0
pydantic==2.0.0

# Logging and monitoring
python-logging-loki==0.3.2
prometheus-client==0.18.0

# Testing
pytest==7.4.0
pytest-asyncio==0.21.0
pytest-cov==4.1.0

# Code quality
black==23.10.0
flake8==6.1.0
mypy==1.6.0

# Utilities
python-dotenv==1.0.0
pyyaml==6.0
click==8.1.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 1.5 Environment Configuration

Create an `.env` file to store sensitive configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your specific values
nano .env
```

The `.env` file should contain:

```
# LLM Configuration
LLM_PROVIDER=openai  # Options: openai, anthropic, local
LLM_MODEL=gpt-4-turbo
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Local LLM Configuration (if using local models)
LOCAL_MODEL_PATH=/path/to/model
LOCAL_MODEL_NAME=llama-2-70b-chat

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/mantus.log

# System Configuration
MANTUS_ENVIRONMENT=development  # development, staging, production
DEBUG_MODE=true
```

### 1.6 Docker Setup (Optional but Recommended)

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/cache data/models

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run Mantus
CMD ["python", "-m", "uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create a `docker-compose.yml` for local development:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  mantus:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - LOG_LEVEL=DEBUG
    depends_on:
      redis:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    command: python -m uvicorn core.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  redis_data:
```

Start the development environment:

```bash
docker-compose up -d
```

### 1.7 Logging and Monitoring Setup

Configure logging in `config/logging_config.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: detailed
    filename: logs/mantus.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  mantus:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

---

## Phase 2: LLM Integration and Core Intelligence

### 2.1 Understanding LLM Providers

Mantus can integrate with multiple LLM providers. Each has distinct characteristics:

| Provider | Model | Context Window | Cost | Strengths |
|---|---|---|---|---|
| **OpenAI** | GPT-4 Turbo | 128K tokens | $0.01-0.03/1K tokens | Reasoning, instruction-following |
| **Anthropic** | Claude 3.5 Sonnet | 200K tokens | $0.003-0.015/1K tokens | Long context, safety |
| **Meta** | Llama 2 (70B) | 4K tokens | Free (self-hosted) | Open-source, customizable |
| **DeepSeek** | DeepSeek-V3 | 128K tokens | Low-cost | Technical reasoning, efficiency |

For production Mantus deployment, we recommend starting with **Claude 3.5 Sonnet** or **GPT-4 Turbo** due to their superior reasoning capabilities and tool-use support [1][2].

### 2.2 LLM Wrapper Implementation

Create `core/llm_component.py` to abstract LLM interactions:

```python
"""
LLM Component: Abstraction layer for LLM providers
Supports OpenAI, Anthropic, and local models
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Generate text from the LLM"""
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with tool calling capability"""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        openai.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Generate text using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with tool calling"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "tool_calls": response.choices[0].message.tool_calls,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            self.logger.error(f"Error generating with tools: {e}")
            raise


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM implementation"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not provided")
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            raise

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with tool calling"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=system_prompt or "",
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return {
                "content": response.content[0].text if response.content else None,
                "tool_calls": [block for block in response.content if hasattr(block, 'type') and block.type == 'tool_use'],
                "finish_reason": response.stop_reason
            }
        except Exception as e:
            self.logger.error(f"Error generating with tools: {e}")
            raise


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create(config: LLMConfig) -> BaseLLM:
        """Create an LLM instance based on configuration"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAILLM(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicLLM(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create OpenAI LLM
    openai_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo",
        temperature=0.7,
        max_tokens=2048
    )
    llm = LLMFactory.create(openai_config)

    # Simple text generation
    response = llm.generate(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant."
    )
    print(f"Response: {response}")

    # Tool-based generation
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]

    response_with_tools = llm.generate_with_tools(
        prompt="What's the weather in Paris?",
        tools=tools,
        system_prompt="You are a helpful assistant with access to weather tools."
    )
    print(f"Response with tools: {response_with_tools}")
```

### 2.3 Configuration Management

Create `config/llm_config.yaml` for easy LLM configuration:

```yaml
# LLM Configuration for Mantus

# Production configuration
production:
  provider: openai
  model: gpt-4-turbo
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.9
  timeout: 30
  retry_attempts: 3
  retry_delay: 1

# Development configuration
development:
  provider: openai
  model: gpt-4-turbo
  temperature: 0.9
  max_tokens: 2048
  top_p: 0.95
  timeout: 60
  retry_attempts: 5
  retry_delay: 2

# Alternative providers
anthropic:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 4096
  timeout: 30

local:
  provider: local
  model: llama-2-70b-chat
  model_path: /models/llama-2-70b-chat
  temperature: 0.7
  max_tokens: 4096
```

### 2.4 Testing LLM Integration

Create `tests/unit/test_llm_component.py`:

```python
"""
Unit tests for LLM component
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.llm_component import (
    LLMConfig, LLMProvider, OpenAILLM, AnthropicLLM, LLMFactory
)


class TestOpenAILLM:
    """Test OpenAI LLM implementation"""

    @pytest.fixture
    def config(self):
        return LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )

    @patch('openai.OpenAI')
    def test_generate(self, mock_openai, config):
        """Test text generation"""
        # Mock the OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Paris is the capital of France."
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(config)
        response = llm.generate("What is the capital of France?")

        assert response == "Paris is the capital of France."
        mock_openai.return_value.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_generate_with_tools(self, mock_openai, config):
        """Test text generation with tool calling"""
        # Mock the OpenAI response with tool calls
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I'll check the weather for you."
        mock_response.choices[0].message.tool_calls = [
            {"id": "1", "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}}
        ]
        mock_response.choices[0].finish_reason = "tool_calls"
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        llm = OpenAILLM(config)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        response = llm.generate_with_tools("What's the weather?", tools)

        assert response["finish_reason"] == "tool_calls"
        assert len(response["tool_calls"]) > 0


class TestLLMFactory:
    """Test LLM factory"""

    def test_create_openai(self):
        """Test creating OpenAI LLM"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo",
            api_key="test-key"
        )
        with patch('openai.OpenAI'):
            llm = LLMFactory.create(config)
            assert isinstance(llm, OpenAILLM)

    def test_unsupported_provider(self):
        """Test unsupported provider"""
        config = LLMConfig(
            provider="unsupported",
            model="test"
        )
        with pytest.raises(ValueError):
            LLMFactory.create(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run the tests:

```bash
pytest tests/unit/test_llm_component.py -v
```

### 2.5 API Key Management and Security

Implement secure credential management in `core/secrets_manager.py`:

```python
"""
Secrets Manager: Secure handling of API keys and credentials
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class SecretsManager:
    """Manage sensitive credentials securely"""

    @staticmethod
    def get_openai_api_key() -> str:
        """Get OpenAI API key"""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return key

    @staticmethod
    def get_anthropic_api_key() -> str:
        """Get Anthropic API key"""
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return key

    @staticmethod
    def get_redis_url() -> str:
        """Get Redis connection URL"""
        host = os.getenv("REDIS_HOST", "localhost")
        port = os.getenv("REDIS_PORT", "6379")
        db = os.getenv("REDIS_DB", "0")
        return f"redis://{host}:{port}/{db}"

    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> str:
        """Get a secret from environment"""
        value = os.getenv(key)
        if not value and default is None:
            raise ValueError(f"Secret '{key}' not found in environment")
        return value or default
```

---

## References

[1] OpenAI. (2024). "GPT-4 Turbo Model Card." Retrieved from https://openai.com/research/gpt-4

[2] Anthropic. (2024). "Claude 3.5 Sonnet: Introducing the next generation of Claude." Retrieved from https://www.anthropic.com/news/claude-3-5-sonnet

[3] Srinivasan, A. (2025). "Building Production-Ready AI Agents: A Full-Stack Blueprint for Reliability and Scalability." AI with Aish. Retrieved from https://aishwaryasrinivasan.substack.com/p/building-production-ready-ai-agents

[4] Microsoft Azure. (2025). "AI Agent Orchestration Patterns." Azure Architecture Center. Retrieved from https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns

[5] Perrone, P. (2025). "Why Most AI Agents Fail in Production (And How to Build Ones That Don't)." Data Science Collective. Retrieved from https://medium.com/data-science-collective/why-most-ai-agents-fail-in-production-and-how-to-build-ones-that-dont-f6f604bcd075

---

**End of Phase 1-2 Manual**

*Next: Phase 3-4 will cover Tool Orchestration, Memory Systems, and Planning Mechanisms.*

