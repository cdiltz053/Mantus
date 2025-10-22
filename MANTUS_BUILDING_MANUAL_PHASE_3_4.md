# Mantus Building Manual: Phase 3-4
## Tool Orchestration, Memory Systems, and Planning Mechanisms

**Author:** Manus AI  
**Version:** 1.0  
**Date:** October 22, 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Phase 3: Tool Orchestration and Integration](#phase-3-tool-orchestration-and-integration)
3. [Phase 4: Memory Systems and Knowledge Management](#phase-4-memory-systems-and-knowledge-management)
4. [References](#references)

---

## Introduction

Phases 3 and 4 transform Mantus from a simple LLM wrapper into a sophisticated autonomous agent capable of reasoning, planning, and remembering. These phases implement the critical infrastructure that enables Mantus to interact with external systems, maintain context across conversations, and make informed decisions based on accumulated knowledge.

### Key Capabilities Enabled

**Tool Orchestration:** Mantus can now invoke external tools, APIs, and services in response to user requests.  
**Memory Management:** Both short-term (episodic) and long-term (semantic) memory enable context retention and learning.  
**Planning and Reasoning:** Mantus can decompose complex goals into actionable steps and execute them strategically.

---

## Phase 3: Tool Orchestration and Integration

### 3.1 Tool Registry and Schema Definition

Create `core/tool_registry.py` to manage available tools:

```python
"""
Tool Registry: Central management of available tools and their schemas
"""

import json
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools"""
    SYSTEM = "system"
    WEB = "web"
    DATA = "data"
    MEDIA = "media"
    SPECIALIZED = "specialized"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Optional[Any] = None


@dataclass
class ToolSchema:
    """Schema definition for a tool"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: str  # Description of return value
    examples: Optional[List[Dict[str, Any]]] = None


class Tool:
    """Base class for tools"""

    def __init__(
        self,
        schema: ToolSchema,
        handler: Callable,
        timeout: int = 30
    ):
        self.schema = schema
        self.handler = handler
        self.timeout = timeout
        self.logger = logging.getLogger(f"Tool.{schema.name}")

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        try:
            # Validate parameters
            self._validate_parameters(kwargs)
            
            # Execute the handler
            result = await self.handler(**kwargs)
            
            self.logger.info(f"Tool '{self.schema.name}' executed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool: {e}")
            raise

    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate parameters against schema"""
        for param in self.schema.parameters:
            if param.required and param.name not in params:
                raise ValueError(f"Missing required parameter: {param.name}")
            
            if param.name in params:
                value = params[param.name]
                if not self._validate_type(value, param.type):
                    raise TypeError(
                        f"Parameter '{param.name}' has incorrect type. "
                        f"Expected {param.type}, got {type(value).__name__}"
                    )

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        expected = type_mapping.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool schema to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.schema.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


class ToolRegistry:
    """Central registry for all available tools"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.logger = logging.getLogger("ToolRegistry")

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        if tool.schema.name in self.tools:
            self.logger.warning(f"Tool '{tool.schema.name}' already registered. Overwriting.")
        
        self.tools[tool.schema.name] = tool
        self.logger.info(f"Tool '{tool.schema.name}' registered successfully")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category"""
        return [
            tool for tool in self.tools.values()
            if tool.schema.category == category
        ]

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in OpenAI format"""
        return [tool.to_openai_schema() for tool in self.tools.values()]

    def list_tools(self) -> Dict[str, str]:
        """List all tools with descriptions"""
        return {
            name: tool.schema.description
            for name, tool in self.tools.items()
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry"""
    get_registry().register(tool)
```

### 3.2 Tool Orchestrator

Create `core/tool_orchestrator.py` to manage tool invocation:

```python
"""
Tool Orchestrator: Manages tool selection and execution
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
import logging

from core.tool_registry import get_registry, Tool
from core.llm_component import BaseLLM

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """Orchestrates tool selection and execution"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.registry = get_registry()
        self.logger = logging.getLogger("ToolOrchestrator")
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_with_tools(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5
    ) -> str:
        """
        Execute a task using tools iteratively
        
        This implements an agentic loop where:
        1. LLM analyzes the task and decides which tools to use
        2. Tools are executed
        3. Results are fed back to the LLM
        4. Process repeats until task is complete
        """
        conversation_history = []
        
        if system_prompt:
            conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
        
        conversation_history.append({
            "role": "user",
            "content": user_prompt
        })

        for iteration in range(max_iterations):
            self.logger.info(f"Iteration {iteration + 1}/{max_iterations}")

            # Get available tools
            tools = self.registry.get_openai_schemas()

            # Call LLM with tools
            try:
                response = self.llm.generate_with_tools(
                    prompt=user_prompt,
                    tools=tools,
                    system_prompt=system_prompt
                )
            except Exception as e:
                self.logger.error(f"Error calling LLM: {e}")
                return f"Error: {str(e)}"

            # Check if LLM wants to use tools
            if response.get("finish_reason") != "tool_calls" or not response.get("tool_calls"):
                # LLM has finished
                return response.get("content", "No response")

            # Execute tool calls
            tool_results = []
            for tool_call in response.get("tool_calls", []):
                try:
                    result = await self._execute_tool_call(tool_call)
                    tool_results.append({
                        "tool": tool_call.function.name,
                        "result": result,
                        "success": True
                    })
                except Exception as e:
                    self.logger.error(f"Error executing tool: {e}")
                    tool_results.append({
                        "tool": tool_call.function.name,
                        "result": str(e),
                        "success": False
                    })

            # Add tool results to conversation
            conversation_history.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": response.get("tool_calls")
            })

            # Format tool results for LLM
            results_text = self._format_tool_results(tool_results)
            conversation_history.append({
                "role": "user",
                "content": f"Tool execution results:\n{results_text}"
            })

            # Store execution in history
            self.execution_history.append({
                "iteration": iteration + 1,
                "tool_calls": [tc.function.name for tc in response.get("tool_calls", [])],
                "results": tool_results
            })

        return "Max iterations reached without completion"

    async def _execute_tool_call(self, tool_call: Any) -> Any:
        """Execute a single tool call"""
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        tool = self.registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        # Execute tool
        result = await tool.execute(**tool_args)
        
        return result

    def _format_tool_results(self, results: List[Dict[str, Any]]) -> str:
        """Format tool results for LLM consumption"""
        formatted = []
        for result in results:
            status = "✓" if result["success"] else "✗"
            formatted.append(
                f"{status} {result['tool']}: {result['result']}"
            )
        return "\n".join(formatted)

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history"""
        return self.execution_history

    def clear_history(self) -> None:
        """Clear execution history"""
        self.execution_history = []
```

### 3.3 Built-in Tools Implementation

Create `core/builtin_tools.py` with essential tools:

```python
"""
Built-in Tools: Essential tools for Mantus
"""

import asyncio
import subprocess
from typing import Dict, Any
import logging

from core.tool_registry import (
    Tool, ToolSchema, ToolParameter, ToolCategory, register_tool
)

logger = logging.getLogger(__name__)


# Shell Execution Tool
async def execute_shell_command(command: str, timeout: int = 30) -> str:
    """Execute a shell command"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return f"Exit code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error: {str(e)}"


shell_tool = Tool(
    schema=ToolSchema(
        name="execute_shell",
        description="Execute a shell command and return the output",
        category=ToolCategory.SYSTEM,
        parameters=[
            ToolParameter(
                name="command",
                type="string",
                description="The shell command to execute",
                required=True
            ),
            ToolParameter(
                name="timeout",
                type="number",
                description="Timeout in seconds",
                required=False,
                default=30
            )
        ],
        returns="Command output and exit code"
    ),
    handler=execute_shell_command
)


# File Reading Tool
async def read_file(filepath: str) -> str:
    """Read a file and return its contents"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


file_read_tool = Tool(
    schema=ToolSchema(
        name="read_file",
        description="Read the contents of a file",
        category=ToolCategory.SYSTEM,
        parameters=[
            ToolParameter(
                name="filepath",
                type="string",
                description="Path to the file to read",
                required=True
            )
        ],
        returns="File contents as string"
    ),
    handler=read_file
)


# Web Search Tool (placeholder)
async def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information"""
    # This is a placeholder - in production, integrate with a search API
    return f"Search results for '{query}' (placeholder - integrate with real search API)"


web_search_tool = Tool(
    schema=ToolSchema(
        name="web_search",
        description="Search the web for information",
        category=ToolCategory.WEB,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            ),
            ToolParameter(
                name="num_results",
                type="number",
                description="Number of results to return",
                required=False,
                default=5
            )
        ],
        returns="Search results"
    ),
    handler=web_search
)


# Register all built-in tools
def register_builtin_tools():
    """Register all built-in tools"""
    register_tool(shell_tool)
    register_tool(file_read_tool)
    register_tool(web_search_tool)
    logger.info("Built-in tools registered")
```

---

## Phase 4: Memory Systems and Knowledge Management

### 4.1 Memory Architecture

Create `core/memory.py` for memory management:

```python
"""
Memory System: Episodic and semantic memory for Mantus
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

import redis

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    content: str
    timestamp: float
    type: str  # "episodic" or "semantic"
    metadata: Dict[str, Any]
    ttl: Optional[int] = None  # Time to live in seconds


class EpisodicMemory:
    """Short-term, session-specific memory"""

    def __init__(self, redis_client: redis.Redis, session_id: str):
        self.redis = redis_client
        self.session_id = session_id
        self.logger = logging.getLogger("EpisodicMemory")
        self.prefix = f"episodic:{session_id}:"

    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: int = 3600  # Default 1 hour
    ) -> str:
        """Store an episodic memory"""
        memory_id = f"{self.prefix}{int(time.time() * 1000)}"
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            timestamp=time.time(),
            type="episodic",
            metadata=metadata or {},
            ttl=ttl
        )
        
        self.redis.setex(
            memory_id,
            ttl,
            json.dumps(asdict(entry))
        )
        
        self.logger.info(f"Stored episodic memory: {memory_id}")
        return memory_id

    async def retrieve(self, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve recent episodic memories"""
        pattern = f"{self.prefix}*"
        keys = self.redis.keys(pattern)
        
        entries = []
        for key in keys[-limit:]:  # Get most recent
            data = self.redis.get(key)
            if data:
                entry_dict = json.loads(data)
                entries.append(MemoryEntry(**entry_dict))
        
        return sorted(entries, key=lambda x: x.timestamp, reverse=True)

    async def clear(self) -> None:
        """Clear all episodic memories for this session"""
        pattern = f"{self.prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
        self.logger.info(f"Cleared episodic memory for session {self.session_id}")


class SemanticMemory:
    """Long-term, persistent knowledge memory"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger("SemanticMemory")
        self.prefix = "semantic:"

    async def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store semantic knowledge"""
        entry = MemoryEntry(
            id=f"{self.prefix}{key}",
            content=content,
            timestamp=time.time(),
            type="semantic",
            metadata=metadata or {}
        )
        
        self.redis.set(
            f"{self.prefix}{key}",
            json.dumps(asdict(entry))
        )
        
        self.logger.info(f"Stored semantic memory: {key}")

    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve semantic knowledge"""
        data = self.redis.get(f"{self.prefix}{key}")
        if data:
            return MemoryEntry(**json.loads(data))
        return None

    async def search(self, pattern: str) -> List[MemoryEntry]:
        """Search semantic memories by pattern"""
        keys = self.redis.keys(f"{self.prefix}{pattern}*")
        
        entries = []
        for key in keys:
            data = self.redis.get(key)
            if data:
                entries.append(MemoryEntry(**json.loads(data)))
        
        return entries

    async def update(self, key: str, content: str) -> None:
        """Update semantic knowledge"""
        existing = await self.retrieve(key)
        if existing:
            existing.content = content
            existing.timestamp = time.time()
            self.redis.set(
                f"{self.prefix}{key}",
                json.dumps(asdict(existing))
            )
            self.logger.info(f"Updated semantic memory: {key}")


class MemoryManager:
    """Unified memory management"""

    def __init__(self, redis_url: str, session_id: str):
        self.redis = redis.from_url(redis_url)
        self.session_id = session_id
        self.episodic = EpisodicMemory(self.redis, session_id)
        self.semantic = SemanticMemory(self.redis)
        self.logger = logging.getLogger("MemoryManager")

    async def store_conversation(
        self,
        role: str,
        message: str
    ) -> None:
        """Store a conversation turn"""
        await self.episodic.store(
            content=message,
            metadata={"role": role},
            ttl=86400  # 24 hours
        )

    async def get_conversation_history(self, limit: int = 20) -> List[Dict[str, str]]:
        """Get recent conversation history"""
        entries = await self.episodic.retrieve(limit=limit)
        return [
            {"role": entry.metadata.get("role", "unknown"), "content": entry.content}
            for entry in entries
        ]

    async def store_knowledge(
        self,
        key: str,
        value: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store persistent knowledge"""
        await self.semantic.store(key, value, metadata)

    async def retrieve_knowledge(self, key: str) -> Optional[str]:
        """Retrieve persistent knowledge"""
        entry = await self.semantic.retrieve(key)
        return entry.content if entry else None
```

### 4.2 Planning and Task Decomposition

Create `core/planner.py` for task planning:

```python
"""
Planner: Task decomposition and planning
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from core.llm_component import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task or subtask"""
    id: str
    description: str
    dependencies: List[str]  # IDs of tasks that must complete first
    estimated_duration: int  # seconds
    priority: int  # 1-10, higher is more important
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None


class TaskPlanner:
    """Plans and decomposes complex goals into tasks"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.logger = logging.getLogger("TaskPlanner")
        self.tasks: Dict[str, Task] = {}

    async def plan(self, goal: str) -> List[Task]:
        """
        Decompose a goal into actionable tasks
        """
        prompt = f"""
        You are a task planning expert. Decompose the following goal into a sequence of concrete, actionable tasks.
        
        Goal: {goal}
        
        For each task, provide:
        1. A clear description
        2. Any dependencies on other tasks
        3. Estimated duration in seconds
        4. Priority (1-10)
        
        Format your response as a JSON array of tasks.
        """

        system_prompt = """You are a task planning expert. You decompose complex goals into actionable subtasks.
        Always respond with valid JSON."""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

        # Parse response and create tasks
        try:
            import json
            task_data = json.loads(response)
            tasks = []
            for i, data in enumerate(task_data):
                task = Task(
                    id=f"task_{i}",
                    description=data.get("description", ""),
                    dependencies=data.get("dependencies", []),
                    estimated_duration=data.get("estimated_duration", 300),
                    priority=data.get("priority", 5)
                )
                self.tasks[task.id] = task
                tasks.append(task)
            
            self.logger.info(f"Planned {len(tasks)} tasks for goal: {goal}")
            return tasks
        except Exception as e:
            self.logger.error(f"Error parsing task plan: {e}")
            return []

    def get_executable_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute"""
        executable = []
        for task in self.tasks.values():
            if task.status == "pending":
                # Check if all dependencies are met
                deps_met = all(
                    self.tasks[dep_id].status == "completed"
                    for dep_id in task.dependencies
                )
                if deps_met:
                    executable.append(task)
        
        # Sort by priority
        return sorted(executable, key=lambda t: t.priority, reverse=True)

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None
    ) -> None:
        """Update task status"""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            if result:
                self.tasks[task_id].result = result
            self.logger.info(f"Task {task_id} status updated to {status}")
```

---

## References

[1] Srinivasan, A. (2025). "Building Production-Ready AI Agents: A Full-Stack Blueprint for Reliability and Scalability." AI with Aish. Retrieved from https://aishwaryasrinivasan.substack.com/p/building-production-ready-ai-agents

[2] Microsoft Azure. (2025). "AI Agent Orchestration Patterns." Azure Architecture Center. Retrieved from https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns

[3] OpenAI. (2024). "Function Calling." OpenAI API Documentation. Retrieved from https://platform.openai.com/docs/guides/function-calling

[4] Anthropic. (2024). "Tool Use." Anthropic Documentation. Retrieved from https://docs.anthropic.com/claude/reference/tool-use

---

**End of Phase 3-4 Manual**

*Next: Phase 5-6 will cover Self-Correction, Multimodal Integration, and Continuous Learning.*

