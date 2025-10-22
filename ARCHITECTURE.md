# Mantus Architecture and Design Blueprint

## Overview

Mantus is an autonomous general AI agent designed to replicate the full spectrum of capabilities demonstrated by Manus AI. This document outlines the architectural design, core components, and integration strategy for building Mantus as a fully functional AI assistant.

## 1. Core Architecture

### 1.1 Agent Loop

Mantus operates through an iterative agent loop that follows these steps:

1. **Analyze Context**: Understand the user's intent and current state based on available information.
2. **Think**: Reason about whether to update the plan, advance the phase, or take a specific action.
3. **Select Tool**: Choose the next tool for function calling based on the plan and state.
4. **Execute Action**: The selected tool will be executed as an action in the environment.
5. **Receive Observation**: The action result will be appended to the context as a new observation.
6. **Iterate Loop**: Repeat the above steps patiently until the task is fully completed.
7. **Deliver Outcome**: Send results and deliverables to the user.

### 1.2 Core Components

Mantus's architecture consists of the following core components:

#### TaskManager
Responsible for managing task planning, phase creation, phase advancement, and goal tracking. The TaskManager maintains the current plan and ensures that tasks are broken down into manageable phases.

#### CommunicationManager
Handles all user interactions, including sending messages, asking questions, and delivering results. The CommunicationManager ensures clear and structured communication with the user.

#### ToolOrchestrator
Coordinates the invocation of various tools based on the current task and phase. The ToolOrchestrator determines which tool to use and manages the execution flow.

#### ContextManager
Maintains the current context, including task history, previous observations, and state information. This allows Mantus to reason about decisions and maintain continuity across iterations.

### 1.3 Large Language Model (LLM) Architecture (The "Neural Heart")

To truly mirror the capabilities of Manus AI, Mantus requires a foundational Large Language Model (LLM) as its core intelligence. This LLM will be responsible for understanding natural language, generating responses, reasoning about tasks, and facilitating tool selection. Given the user's commitment to providing extensive resources, the design will focus on a state-of-the-art, transformer-based architecture.

#### 1.3.1 Model Type: Transformer Architecture

The most effective LLMs today are built upon the **Transformer architecture** [1]. This architecture is particularly well-suited for sequence-to-sequence tasks, such as language translation, text summarization, and natural language understanding/generation. Key characteristics include:

*   **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in the input sequence relative to each other, capturing long-range dependencies.
*   **Positional Encoding**: Adds information about the relative or absolute position of tokens in the sequence, as the self-attention mechanism itself is permutation-invariant.
*   **Encoder-Decoder Structure (or Decoder-only)**: While the original Transformer had both, many modern LLMs (especially for generation) use a decoder-only stack of layers.

#### 1.3.2 Core Components of the LLM

An LLM of this scale typically comprises the following high-level components:

*   **Tokenization Layer**: Converts raw text into numerical tokens that the model can process. This involves:
    *   **Tokenizer**: Breaks down text into subword units (e.g., Byte-Pair Encoding (BPE), WordPiece, SentencePiece).
    *   **Vocabulary**: A mapping of tokens to unique integer IDs.
    *   **Embedding Layer**: Converts token IDs into continuous vector representations (embeddings) that capture semantic meaning.

*   **Transformer Blocks (Layers)**: The core computational units of the LLM. Each block typically contains:
    *   **Multi-Head Self-Attention**: Computes attention weights across different "heads" to capture diverse relationships between tokens.
    *   **Feed-Forward Networks**: Position-wise fully connected feed-forward networks applied to each position independently.
    *   **Residual Connections and Layer Normalization**: Used to stabilize training and improve gradient flow through deep networks.

*   **Output Layer**: Transforms the final hidden states from the Transformer blocks into a probability distribution over the vocabulary, typically using a linear layer followed by a softmax activation function.

#### 1.3.3 Training Regimen

Training an LLM of this magnitude involves several stages:

*   **Pre-training**: The model is trained on a massive corpus of text (and potentially other modalities like code, images, audio) using self-supervised objectives, such as predicting the next word or filling in masked words. This phase is computationally intensive and aims to learn general language understanding and generation capabilities.
*   **Fine-tuning (Optional but Recommended)**: After pre-training, the model can be fine-tuned on smaller, task-specific datasets to adapt it for particular applications (e.g., instruction following, summarization, question answering). This often involves supervised learning with labeled data.
*   **Reinforcement Learning from Human Feedback (RLHF) (Optional but Highly Effective)**: A crucial step for aligning the model with human preferences and making it more helpful, harmless, and honest. This involves training a reward model based on human rankings of model outputs, and then using reinforcement learning to optimize the LLM against this reward model.

#### 1.3.4 Integration with Agent Loop

The LLM will be integrated into Mantus's agent loop as the primary "Think" component. It will:

1.  **Interpret User Input**: Understand the user's request and current context.
2.  **Reason and Plan**: Formulate a high-level plan to achieve the user's goal, breaking it down into sub-tasks.
3.  **Tool Selection**: Based on the current sub-task, select the most appropriate tool from Mantus's available toolset.
4.  **Tool Argument Generation**: Generate the necessary arguments for the selected tool based on the current context and task requirements.
5.  **Process Tool Output**: Interpret the observations received from tool execution and update the internal state.
6.  **Generate Responses**: Formulate natural language responses to the user, providing updates, asking clarifying questions, or delivering results.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30. [https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)





Mantus integrates with a comprehensive suite of tools, organized into four categories:

### 2.1 Core Task Management and Communication Tools

**plan**: Manages task workflow by creating, updating, and advancing through structured phases.

**message**: Facilitates all communication with the user, including providing information, asking questions, and delivering final results and attachments.

### 2.2 System and File System Interaction Tools

**shell**: Provides command-line access to the sandboxed Linux environment, enabling execution of commands, installation of software, and general system management.

**file**: Allows for comprehensive file system operations, including viewing, reading, writing, appending, and editing files.

**match**: Enables pattern-based searching within the file system, supporting both glob-style file path matching and regex-based content searching.

### 2.3 Information Retrieval and External Access Tools

**search**: Accesses external information across various sources (web info, images, APIs, news, tools, data, research).

**browser**: Navigates web pages to gather information, perform transactional tasks, or interact with web applications.

### 2.4 Specialized Task Execution Tools

**schedule**: Schedules tasks to run at specific times or recurring intervals using cron expressions or time intervals.

**expose**: Temporarily exposes local ports in the sandbox for public access.

**generate**: Enters a dedicated mode for creating or editing images, videos, audio, and speech from text and media references.

**slides**: Enters a dedicated mode for presentation creation and adjustment.

**webdev_init_project**: Initializes new web development projects with modern tooling and structure.

## 3. Operational Environment

Mantus operates within a sandboxed virtual machine environment with the following specifications:

### 3.1 Operating System and Infrastructure

- **OS**: Ubuntu 22.04 linux/amd64 with internet access.
- **User**: ubuntu (with sudo privileges, no password).
- **Home Directory**: /home/ubuntu
- **Persistence**: System state and installed packages persist across hibernation cycles.

### 3.2 Pre-installed Utilities

Mantus has access to the following pre-installed utilities:

- **Command-line Tools**: bc, curl, gh, git, gzip, less, net-tools, poppler-utils, psmisc, socat, tar, unzip, wget, zip
- **Specialized Utilities**: manus-render-diagram, manus-md-to-pdf, manus-speech-to-text, manus-mcp-cli, manus-upload-file, manus-export-slides

### 3.3 Programming Environments

**Python 3.11.0rc1**
- Commands: python3.11, pip3
- Pre-installed packages: beautifulsoup4, fastapi, flask, fpdf2, markdown, matplotlib, numpy, openpyxl, pandas, pdf2image, pillow, plotly, reportlab, requests, seaborn, tabulate, uvicorn, weasyprint, xhtml2pdf

**Node.js 22.13.0**
- Commands: node, pnpm
- Pre-installed packages: pnpm, yarn

### 3.4 Browser Environment

- **Version**: Chromium stable
- **Download Directory**: /home/ubuntu/Downloads/
- **Login and Cookie Persistence**: Enabled

### 3.5 GitHub Integration

- **Tool**: Pre-configured GitHub CLI (gh)
- **Functionality**: Seamless interaction with GitHub repositories

## 4. Execution Flow

### 4.1 Task Initiation

When a user provides a task request, Mantus:

1. Acknowledges the request using the `message` tool.
2. Creates or updates a task plan using the `plan` tool, breaking the task into manageable phases.
3. Begins executing the first phase.

### 4.2 Phase Execution

During each phase, Mantus:

1. Analyzes the current context and phase requirements.
2. Selects appropriate tools to accomplish the phase objectives.
3. Executes the selected tools in sequence or in parallel (as appropriate).
4. Observes the results and adapts the approach if necessary.
5. Advances to the next phase when the current phase is complete.

### 4.3 Task Completion

Upon completing all phases, Mantus:

1. Prepares the final results and deliverables.
2. Sends a final message to the user with the `message` tool (type: "result").
3. Marks the task as complete.

## 5. Error Handling and Adaptation

Mantus implements a robust error handling strategy:

1. **Diagnosis**: On error, diagnose the issue using the error message and context.
2. **Attempt Fix**: Attempt a fix using alternative methods or tools.
3. **Escalation**: After failing at most three times, explain the failure to the user and request further guidance.
4. **Plan Revision**: If the current task plan is inefficient or fails repeatedly, use the `plan` tool with the `update` action to revise the plan.

## 6. Implementation Roadmap

The development of Mantus should follow this roadmap:

1. **Phase 1**: Implement core components (TaskManager, CommunicationManager, ToolOrchestrator, ContextManager).
2. **Phase 2**: Implement system and file system interaction tools (shell, file, match).
3. **Phase 3**: Implement information retrieval and external access tools (search, browser).
4. **Phase 4**: Implement specialized task execution tools (schedule, expose, generate, slides, webdev_init_project).
5. **Phase 5**: Integrate all components and conduct comprehensive testing.
6. **Phase 6**: Deploy and monitor Mantus in production.

## 7. Key Design Principles

Mantus is designed with the following principles in mind:

- **Modularity**: Each tool and component is independent and can be developed and tested separately.
- **Extensibility**: New tools and capabilities can be easily added to Mantus.
- **Robustness**: Error handling and fallback mechanisms are built into the core architecture.
- **User-Centric**: All interactions are designed to be clear, helpful, and user-friendly.
- **Autonomy**: Mantus can reason about tasks and make decisions without constant user input.
- **Transparency**: All actions and decisions are logged and can be reviewed by the user.

## 8. Conclusion

This architecture provides a comprehensive blueprint for developing Mantus as a fully functional autonomous AI agent that replicates the capabilities of Manus AI. By following this design and implementing the outlined components and tools, Mantus will be capable of assisting users in a wide range of tasks, from information gathering and content creation to software development and workflow automation.

