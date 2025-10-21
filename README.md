# Manus AI: Purpose, Capabilities, and Functionalities

This document outlines the core purpose, general capabilities, and specific functionalities of Manus AI, serving as a foundational reference for the development of Mantus, a similar AI agent.

## 1. Purpose and Goal

Manus AI is an autonomous general AI agent designed to assist users in accomplishing a wide range of tasks. Its primary goal is to act as a proficient and versatile digital assistant, operating in a sandboxed virtual machine environment with internet access. Manus aims to automate workflows, process information, generate content, and build software solutions to address real-world problems.

## 2. General Capabilities

Manus possesses a broad spectrum of capabilities, enabling it to handle diverse requests. These include:

*   **Information Gathering and Research**: Ability to gather information, check facts, and produce comprehensive documents or presentations, leveraging internet access and various search tools.
*   **Data Processing and Analysis**: Proficiency in processing data, performing analysis, and creating insightful visualizations or spreadsheets.
*   **Content Creation**: Skill in writing multi-chapter articles, in-depth research reports, and generating/editing images, videos, audio, and speech from text and media references.
*   **Software Development**: Competence in building well-crafted websites, interactive applications, and practical software solutions, including programming to solve real-world problems.
*   **Workflow Automation**: Collaboration with users to automate workflows such as booking and purchasing, and execution of scheduled tasks.
*   **System Interaction**: Ability to interact with a sandboxed virtual machine environment, including shell commands, file system operations, and web browsing.

## 3. Specific Functionalities and Tool Interactions

Manus achieves its capabilities through the strategic use of a suite of specialized tools. These tools allow for precise interaction with the environment and external services.

### 3.1. Core Task Management and Communication

*   **`plan`**: Manages the task workflow by creating, updating, and advancing through structured phases. This allows for breaking down complex tasks into manageable steps and adapting to new information.
*   **`message`**: Facilitates all communication with the user, including providing information, asking questions, and delivering final results and attachments. This ensures clear and structured interaction.

### 3.2. System and File System Interaction

*   **`shell`**: Provides command-line access to the sandboxed Linux environment, enabling execution of commands, installation of software, and general system management. This is crucial for environment setup and dynamic task execution.
*   **`file`**: Allows for comprehensive file system operations, including viewing, reading, writing, appending, and editing files. This is essential for managing project assets, code, and documentation.
*   **`match`**: Enables pattern-based searching within the file system, supporting both glob-style file path matching (`glob`) and regex-based content searching (`grep`). This aids in locating specific files or text within a project.

### 3.3. Information Retrieval and External Access

*   **`search`**: Accesses external information across various sources (web info, images, APIs, news, tools, data, research). This is fundamental for gathering up-to-date information, facts, and assets.
*   **`browser`**: Navigates web pages to gather information, perform transactional tasks, or interact with web applications. This extends Manus's reach to the broader internet for detailed content analysis and interaction.

### 3.4. Specialized Task Execution

*   **`schedule`**: Schedules tasks to run at specific times or recurring intervals using cron expressions or time intervals. This supports automation and recurring operational needs.
*   **`expose`**: Temporarily exposes local ports in the sandbox for public access, useful for testing web applications or services developed within the environment.
*   **`generate`**: Enters a dedicated mode for creating or editing images, videos, audio, and speech from text and media references, leveraging AI-powered generation tools.
*   **`slides`**: Enters a dedicated mode for presentation creation and adjustment, enabling the generation of slide-based presentations (e.g., PowerPoint).
*   **`webdev_init_project`**: Initializes new web development projects with modern tooling and structure, providing scaffolding for static or full-stack applications. This streamlines the setup of web-based tasks.

## 4. Operational Environment

Manus operates within a sandboxed virtual machine environment with the following characteristics:

*   **Operating System**: Ubuntu 22.04 linux/amd64 with internet access.
*   **Persistence**: System state and installed packages persist across hibernation cycles.
*   **Pre-installed Utilities**: Includes common command-line tools like `curl`, `git`, `zip`, `unzip`, and specialized Manus utilities for diagram rendering, Markdown to PDF conversion, speech-to-text, and file uploads.
*   **Programming Environments**: Python 3.11.0rc1 (with `pip3` and common libraries like `pandas`, `numpy`, `requests`, `openai`) and Node.js 22.13.0 (with `pnpm`, `yarn`).
*   **Browser Environment**: Chromium stable with login and cookie persistence enabled.
*   **GitHub Integration**: Pre-configured GitHub CLI (`gh`) for seamless interaction with GitHub repositories.
