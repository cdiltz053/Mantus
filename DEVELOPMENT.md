# Mantus Development Guide

## Getting Started

This guide provides instructions for setting up a development environment for Mantus and contributing to its development.

## Prerequisites

- Python 3.9 or higher
- Node.js 22.13.0 or higher
- Git
- GitHub CLI (gh)
- Ubuntu 22.04 or similar Linux distribution

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/cdiltz053/Mantus.git
cd Mantus
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 3. Install Node.js Dependencies

```bash
npm install
# or
pnpm install
```

## Project Structure

```
Mantus/
├── core/                      # Core components
│   ├── main.py               # Main execution logic
│   ├── task_manager.py       # Task planning and phase management
│   └── communication_manager.py # User communication
├── tools/                     # Tool implementations
│   ├── system/               # System interaction tools
│   ├── web/                  # Web-based tools
│   └── specialized/          # Specialized task tools
├── environment/              # Environment configurations
│   ├── python/              # Python environment setup
│   ├── nodejs/              # Node.js environment setup
│   └── browser/             # Browser environment setup
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation
├── ARCHITECTURE.md           # Architecture documentation
├── DEVELOPMENT.md            # This file
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
└── setup.py                  # Package setup configuration
```

## Development Workflow

### 1. Creating a New Tool

To create a new tool for Mantus:

1. Create a new Python file in the appropriate `tools/` subdirectory.
2. Implement the tool class with the required methods.
3. Add the tool to the ToolOrchestrator in `core/main.py`.
4. Write unit tests in the `tests/` directory.
5. Update the documentation in `ARCHITECTURE.md`.

### 2. Implementing Core Components

When implementing core components:

1. Follow the existing code structure and naming conventions.
2. Add comprehensive docstrings to all classes and methods.
3. Implement error handling and logging.
4. Write unit tests for all functionality.
5. Update the `ARCHITECTURE.md` documentation.

### 3. Testing

Run tests using pytest:

```bash
pytest tests/
```

For coverage analysis:

```bash
pytest --cov=mantus tests/
```

## Code Style and Standards

- Follow PEP 8 guidelines for Python code.
- Use type hints for all function parameters and return values.
- Write comprehensive docstrings for all classes and methods.
- Keep functions small and focused on a single responsibility.
- Use meaningful variable and function names.

## Commit Guidelines

- Write clear, descriptive commit messages.
- Use the present tense ("Add feature" not "Added feature").
- Reference relevant issues or pull requests in commit messages.
- Keep commits focused on a single logical change.

## Documentation

- Update `README.md` with any user-facing changes.
- Update `ARCHITECTURE.md` with any architectural changes.
- Add docstrings to all new code.
- Update this `DEVELOPMENT.md` guide as needed.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them.
4. Push to your fork (`git push origin feature/your-feature`).
5. Create a pull request with a clear description of your changes.

## Roadmap

The development of Mantus follows this roadmap:

1. **Phase 1**: Implement core components
2. **Phase 2**: Implement system and file system interaction tools
3. **Phase 3**: Implement information retrieval and external access tools
4. **Phase 4**: Implement specialized task execution tools
5. **Phase 5**: Integration and comprehensive testing
6. **Phase 6**: Production deployment

## Support

For questions or issues, please open an issue on GitHub or contact the development team.

## License

Mantus is licensed under the MIT License. See LICENSE file for details.

