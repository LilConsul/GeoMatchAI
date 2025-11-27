# Contributing to GeoMatchAI

Thank you for your interest in contributing to GeoMatchAI! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- **Python 3.13**
- **UV package manager** ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/))
- **Git**
- **CUDA 12.8** (optional, for GPU acceleration)

### Getting Started

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/GeoMatchAI.git
   cd GeoMatchAI
   ```

2. **Create and sync virtual environment**
   ```powershell
   uv sync --all-groups
   ```

3. **Activate the virtual environment**
   ```powershell
   .venv\Scripts\activate
   ```

4. **Set up environment variables**
   ```powershell
   copy .env.example .env
   # Edit .env and add your Mapillary API key
   ```

## Development Workflow

### Running Tests

```powershell
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_verifier.py

# Run with coverage
uv run pytest --cov=geomatchai --cov-report=html
```

### Code Quality

```powershell
# Format code with ruff
uv run ruff format .

# Lint code
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Type checking
uv run mypy src/geomatchai
```

### Pre-commit Hooks

We recommend using pre-commit hooks to ensure code quality:

```powershell
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

## Code Style Guidelines

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Maximum line length: **100 characters**
- Use **Google-style docstrings**

Example:
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Short description of the function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When input is invalid
        
    Example:
        >>> my_function("test", 42)
        True
    """
    pass
```

### Logging

- Use the `logging` module, **never `print()`** in production code
- Use appropriate log levels:
  - `DEBUG`: Detailed diagnostic information
  - `INFO`: General informational messages
  - `WARNING`: Warning messages for potentially harmful situations
  - `ERROR`: Error messages for serious problems
  - `CRITICAL`: Critical errors that may cause program termination

Example:
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Processing started")
logger.debug(f"Processing {count} items")
logger.warning("Threshold may be too low")
logger.error(f"Failed to process image: {e}")
```

### Error Handling

- Use custom exception classes from `geomatchai.exceptions`
- Include context in error messages
- Use `raise ... from e` for exception chaining

Example:
```python
from geomatchai.exceptions import PreprocessingError

try:
    result = process_image(img)
except Exception as e:
    logger.error(f"Failed to process image: {e}")
    raise PreprocessingError(
        f"Image preprocessing failed for {img_path}"
    ) from e
```

## Project Structure

```
GeoMatchAI/
├── src/geomatchai/       # Main package
│   ├── __init__.py       # Package exports
│   ├── config.py         # Configuration management
│   ├── exceptions.py     # Custom exceptions
│   ├── preprocessing/    # Image preprocessing
│   ├── models/           # Feature extraction models
│   ├── gallery/          # Gallery builder
│   ├── verification/     # Verification logic
│   └── fetchers/         # Image fetchers
├── tests/                # Test files
├── pyproject.toml        # Project configuration
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest fixtures for common resources
- Mock external APIs (Mapillary) in tests
- Use descriptive test names

Example:
```python
import pytest
from geomatchai import Preprocessor

@pytest.fixture
def preprocessor():
    return Preprocessor(device="cpu")

def test_preprocessor_validates_image_size(preprocessor):
    """Test that preprocessor rejects oversized images."""
    # Test implementation
    pass
```

## Pull Request Process

1. **Create a feature branch**
   ```powershell
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```powershell
   uv run ruff check .
   uv run pytest
   ```

4. **Commit your changes**
   ```powershell
   git add .
   git commit -m "feat: add new feature"
   ```

5. **Push to your fork**
   ```powershell
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure all CI checks pass

## Commit Message Convention

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```
feat: add support for EfficientNet-B5
fix: correct threshold validation in verifier
docs: update installation instructions
```

## Questions or Issues?

- **Bug reports**: Open an issue with detailed reproduction steps
- **Feature requests**: Open an issue describing the feature and use case
- **Questions**: Open a discussion or issue

## License

By contributing to GeoMatchAI, you agree that your contributions will be licensed under the MIT License.

