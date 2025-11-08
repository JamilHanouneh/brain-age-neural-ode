# Contributing to Brain Aging Neural ODE

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests: `pytest tests/`
5. Commit with clear messages: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Submit a Pull Request

## Development Setup

git clone https://github.com/JamilHanouneh/brain-age-neural-ode.git
cd brain-aging-neural-ode
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .


## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Keep functions focused and short

## Testing

Add tests for new features in `tests/`:
pytest tests/ -v


## Reporting Issues

Use GitHub Issues with:
- Clear title
- Detailed description
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, GPU/CPU)

## Questions?

Contact: Jamil.hanouneh1997@gmail.com

