# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands
- Installation: `pip install -r requirements.txt`
- Run application: `python main.py` or `python video_enhanced.py`
- Debug mode: Add `--debug` flag to enable verbose logging
- Test: Manual testing (no automated tests detected)

## Code Style Guidelines
- Imports: Group by standard library, third-party, then local modules
- Naming: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants
- Types: Use Python type hints (List, Dict, Optional, etc. from typing module)
- Documentation: Google-style docstrings with Args/Returns sections
- Error handling: Use try/except with specific exceptions, include logging, provide fallbacks
- Logging: Use Python's logging module with appropriate levels (info, warning, error)
- Functions: Small, single-purpose, descriptive names
- Comments: Document "why" not "what", include Japanese for domain-specific explanations
- Resources: Clean up resources in finally blocks

Update this file as the codebase evolves.