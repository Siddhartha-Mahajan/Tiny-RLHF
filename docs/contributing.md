# Contributing

1. Fork the repository and create a feature branch.
2. Install development dependencies:

   ```bash
   pip install -e .[dev]
   ```

3. Run the formatting and linting helpers:

   ```bash
   tools/format.sh
   tools/lint.sh
   ```

4. Add tests under `tests/` and ensure they pass:

   ```bash
   pytest
   ```

5. Submit a pull request describing your changes and how to reproduce them.

## Code Style

- Follow the existing module structure when adding new pipelines, trainers, or adapters.
- Use Pydantic models for configuration validation.
- Prefer dependency injection (pass components via constructors) over global state.
