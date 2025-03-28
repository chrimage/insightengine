# InsightEngine Tests

This directory contains tests for the InsightEngine project.

## Structure

```
tests/
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions
├── fixtures/             # Test fixtures and mock data
└── conftest.py           # Pytest configuration and fixtures
```

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test modules:

```bash
pytest tests/unit/test_database.py
```

To run tests with coverage:

```bash
pytest --cov=memory_ai
```

## Test Guidelines

1. Unit tests should focus on testing one function or method in isolation
2. Use mocks for external dependencies like LLM APIs
3. Integration tests should test interactions between components
4. Fixtures should be used to setup common test data
5. Tests should be independent and not rely on other tests

## Adding New Tests

1. Create a new test file in the appropriate directory
2. Import the module to be tested
3. Create test functions that start with `test_`
4. Use pytest fixtures for setup and teardown
5. Add assertions to verify expected behavior