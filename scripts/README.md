# Utility Scripts

This directory contains utility scripts for managing the InsightEngine system.

## Available Scripts

- `setup_dev.sh` - Sets up the development environment
- `initialize_db.py` - Creates and initializes an empty database
- `backup_db.py` - Creates a backup of the database
- `import_conversations.py` - Imports conversations from various formats
- `evaluate_performance.py` - Analyzes system performance metrics

## Usage

Most scripts support the `--help` flag for usage information:

```bash
python scripts/initialize_db.py --help
```

## Adding New Scripts

When adding new utility scripts:

1. Use a descriptive name that indicates the script's purpose
2. Include a docstring that explains what the script does
3. Use argparse for command-line arguments
4. Add error handling and proper exit codes
5. Update this README with the new script