# Database Migrations

This directory contains database migration scripts using Alembic.

## Setup

To create a new migration:

```bash
alembic revision -m "description of your migration"
```

To upgrade the database to the latest version:

```bash
alembic upgrade head
```

To downgrade the database:

```bash
alembic downgrade -1  # Go back one revision
```

## Migration Guidelines

1. Each migration should be atomic and focused on a single change
2. Always include both upgrade and downgrade paths
3. Keep migrations forward-compatible with existing data
4. Test migrations on sample data before applying to production

## Structure

- `versions/` - Contains the actual migration scripts
- `env.py` - Alembic environment configuration
- `script.py.mako` - Template for migration scripts