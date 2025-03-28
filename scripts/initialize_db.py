#!/usr/bin/env python3
"""
Initialize an empty database with the InsightEngine schema.

This script creates a new database file with all the required tables,
indexes, and vector stores for the InsightEngine system.
"""

import argparse
import os
import sys
from pathlib import Path
import sqlalchemy as sa

# Add the parent directory to the path so we can import memory_ai
sys.path.append(str(Path(__file__).parent.parent))

from memory_ai.core.database import get_db_manager
from memory_ai.core.config import get_settings, settings


def initialize_database(db_path=None, vector_db_path=None, force=False):
    """Initialize a new database with the InsightEngine schema.
    
    Args:
        db_path: Path to the database file
        vector_db_path: Path to the vector store
        force: Whether to overwrite an existing database
    
    Returns:
        bool: True if successful
    """
    # Override settings if provided
    if db_path:
        settings.database.db_path = db_path
    
    if vector_db_path:
        settings.database.vector_db_path = vector_db_path
    
    # Check if database already exists
    if os.path.exists(settings.database.db_path) and not force:
        print(f"Error: Database file already exists at {settings.database.db_path}")
        print("Use --force to overwrite it.")
        return False
    
    # Check if vector db already exists
    if os.path.exists(settings.database.vector_db_path) and not force:
        print(f"Error: Vector store already exists at {settings.database.vector_db_path}")
        print("Use --force to overwrite it.")
        return False
    
    # Create directories if they don't exist
    db_dir = Path(settings.database.db_path).parent
    if db_dir and not db_dir.exists():
        print(f"Creating directory: {db_dir}")
        db_dir.mkdir(parents=True, exist_ok=True)
    
    vector_db_dir = Path(settings.database.vector_db_path).parent
    if vector_db_dir and not vector_db_dir.exists():
        print(f"Creating directory: {vector_db_dir}")
        vector_db_dir.mkdir(parents=True, exist_ok=True)
    
    # Delete existing files if force is True
    if os.path.exists(settings.database.db_path) and force:
        print(f"Removing existing database: {settings.database.db_path}")
        os.remove(settings.database.db_path)
    
    # Initialize the database
    print(f"Initializing database at {settings.database.db_path}")
    print(f"Using vector store at {settings.database.vector_db_path}")
    
    # This will create the tables and initialize the schema
    db_manager = get_db_manager()
    
    try:
        # Test that we can connect to the database
        with db_manager.engine.connect() as conn:
            print(f"Successfully connected to database at {settings.database.db_path}")
            
            # Explicitly create database tables
            print("Creating database tables...")
            from memory_ai.core.database import Base
            Base.metadata.create_all(db_manager.engine)
            print("Tables created successfully")
            
            # Run a test query
            print("Running test query...")
            result = conn.execute(sa.text("SELECT sqlite_version()"))
            version = result.scalar()
            print(f"Connected to SQLite version: {version}")
            
            return True
    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        print("\nDiagnostics:")
        print(f"Database path: {settings.database.db_path}")
        print(f"Database directory exists: {os.path.exists(os.path.dirname(settings.database.db_path) or '.')}")
        print(f"Database file exists: {os.path.exists(settings.database.db_path)}")
        print(f"SQLAlchemy URL: {db_manager.engine.url}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Initialize an empty InsightEngine database.")
    parser.add_argument("--db-path", help="Path to the database file")
    parser.add_argument("--vector-db-path", help="Path to the vector store")
    parser.add_argument("--force", action="store_true", help="Overwrite existing database")
    
    args = parser.parse_args()
    
    success = initialize_database(
        db_path=args.db_path,
        vector_db_path=args.vector_db_path,
        force=args.force
    )
    
    if success:
        print("\nNext steps:")
        print("1. Import conversations with memory_ai.tools.index")
        print("2. Generate summaries with memory_ai.tools.summarize")
        print("3. Start chatting with memory_ai.tools.interact")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()