# Tool for generating summaries and related actions

import argparse
import os
import sys
import traceback
import json
from datetime import datetime

from memory_ai.core.database import MemoryDatabase
from memory_ai.memory.summary import RollingSummaryProcessor
from memory_ai.utils.gemini import GeminiClient

# --- Helper Functions for Display ---

def print_separator():
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

def print_summary_details(summary):
    """Prints common details for global and conversation summaries."""
    if not summary:
        return

    # Print metadata if available
    if 'metadata' in summary and summary.get('metadata'):
        metadata = summary['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                print("Warning: Could not parse metadata JSON.")
                metadata = {} # Reset to empty dict if parsing fails

        if isinstance(metadata, dict):
            if 'key_points' in metadata and metadata['key_points']:
                print("KEY POINTS:")
                for point in metadata['key_points']:
                    print(f"- {point}")
                print()

            if 'themes' in metadata and metadata['themes']:
                print(f"THEMES: {', '.join(metadata['themes'])}")

            if 'entities' in metadata and metadata['entities']:
                print(f"ENTITIES: {', '.join(metadata['entities'][:10])}")

            if 'sentiment' in metadata:
                print(f"SENTIMENT: {metadata['sentiment']}")

            if 'technical_level' in metadata:
                print(f"TECHNICAL LEVEL: {metadata['technical_level']}")

            if 'quality_score' in metadata:
                 print(f"Quality Score: {metadata['quality_score']:.2f}")

            if 'topics' in metadata and metadata['topics']:
                 print(f"Topics: {', '.join(metadata['topics'][:10])}")
            print() # Add a newline after metadata block

    print(summary['summary_text'])


# --- Action Functions ---

def view_active_summary(db_path):
    """View the active global summary."""
    print(f"🔍 Viewing active summary from {db_path}...")
    print_separator()
    db = None
    try:
        db = MemoryDatabase(db_path)
        summary = db.get_active_summary()
        if summary:
            print(f"Summary version: {summary['version']}\n")
            if 'conversation_range' in summary:
                 print(f"Covers {len(summary['conversation_range'])} conversations\n")
            print_summary_details(summary)
            print(f"\nLast updated: {datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print('No active global summary found in the database.')
    except Exception as e:
        print(f"Error viewing active summary: {e}")
        traceback.print_exc()
    finally:
        if db:
            db.close()
    print_separator()

def view_conversation_summary(db_path, conv_id):
    """View summary for a specific conversation."""
    print(f"🔍 Viewing summary for conversation {conv_id} from {db_path}...")
    print_separator()
    db = None
    try:
        db = MemoryDatabase(db_path)
        summary = db.get_conversation_summary(conv_id)

        if summary:
            print(f"CONVERSATION SUMMARY: {summary['conversation_id']}\n")
            print_summary_details(summary)

            # Check if we have conversation information
            c = db.conn.cursor()
            c.execute('SELECT title, timestamp FROM conversations WHERE id = ?', (summary['conversation_id'],))
            conv_data = c.fetchone()

            if conv_data:
                title, timestamp = conv_data
                print(f"\nConversation: {title}")
                print(f"Date: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print('No summary found for this conversation.')

            # Check if conversation exists
            c = db.conn.cursor()
            c.execute('SELECT id, title FROM conversations WHERE id = ?', (conv_id,))
            conv = c.fetchone()

            if conv:
                print(f"Conversation exists: {conv[1]}")
                print("But no summary has been generated yet.")
            else:
                print(f"Conversation with ID '{conv_id}' not found in database.")
    except Exception as e:
        print(f"Error viewing conversation summary: {e}")
        traceback.print_exc()
    finally:
        if db:
            db.close()
    print_separator()

def search_summaries_by_theme(db_path, query):
    """Search for summaries related to a specific theme or topic."""
    print(f"🔍 Searching for summaries related to '{query}' in {db_path}...")
    print_separator()
    db = None
    gemini = None
    try:
        db = MemoryDatabase(db_path)
        embedding = None

        # Try to use vector search if possible
        try:
            gemini = GeminiClient()
            embedding = gemini.generate_embedding(query)
            print('Using vector similarity search with embeddings')
        except Exception as e:
            print(f'Falling back to text search: {e}')

        theme_summaries = db.get_summaries_by_theme(query, embedding=embedding)

        if theme_summaries:
            print(f"Found {len(theme_summaries)} matching summaries\n")
            for i, summary in enumerate(theme_summaries, 1):
                print(f"SUMMARY #{i} (Version: {summary['version']}):")

                # Print relevance score if available
                if 'relevance' in summary:
                    print(f"Relevance: {summary['relevance']:.2f}")

                # Print metadata if available (simplified for brevity in search results)
                if 'metadata' in summary and summary.get('metadata'):
                     metadata = summary['metadata']
                     if isinstance(metadata, str):
                         try: metadata = json.loads(metadata)
                         except: metadata = {}
                     if isinstance(metadata, dict):
                         if 'themes' in metadata and metadata['themes']:
                             print(f"Themes: {', '.join(metadata['themes'])}")
                         if 'topics' in metadata and metadata['topics']:
                             print(f"Topics: {', '.join(metadata['topics'][:5])}") # Show fewer topics
                         if 'quality_score' in metadata:
                             print(f"Quality Score: {metadata['quality_score']:.2f}")

                # Print timestamp
                created = datetime.fromtimestamp(summary['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Created: {created}")

                print(f"\n{summary['summary_text'][:500]}...") # Shorter preview
                print("\n" + "-"*50 + "\n")
        else:
            print('No summaries found matching your query.')
    except Exception as e:
        print(f"Error searching themes: {e}")
        traceback.print_exc()
    finally:
        if db:
            db.close()
    print_separator()

def list_conversations_without_summaries(db_path):
    """List conversations that do not have summaries."""
    print(f"🔍 Listing conversations without summaries in {db_path}...")
    print_separator()
    db = None
    try:
        db = MemoryDatabase(db_path)
        c = db.conn.cursor()

        # Get conversations without summaries
        c.execute('''
        SELECT id, title, timestamp, message_count
        FROM conversations
        WHERE summary IS NULL OR summary = ''
        ORDER BY timestamp DESC
        ''')

        conversations = c.fetchall()
        if conversations:
            print(f'Found {len(conversations)} conversations without summaries:\n')

            for conv in conversations:
                conv_id, title, timestamp, message_count = conv
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                print(f'ID: {conv_id}')
                print(f'Title: {title}')
                print(f'Date: {date}')
                print(f'Messages: {message_count}')
                print('-' * 40)
        else:
            print('All conversations seem to have summaries.')
    except Exception as e:
        print(f"Error listing conversations: {e}")
        traceback.print_exc()
    finally:
        if db:
            db.close()
    print_separator()

def generate_summary_embeddings(db_path):
    """Generate embeddings for existing summaries."""
    print(f"🔍 Generating embeddings for existing summaries in {db_path}...")
    print_separator()
    db = None
    gemini = None
    try:
        db = MemoryDatabase(db_path)
        gemini = GeminiClient()
        processor = RollingSummaryProcessor(db, gemini)

        # Generate embeddings for existing summaries
        processed_global, processed_conv = processor.generate_embeddings_for_summaries()
        print(f'Generated embeddings for {processed_global} global summaries and {processed_conv} conversation summaries')
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        traceback.print_exc()
    finally:
        if db:
            db.close()
    print_separator()


def run_main_summarization(args):
    """Run the main summarization process based on arguments."""
    print(f"🧠 Generating summaries from conversations in {args.db}...")
    print_separator()

    db = None
    gemini = None
    try:
        db = MemoryDatabase(args.db)

        # Count stats before processing
        conv_count_before = db.conn.execute("SELECT COUNT(*) FROM conversations;").fetchone()[0]
        msg_count_before = db.conn.execute("SELECT COUNT(*) FROM messages;").fetchone()[0]
        summary_count_before = db.conn.execute("SELECT COUNT(*) FROM rolling_summaries;").fetchone()[0]

        if args.verbose:
            print("Settings:")
            print(f"  Database:       {args.db}")
            print(f"  Batch size:     {args.batch_size} conversations")
            print(f"  Rebuild:        {'Yes (deleting existing summaries)' if args.rebuild else 'No'}")
            print(f"  Forgetting:     {'Enabled ({args.days_threshold} days)' if args.forget else 'Disabled'}")
            print(f"  Conv Summaries: {'Yes' if args.conversation_summaries else 'No (unless needed after global)'}")
            print(f"  Embeddings:     {'Yes' if args.generate_embeddings else 'No'}")
            print(f"  Evaluation:     {'Yes' if args.evaluate_summaries else 'No'}")
            print(f"  Workers:        {args.workers}")
            print(f"  Limit:          {args.limit if args.limit else 'None'}")
            print()

        print("Database Stats (Before):")
        print(f"  📚 Conversations: {conv_count_before}")
        print(f"  💬 Messages: {msg_count_before}")
        print(f"  📝 Summaries: {summary_count_before}")
        print()

        # --- Rebuild Logic ---
        if args.rebuild:
            print('Rebuilding summaries: deleting all existing summaries...')
            db.conn.execute('DELETE FROM rolling_summaries')
            c = db.conn.cursor()
            c.execute('UPDATE conversations SET summary = NULL, metadata = NULL')
            c.execute('DELETE FROM embeddings WHERE id LIKE "summary_%"')
            db.conn.commit()
            print('All existing summaries deleted. Starting fresh.')

        gemini = GeminiClient()
        # TODO: Pass evaluation flag to processor if needed
        processor = RollingSummaryProcessor(db, gemini, batch_size=args.batch_size)

        limit_to_use = args.limit if args.limit and args.limit > 0 else None

        # --- Main Processing Logic ---
        processed_conv_count = 0
        if args.conversation_summaries and not args.rebuild:
            # Only generate conversation-specific summaries if explicitly requested and not rebuilding
            print('Processing only conversation-specific summaries...')
            try:
                processed_conv_count = processor.process_conversation_specific_summaries(limit=limit_to_use)
                print(f'Generated {processed_conv_count} conversation-specific summaries!')
            except Exception as e:
                print(f'ERROR during conversation summary generation: {e}')
                traceback.print_exc()
        else:
            # Generate global rolling summaries (or if rebuilding)
            print('Processing global rolling summaries...')
            try:
                processor.process_conversations() # Limit doesn't apply to global rolling
                print('Global summary processing completed successfully!')

                # Always check and process conversation summaries for new batches after global run
                unsummarized_count = db.conn.execute(
                    'SELECT COUNT(*) FROM conversations WHERE summary IS NULL OR summary = ""'
                ).fetchone()[0]

                if unsummarized_count > 0:
                    print(f'Found {unsummarized_count} conversations needing summaries after global run.')
                    processed_conv_count = processor.process_conversation_specific_summaries(limit=limit_to_use)
                    print(f'Generated {processed_conv_count} additional conversation-specific summaries')

            except Exception as e:
                print(f'ERROR during global summary generation: {e}')
                traceback.print_exc()

        # --- Generate Embeddings ---
        if args.generate_embeddings and (processed_conv_count > 0 or args.rebuild):
             print("Generating embeddings for new/updated summaries...")
             processor.generate_embeddings_for_summaries()

        # --- Apply Forgetting ---
        if args.forget:
            print(f'Applying forgetting mechanism (threshold: {args.days_threshold} days)')
            try:
                updated_summary = processor.implement_forgetting(days_threshold=args.days_threshold)
                if updated_summary:
                    print(f'Forgetting mechanism applied. New summary version: {updated_summary["version"]}')
                else:
                    print('No summaries available to apply forgetting mechanism.')
            except Exception as e:
                print(f'ERROR during forgetting mechanism: {e}')
                traceback.print_exc()

        # --- Final Stats ---
        cur = db.conn.cursor()
        cur.execute('SELECT COUNT(*) FROM rolling_summaries')
        global_count_after = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM conversations WHERE summary IS NOT NULL AND summary != ""')
        conv_summary_count_after = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM conversations')
        total_conv_after = cur.fetchone()[0]

        print_separator()
        print(f'✅ Summary generation process complete.')
        print(f'- Global summaries: {global_count_after}')
        if total_conv_after > 0:
            percent_summarized = int(conv_summary_count_after / total_conv_after * 100)
            print(f'- Conversation summaries: {conv_summary_count_after}/{total_conv_after} ({percent_summarized}%)')
        else:
            print('- Conversation summaries: 0/0 (0%)')

        return True # Indicate main process ran

    except ImportError as e:
        print(f'ERROR: Failed to import required modules: {e}')
        traceback.print_exc()
    except Exception as e:
        print(f'Unexpected error during main summarization: {e}')
        traceback.print_exc()
    finally:
        if db:
            db.close()

    return False # Indicate main process did not complete successfully


def prompt_to_view_summaries(args):
    """After generation, ask the user if they want to view summaries."""
    print()
    db = None
    try:
        db = MemoryDatabase(args.db)
        if args.conversation_summaries:
            answer = input("Would you like to view a specific conversation summary? (y/n) ").lower()
            if answer == 'y':
                print("\nRecent conversations with summaries:")
                print_separator()
                c = db.conn.cursor()
                c.execute('''
                SELECT id, title, timestamp
                FROM conversations
                WHERE summary IS NOT NULL AND summary != ''
                ORDER BY timestamp DESC
                LIMIT 5
                ''')
                convs = c.fetchall()
                if convs:
                    for i, conv in enumerate(convs, 1):
                        conv_id, title, timestamp = conv
                        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                        print(f"{i}. {title} - {date} (ID: {conv_id})")
                else:
                    print('No conversation summaries found.')
                print_separator()
                conv_id_to_view = input("Enter a conversation ID to view its summary: ")
                if conv_id_to_view:
                    print()
                    view_conversation_summary(args.db, conv_id_to_view)
        else:
            answer = input("Would you like to view the active global summary? (y/n) ").lower()
            if answer == 'y':
                print()
                view_active_summary(args.db)
    except Exception as e:
        print(f"Error during interactive prompt: {e}")
    finally:
        if db:
            db.close()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate, view, and manage conversation summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # --- Arguments from Shell Script ---
    parser.add_argument("--db", default="memory.db", help="Database path")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of conversations to process per batch for global summary")
    parser.add_argument("--forget", action="store_true", help="Apply forgetting mechanism after processing")
    parser.add_argument("--days", type=int, default=180, dest="days_threshold", help="Days threshold for forgetting mechanism")
    parser.add_argument("--verbose", action="store_true", help="Show detailed process information")
    parser.add_argument("--debug", action="store_true", help="Show API debug information (sets verbose)")
    parser.add_argument("--rebuild", action="store_true", help="Delete all existing summaries and rebuild from scratch")
    parser.add_argument("--conversation-summaries", "--conv", action="store_true", dest="conversation_summaries", help="Generate summaries for individual conversations")
    parser.add_argument("--no-embeddings", action="store_false", dest="generate_embeddings", help="Don't generate embeddings for new/updated summaries")
    parser.add_argument("--no-evaluation", action="store_false", dest="evaluate_summaries", help="Don't evaluate summary quality (placeholder)") # TODO: Implement evaluation toggle
    parser.add_argument("--workers", type=int, default=4, help="Maximum parallel workers for certain tasks (e.g., conversation summaries)") # TODO: Implement worker usage
    parser.add_argument("--limit", type=int, default=0, help="Limit number of conversations to process (0 for no limit)")

    # --- Action Arguments ---
    parser.add_argument("--embeddings", action="store_true", dest="generate_embeddings_only", help="Generate embeddings for existing summaries and exit")
    parser.add_argument("--themes", metavar="QUERY", help="Search summaries related to QUERY and exit")
    parser.add_argument("--view", action="store_true", help="View the active global summary and exit")
    parser.add_argument("--view-conv", metavar="CONV_ID", help="View summary for a specific conversation ID and exit")
    parser.add_argument("--list-convs", action="store_true", help="List conversations without summaries and exit")

    args = parser.parse_args()

    # --- Set Environment Variables based on args (for underlying modules) ---
    if args.verbose or args.debug:
        os.environ['VERBOSE_EMBEDDINGS'] = 'true'
    if args.debug:
        os.environ['DEBUG_EMBEDDINGS'] = 'true'
        os.environ['DEBUG_EVALUATION'] = 'true' # Assuming evaluation module checks this

    # Use dummy embeddings if db path suggests testing
    if "test" in args.db.lower():
         os.environ['USE_DUMMY_EMBEDDINGS'] = 'true'

    # --- Execute Action or Main Process ---
    main_process_ran = False
    if args.generate_embeddings_only:
        generate_summary_embeddings(args.db)
    elif args.themes:
        search_summaries_by_theme(args.db, args.themes)
    elif args.view:
        view_active_summary(args.db)
    elif args.view_conv:
        view_conversation_summary(args.db, args.view_conv)
    elif args.list_convs:
        list_conversations_without_summaries(args.db)
    else:
        # Default action: run the main summarization process
        main_process_ran = run_main_summarization(args)

    # --- Interactive Prompt (only if main process ran and not verbose) ---
    if main_process_ran and not args.verbose:
        prompt_to_view_summaries(args)
