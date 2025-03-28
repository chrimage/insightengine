"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2025-03-27

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create conversations table
    op.create_table(
        'conversations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('model', sa.String()),
        sa.Column('message_count', sa.Integer(), default=0),
        sa.Column('summary', sa.String()),
        sa.Column('quality_score', sa.Float()),
        sa.Column('token_count', sa.Integer()),
        sa.Column('metadata_json', sa.String(), default="{}")
    )
    
    # Create messages table
    op.create_table(
        'messages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('metadata_json', sa.String(), default="{}")
    )
    
    # Create chunks table
    op.create_table(
        'chunks',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), sa.ForeignKey('conversations.id'), nullable=False),
        sa.Column('message_ids', sa.String()),
        sa.Column('content', sa.String(), nullable=False),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('metadata_json', sa.String(), default="{}")
    )
    
    # Create embeddings table
    op.create_table(
        'embeddings',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('source_id', sa.String(), nullable=False),
        sa.Column('vector_blob', sa.LargeBinary()),
        sa.Column('model', sa.String(), nullable=False),
        sa.Column('dimensions', sa.Integer(), nullable=False),
        sa.Column('source_type', sa.String(), nullable=False),
        sa.Column('vector_id', sa.Integer())
    )
    
    # Create rolling_summaries table
    op.create_table(
        'rolling_summaries',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('summary_text', sa.String(), nullable=False),
        sa.Column('conversation_range', sa.String()),
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('metadata_json', sa.String(), default="{}")
    )
    
    # Create insights table
    op.create_table(
        'insights',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('text', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.Float(), nullable=False),
        sa.Column('updated_at', sa.Float(), nullable=False),
        sa.Column('evidence', sa.String()),
        sa.Column('application_count', sa.Integer(), default=0),
        sa.Column('success_rate', sa.Float()),
        sa.Column('applications', sa.String()),
        sa.Column('metadata_json', sa.String(), default="{}")
    )
    
    # Create evaluations table
    op.create_table(
        'evaluations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('timestamp', sa.Float(), nullable=False),
        sa.Column('query', sa.String(), nullable=False),
        sa.Column('evaluation', sa.String(), nullable=False),
        sa.Column('structured', sa.Boolean(), default=True),
        sa.Column('meta_evaluation', sa.String())
    )
    
    # Create indexes
    op.create_index('idx_messages_conversation_id', 'messages', ['conversation_id'])
    op.create_index('idx_conversations_timestamp', 'conversations', ['timestamp'])
    op.create_index('idx_embeddings_source_id', 'embeddings', ['source_id'])
    op.create_index('idx_insights_category', 'insights', ['category'])
    op.create_index('idx_evaluations_timestamp', 'evaluations', ['timestamp'])


def downgrade() -> None:
    # Drop tables in reverse order of dependencies
    op.drop_table('evaluations')
    op.drop_table('insights')
    op.drop_table('rolling_summaries')
    op.drop_table('embeddings')
    op.drop_table('chunks')
    op.drop_table('messages')
    op.drop_table('conversations')