"""Add pgvector patch embeddings

Revision ID: 2b7c7d44e7c1
Revises: 8f6d9a7b2c31
Create Date: 2026-06-06 18:05:00.000000

"""
from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2b7c7d44e7c1"
down_revision = "8f6d9a7b2c31"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "scoutimagepatch",
        sa.Column("embedding", Vector(512), nullable=True),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_scoutimagepatch_embedding_hnsw "
        "ON scoutimagepatch USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_scoutimagepatch_embedding_hnsw")
    op.drop_column("scoutimagepatch", "embedding")
