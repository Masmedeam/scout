"""Add Scout geospatial image index models

Revision ID: 8f6d9a7b2c31
Revises: fe56fa70289e
Create Date: 2026-06-06 16:45:00.000000

"""
from alembic import op
import sqlalchemy as sa
import sqlmodel.sql.sqltypes


# revision identifiers, used by Alembic.
revision = "8f6d9a7b2c31"
down_revision = "fe56fa70289e"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "scoutlocation",
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("west", sa.Float(), nullable=False),
        sa.Column("south", sa.Float(), nullable=False),
        sa.Column("east", sa.Float(), nullable=False),
        sa.Column("north", sa.Float(), nullable=False),
        sa.Column(
            "description", sqlmodel.sql.sqltypes.AutoString(length=500), nullable=True
        ),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scoutlocation_name"), "scoutlocation", ["name"], unique=True)

    op.create_table(
        "scoutrasterasset",
        sa.Column(
            "source_provider",
            sqlmodel.sql.sqltypes.AutoString(length=100),
            nullable=False,
        ),
        sa.Column(
            "source_uri", sqlmodel.sql.sqltypes.AutoString(length=1000), nullable=True
        ),
        sa.Column("license", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=True),
        sa.Column(
            "capture_date", sqlmodel.sql.sqltypes.AutoString(length=50), nullable=True
        ),
        sa.Column("file_path", sqlmodel.sql.sqltypes.AutoString(length=1000), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("west", sa.Float(), nullable=False),
        sa.Column("south", sa.Float(), nullable=False),
        sa.Column("east", sa.Float(), nullable=False),
        sa.Column("north", sa.Float(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("location_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["location_id"], ["scoutlocation.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "scoutsearchrun",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "query_image_path",
            sqlmodel.sql.sqltypes.AutoString(length=1000),
            nullable=False,
        ),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("predicted_lat", sa.Float(), nullable=True),
        sa.Column("predicted_lon", sa.Float(), nullable=True),
        sa.Column("matches", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "scoutimagepatch",
        sa.Column("file_path", sqlmodel.sql.sqltypes.AutoString(length=1000), nullable=False),
        sa.Column("pixel_x", sa.Integer(), nullable=False),
        sa.Column("pixel_y", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("west", sa.Float(), nullable=False),
        sa.Column("south", sa.Float(), nullable=False),
        sa.Column("east", sa.Float(), nullable=False),
        sa.Column("north", sa.Float(), nullable=False),
        sa.Column("center_lat", sa.Float(), nullable=False),
        sa.Column("center_lon", sa.Float(), nullable=False),
        sa.Column(
            "embedding_model",
            sqlmodel.sql.sqltypes.AutoString(length=100),
            nullable=False,
        ),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("redis_key", sqlmodel.sql.sqltypes.AutoString(length=255), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raster_asset_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["raster_asset_id"], ["scoutrasterasset.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_scoutimagepatch_redis_key"),
        "scoutimagepatch",
        ["redis_key"],
        unique=False,
    )


def downgrade():
    op.drop_index(op.f("ix_scoutimagepatch_redis_key"), table_name="scoutimagepatch")
    op.drop_table("scoutimagepatch")
    op.drop_table("scoutsearchrun")
    op.drop_table("scoutrasterasset")
    op.drop_index(op.f("ix_scoutlocation_name"), table_name="scoutlocation")
    op.drop_table("scoutlocation")
