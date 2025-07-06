"""Database module using SQLAlchemy for storing prompts, responses and trees."""

from datetime import datetime
import json
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class PromptResponse(Base):
    """ORM model representing a prompt and generated response."""

    __tablename__ = "prompt_response"

    id = Column(Integer, primary_key=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Database:
    """Database wrapper for simple CRUD operations."""

    def __init__(self, url: str = "sqlite:///loom.db") -> None:
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)
        self._upgrade_schema()
        self.Session = sessionmaker(bind=self.engine)

    def _upgrade_schema(self) -> None:
        """Ensure new columns exist for older databases."""
        inspector = inspect(self.engine)
        if "tree" not in inspector.get_table_names():
            return
        columns = {col["name"] for col in inspector.get_columns("tree")}
        with self.engine.begin() as conn:
            if "length" not in columns:
                conn.execute(text("ALTER TABLE tree ADD COLUMN length INTEGER DEFAULT 100"))
            if "temperature" not in columns:
                conn.execute(text("ALTER TABLE tree ADD COLUMN temperature FLOAT DEFAULT 0.8"))
            if "model" not in columns:
                conn.execute(text("ALTER TABLE tree ADD COLUMN model TEXT DEFAULT 'sshleifer/tiny-gpt2'"))
            if "variants" not in columns:
                conn.execute(text("ALTER TABLE tree ADD COLUMN variants INTEGER DEFAULT 3"))

    def add_record(self, prompt: str, response: str) -> None:
        """Add a generated response to the database."""
        session = self.Session()
        record = PromptResponse(prompt=prompt, response=response)
        session.add(record)
        session.commit()
        session.close()

    def get_all(self):
        """Retrieve all stored records sorted by timestamp."""
        session = self.Session()
        records = (
            session.query(PromptResponse)
            .order_by(PromptResponse.timestamp.desc())
            .all()
        )
        session.close()
        return records

    def add_tree(
        self,
        tree_json: str,
        length: int = 100,
        temperature: float = 0.8,
        model: str = "sshleifer/tiny-gpt2",
        variants: int = 3,
    ) -> int:
        """Store a full generation tree and associated settings."""
        session = self.Session()
        record = TreeRecord(
            data=tree_json,
            length=length,
            temperature=temperature,
            model=model,
            variants=variants,
        )
        session.add(record)
        session.commit()
        tree_id = record.id
        session.close()
        return tree_id

    def update_tree(
        self,
        tree_id: int,
        tree_json: str,
        length: int | None = None,
        temperature: float | None = None,
        model: str | None = None,
        variants: int | None = None,
    ) -> None:
        """Update an existing tree record."""
        session = self.Session()
        record = session.query(TreeRecord).get(tree_id)
        if record:
            record.data = tree_json
            if length is not None:
                record.length = length
            if temperature is not None:
                record.temperature = temperature
            if model is not None:
                record.model = model
            if variants is not None:
                record.variants = variants
            record.timestamp = datetime.utcnow()
            session.commit()
        session.close()

    def get_all_trees(self):
        """Retrieve all saved trees ordered by timestamp."""
        session = self.Session()
        records = session.query(TreeRecord).order_by(TreeRecord.timestamp.desc()).all()
        session.close()
        return records

    def get_tree(self, tree_id: int):
        """Return a single tree by id."""
        session = self.Session()
        record = session.query(TreeRecord).get(tree_id)
        session.close()
        return record

    def get_blank_tree(self):
        """Return an existing blank tree if one exists."""
        session = self.Session()
        records = (
            session.query(TreeRecord)
            .order_by(TreeRecord.timestamp.desc())
            .all()
        )
        for rec in records:
            try:
                data = json.loads(rec.data)
                if data == [{"id": 0, "text": "", "parent": None}]:
                    session.close()
                    return rec
            except Exception:
                continue
        session.close()
        return None

    def delete_tree(self, tree_id: int) -> None:
        """Delete a tree by id."""
        session = self.Session()
        record = session.query(TreeRecord).get(tree_id)
        if record:
            session.delete(record)
            session.commit()
        session.close()

    def duplicate_tree(self, tree_id: int) -> int | None:
        """Duplicate a tree record and return the new id."""
        session = self.Session()
        record = session.query(TreeRecord).get(tree_id)
        if not record:
            session.close()
            return None
        new_record = TreeRecord(
            data=record.data,
            length=record.length,
            temperature=record.temperature,
            model=record.model,
            variants=record.variants,
        )
        session.add(new_record)
        session.commit()
        new_id = new_record.id
        session.close()
        return new_id


class TreeRecord(Base):
    """ORM model storing an entire generation tree as JSON."""

    __tablename__ = "tree"

    id = Column(Integer, primary_key=True)
    data = Column(Text, nullable=False)
    length = Column(Integer, default=100)
    temperature = Column(Float, default=0.8)
    model = Column(String, default="sshleifer/tiny-gpt2")
    variants = Column(Integer, default=3)
    timestamp = Column(DateTime, default=datetime.utcnow)


