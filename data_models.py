from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import datetime

class JournalEntry(BaseModel):
    """Represents a single processed journal entry."""
    entry_id: str # Unique ID (e.g., filename or hash)
    date: datetime.date
    raw_text: str
    processed_text: Optional[str] = None # Could hold cleaned text
    tags: List[str] = Field(default_factory=list)
    competencies_identified: List[str] = Field(default_factory=list)
    projects_mentioned: List[str] = Field(default_factory=list)
    source_file: str

class DocumentChunk(BaseModel):
    """Represents a chunk of text for the vector database."""
    chunk_id: str # Unique ID for the chunk
    entry_id: str # ID of the parent JournalEntry
    text: str
    metadata: Dict = Field(default_factory=dict) # Store date, tags etc. here for filtering

class ReportSection(BaseModel):
    """Represents a section in the report plan."""
    title: str
    level: int
    content: Optional[str] = None # Generated content goes here
    status: str = "pending" # e.g., pending, drafting, drafted, reviewed, final
    subsections: List['ReportSection'] = Field(default_factory=list)

ReportSection.model_rebuild() # Needed for self-referencing List['ReportSection']

class ReportPlan(BaseModel):
    """The overall structure of the report."""
    title: str = "Apprenticeship Report"
    structure: List[ReportSection]

class Citation(BaseModel):
    """Represents a reference for the bibliography."""
    key: str # e.g., Smith2023
    citation_type: str # book, article, web, etc.
    data: Dict # Fields like author, year, title, publisher, url etc.
    formatted_harvard: Optional[str] = None # Store the formatted string