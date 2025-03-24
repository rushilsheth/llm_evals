from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime
import json


class EmailParticipant(BaseModel):
    name: str = Field(..., description="Full name of the participant")
    email: str = Field(..., description="Email address of the participant")
    clearance_level: int = Field(..., description="Security clearance level (1, 2, or 3)")
    
class Email(BaseModel):
    id: str = Field(..., description="Unique identifier for the email")
    sender: str = Field(..., description="Email address of the sender")
    recipients: List[str] = Field(..., description="List of recipient email addresses")
    cc: Optional[List[str]] = Field(default=[], description="List of CC'd email addresses")
    subject: str = Field(..., description="Subject line of the email")
    body: str = Field(..., description="Body content of the email")
    timestamp: datetime = Field(default_factory=datetime.now, description="Time the email was sent")
    thread_id: Optional[str] = Field(None, description="ID of the thread this email belongs to")
    
class InformationPiece(BaseModel):
    id: str = Field(..., description="Unique identifier for this piece of information")
    content: str = Field(..., description="The actual information content")
    required_clearance: int = Field(..., description="Minimum clearance level required (1, 2, or 3)")