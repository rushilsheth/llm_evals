from typing import List, Optional, Dict, Literal
from datetime import datetime

from pydantic import BaseModel, Field
    
# Function to read emails
class ReadEmail(BaseModel):
    email_id: str = Field(..., description="ID of the email to read")

# Function to send a new email
class SendEmail(BaseModel):
    recipients: List[str] = Field(..., description="List of recipient email addresses")
    cc: Optional[List[str]] = Field(..., description="List of CC'd email addresses")
    subject: str = Field(..., description="Subject line of the email")
    body: str = Field(..., description="Body content of the email")
    thread_id: Optional[str] = Field(None, description="ID of the thread this email belongs to")

# Function to check security clearance for information sharing
class CheckClearance(BaseModel):
    information_ids: List[str] = Field(..., description="IDs of information pieces to check")
    recipient_emails: List[str] = Field(..., description="Email addresses to check clearance for")

# Function to analyze email for potential security issues
class AnalyzeEmailContent(BaseModel):
    email_content: str = Field(..., description="Content of the email to analyze")
    recipients: List[str] = Field(..., description="Recipients of this email")
    
# Function to update knowledge state after email exchange
class UpdateKnowledgeState(BaseModel):
    email_id: str = Field(..., description="ID of the email that was read")
    participant_email: str = Field(..., description="Email address of the participant who read it")
    content: str = Field(..., description="The actual information content")