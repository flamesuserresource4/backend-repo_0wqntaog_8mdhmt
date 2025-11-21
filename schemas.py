"""
Database Schemas for PlayerStock MVP

Each Pydantic model maps to a MongoDB collection. Collection name is the lowercase of the class name.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime

# Authentication & Users
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Unique email address")
    password_hash: str = Field(..., description="BCrypt password hash")
    avatar_url: Optional[str] = Field(None, description="Profile avatar URL")
    bio: Optional[str] = Field(None, description="Short bio")
    is_active: bool = Field(True)
    is_admin: bool = Field(False)

# Wallet & Transactions
class WalletTransaction(BaseModel):
    user_id: str = Field(..., description="User ObjectId as string")
    type: Literal["deposit", "withdrawal"]
    amount: float = Field(..., gt=0)
    currency: str = Field("USD")
    status: Literal["pending", "completed", "failed", "cancelled"] = "pending"
    provider: Optional[Literal["stripe", "paystack", "flutterwave", "manual"]] = None
    reference: Optional[str] = None
    meta: Optional[dict] = None

# Players & Pricing
class Player(BaseModel):
    name: str
    team: str
    nationality: Optional[str] = None
    position: Optional[str] = None
    league: Optional[str] = None
    cwc_status: Optional[Literal["current", "upcoming", "eliminated", "qualifying"]] = None
    image_url: Optional[str] = None
    is_active: bool = True
    momentum_score: float = 0.0

class PriceTick(BaseModel):
    player_id: str
    price: float = Field(..., gt=0)
    source: Literal["engine", "manual", "import"] = "engine"
    event: Optional[str] = None  # e.g., goal, assist, save, card
    ts: Optional[datetime] = None

# Trading & Portfolio
class Trade(BaseModel):
    user_id: str
    player_id: str
    side: Literal["buy", "sell"]
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    total: float = Field(..., gt=0)

class Position(BaseModel):
    user_id: str
    player_id: str
    quantity: float
    avg_price: float

# Community
class Comment(BaseModel):
    user_id: str
    player_id: str
    text: str
    reactions: Optional[List[Literal["like", "hot"]]] = []

class ChatMessage(BaseModel):
    user_id: str
    message: str

# Contests
class Contest(BaseModel):
    name: str
    description: Optional[str] = None
    start_at: datetime
    end_at: datetime
    pick_count: int = 5
    is_active: bool = True

class ContestEntry(BaseModel):
    contest_id: str
    user_id: str
    player_ids: List[str]
    score: float = 0.0

# Badges
class Badge(BaseModel):
    code: str
    label: str
    description: Optional[str] = None

class UserBadge(BaseModel):
    user_id: str
    badge_code: str
    awarded_at: Optional[datetime] = None

# Alerts
class Alert(BaseModel):
    user_id: str
    player_id: str
    direction: Literal["up", "down"]
    threshold: float
    active: bool = True
