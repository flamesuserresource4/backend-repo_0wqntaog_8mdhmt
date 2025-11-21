import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Set

from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext

from database import db, create_document, get_documents
from schemas import (
    User,
    WalletTransaction,
    Player,
    PriceTick,
    Trade,
    Comment,
    ChatMessage,
)

# App setup
app = FastAPI(title="PlayerStock API")

# Explicitly set allowed origins to the deployed frontend and localhost for safety
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://ta-01kakqd14e9qgacennf3755jrn-3000.wo-yqosysqdylcp6jeriqprmf0ji.w.modal.host",
)
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://localhost:3000",
    "https://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # We use Bearer tokens, not cookie credentials
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# Security / Auth helpers
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AuthBody(BaseModel):
    email: EmailStr
    password: str
    name: Optional[str] = None


# Mongo helpers
from bson import ObjectId


def to_str_id(doc: Dict[str, Any]) -> Dict[str, Any]:
    if not doc:
        return doc
    d = {**doc}
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # Convert nested ObjectIds
    for k, v in list(d.items()):
        if isinstance(v, ObjectId):
            d[k] = str(v)
    return d


# Auth dependencies
class AuthedUser(BaseModel):
    id: str
    email: EmailStr
    name: Optional[str] = None


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(authorization: str = Header(None)) -> AuthedUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        name: str = payload.get("name")
        if user_id is None or email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return AuthedUser(id=user_id, email=email, name=name)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Simple WebSocket manager for real-time broadcasts
class ConnectionManager:
    def __init__(self):
        self.ticks: Set[WebSocket] = set()
        self.chat: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket, channel: str):
        await ws.accept()
        if channel == "ticks":
            self.ticks.add(ws)
        elif channel == "chat":
            self.chat.add(ws)

    def disconnect(self, ws: WebSocket):
        self.ticks.discard(ws)
        self.chat.discard(ws)

    async def broadcast_ticks(self, data: dict):
        dead = []
        for ws in list(self.ticks):
            try:
                await ws.send_json({"type": "tick", **data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_chat(self, data: dict):
        dead = []
        for ws in list(self.chat):
            try:
                await ws.send_json({"type": "chat", **data})
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# Routes
@app.get("/")
def read_root():
    return {"message": "PlayerStock API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = getattr(db, "name", None) or "unknown"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# Auth endpoints
@app.post("/auth/register", response_model=Token)
def register(body: AuthBody):
    # Ensure email uniqueness
    existing = db["user"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=body.name or body.email.split("@")[0],
        email=body.email,
        password_hash=get_password_hash(body.password),
        is_active=True,
        is_admin=False,
    )
    user_id = create_document("user", user)
    token = create_access_token({"sub": user_id, "email": user.email, "name": user.name})
    return Token(access_token=token)


@app.post("/auth/login", response_model=Token)
def login(body: AuthBody):
    doc = db["user"].find_one({"email": body.email})
    if not doc:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    if not verify_password(body.password, doc.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    token = create_access_token({"sub": str(doc["_id"]), "email": doc["email"], "name": doc.get("name")})
    return Token(access_token=token)


@app.get("/me")
def me(user: AuthedUser = Depends(get_current_user)):
    doc = db["user"].find_one({"_id": ObjectId(user.id)})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return to_str_id(doc)


# Players
class UpsertPlayerBody(BaseModel):
    name: str
    team: str
    nationality: Optional[str] = None
    position: Optional[str] = None
    cwc_status: Optional[str] = None
    image_url: Optional[str] = None


@app.get("/players")
def list_players(status: Optional[str] = None, q: Optional[str] = None):
    filt: Dict[str, Any] = {}
    if status:
        filt["cwc_status"] = status
    if q:
        filt["name"] = {"$regex": q, "$options": "i"}
    players = [to_str_id(d) for d in db["player"].find(filt).limit(200)]
    return players


@app.post("/players")
async def create_player(body: UpsertPlayerBody, user: AuthedUser = Depends(get_current_user)):
    # Any authenticated user can create for now
    p = Player(
        name=body.name, team=body.team, nationality=body.nationality,
        position=body.position, cwc_status=body.cwc_status, image_url=body.image_url,
        is_active=True
    )
    pid = create_document("player", p)
    # Seed an initial price if none
    if db["pricetick"].count_documents({"player_id": pid}) == 0:
        create_document("pricetick", PriceTick(player_id=pid, price=10.0))
        # Broadcast a synthetic initial tick
        await manager.broadcast_ticks({"player_id": pid, "price": 10.0, "event": "seed"})
    return {"id": pid}


@app.get("/players/{player_id}")
def get_player(player_id: str):
    doc = db["player"].find_one({"_id": ObjectId(player_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    return to_str_id(doc)


@app.get("/players/{player_id}/prices")
def get_prices(player_id: str, limit: int = 100):
    items = [to_str_id(d) for d in db["pricetick"].find({"player_id": player_id}).sort("_id", -1).limit(limit)]
    return list(reversed(items))


class TickBody(BaseModel):
    price: float
    event: Optional[str] = None


@app.post("/players/{player_id}/tick")
async def add_tick(player_id: str, body: TickBody, user: AuthedUser = Depends(get_current_user)):
    if db["player"].count_documents({"_id": ObjectId(player_id)}) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    tid = create_document("pricetick", PriceTick(player_id=player_id, price=body.price, event=body.event))
    # Broadcast to live subscribers
    await manager.broadcast_ticks({"player_id": player_id, "price": body.price, "event": body.event})
    return {"id": tid}


# Trading
class TradeBody(BaseModel):
    player_id: str
    side: str  # buy/sell
    quantity: float


@app.post("/trade")
def place_trade(body: TradeBody, user: AuthedUser = Depends(get_current_user)):
    # Get latest price
    last_tick = db["pricetick"].find({"player_id": body.player_id}).sort("_id", -1).limit(1)
    price = None
    for t in last_tick:
        price = t.get("price")
    if price is None:
        raise HTTPException(status_code=400, detail="No price available for player")

    side = body.side.lower()
    if side not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail="Invalid side")

    total = round(price * body.quantity, 2)

    # Check wallet balance for buy
    if side == "buy":
        wallet_balance = get_wallet_balance(user.id)
        if wallet_balance < total:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        # Lock funds by creating a negative pending transaction? Simplify by recording trade only
    else:
        # Optional: enforce position availability (skipped for MVP)
        pass

    trade = Trade(
        user_id=user.id,
        player_id=body.player_id,
        side=side,
        quantity=body.quantity,
        price=price,
        total=total,
    )
    trade_id = create_document("trade", trade)

    # Mirror to wallet: deduct/add cash via synthetic transaction for simple accounting
    cash_delta = -total if side == "buy" else total
    create_document(
        "wallettransaction",
        WalletTransaction(
            user_id=user.id,
            type="deposit" if cash_delta > 0 else "withdrawal",
            amount=abs(cash_delta),
            currency="USD",
            status="completed",
            provider="manual",
            reference=f"trade:{trade_id}",
        ),
    )

    return {"id": trade_id, "price": price, "total": total}


@app.get("/portfolio")
def portfolio(user: AuthedUser = Depends(get_current_user)):
    # Aggregate positions from trades
    trades = list(db["trade"].find({"user_id": user.id}))
    positions: Dict[str, Dict[str, float]] = {}
    for t in trades:
        pid = t["player_id"]
        qty = float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
        cost = float(t["price"]) * float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
        p = positions.setdefault(pid, {"quantity": 0.0, "cost": 0.0})
        p["quantity"] += qty
        p["cost"] += cost

    # Current prices
    def latest_price(pid: str) -> Optional[float]:
        doc = db["pricetick"].find({"player_id": pid}).sort("_id", -1).limit(1)
        for d in doc:
            return float(d.get("price"))
        return None

    items = []
    total_equity = 0.0
    for pid, p in positions.items():
        if abs(p["quantity"]) < 1e-9:
            continue
        price = latest_price(pid) or 0.0
        value = price * p["quantity"]
        avg_price = (p["cost"] / p["quantity"]) if p["quantity"] != 0 else 0.0
        pnl = (price - avg_price) * p["quantity"]
        total_equity += value
        player = db["player"].find_one({"_id": ObjectId(pid)}) or {}
        items.append({
            "player_id": pid,
            "player_name": player.get("name"),
            "quantity": round(p["quantity"], 4),
            "avg_price": round(avg_price, 4),
            "price": round(price, 4),
            "value": round(value, 2),
            "pnl": round(pnl, 2),
        })

    return {"positions": items, "equity": round(total_equity, 2), "cash": round(get_wallet_balance(user.id), 2)}


# Wallet
class WalletAction(BaseModel):
    amount: float
    provider: Optional[str] = None
    reference: Optional[str] = None


@app.post("/wallet/deposit")
def deposit(body: WalletAction, user: AuthedUser = Depends(get_current_user)):
    if body.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    tx = WalletTransaction(
        user_id=user.id,
        type="deposit",
        amount=body.amount,
        currency="USD",
        status="completed",  # In real integration, start as pending
        provider=(body.provider or "manual"),
        reference=body.reference,
    )
    txid = create_document("wallettransaction", tx)
    return {"id": txid, "balance": round(get_wallet_balance(user.id), 2)}


@app.post("/wallet/withdraw")
def withdraw(body: WalletAction, user: AuthedUser = Depends(get_current_user)):
    if body.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    if get_wallet_balance(user.id) < body.amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    tx = WalletTransaction(
        user_id=user.id,
        type="withdrawal",
        amount=body.amount,
        currency="USD",
        status="completed",
        provider=(body.provider or "manual"),
        reference=body.reference,
    )
    txid = create_document("wallettransaction", tx)
    return {"id": txid, "balance": round(get_wallet_balance(user.id), 2)}


@app.get("/wallet/transactions")
def wallet_transactions(user: AuthedUser = Depends(get_current_user)):
    items = [to_str_id(d) for d in db["wallettransaction"].find({"user_id": user.id}).sort("_id", -1).limit(200)]
    return items


def get_wallet_balance(user_id: str) -> float:
    agg = db["wallettransaction"].aggregate([
        {"$match": {"user_id": user_id, "status": "completed"}},
        {"$group": {
            "_id": None,
            "deposits": {"$sum": {"$cond": [[{"$eq": ["$type", "deposit"]}, "$amount", 0]]}},
            "withdrawals": {"$sum": {"$cond": [[{"$eq": ["$type", "withdrawal"]}, "$amount", 0]]}},
        }},
    ])
    total_deposits = 0.0
    total_withdrawals = 0.0
    for row in agg:
        total_deposits = float(row.get("deposits", 0))
        total_withdrawals = float(row.get("withdrawals", 0))
    return total_deposits - total_withdrawals


# Comments & Chat
class CommentBody(BaseModel):
    text: str


@app.post("/players/{player_id}/comments")
def add_comment(player_id: str, body: CommentBody, user: AuthedUser = Depends(get_current_user)):
    if db["player"].count_documents({"_id": ObjectId(player_id)}) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    cid = create_document("comment", Comment(user_id=user.id, player_id=player_id, text=body.text))
    return {"id": cid}


@app.get("/players/{player_id}/comments")
def list_comments(player_id: str):
    items = [to_str_id(d) for d in db["comment"].find({"player_id": player_id}).sort("_id", -1).limit(100)]
    return items


class ChatBody(BaseModel):
    message: str


@app.post("/chat")
async def post_chat(body: ChatBody, user: AuthedUser = Depends(get_current_user)):
    mid = create_document("chatmessage", ChatMessage(user_id=user.id, message=body.message))
    # Broadcast new chat message
    await manager.broadcast_chat({"id": mid, "message": body.message})
    return {"id": mid}


@app.get("/chat")
def get_chat(limit: int = 50):
    items = [to_str_id(d) for d in db["chatmessage"].find({}).sort("_id", -1).limit(limit)]
    return list(reversed(items))


# Leaderboard (by equity growth proxy: realized + unrealized from trades)
@app.get("/leaderboard")
def leaderboard():
    # Get distinct users from trades
    user_ids = db["trade"].distinct("user_id")
    leaders = []
    for uid in user_ids:
        # Build portfolio for user
        trades = list(db["trade"].find({"user_id": uid}))
        positions: Dict[str, Dict[str, float]] = {}
        for t in trades:
            pid = t["player_id"]
            qty = float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
            cost = float(t["price"]) * float(t["quantity"]) * (1 if t["side"] == "buy" else -1)
            p = positions.setdefault(pid, {"quantity": 0.0, "cost": 0.0})
            p["quantity"] += qty
            p["cost"] += cost
        equity = 0.0
        for pid, p in positions.items():
            if abs(p["quantity"]) < 1e-9:
                continue
            doc = db["pricetick"].find({"player_id": pid}).sort("_id", -1).limit(1)
            price = None
            for d in doc:
                price = float(d.get("price"))
            if price is None:
                price = 0.0
            equity += price * p["quantity"]
        cash = get_wallet_balance(uid)
        leaders.append({"user_id": uid, "net_worth": round(equity + cash, 2)})
    leaders.sort(key=lambda x: x["net_worth"], reverse=True)
    return leaders[:50]


# WebSocket endpoints
@app.websocket("/ws/ticks")
async def ws_ticks(ws: WebSocket):
    await manager.connect(ws, "ticks")
    try:
        while True:
            # Keep the connection alive; we don't expect messages from client for now
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await manager.connect(ws, "chat")
    try:
        while True:
            # If client sends a message directly on WS, echo to broadcast as chat
            data = await ws.receive_text()
            mid = create_document("chatmessage", ChatMessage(user_id="anon", message=data))
            await manager.broadcast_chat({"id": mid, "message": data})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
