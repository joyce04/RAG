import os
import httpx
from fastapi import FastAPI, Depends, HTTPException, Header, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from database import engine, get_db
import models

load_dotenv()

# Initialize DB
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Adaptive RAG API")

# Setup CORS - to allow frontend to communicate with backend cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://frontend:3000")

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    # Read the cookie from the incoming request
    cookies = request.cookies
    if not cookies:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Call NextAuth endpoint to verify the session
    async with httpx.AsyncClient() as client:
        try:
            # We must pass the cookies to NextAuth to identify the session
            response = await client.get(f"{FRONTEND_URL}/api/auth/session", cookies=cookies)
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid session")
            
            session_data = response.json()
            if not session_data or "user" not in session_data:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            user_info = session_data["user"]
            email = user_info.get("email")
            name = user_info.get("name")
            image = user_info.get("image")
            
            if not email:
                raise HTTPException(status_code=401, detail="Email not found in session")
            
            # Upsert user in our database
            user = db.query(models.User).filter(models.User.email == email).first()
            if not user:
                user = models.User(email=email, name=name, avatar_url=image)
                db.add(user)
                db.commit()
                db.refresh(user)
            
            return user
            
        except httpx.RequestError as e:
            print(f"Error reaching frontend for auth: {e}")
            # Fallback for local dev if frontend isn't reachable via docker hostname
            if "frontend:3000" in FRONTEND_URL:
                try:
                    response = await client.get("http://localhost:3000/api/auth/session", cookies=cookies)
                    if response.status_code == 200 and response.json().get("user"):
                        user_info = response.json()["user"]
                        email = user_info.get("email")
                        user = db.query(models.User).filter(models.User.email == email).first()
                        if not user:
                            user = models.User(email=email, name=user_info.get("name"), avatar_url=user_info.get("image"))
                            db.add(user)
                            db.commit()
                            db.refresh(user)
                        return user
                except Exception as ex:
                    print(f"Fallback auth failed: {ex}")
            raise HTTPException(status_code=401, detail="Authentication server unreachable")

api_router = APIRouter(prefix="/api/v1")

@api_router.get("/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

class SessionCreate(BaseModel):
    title: str = "New Chat"

@api_router.get("/sessions")
def list_sessions(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(models.Session).filter(models.Session.user_id == current_user.id).order_by(models.Session.created_at.desc()).all()

@api_router.post("/sessions")
def create_session(session_data: SessionCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_session = models.Session(user_id=current_user.id, title=session_data.title)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

@api_router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(models.Session).filter(models.Session.id == session_id, models.Session.user_id == current_user.id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess.messages

class ChatRequest(BaseModel):
    session_id: str
    question: str

@api_router.post("/chat")
def chat_request(req: ChatRequest, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.query(models.Session).filter(models.Session.id == req.session_id, models.Session.user_id == current_user.id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Save User Message
    user_msg = models.Message(session_id=sess.id, role="user", content=req.question)
    db.add(user_msg)
    db.commit()

    # Call LangGraph
    from graph.graph import app as graph_app
    print(f"Invoking graph with question: {req.question}")
    result = graph_app.invoke({'question': req.question})
    
    answer_text = result.get('generation', "Sorry, I couldn't generate an answer.")
    references = result.get('references', [])

    # Save Assistant Message
    assistant_msg = models.Message(session_id=sess.id, role="assistant", content=answer_text, references=references)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    
    # Update title optionally if it is "New Chat"
    if sess.title == "New Chat":
        sess.title = req.question[:30] + "..."
        db.commit()

    return assistant_msg

app.include_router(api_router)

# Mount static files for PDFs
# Documents are inside data/korean_competition_law_cases
docs_path = os.path.join(os.path.dirname(__file__), "data", "korean_competition_law_cases")
if os.path.exists(docs_path):
    app.mount("/api/v1/docs", StaticFiles(directory=docs_path), name="docs")
else:
    print(f"Warning: Docs path {docs_path} does not exist.")

app.include_router(api_router)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
