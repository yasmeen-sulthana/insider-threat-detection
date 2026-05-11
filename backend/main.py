import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, model

app = FastAPI(title="Insider Threat Detection API", version="1.0.0")

# ✅ FIXED CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all ports (important)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

app.include_router(upload.router)
app.include_router(model.router)

@app.get("/")
def root():
    return {"message": "Insider Threat Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}