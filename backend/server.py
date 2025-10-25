from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from route_optimizer import RouteOptimizer

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize RouteOptimizer
try:
    route_optimizer = RouteOptimizer(
        graph_path=str(ROOT_DIR / 'station_graph.pkl'),
        model_path=str(ROOT_DIR / 'railway_model.keras'),
        scaler_path=str(ROOT_DIR / 'scaler.pkl'),
        mappings_path=str(ROOT_DIR / 'station_mappings.pkl')
    )
    print("✅ Route optimizer initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize route optimizer: {e}")
    print("Please run data preprocessing and model training first.")
    route_optimizer = None

# Define Models
class Station(BaseModel):
    code: str
    name: str

class RouteSegment(BaseModel):
    from_station: str = Field(alias='from')
    from_name: str
    to: str
    to_name: str
    distance: float
    time: float
    
    model_config = ConfigDict(populate_by_name=True)

class RouteResponse(BaseModel):
    path: List[Station]
    route_details: List[RouteSegment]
    total_distance: float
    total_time: float
    total_time_hours: float

class RouteRequest(BaseModel):
    source: str
    destination: str

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Railway Path Optimization System API", "status": "active"}

@api_router.get("/stations", response_model=List[Station])
async def get_stations():
    """
    Get all available railway stations
    """
    if route_optimizer is None:
        raise HTTPException(status_code=503, detail="Route optimizer not initialized. Please run preprocessing and training first.")
    
    try:
        stations = route_optimizer.get_all_stations()
        return stations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/route", response_model=RouteResponse)
async def find_route(request: RouteRequest):
    """
    Find optimal route between two stations
    """
    if route_optimizer is None:
        raise HTTPException(status_code=503, detail="Route optimizer not initialized. Please run preprocessing and training first.")
    
    try:
        result = route_optimizer.find_optimal_route(request.source, request.destination)
        
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"No route found between {request.source} and {request.destination}"
            )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/station/{station_code}/connections")
async def get_connections(station_code: str):
    """
    Get all direct connections from a station
    """
    if route_optimizer is None:
        raise HTTPException(status_code=503, detail="Route optimizer not initialized")
    
    try:
        connections = route_optimizer.get_station_connections(station_code)
        return {"station_code": station_code, "connections": connections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "optimizer_ready": route_optimizer is not None
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()