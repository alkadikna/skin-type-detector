from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from datetime import datetime
import os

# MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://host.docker.internal:27017")
DATABASE_NAME = "skin-type-detector"
COLLECTION_NAME = "skin-type-db"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DATABASE_NAME]

async def get_db():
    return db

async def save_prediction(db, user_id: str, predicted_type: str, confidence: float):
    doc = {
        "user_id": user_id,
        "predicted_skin_type": predicted_type,
        "confidence": confidence,
        "created_at": datetime.utcnow(),
        "actual_skin_type": None
    }
    result = await db[COLLECTION_NAME].insert_one(doc)
    return result.inserted_id

async def update_feedback(db, prediction_id: str, actual_type: str):
    result = await db[COLLECTION_NAME].update_one(
        {"_id": ObjectId(prediction_id)},
        {"$set": {"actual_skin_type": actual_type}}
    )
    return result.modified_count > 0