# tasks/cleanup.py

from celery import shared_task
from sqlalchemy import text
from db import get_master_db_session
import logging

logger = logging.getLogger(__name__)

@shared_task
def cleanup_old_metadata():
    try:
        db = next(get_master_db_session())
        result = db.execute(text("""
            DELETE FROM visualizations_metadata
            WHERE created_at < NOW() - INTERVAL '180 days'
        """))
        db.commit()
        logger.info(f"✅ Cleanup complete. Deleted {result.rowcount} old metadata records.")
        return {"deleted": result.rowcount}
    except Exception as e:
        logger.exception("❌ Failed to clean up metadata.")
        return {"error": str(e)}
