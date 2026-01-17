"""
Angel Intelligence - Database Connection Management

Provides database connection pooling and management for MySQL.
Uses the 'ai' database connection as specified in the schema.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Any, Dict, List

import mysql.connector
from mysql.connector import pooling, Error as MySQLError

from src.config import get_settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Database connection manager with connection pooling.
    
    Provides context manager support for automatic connection cleanup.
    """
    
    _pool: Optional[pooling.MySQLConnectionPool] = None
    
    def __init__(self):
        """Initialise database connection configuration."""
        settings = get_settings()
        self.config = {
            'host': settings.ai_db_host,
            'port': settings.ai_db_port,
            'database': settings.ai_db_database,
            'user': settings.ai_db_username,
            'password': settings.ai_db_password,
        }
    
    @classmethod
    def get_pool(cls) -> pooling.MySQLConnectionPool:
        """Get or create the connection pool."""
        if cls._pool is None:
            settings = get_settings()
            cls._pool = pooling.MySQLConnectionPool(
                pool_name="angel_intelligence_pool",
                pool_size=5,
                pool_reset_session=True,
                host=settings.ai_db_host,
                port=settings.ai_db_port,
                database=settings.ai_db_database,
                user=settings.ai_db_username,
                password=settings.ai_db_password,
            )
            logger.info(f"Created database connection pool for {settings.ai_db_database}")
        return cls._pool
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a database connection from the pool.
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM ai_call_recordings")
        """
        conn = None
        try:
            conn = self.get_pool().get_connection()
            yield conn
        except MySQLError as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn and conn.is_connected():
                conn.close()
    
    @contextmanager
    def get_cursor(self, dictionary: bool = True) -> Generator:
        """
        Get a database cursor with automatic connection management.
        
        Usage:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM ai_call_recordings")
                rows = cursor.fetchall()
        """
        with self.get_connection() as conn:
            cursor = conn.cursor(dictionary=dictionary)
            try:
                yield cursor
                conn.commit()
            except MySQLError as e:
                conn.rollback()
                logger.error(f"Database error, rolling back: {e}")
                raise
            finally:
                cursor.close()
    
    def execute(self, query: str, params: tuple = None) -> int:
        """
        Execute a single query and return affected row count.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.rowcount
    
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """
        Execute a query and fetch one row.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Dictionary of column: value or None if no rows
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchone()
    
    def fetch_all(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a query and fetch all rows.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List of dictionaries
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.fetchall()
    
    def insert(self, query: str, params: tuple = None) -> int:
        """
        Execute an INSERT query and return the last insert ID.
        
        Args:
            query: INSERT SQL query
            params: Query parameters
            
        Returns:
            Last inserted row ID
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params or ())
            return cursor.lastrowid


# Global database connection instance
_db: Optional[DatabaseConnection] = None


def get_db_connection() -> DatabaseConnection:
    """Get the global database connection instance."""
    global _db
    if _db is None:
        _db = DatabaseConnection()
    return _db
