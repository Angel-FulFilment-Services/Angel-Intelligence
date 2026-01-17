"""
Angel Intelligence - API Authentication

Bearer token authentication for all API requests.
Token is configured via API_AUTH_TOKEN environment variable.
"""

import logging
from typing import Optional

from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config import get_settings

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Authentication error."""
    pass


class BearerAuth(HTTPBearer):
    """Bearer token authentication scheme."""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication scheme. Use Bearer token."
                )
            
            if not self._verify_token(credentials.credentials):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or expired token."
                )
            
            return credentials.credentials
        
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing."
        )
    
    def _verify_token(self, token: str) -> bool:
        """Verify the bearer token against configured API token."""
        settings = get_settings()
        
        if not settings.api_auth_token:
            logger.warning("API_AUTH_TOKEN not configured - authentication disabled")
            return True
        
        if len(settings.api_auth_token) < 64:
            logger.warning("API_AUTH_TOKEN is less than 64 characters - insecure configuration")
        
        return token == settings.api_auth_token


# Global auth instance
bearer_auth = BearerAuth()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(bearer_auth)
) -> str:
    """
    Dependency for verifying bearer token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(token: str = Depends(verify_token)):
            return {"message": "Authenticated"}
    """
    return credentials
