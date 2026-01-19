"""
Angel Intelligence - Interactive Service Proxy

HTTP client for proxying requests to the interactive service.
Used when running as API-only pod (WORKER_MODE=api) to delegate
AI workloads to dedicated interactive worker pods.

The Kubernetes service 'angel-intelligence-interactive' provides
automatic load balancing across all interactive pods.
"""

import logging
import httpx
from typing import Optional, List, Dict, Any

from src.config import get_settings

logger = logging.getLogger(__name__)


class InteractiveServiceProxy:
    """
    HTTP client for proxying to the interactive service.
    
    This allows the API pod to delegate AI workloads to interactive
    worker pods via the K8s service load balancer.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.interactive_service_url
        
        if not self.base_url:
            raise ValueError(
                "INTERACTIVE_SERVICE_URL must be set when WORKER_MODE=api. "
                "In K8s, use: http://angel-intelligence-interactive:8000"
            )
        
        # Remove trailing slash for consistency
        self.base_url = self.base_url.rstrip('/')
        
        # HTTP client with reasonable timeouts for AI inference
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=300.0,    # Read timeout (5 mins for slow inference)
                write=30.0,    # Write timeout
                pool=10.0      # Pool acquisition timeout
            ),
            headers={
                "Content-Type": "application/json",
            }
        )
        
        # Async client for async endpoints
        self._async_client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"InteractiveServiceProxy initialised: base_url={self.base_url}")
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            # Get auth token from settings for internal service calls
            headers = {"Content-Type": "application/json"}
            if self.settings.api_auth_token:
                headers["Authorization"] = f"Bearer {self.settings.api_auth_token}"
            
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=300.0,
                    write=30.0,
                    pool=10.0
                ),
                headers=headers
            )
        return self._async_client
    
    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Proxy a chat request to the interactive service.
        
        Returns the same format as InteractiveService.chat()
        """
        try:
            payload = {
                "message": message,
                "max_tokens": max_tokens,
            }
            if context:
                payload["context"] = context
            if conversation_history:
                payload["conversation_history"] = conversation_history
            
            logger.info(f"Proxying chat request to {self.base_url}/internal/chat")
            
            response = self.client.post("/internal/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Chat proxy succeeded: {len(result.get('response', ''))} chars")
            
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"Chat proxy timeout: {e}")
            return {
                "response": "The AI service is taking longer than expected. Please try again.",
                "error": True,
                "error_type": "timeout",
                "generation_time": 0.0,
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Chat proxy HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "response": f"AI service error: {e.response.status_code}",
                "error": True,
                "error_type": "http_error",
                "generation_time": 0.0,
            }
        except Exception as e:
            logger.error(f"Chat proxy error: {e}")
            return {
                "response": f"Failed to reach AI service: {str(e)}",
                "error": True,
                "error_type": "connection_error",
                "generation_time": 0.0,
            }
    
    async def chat_async(
        self,
        message: str,
        context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Async version of chat proxy.
        """
        try:
            payload = {
                "message": message,
                "max_tokens": max_tokens,
            }
            if context:
                payload["context"] = context
            if conversation_history:
                payload["conversation_history"] = conversation_history
            
            logger.info(f"Proxying async chat request to {self.base_url}/internal/chat")
            
            client = self._get_async_client()
            response = await client.post("/internal/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Async chat proxy succeeded: {len(result.get('response', ''))} chars")
            
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"Async chat proxy timeout: {e}")
            return {
                "response": "The AI service is taking longer than expected. Please try again.",
                "error": True,
                "error_type": "timeout",
                "generation_time": 0.0,
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Async chat proxy HTTP error: {e.response.status_code} - {e.response.text}")
            return {
                "response": f"AI service error: {e.response.status_code}",
                "error": True,
                "error_type": "http_error",
                "generation_time": 0.0,
            }
        except Exception as e:
            logger.error(f"Async chat proxy error: {e}")
            return {
                "response": f"Failed to reach AI service: {str(e)}",
                "error": True,
                "error_type": "connection_error",
                "generation_time": 0.0,
            }
    
    def generate_summary(
        self,
        transcript: str,
        summary_type: str = "brief",
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Proxy a summary generation request to the interactive service.
        """
        try:
            payload = {
                "transcript": transcript,
                "summary_type": summary_type,
            }
            if custom_prompt:
                payload["custom_prompt"] = custom_prompt
            
            logger.info(f"Proxying summary request to {self.base_url}/internal/summary")
            
            response = self.client.post("/internal/summary", json=payload)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Summary proxy error: {e}")
            return {
                "summary": f"Failed to generate summary: {str(e)}",
                "error": True,
                "generation_time": 0.0,
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of the interactive service.
        """
        try:
            response = self.client.get("/internal/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }
    
    def close(self):
        """Close HTTP clients."""
        self.client.close()
        if self._async_client:
            # Note: async client should be closed with await
            pass
    
    async def aclose(self):
        """Async close HTTP clients."""
        self.client.close()
        if self._async_client:
            await self._async_client.aclose()


# Singleton instance
_proxy_instance: Optional[InteractiveServiceProxy] = None


def get_interactive_proxy() -> InteractiveServiceProxy:
    """Get or create the interactive service proxy singleton."""
    global _proxy_instance
    if _proxy_instance is None:
        _proxy_instance = InteractiveServiceProxy()
    return _proxy_instance
