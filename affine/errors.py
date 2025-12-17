from typing import Optional, Any
from datetime import datetime

class AffineError(Exception):
    """Base exception for all Affine errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        self.timestamp = datetime.now()

class NetworkError(AffineError):
    """Raised when a network request fails (e.g. connection error, timeout)."""
    def __init__(self, message: str, url: Optional[str] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.url = url
        self.original_error = original_error

    def __str__(self):
        return f"NetworkError(url={self.url}): {self.message}"

class ValidationError(AffineError):
    """Raised when data validation fails."""
    pass

class ApiResponseError(AffineError):
    """Raised when the API returns an error response (non-2xx or malformed)."""
    def __init__(self, message: str, status_code: int, url: str, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.body = body

    def __str__(self):
        return f"ApiResponseError(status={self.status_code}, url={self.url}): {self.message}"
