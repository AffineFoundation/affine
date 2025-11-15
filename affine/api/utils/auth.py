"""
Authentication Utilities

Signature verification and authentication helpers.
"""

import json
from typing import Optional, Tuple
from fastapi import Request, HTTPException, status
from bittensor import Keypair


def verify_signature(payload: str, signature: str, hotkey: str) -> bool:
    """
    Verify that a signature matches the payload and hotkey.
    
    Args:
        payload: JSON payload string
        signature: Hex-encoded signature
        hotkey: SS58 address of the signing keypair
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        # Create keypair from SS58 address
        keypair = Keypair(ss58_address=hotkey)
        
        # Convert signature from hex
        signature_bytes = bytes.fromhex(signature.replace("0x", ""))
        
        # Verify signature
        return keypair.verify(payload.encode(), signature_bytes)
    except Exception as e:
        print(f"Signature verification error: {e}")
        return False


def get_hotkey_from_request(request: Request) -> str:
    """
    Extract hotkey from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Hotkey from X-Hotkey header
        
    Raises:
        HTTPException: If hotkey header is missing
    """
    hotkey = request.headers.get("X-Hotkey")
    if not hotkey:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Hotkey header"
        )
    return hotkey


def get_signature_from_request(request: Request) -> str:
    """
    Extract signature from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Signature from X-Signature header
        
    Raises:
        HTTPException: If signature header is missing
    """
    signature = request.headers.get("X-Signature")
    if not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Signature header"
        )
    return signature


async def verify_request_signature(request: Request, payload: dict) -> Tuple[str, bool]:
    """
    Verify the signature of a request against its payload.
    
    Args:
        request: FastAPI request object
        payload: Request payload as dict
        
    Returns:
        Tuple of (hotkey, is_valid)
        
    Raises:
        HTTPException: If authentication headers are missing
    """
    hotkey = get_hotkey_from_request(request)
    signature = get_signature_from_request(request)
    
    # Serialize payload to JSON (same format as when signing)
    payload_str = json.dumps(payload, separators=(',', ':'))
    
    # Verify signature
    is_valid = verify_signature(payload_str, signature, hotkey)
    
    return hotkey, is_valid


def require_signature(func):
    """
    Decorator to require signature verification on an endpoint.
    
    Usage:
        @router.post("/endpoint")
        @require_signature
        async def endpoint(request: Request, data: MyModel):
            ...
    """
    async def wrapper(request: Request, *args, **kwargs):
        # Get payload from kwargs (should be a Pydantic model)
        payload = None
        for arg in args:
            if hasattr(arg, 'model_dump'):
                payload = arg.model_dump()
                break
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No payload found for signature verification"
            )
        
        hotkey, is_valid = await verify_request_signature(request, payload)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
        
        # Add hotkey to request state for use in endpoint
        request.state.authenticated_hotkey = hotkey
        
        return await func(request, *args, **kwargs)
    
    return wrapper