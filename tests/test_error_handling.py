import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock, patch
from affine.utils.api_client import APIClient
from affine.utils.errors import NetworkError, ApiResponseError

@pytest.mark.asyncio
async def test_api_timeout():
    # Mock session and get to raise timeout
    mock_session = MagicMock()
    mock_get = MagicMock()
    # Correct way to mock async context manager exception
    mock_get.__aenter__.side_effect = aiohttp.ServerTimeoutError("Timeout")
    mock_session.get.return_value = mock_get
    
    client = APIClient("http://test.com", mock_session)
    
    with pytest.raises(NetworkError) as exc:
        await client.get("/timeout")
    assert "Timeout" in str(exc.value)

@pytest.mark.asyncio
async def test_api_bad_json():
    # Mock response with 200 OK but bad JSON
    mock_response = AsyncMock()
    mock_response.status = 200
    # json() raises ValueError
    mock_response.json.side_effect = ValueError("Bad JSON")
    mock_response.text.return_value = "<html>Not JSON</html>"
    
    mock_session = MagicMock()
    mock_get = MagicMock()
    mock_get.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_get
    
    client = APIClient("http://test.com", mock_session)
    
    with pytest.raises(ApiResponseError) as exc:
        await client.get("/bad-json")
    
    assert "Invalid JSON" in str(exc.value)
    assert exc.value.status_code == 200
    assert "<html>" in exc.value.body

@pytest.mark.asyncio
async def test_api_404():
    # Mock 404
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.text.return_value = "Not Found"
    
    mock_session = MagicMock()
    mock_get = MagicMock()
    mock_get.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_get
    
    client = APIClient("http://test.com", mock_session)
    
    with pytest.raises(ApiResponseError) as exc:
        await client.get("/404")
    
    assert exc.value.status_code == 404
    assert "Not Found" in str(exc.value)

@pytest.mark.asyncio
async def test_api_500():
    # Mock 500
    # Use POST to check it behaves same
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text.return_value = "Internal Server Error"
    
    mock_session = MagicMock()
    mock_post = MagicMock()
    mock_post.__aenter__.return_value = mock_response
    mock_session.post.return_value = mock_post
    
    client = APIClient("http://test.com", mock_session)
    
    with pytest.raises(ApiResponseError) as exc:
        await client.post("/500")
        
    assert exc.value.status_code == 500
    assert "Internal Server Error" in str(exc.value)
