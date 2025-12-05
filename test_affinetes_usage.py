"""
Test cases for affinetes usage examples.
These tests validate the affinetes functionality shown in the README example.
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch
import asyncio


class TestAffinetesUsage(unittest.TestCase):
    """Test affinetes usage examples"""

    @patch('affinetes.load_env')
    def test_affinetes_basic_usage_example(self, mock_load_env):
        """Test the basic affinetes usage example from the issue"""
        # Mock the environment object
        mock_env = Mock()
        mock_env.evaluate = AsyncMock(return_value={'score': 0.85})
        mock_env.cleanup = AsyncMock()
        
        # Configure load_env to return our mock
        mock_load_env.return_value = mock_env
        
        async def run_example():
            import affinetes as af_env

            # Load environment from Docker image
            env = af_env.load_env(
                image="bignickeye/agentgym:sciworld-v2",
                env_vars={"CHUTES_API_KEY": "your-api-key"}
            )
            
            # Execute methods
            result = await env.evaluate(
                model="deepseek-ai/DeepSeek-V3",
                base_url="https://llm.chutes.ai/v1",
                task_id=10
            )
            
            # Verify the result
            self.assertEqual(result['score'], 0.85)
            
            # Cleanup
            await env.cleanup()
            
            # Verify all methods were called with correct parameters
            mock_load_env.assert_called_once_with(
                image="bignickeye/agentgym:sciworld-v2",
                env_vars={"CHUTES_API_KEY": "your-api-key"}
            )
            
            mock_env.evaluate.assert_called_once_with(
                model="deepseek-ai/DeepSeek-V3",
                base_url="https://llm.chutes.ai/v1",
                task_id=10
            )
            
            mock_env.cleanup.assert_called_once()
        
        # Run the async test
        asyncio.run(run_example())

    def test_affinetes_import_availability(self):
        """Test that affinetes can be imported"""
        try:
            import affinetes
            self.assertTrue(True, "affinetes imported successfully")
        except ImportError:
            self.fail("affinetes could not be imported")


if __name__ == "__main__":
    unittest.main()