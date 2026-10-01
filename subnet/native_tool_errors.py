"""Prospective MCP ToolError observations; no infrastructure-error suppression.

The opt-in adapter must pin this module and its policy in a new environment
contract. This helper alone does not change any running epoch or adapter.
"""
from mcp.server.fastmcp.exceptions import ToolError

POLICY='native-mcp-toolerror-observation-v1'

async def call_tool(manager,name,arguments):
    try:
        return {'result':await manager.call_tool(name,arguments),'is_error':False}
    except ToolError as error:
        # Native MCP Server.call_tool also exposes str(error) in a TextContent
        # error result. Keep that exact text for model context and replay.
        return {'result':str(error),'is_error':True,'error_type':type(error).__name__}
