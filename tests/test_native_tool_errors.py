import unittest
from mcp import types
from mcp.server.fastmcp import FastMCP
from subnet.native_tool_errors import call_tool

class NativeToolErrorControls(unittest.IsolatedAsyncioTestCase):
    def server(self):
        server=FastMCP('owned-control')
        @server.tool()
        def add(x:int,y:int)->int:return x+y
        return server
    async def test_unknown_tool_matches_actual_native_mcp_dispatch_error(self):
        server=self.server()
        result=await call_tool(server._tool_manager,'missing_tool',{})
        handler=server._mcp_server.request_handlers[types.CallToolRequest]
        native=await handler(types.CallToolRequest(params=types.CallToolRequestParams(name='missing_tool',arguments={})))
        self.assertTrue(native.root.isError)
        self.assertTrue(result['is_error'])
        self.assertEqual(result['result'],native.root.content[0].text)
    async def test_successful_native_tool_result_is_preserved(self):
        server=self.server();result=await call_tool(server._tool_manager,'add',{'x':2,'y':3})
        self.assertEqual(result,{'result':5,'is_error':False})
    async def test_infrastructure_failure_is_not_relabelled_as_model_error(self):
        class BrokenManager:
            async def call_tool(self,*args):raise RuntimeError('isolated runtime unavailable')
        with self.assertRaisesRegex(RuntimeError,'runtime unavailable'):await call_tool(BrokenManager(),'add',{})

if __name__=='__main__':unittest.main()
