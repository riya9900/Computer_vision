# OrbitWatch: Satellite Change Watchdog

CSE471 Computer Vision project. Draw an area on a map; background workers compare fresh Sentinel-2 satellite images over time, a model built on the Clay foundation model finds what changed, and an MCP server lets an AI agent use the system.

## Folders

| Folder | Purpose | Stack |
|---|---|---|
| `client/` | Web UI | Next.js |
| `server/` | REST API | FastAPI |
| `worker/` | Background jobs | Temporal |
| `mcp/` | MCP server for agents | FastMCP |
