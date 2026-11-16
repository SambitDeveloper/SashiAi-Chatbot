"""Repository that queries MongoDB through the official MongoDB MCP Server (read-only).

The backend starts `mongodb-mcp-server --readOnly` as a child process (stdio) and runs every query with its
`aggregate` tool, starting from the resident view in app/data/resident_schema.yaml (soft-delete filter, staff and
facility joins, field mapping, field whitelist). Read-only mode removes every insert/update/delete tool from the
server, so this layer cannot change data even if asked to. Filters are built by `to_mongo_filter` from a validated
QueryPlan, never from raw user text.

Server settings applied here (see MongoDB MCP Server docs):
  • MDB_MCP_READ_ONLY=true            only read, connect and metadata tools
  • MDB_MCP_TELEMETRY=disabled        the server's default is "enabled"; resident data must not leave
  • MDB_MCP_LOGGERS=mcp               no query logs written to disk or stderr
  • MDB_MCP_MAX_DOCUMENTS_PER_QUERY   server default 100; raised so name lists are complete
"""

import asyncio
import contextlib
import glob
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from app.data.resident_view import ResidentView, ViewRepository

log = logging.getLogger(__name__)


def _node_ok(version: tuple[int, int]) -> bool:
    """mongodb-mcp-server 3.x engines: ^20.19.0 || ^22.13.0 || >= 24.0.0"""
    major, minor = version
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 13) or major >= 24


def _node_version(path: str) -> tuple[int, int] | None:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)", out.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_node(preferred: str | None = None) -> str:
    """Path of a Node.js binary that can run the MongoDB MCP Server.

    The server is a `#!/usr/bin/env node` script, so it runs on the FIRST `node` on PATH. On many machines that is
    an old system Node (e.g. Ubuntu's 18.x) while a newer one is installed with nvm. This picks: MCP_NODE_PATH if set,
    otherwise the first suitable `node` on PATH, otherwise the newest suitable nvm / common install.
    """
    if preferred:
        version = _node_version(preferred)
        if version is None or not _node_ok(version):
            raise McpError(f"MCP_NODE_PATH={preferred} is not Node.js 20.19+, 22.13+ or 24+ (found {version}).")
        return preferred

    on_path = [str(Path(d) / "node") for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    installed = sorted(
        glob.glob(os.path.expanduser("~/.nvm/versions/node/*/bin/node"))
        + glob.glob(os.path.expanduser("~/.volta/tools/image/node/*/bin/node"))
        + ["/usr/local/bin/node", "/opt/homebrew/bin/node"],
        key=lambda p: _node_version(p) or (0, 0),
        reverse=True,
    )
    seen: dict[str, tuple[int, int] | None] = {}
    for path in on_path + installed:
        real = os.path.realpath(path)
        if real in seen or not os.access(real, os.X_OK):
            continue
        seen[real] = version = _node_version(real)
        if version and _node_ok(version):
            return path
    found = ", ".join(f"{p} (v{v[0]}.{v[1]})" for p, v in seen.items() if v) or "none"
    raise McpError(
        "The MongoDB MCP Server needs Node.js 20.19+, 22.13+ or 24+. Found: "
        f"{found}. Install one (e.g. `nvm install 22`) or set MCP_NODE_PATH to its node binary."
    )


class McpError(RuntimeError):
    """The MCP server reported an error or could not be reached."""


class McpResidentRepository(ViewRepository):
    name = "mongodb-mcp"

    def __init__(
        self,
        command: str,
        connection_string: str,
        database: str,
        collection: str,
        max_documents: int = 5000,
        timeout_seconds: float = 20.0,
        view: ResidentView | None = None,
        node_path: str | None = None,
    ):
        super().__init__(view, max_rows=max_documents)
        self._command = shlex.split(command)
        self._node_path = node_path
        self._node_ready = False
        self._env = {
            **os.environ,
            "MDB_MCP_CONNECTION_STRING": connection_string,
            "MDB_MCP_READ_ONLY": "true",
            "MDB_MCP_TELEMETRY": "disabled",
            "MDB_MCP_LOGGERS": "mcp",  # log over the MCP channel only: nothing on disk, no debug noise in stderr
            "MDB_MCP_MAX_DOCUMENTS_PER_QUERY": str(max_documents),
        }
        self._base = {"connectionId": "preconfigured", "database": database, "collection": collection}
        self._timeout = timeout_seconds
        self._owner: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._session = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ lifecycle
    async def init(self) -> None:
        await self._connect()
        await self._check_schema()  # clear error if the collection does not match the expected fields

    async def close(self) -> None:
        owner, self._owner, self._session = self._owner, None, None
        if owner is not None:
            self._stop.set()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(owner, 10)

    def _prepare_node(self) -> None:
        """Put a compatible Node.js first on the server's PATH (its `#!/usr/bin/env node` uses the first one)."""
        node = find_node(self._node_path)
        node_dir = str(Path(node).parent)
        path = self._env.get("PATH", "")
        if not path.startswith(node_dir + os.pathsep):
            self._env["PATH"] = node_dir + os.pathsep + path
        if self._command and Path(self._command[0]).name == "node":
            self._command[0] = node
        log.info("MongoDB MCP server uses Node.js", extra={"node": node})

    async def _connect(self) -> None:
        await self.close()
        if not self._node_ready:
            await asyncio.to_thread(self._prepare_node)
            self._node_ready = True
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._owner = asyncio.create_task(self._own_connection(ready, self._stop), name="mongodb-mcp-connection")
        try:
            self._session = await asyncio.wait_for(asyncio.shield(ready), self._timeout + 30)
        except BaseException:
            await self.close()
            raise
        log.info("connected to MongoDB MCP server", extra={"tools": ["aggregate", "collection-schema"]})

    async def _own_connection(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        """Open, hold and close the MCP connection inside ONE task.

        The MCP client library (anyio) requires the connection to be closed by the same task that opened it;
        closing it from another task raises "Attempted to exit cancel scope in a different task".
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=self._command[0], args=self._command[1:] + ["--readOnly"], env=self._env)
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = {t.name for t in (await session.list_tools()).tools}
                    missing = {"aggregate"} - tools
                    if missing:
                        raise McpError(f"MongoDB MCP server is missing tools: {sorted(missing)}")
                    if tools & {"insert-many", "update-many", "delete-many", "drop-collection"}:
                        raise McpError("MongoDB MCP server is not in read-only mode; refusing to use it")
                    ready.set_result(session)
                    await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc if isinstance(exc, Exception) else McpError("MongoDB MCP server stopped"))
            elif not stop.is_set():
                log.warning("MongoDB MCP server connection ended unexpectedly; will reconnect on next query")
        finally:
            if self._owner is asyncio.current_task():
                self._session = None

    async def _check_schema(self) -> None:
        """Fail at startup with a readable message when the collection lacks fields the mapping needs."""
        try:
            data = await self._call("collection-schema", {})
        except McpError as exc:
            log.warning("could not read collection schema; skipping field check", extra={"error": str(exc)[:200]})
            data = {}
        fields = set((data.get("schema") or {}).keys())
        missing = sorted(set(self.view.required_source_fields) - fields) if fields else []
        if missing:
            raise McpError(
                f"Collection {self._base['database']}.{self._base['collection']} lacks fields used by "
                f"{self.view.schema_file.name}: {', '.join(missing)}. Update the mapping in that file."
            )
        # the mapping itself must run on this database (bad expression, missing join collection, old server, ...)
        await self._aggregate(self.view.pipeline(None, {"$limit": 1}))

    async def _call(self, tool: str, args: dict[str, Any]) -> dict:
        """Call one MCP tool; reconnect once if the server process went away."""
        for attempt in (1, 2):
            try:
                async with self._lock:
                    if self._session is None:
                        await self._connect()
                    result = await asyncio.wait_for(
                        self._session.call_tool(tool, {**self._base, **args}), self._timeout
                    )
                break
            except (McpError, TimeoutError):
                raise
            except Exception as exc:  # broken pipe, closed process, ...
                if attempt == 2:
                    raise McpError(f"MongoDB MCP call '{tool}' failed: {type(exc).__name__}: {exc}") from exc
                log.warning("MongoDB MCP connection lost; reconnecting", extra={"tool": tool})
                await self.close()
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)[:300]
            raise McpError(f"MongoDB MCP tool '{tool}' returned an error: {text}")
        data = result.structured_content or {}
        if data.get("appliedLimits"):
            log.warning("MongoDB MCP result was capped", extra={"tool": tool, "limits": data["appliedLimits"]})
        return data

    # ------------------------------------------------------------------ repository interface (see ViewRepository)
    async def _aggregate(self, pipeline: list[dict]) -> list[dict]:
        return (await self._call("aggregate", {"pipeline": pipeline})).get("documents", [])
