"""内置工具实现。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .base import BaseTool, ToolResult


def _resolve_path(path: str, *, base_dir: Path | str) -> Path:
    base_path = Path(base_dir).resolve()
    target_path = (base_path / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        common_path = os.path.commonpath([str(base_path), str(target_path)])
    except ValueError as exc:
        raise ValueError("目标路径不在允许范围内") from exc
    if common_path != str(base_path):
        raise ValueError("目标路径不在允许范围内")
    return target_path


class ReadTool(BaseTool):
    def __init__(self, *, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        super().__init__(name="read", description="Read a text file from the workspace")

    def args_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "start_line": {"type": "integer", "description": "1-based inclusive start line", "default": 1},
                "end_line": {"type": "integer", "description": "1-based inclusive end line"},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def invoke(
        self,
        *,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            target_path = _resolve_path(path, base_dir=self.base_dir)
            content = target_path.read_text(encoding=encoding)
            lines = content.splitlines()
            start_index = max(start_line - 1, 0)
            end_index = len(lines) if end_line is None else min(end_line, len(lines))
            snippet = "\n".join(lines[start_index:end_index])
            return ToolResult(
                status="success",
                data={
                    "path": str(target_path),
                    "content": snippet,
                    "start_line": start_line,
                    "end_line": end_index,
                },
            )
        except Exception as exc:
            return ToolResult(status="error", message=str(exc))


class WriteTool(BaseTool):
    def __init__(self, *, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        super().__init__(name="write", description="Write text content into a file in the workspace")

    def args_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root"},
                "content": {"type": "string", "description": "Text content to write"},
                "append": {"type": "boolean", "default": False},
                "overwrite": {"type": "boolean", "default": True},
                "create_dirs": {"type": "boolean", "default": True},
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def invoke(
        self,
        *,
        path: str,
        content: str,
        append: bool = False,
        overwrite: bool = True,
        create_dirs: bool = True,
        encoding: str = "utf-8",
    ) -> ToolResult:
        try:
            target_path = _resolve_path(path, base_dir=self.base_dir)
            if create_dirs:
                target_path.parent.mkdir(parents=True, exist_ok=True)

            if target_path.exists() and not append and not overwrite:
                raise ValueError("目标文件已存在，且未允许覆盖")

            mode = "a" if append else "w"
            with target_path.open(mode, encoding=encoding) as file:
                file.write(content)

            return ToolResult(
                status="success",
                data={
                    "path": str(target_path),
                    "bytes_written": len(content.encode(encoding)),
                    "append": append,
                },
            )
        except Exception as exc:
            return ToolResult(status="error", message=str(exc))


class BashTool(BaseTool):
    def __init__(self, *, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir).resolve()
        super().__init__(name="bash", description="Run a shell command in the workspace")

    def args_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "number", "default": 30.0},
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def invoke(self, *, command: str, timeout: float = 30.0) -> ToolResult:
        try:
            shell_command = self._build_shell_command(command)
            completed = subprocess.run(
                shell_command,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            status = "success" if completed.returncode == 0 else "error"
            return ToolResult(
                status=status,
                data={
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "exit_code": completed.returncode,
                },
                message="" if completed.returncode == 0 else "命令执行失败",
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                status="error",
                data={"stdout": exc.stdout or "", "stderr": exc.stderr or "", "exit_code": None},
                message=f"命令执行超时: {timeout}s",
            )
        except Exception as exc:
            return ToolResult(status="error", message=str(exc))

    def _build_shell_command(self, command: str) -> list[str]:
        if os.name == "nt":
            return ["powershell", "-NoProfile", "-Command", command]
        bash_path = shutil.which("bash")
        if bash_path:
            return [bash_path, "-lc", command]
        return ["sh", "-lc", command]