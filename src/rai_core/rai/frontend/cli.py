import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict
from langchain_core.runnables import RunnableConfig

from rai.memory.graph import MemoryAgentContext
from rai.memory.long_term import format_long_term_item, list_long_term_memory_items
from rai.memory.manager import MemoryManager
from rai.memory.session import (
    delete_session,
    get_session_ids,
    graph_config,
    load_thread_state,
)
from rai.memory.users import delete_user, get_user_ids
from rai.messages import preprocess_image

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
except ImportError:  # pragma: no cover - fallback is exercised by plain input users.
    PromptSession = None
    FileHistory = None

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
except ImportError:  # pragma: no cover - fallback is exercised by plain print users.
    Console = None
    Markdown = None
    Panel = None
    Syntax = None
    Table = None


GraphFactory = Callable[[str], Any]

HELP_TEXT = """Available commands:
/help
/status
/tools
/exit
/new
/sessions
/resume <thread_id>
/session <thread_id>
/users
/user
/user <user_id>
/memory
/memory facts
/memory locations
/delete-session <thread_id>
/delete-memory <facts|locations> <key>
/delete-user <user_id>
/clear
/export-session <path>
/image <path> <message>"""


@dataclass
class CliTurn:
    text: str
    images: list[str] = field(default_factory=list)


@dataclass
class CliCommandResult:
    handled: bool
    should_exit: bool = False
    message: str | None = None
    turn: CliTurn | None = None


@dataclass
class MemoryCliSession:
    memory_mgr: MemoryManager
    graph: Any
    namespace: str
    user_id: str = "default"
    thread_id: str = field(default_factory=lambda: f"session-{int(time.time())}")
    graph_factory: GraphFactory | None = None
    tools: Sequence[Any] = field(default_factory=tuple)
    welcome_message_factory: Callable[[], AIMessage] = lambda: AIMessage(
        content="New conversation started."
    )
    messages: list[Any] = field(default_factory=list)
    summary: str = ""

    def __post_init__(self) -> None:
        self.reload_thread()

    def reload_thread(self) -> None:
        restored_messages, restored_summary = load_thread_state(self.graph, self.thread_id)
        self.messages = restored_messages or [self.welcome_message_factory()]
        self.summary = restored_summary

    def new_session(self) -> str:
        self.thread_id = f"session-{int(time.time())}"
        self.messages = [self.welcome_message_factory()]
        self.summary = ""
        return self.thread_id

    def resume_session(self, thread_id: str) -> list[Any]:
        self.thread_id = thread_id
        self.reload_thread()
        return self.messages

    def list_sessions(self) -> list[str]:
        return get_session_ids(self.memory_mgr)

    def delete_session(self, thread_id: str) -> str:
        delete_session(self.memory_mgr, thread_id)
        if thread_id == self.thread_id:
            new_thread_id = self.new_session()
            return f"Deleted current session {thread_id}; started {new_thread_id}."
        return f"Deleted session {thread_id}."

    def list_users(self) -> list[str]:
        return get_user_ids(self.memory_mgr, self.namespace)

    def list_long_term_memory(
        self,
        kind: str | None = None,
    ) -> list[tuple[str, Any, str, Any]]:
        items = list_long_term_memory_items(
            self.memory_mgr.store,
            self.namespace,
            self.user_id,
        )
        if kind is None:
            return items
        schema = _normalize_memory_kind(kind)
        return [item for item in items if item[0] == schema]

    def delete_long_term_memory(self, kind: str, key: str) -> str:
        schema = _normalize_memory_kind(kind)
        self.memory_mgr.store.delete((self.namespace, self.user_id, schema), key)
        return f"Deleted {schema} memory key={key} for user={self.user_id}."

    def delete_user(self, user_id: str) -> str:
        if user_id == "default":
            return "Refusing to delete default user."
        deleted = delete_user(self.memory_mgr, self.namespace, user_id)
        if user_id == self.user_id:
            self.set_user("default")
            return f"Deleted user {user_id} ({deleted} item(s)); switched to default."
        return f"Deleted user {user_id} ({deleted} item(s))."

    def export_session(self, path: str | Path) -> str:
        export_path = Path(path).expanduser()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "summary": self.summary,
            "messages": messages_to_dict(self.messages),
        }
        export_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return f"Exported session {self.thread_id} to {export_path}."

    def status(self) -> dict[str, Any]:
        config = getattr(self.memory_mgr, "_config", None)
        return {
            "user": self.user_id,
            "namespace": self.namespace,
            "thread": self.thread_id,
            "summary": self.summary,
            "memory_backend": getattr(config, "backend", "unknown"),
        }

    def tool_summaries(self) -> list[tuple[str, str]]:
        summaries = []
        for tool in self.tools:
            name = getattr(tool, "name", tool.__class__.__name__)
            description = getattr(tool, "description", "")
            summaries.append((str(name), str(description)))
        return summaries

    def set_user(self, user_id: str) -> None:
        self.user_id = user_id
        if self.graph_factory is not None:
            self.graph = self.graph_factory(user_id)
        self.reload_thread()

    def invoke(self, turn: CliTurn) -> list[Any]:
        human_msg = HumanMessage(content=turn.text)
        transient_images = encode_image_paths(turn.images)
        context = MemoryAgentContext(
            user_id=self.user_id,
            namespace=self.namespace,
            transient_images=transient_images or None,
        )
        result = self.graph.invoke(
            input={"messages": [human_msg]},
            config=RunnableConfig(
                {
                    "recursion_limit": 100,
                    "configurable": graph_config(self.thread_id).get("configurable", {}),
                }
            ),
            context=context,
        )
        old_count = len(self.messages)
        if result and "messages" in result:
            self.messages = result["messages"]
            self.summary = result.get("summary", "")
        return self.messages[old_count:]


def parse_cli_input(line: str) -> CliCommandResult:
    stripped = line.strip()
    if not stripped:
        return CliCommandResult(handled=True)
    if stripped in {"/exit", "/quit"}:
        return CliCommandResult(handled=True, should_exit=True)
    if stripped == "/help":
        return CliCommandResult(handled=True, message="help")
    if stripped == "/status":
        return CliCommandResult(handled=True, message="status")
    if stripped == "/tools":
        return CliCommandResult(handled=True, message="tools")
    if stripped == "/clear":
        return CliCommandResult(handled=True, message="clear")
    if stripped == "/new":
        return CliCommandResult(handled=True, message="new")
    if stripped == "/sessions":
        return CliCommandResult(handled=True, message="sessions")
    if stripped.startswith("/delete-session"):
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return CliCommandResult(
                handled=True,
                message="usage: /delete-session <thread_id>",
            )
        return CliCommandResult(
            handled=True, message=f"delete-session:{parts[1].strip()}"
        )
    if stripped.startswith("/delete-memory"):
        parts = stripped.split(maxsplit=2)
        if len(parts) != 3 or parts[1].strip() not in {"facts", "locations", "spatial"}:
            return CliCommandResult(
                handled=True,
                message="usage: /delete-memory <facts|locations> <key>",
            )
        return CliCommandResult(
            handled=True,
            message=f"delete-memory:{parts[1].strip()}:{parts[2].strip()}",
        )
    if stripped.startswith("/delete-user"):
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return CliCommandResult(
                handled=True,
                message="usage: /delete-user <user_id>",
            )
        return CliCommandResult(handled=True, message=f"delete-user:{parts[1].strip()}")
    if stripped.startswith("/export-session"):
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return CliCommandResult(
                handled=True,
                message="usage: /export-session <path>",
            )
        return CliCommandResult(
            handled=True,
            message=f"export-session:{parts[1].strip()}",
        )
    if stripped.startswith(("/resume", "/session")):
        command_name = stripped.split(maxsplit=1)[0]
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return CliCommandResult(
                handled=True,
                message=f"usage: {command_name} <thread_id>",
            )
        resume_args = parts[1].strip().split()
        quiet = False
        if resume_args and resume_args[-1] == "--quiet":
            quiet = True
            resume_args = resume_args[:-1]
        if len(resume_args) != 1:
            return CliCommandResult(
                handled=True,
                message=f"usage: {command_name} <thread_id> [--quiet]",
            )
        suffix = ":quiet" if quiet else ""
        return CliCommandResult(handled=True, message=f"resume:{resume_args[0]}{suffix}")
    if stripped in {"/users", "/user"}:
        return CliCommandResult(handled=True, message="users")
    if stripped.startswith("/memory"):
        parts = stripped.split(maxsplit=1)
        if len(parts) == 1:
            return CliCommandResult(handled=True, message="memory")
        kind = parts[1].strip().lower()
        if kind not in {"facts", "locations", "spatial"}:
            return CliCommandResult(
                handled=True,
                message="usage: /memory [facts|locations]",
            )
        return CliCommandResult(handled=True, message=f"memory:{kind}")
    if stripped.startswith("/user"):
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            return CliCommandResult(
                handled=True, message="usage: /user <user_id>"
            )
        return CliCommandResult(handled=True, message=f"user:{parts[1].strip()}")
    if stripped.startswith("/image"):
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            return CliCommandResult(
                handled=True, message="usage: /image <path> <message>"
            )
        return CliCommandResult(
            handled=False,
            turn=CliTurn(text=parts[2].strip(), images=[parts[1].strip()]),
        )
    return CliCommandResult(handled=False, turn=CliTurn(text=line))


def encode_image_paths(image_paths: Sequence[str]) -> list[str]:
    return [preprocess_image(str(Path(path).expanduser())) for path in image_paths]


def _normalize_memory_kind(kind: str) -> str:
    return "spatial" if kind in {"locations", "spatial"} else kind


class CliRenderer:
    def __init__(self, console: Any | None = None) -> None:
        self.console = console or (Console() if Console is not None else None)

    def system(self, message: str) -> None:
        if self.console is None:
            print(message)
            return
        self.console.print(f"[dim]{message}[/dim]")

    def user(self, message: str) -> None:
        if self.console is None:
            print(f"> {message}")
            return
        self.console.print(Panel(message, title="User", border_style="blue"))

    def messages(self, messages: Iterable[Any]) -> None:
        for message in messages:
            self.message(message)

    def message(self, message: Any) -> None:
        if isinstance(message, AIMessage):
            if message.content:
                self.assistant(str(message.content))
            for tool_call in message.tool_calls or []:
                self.tool_call(tool_call.get("name", "tool"), tool_call.get("args", {}))
            return
        if isinstance(message, ToolMessage):
            self.tool_output(message.name or "tool", message.content)
            return
        if isinstance(message, HumanMessage):
            self.user(str(message.content))

    def assistant(self, message: str) -> None:
        if self.console is None:
            print(f"Assistant: {message}")
            return
        self.console.print(Panel(Markdown(message), title="Assistant", border_style="green"))

    def tool_call(self, name: str, args: Any) -> None:
        formatted = _format_json(args)
        if self.console is None:
            print(f"Tool call: {name}\n{formatted}")
            return
        self.console.print(
            Panel(
                Syntax(formatted, "json", word_wrap=True),
                title=f"Tool call: {name}",
                border_style="yellow",
            )
        )

    def tool_output(self, name: str, content: Any) -> None:
        text = str(content)
        if self.console is None:
            print(f"Tool result: {name}\n{text}")
            return
        self.console.print(
            Panel(text, title=f"Tool result: {name}", border_style="magenta")
        )

    def sessions(self, session_ids: Sequence[str], current_thread_id: str) -> None:
        if self.console is None or Table is None:
            print("\n".join(session_ids) if session_ids else "No sessions.")
            return
        table = Table(title="Sessions")
        table.add_column("Thread ID")
        table.add_column("Current")
        for thread_id in session_ids:
            table.add_row(thread_id, "*" if thread_id == current_thread_id else "")
        self.console.print(table)

    def help(self) -> None:
        if self.console is None:
            print(HELP_TEXT)
            return
        self.console.print(Panel(HELP_TEXT, title="Help", border_style="cyan"))

    def clear(self) -> None:
        if self.console is None:
            print("\033[2J\033[H", end="")
            return
        self.console.clear()

    def status(self, status: dict[str, Any]) -> None:
        if self.console is None or Table is None:
            for key, value in status.items():
                print(f"{key}: {value}")
            return
        table = Table(title="Status")
        table.add_column("Field")
        table.add_column("Value")
        for key, value in status.items():
            table.add_row(str(key), str(value))
        self.console.print(table)

    def tools(self, tools: Sequence[tuple[str, str]]) -> None:
        if self.console is None or Table is None:
            if not tools:
                print("No tools.")
                return
            for name, description in tools:
                print(f"- {name}: {description}")
            return
        table = Table(title="Tools")
        table.add_column("Name")
        table.add_column("Description")
        for name, description in tools:
            table.add_row(name, description)
        self.console.print(table)

    def users(self, user_ids: Sequence[str], current_user_id: str) -> None:
        if self.console is None or Table is None:
            if not user_ids:
                print("No users.")
                return
            for user_id in user_ids:
                marker = "*" if user_id == current_user_id else " "
                print(f"{marker} {user_id}")
            return
        table = Table(title="Users")
        table.add_column("User ID")
        table.add_column("Current")
        for user_id in user_ids:
            table.add_row(user_id, "*" if user_id == current_user_id else "")
        self.console.print(table)

    def long_term_memory(
        self,
        items: Sequence[tuple[str, Any, str, Any]],
        user_id: str,
        namespace: str,
    ) -> None:
        if not items:
            self.system(f"No long-term memory for user={user_id} namespace={namespace}.")
            return
        if self.console is None or Table is None:
            print(f"Long-term memory for user={user_id} namespace={namespace}")
            for schema, _ns, key, value in items:
                print(f"- {schema}: {format_long_term_item(schema, key, value)}")
            return
        table = Table(title=f"Long-term memory: {namespace}/{user_id}")
        table.add_column("Type")
        table.add_column("Key")
        table.add_column("Value")
        for schema, _ns, key, value in items:
            table.add_row(schema, key, format_long_term_item(schema, key, value))
        self.console.print(table)


def run_memory_cli(
    session: MemoryCliSession,
    renderer: CliRenderer | None = None,
    history_path: str | Path | None = None,
) -> None:
    renderer = renderer or CliRenderer()
    prompt_session = _create_prompt_session(history_path)
    renderer.system(
        f"RAI CLI started. user={session.user_id} thread={session.thread_id}. "
        "Commands: /help, /status, /tools, /exit, /new, /sessions, "
        "/resume <thread_id> [--quiet], /users, /user <id>, /memory, /memory facts, "
        "/memory locations, /delete-session <thread_id>, "
        "/delete-memory <facts|locations> <key>, "
        "/delete-user <user_id>, /clear, /export-session <path>, "
        "/image <path> <message>"
    )
    while True:
        try:
            line = _prompt(prompt_session)
        except (KeyboardInterrupt, EOFError):
            renderer.system("Exiting.")
            return

        command = parse_cli_input(line)
        if command.should_exit:
            renderer.system("Exiting.")
            return
        if command.handled:
            _handle_command(command, session, renderer)
            continue
        if command.turn is None:
            continue
        renderer.user(command.turn.text)
        new_messages = session.invoke(command.turn)
        renderer.messages(new_messages)


def shutdown_tool_connectors(tools: Sequence[Any]) -> None:
    """Shutdown unique connector objects attached to tools.

    Tools in an application often share one ROS2Connector. Shutting each unique
    connector once prevents executor threads from keeping CLI processes alive
    after the REPL exits.
    """
    seen_connector_ids: set[int] = set()
    for tool in tools:
        connector = getattr(tool, "connector", None)
        if connector is None:
            continue
        connector_id = id(connector)
        if connector_id in seen_connector_ids:
            continue
        seen_connector_ids.add(connector_id)
        shutdown = getattr(connector, "shutdown", None)
        if callable(shutdown):
            shutdown()


def _handle_command(
    command: CliCommandResult,
    session: MemoryCliSession,
    renderer: CliRenderer,
) -> None:
    if command.message == "help":
        renderer.help()
        return
    if command.message == "status":
        renderer.status(session.status())
        return
    if command.message == "tools":
        renderer.tools(session.tool_summaries())
        return
    if command.message == "clear":
        renderer.clear()
        return
    if command.message == "new":
        thread_id = session.new_session()
        renderer.system(f"Started new session: {thread_id}")
        return
    if command.message and command.message.startswith("delete-session:"):
        thread_id = command.message.removeprefix("delete-session:")
        renderer.system(session.delete_session(thread_id))
        return
    if command.message and command.message.startswith("delete-memory:"):
        _prefix, kind, key = command.message.split(":", maxsplit=2)
        renderer.system(session.delete_long_term_memory(kind, key))
        return
    if command.message and command.message.startswith("delete-user:"):
        user_id = command.message.removeprefix("delete-user:")
        renderer.system(session.delete_user(user_id))
        return
    if command.message and command.message.startswith("export-session:"):
        path = command.message.removeprefix("export-session:")
        renderer.system(session.export_session(path))
        return
    if command.message == "sessions":
        renderer.sessions(session.list_sessions(), session.thread_id)
        return
    if command.message and command.message.startswith("resume:"):
        resume_value = command.message.removeprefix("resume:")
        quiet = resume_value.endswith(":quiet")
        thread_id = resume_value.removesuffix(":quiet")
        messages = session.resume_session(thread_id)
        renderer.system(f"Resumed session: {thread_id}")
        if not quiet:
            renderer.messages(messages)
        return
    if command.message == "users":
        renderer.users(session.list_users(), session.user_id)
        return
    if command.message == "memory":
        renderer.long_term_memory(
            session.list_long_term_memory(),
            session.user_id,
            session.namespace,
        )
        return
    if command.message and command.message.startswith("memory:"):
        kind = command.message.removeprefix("memory:")
        renderer.long_term_memory(
            session.list_long_term_memory(kind),
            session.user_id,
            session.namespace,
        )
        return
    if command.message and command.message.startswith("user:"):
        user_id = command.message.removeprefix("user:")
        session.set_user(user_id)
        renderer.system(f"Switched user to {user_id}")
        return
    if command.message:
        renderer.system(command.message)


def _create_prompt_session(history_path: str | Path | None) -> Any | None:
    if PromptSession is None:
        return None
    history = None
    if history_path is not None and FileHistory is not None:
        history_file = Path(history_path).expanduser()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_file))
    return PromptSession(history=history)


def _prompt(prompt_session: Any | None) -> str:
    if prompt_session is None:
        return input("> ")
    return prompt_session.prompt("> ")


def _format_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except TypeError:
        return str(value)
