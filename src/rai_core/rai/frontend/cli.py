import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from rai.memory.graph import MemoryAgentContext
from rai.memory.long_term import format_long_term_item, list_long_term_memory_items
from rai.memory.manager import MemoryManager
from rai.memory.session import get_session_ids, graph_config, load_thread_state
from rai.memory.users import get_user_ids
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

    def list_sessions(self) -> list[str]:
        return get_session_ids(self.memory_mgr)

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
    if stripped == "/new":
        return CliCommandResult(handled=True, message="new")
    if stripped == "/sessions":
        return CliCommandResult(handled=True, message="sessions")
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
        "Commands: /exit, /new, /sessions, /users, /user <id>, /memory, "
        "/memory facts, /memory locations, /image <path> <message>"
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
    if command.message == "new":
        thread_id = session.new_session()
        renderer.system(f"Started new session: {thread_id}")
        return
    if command.message == "sessions":
        renderer.sessions(session.list_sessions(), session.thread_id)
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
