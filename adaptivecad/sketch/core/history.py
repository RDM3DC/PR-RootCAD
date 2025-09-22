"""Undo/redo scaffolding for sketch documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Command:
    description: str
    redo: Callable[[], None]
    undo: Callable[[], None]

    def execute(self) -> None:
        self.redo()


@dataclass
class _CommandBatch:
    description: str
    commands: List[Command] = field(default_factory=list)

    def append(self, cmd: Command) -> None:
        self.commands.append(cmd)

    def as_command(self) -> Command:
        def redo() -> None:
            for cmd in self.commands:
                cmd.execute()

        def undo() -> None:
            for cmd in reversed(self.commands):
                cmd.undo()

        return Command(self.description, redo=redo, undo=undo)


class CommandStack:
    def __init__(self) -> None:
        self._undo: List[Command] = []
        self._redo: List[Command] = []
        self._active_batch: Optional[_CommandBatch] = None

    def begin_batch(self, description: str) -> None:
        if self._active_batch is not None:
            raise RuntimeError("Nested command batches are not supported")
        self._active_batch = _CommandBatch(description)

    def commit_batch(self) -> None:
        if self._active_batch is None:
            return
        batch = self._active_batch
        self._active_batch = None
        if not batch.commands:
            return
        self.push(batch.as_command(), execute=False)

    def cancel_batch(self) -> None:
        self._active_batch = None

    def push(self, command: Command, *, execute: bool = True) -> None:
        if execute:
            command.execute()
        target = self._active_batch
        if target is not None:
            target.append(command)
            return
        self._undo.append(command)
        self._redo.clear()

    def undo(self) -> Optional[Command]:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        return cmd

    def redo(self) -> Optional[Command]:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        return cmd

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
        self._active_batch = None


__all__ = ["Command", "CommandStack"]
