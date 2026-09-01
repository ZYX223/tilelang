"""Portable metadata for a compiled Sunway kernel package."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


_ARGUMENT_ROLES = frozenset({"input", "output", "inout"})


@dataclass(frozen=True, slots=True)
class SunwayKernelArgument:
    """One tensor pointer in the generated C ABI."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    role: str = "input"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Sunway kernel argument name cannot be empty")
        if self.role not in _ARGUMENT_ROLES:
            raise ValueError(f"Unsupported Sunway kernel argument role {self.role!r}")
        if any(int(dim) <= 0 for dim in self.shape):
            raise ValueError(f"Sunway kernel argument {self.name!r} requires positive static dimensions")


@dataclass(frozen=True, slots=True)
class SunwayKernelManifest:
    """Versioned ABI metadata shared by Dell build tools and the 9A runtime."""

    kernel_name: str
    symbol: str
    arguments: tuple[SunwayKernelArgument, ...]
    schema_version: int = 1
    synchronous: bool = True
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported Sunway manifest schema {self.schema_version}")
        if not self.kernel_name or not self.symbol:
            raise ValueError("Sunway kernel name and symbol cannot be empty")
        if not self.arguments:
            raise ValueError("Sunway kernel manifest requires at least one argument")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kernel_name": self.kernel_name,
            "symbol": self.symbol,
            "synchronous": self.synchronous,
            "arguments": [asdict(argument) for argument in self.arguments],
            "artifacts": dict(self.artifacts),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SunwayKernelManifest:
        arguments = tuple(
            SunwayKernelArgument(
                name=str(argument["name"]),
                dtype=str(argument["dtype"]),
                shape=tuple(int(dim) for dim in argument["shape"]),
                role=str(argument.get("role", "input")),
            )
            for argument in payload["arguments"]
        )
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            kernel_name=str(payload["kernel_name"]),
            symbol=str(payload["symbol"]),
            synchronous=bool(payload.get("synchronous", True)),
            arguments=arguments,
            artifacts={str(key): str(value) for key, value in dict(payload.get("artifacts", {})).items()},
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> SunwayKernelManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Sunway kernel manifest root must be an object")
        return cls.from_dict(payload)
