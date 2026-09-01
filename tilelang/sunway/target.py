"""Sunway target normalization for the TVM ``c`` carrier target."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_normalizer


SUNWAY_TARGET_KEY = "sunway"
_CONFIG_TAG_PREFIX = "tilelang-sunway-v1-"


@dataclass(frozen=True, slots=True)
class SunwayTargetConfig:
    """Backend-only properties that TVM's generic ``c`` target cannot store."""

    arch: str = "sw9a"
    cpe_rows: int = 8
    cpe_cols: int = 8
    ldm_bytes_per_cpe: int = 64 * 1024
    dma_alignment: int = 16
    simd_width: int = 8
    output_dir: Path | None = None
    output_indices: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SunwayTargetConfig:
        defaults = cls()
        output_dir = values.get("output_dir")
        return cls(
            arch=str(values.get("arch", defaults.arch)),
            cpe_rows=int(values.get("cpe_rows", defaults.cpe_rows)),
            cpe_cols=int(values.get("cpe_cols", defaults.cpe_cols)),
            ldm_bytes_per_cpe=int(values.get("ldm_bytes_per_cpe", defaults.ldm_bytes_per_cpe)),
            dma_alignment=int(values.get("dma_alignment", defaults.dma_alignment)),
            simd_width=int(values.get("simd_width", defaults.simd_width)),
            output_dir=Path(str(output_dir)).expanduser() if output_dir is not None else None,
            output_indices=tuple(int(index) for index in values.get("output_indices", defaults.output_indices)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "arch": self.arch,
            "cpe_rows": self.cpe_rows,
            "cpe_cols": self.cpe_cols,
            "ldm_bytes_per_cpe": self.ldm_bytes_per_cpe,
            "dma_alignment": self.dma_alignment,
            "simd_width": self.simd_width,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "output_indices": list(self.output_indices),
        }


_CONFIG_KEYS = frozenset(SunwayTargetConfig.__dataclass_fields__)


def is_sunway_target(target: Target) -> bool:
    """Return whether a normalized target belongs to the Sunway backend."""

    return target.kind.name == "c" and SUNWAY_TARGET_KEY in target.keys


def _encode_config(config: SunwayTargetConfig) -> str:
    payload = json.dumps(config.to_mapping(), sort_keys=True, separators=(",", ":"))
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{_CONFIG_TAG_PREFIX}{encoded}"


def get_sunway_target_config(target: Target) -> SunwayTargetConfig:
    """Decode backend properties carried by this immutable TVM target."""

    tag = str(target.tag)
    if not tag.startswith(_CONFIG_TAG_PREFIX):
        return SunwayTargetConfig()
    encoded = tag.removeprefix(_CONFIG_TAG_PREFIX)
    encoded += "=" * (-len(encoded) % 4)
    try:
        values = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Malformed Sunway target configuration tag") from error
    if not isinstance(values, dict):
        raise ValueError("Malformed Sunway target configuration payload")
    return SunwayTargetConfig.from_mapping(values)


def _carrier_target(
    values: Mapping[str, object] | None = None,
    config: SunwayTargetConfig | None = None,
) -> Target:
    carrier = dict(values or {})
    carrier["kind"] = "c"
    keys = [str(key) for key in carrier.get("keys", ())]
    carrier["keys"] = list(dict.fromkeys([*keys, SUNWAY_TARGET_KEY]))
    carrier["tag"] = _encode_config(config or SunwayTargetConfig())
    return Target(carrier)


def normalize_sunway_target(target: TargetLike) -> Target | None:
    """Normalize ``sunway`` user inputs to a TVM ``kind=c`` carrier."""

    if isinstance(target, Target):
        return target if is_sunway_target(target) else None

    if isinstance(target, Mapping):
        kind = target.get("kind")
        if kind != SUNWAY_TARGET_KEY and not bool(target.get(SUNWAY_TARGET_KEY)):
            return None
        config = SunwayTargetConfig.from_mapping(target)
        carrier = {key: value for key, value in target.items() if key != SUNWAY_TARGET_KEY and key not in _CONFIG_KEYS and key != "tag"}
        return _carrier_target(carrier, config)

    if str(target).strip() == SUNWAY_TARGET_KEY:
        return _carrier_target()
    return None


TARGET_NORMALIZER = register_target_normalizer("sunway", normalize_sunway_target)
