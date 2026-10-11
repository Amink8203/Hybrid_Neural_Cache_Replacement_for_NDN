from __future__ import annotations

import hashlib
from collections.abc import Iterable
from urllib.parse import quote_from_bytes, unquote_to_bytes

import torch
from torch import Tensor


def _canonical_component(component: str) -> str:

    raw = unquote_to_bytes(component)
    return quote_from_bytes(raw, safe="-._~")


def canonicalize_name(name: str) -> str:

    if not isinstance(name, str):
        raise TypeError("name must be a string")
    value = name[4:] if name[:4].lower() == "ndn:" else name
    components = (
        _canonical_component(component)
        for component in value.split("/")
        if component
    )
    joined = "/".join(components)
    return f"/{joined}" if joined else "/"


def split_components(name: str) -> tuple[str, ...]:

    canonical = canonicalize_name(name)
    return () if canonical == "/" else tuple(canonical[1:].split("/"))


class NameTokenizer:

    PAD_ID = 0
    ROOT_ID = 1
    PAD_IDX = PAD_ID
    ROOT_IDX = ROOT_ID

    def __init__(self, vocab_size: int, max_components: int) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("vocab_size must be an integer")
        if vocab_size < 3:
            raise ValueError("vocab_size must be at least 3")
        if isinstance(max_components, bool) or not isinstance(max_components, int):
            raise TypeError("max_components must be an integer")
        if max_components <= 0:
            raise ValueError("max_components must be positive")
        self.vocab_size = vocab_size
        self.max_components = max_components

    def token_for_component(self, component: str) -> int:

        if not isinstance(component, str):
            raise TypeError("component must be a string")
        canonical = _canonical_component(component)
        digest = hashlib.blake2b(
            canonical.encode("ascii"),
            digest_size=8,
            person=b"ndn-cache-v1",
        ).digest()
        return 2 + int.from_bytes(digest, "big") % (self.vocab_size - 2)

    def encode(self, name: str) -> list[int]:

        components = split_components(name)
        if not components:
            return [self.ROOT_ID]
        return [
            self.token_for_component(component)
            for component in components[: self.max_components]
        ]

    def batch_encode(
        self,
        names: Iterable[str],
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:

        encoded = [self.encode(name) for name in names]
        if not encoded:
            return (
                torch.empty((0, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )
        lengths = torch.tensor(
            [len(tokens) for tokens in encoded], dtype=torch.long, device=device
        )
        batch = torch.full(
            (len(encoded), int(lengths.max().item())),
            self.PAD_ID,
            dtype=torch.long,
            device=device,
        )
        for row, tokens in enumerate(encoded):
            batch[row, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=device
            )
        return batch, lengths


__all__ = ["NameTokenizer", "canonicalize_name", "split_components"]
