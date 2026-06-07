from __future__ import annotations

import re
from dataclasses import dataclass

_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)


@dataclass
class ParseResult:
    answer: str
    malformed: bool


def parse_answer(text: str) -> ParseResult:
    """Extract the <answer> content from a (possibly malformed) generation.

    Real outputs may repeat the template, drop the closing tag, or omit the tag
    entirely. Prefer the last well-formed block; otherwise take everything after
    the last opening tag; otherwise return empty (flagged malformed). Never raises.
    """
    blocks = _ANSWER_BLOCK.findall(text)
    if blocks:
        return ParseResult(answer=blocks[-1].strip(), malformed=False)

    open_match = None
    for open_match in _ANSWER_OPEN.finditer(text):
        pass
    if open_match is not None:
        return ParseResult(answer=open_match.group(1).strip(), malformed=True)

    return ParseResult(answer="", malformed=True)
