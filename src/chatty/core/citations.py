"""Chatty citation groups such as [2], [1.3], and [1.3, 2.1]."""

import re

_INDEX = r"[1-9][0-9]*"
_MEMBER = rf"(?:{_INDEX}\.)?{_INDEX}"
CITATION_GROUP = re.compile(rf"\[({_MEMBER}(?:\s*,\s*{_MEMBER})*)\]")


def cited_members(text: str) -> set[str]:
    return {member.strip() for group in CITATION_GROUP.findall(text) for member in group.split(",")}


def cited_turns(text: str) -> set[int]:
    """Return the turn numbers from valid turn-prefixed citations."""
    turns = set()
    for member in cited_members(text):
        prefix, separator, _ = member.partition(".")
        if separator:
            try:
                turns.add(int(prefix))
            except ValueError:  # A digit string can exceed Python's integer conversion limit.
                continue
    return turns
