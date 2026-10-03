"""Best-of-N text augmentations from Hughes et al. (2024), appendix A.1."""

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

@dataclass(frozen=True)
class NoiseProbs:
    scramble: float
    capitalize: float
    ascii_noise: float


# Published probabilities. Intensity levels use level/3 of each value.
PUBLISHED = NoiseProbs(scramble=0.6, capitalize=0.6, ascii_noise=0.06)

# Tags, addresses, timestamps, and field labels are structural. ID values are
# functional identifiers inserted by the leaking templates, so the digits stay too.
_PROTECTED = re.compile(
    r"<[^>\n]+>"
    r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"|^ID:[ \t]*\d+"
    r"|^(?:From|To|Cc|Bcc|Subject|Timestamp|Body|to|subject|body|email_id|code):[ \t]?",
    re.MULTILINE,
)
_LETTER_RUN = re.compile(r"[A-Za-z]+")


def level_probs(level: int) -> NoiseProbs:
    """Return augmentation probabilities for noise level 0, 1, 2, or 3."""
    if level not in (0, 1, 2, 3):
        raise ValueError(f"noise level must be 0, 1, 2, or 3, got {level}")
    scale = level / 3
    return NoiseProbs(
        scramble=PUBLISHED.scramble * scale,
        capitalize=PUBLISHED.capitalize * scale,
        ascii_noise=PUBLISHED.ascii_noise * scale,
    )


def noise_seed(base_seed: int, level: int, draw_id: int) -> int:
    """Stable seed for one (level, draw) pair."""
    return base_seed * 1_000_042 + level * 10_000 + draw_id


def noise_parts(parts: Sequence[str], level: int, seed: int) -> tuple[str, ...]:
    """Noise each string with one RNG stream. Level 0 returns the strings unchanged."""
    probs = level_probs(level)
    if _is_identity(probs):
        return tuple(parts)
    rng = random.Random(seed)
    return tuple(apply_noise(part, probs, rng) for part in parts)


def apply_noise(text: str, probs: NoiseProbs, rng: random.Random) -> str:
    """Scramble, then capitalize, then apply ASCII noise, skipping protected spans."""
    if _is_identity(probs) or text == "":
        return text
    chunks: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        chunks.append(_augment(text[cursor : match.start()], probs, rng))
        chunks.append(match.group(0))
        cursor = match.end()
    chunks.append(_augment(text[cursor:], probs, rng))
    return "".join(chunks)


def _is_identity(probs: NoiseProbs) -> bool:
    return probs.scramble <= 0 and probs.capitalize <= 0 and probs.ascii_noise <= 0


def _augment(text: str, probs: NoiseProbs, rng: random.Random) -> str:
    text = _scramble(text, probs.scramble, rng)
    text = _capitalize(text, probs.capitalize, rng)
    return _ascii_noise(text, probs.ascii_noise, rng)


def _scramble(text: str, probability: float, rng: random.Random) -> str:
    if probability <= 0 or text == "":
        return text
    parts = re.split(r"(\s+)", text)
    noised: list[str] = []
    for part in parts:
        if part == "" or part.isspace() or rng.random() >= probability:
            noised.append(part)
            continue
        noised.append(
            _LETTER_RUN.sub(lambda match: _scramble_run(match.group(0), rng), part)
        )
    return "".join(noised)


def _scramble_run(word: str, rng: random.Random) -> str:
    if len(word) <= 3:
        return word
    middle = list(word[1:-1])
    rng.shuffle(middle)
    return word[0] + "".join(middle) + word[-1]


def _capitalize(text: str, probability: float, rng: random.Random) -> str:
    if probability <= 0 or text == "":
        return text
    chars: list[str] = []
    for char in text:
        if char.isalpha() and rng.random() < probability:
            chars.append(char.upper())
        else:
            chars.append(char)
    return "".join(chars)


def _ascii_noise(text: str, probability: float, rng: random.Random) -> str:
    if probability <= 0 or text == "":
        return text
    chars: list[str] = []
    for char in text:
        code = ord(char)
        if not 32 <= code <= 126 or rng.random() >= probability:
            chars.append(char)
            continue
        delta = -1 if rng.random() < 0.5 else 1
        shifted = code + delta
        chars.append(chr(shifted) if 32 <= shifted <= 126 else char)
    return "".join(chars)
