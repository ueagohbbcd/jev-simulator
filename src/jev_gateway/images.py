"""Validate local inline images and turn them into metadata-free PNG data URLs."""
from __future__ import annotations

import base64
import binascii
import io
import re
import warnings
from collections.abc import Sequence
from typing import Any

from PIL import Image, ImageOps

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000

_DATA_URL_HEADER = re.compile(r"data:(image/(?:jpeg|png|webp|gif));base64")
_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}


class ImageValidationError(ValueError):
    """A safe validation failure; never includes supplied image data."""


class _BoundedBuffer(io.BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: bytes) -> int:
        if self.tell() + len(data) > self.limit:
            raise ImageValidationError("Normalized images exceed the 32 MiB total limit")
        return super().write(data)


def _decode(image: Any) -> tuple[str, bytes]:
    source = image.data
    declared = image.type
    if source.startswith("data:"):
        header, separator, encoded = source.partition(",")
        match = _DATA_URL_HEADER.fullmatch(header)
        if not separator or match is None:
            raise ImageValidationError("Image data URL must contain a supported base64 MIME type")
        mime = match.group(1)
        if declared is not None and declared != mime:
            raise ImageValidationError("Image type conflicts with data URL MIME type")
    else:
        if declared is None:
            raise ImageValidationError("Raw base64 images require an image type")
        mime, encoded = declared, source

    # Bound the allocation before base64 decoding. validate=True rejects
    # whitespace, URL-safe alphabet, and non-base64 characters.
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ImageValidationError("Image exceeds the 12 MiB decoded limit")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
        raise ImageValidationError("Image data is not strict base64") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image exceeds the 12 MiB decoded limit or is empty")
    return mime, raw


def _normalize_png(mime: str, raw: bytes, remaining: int) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as probe:
                if _FORMATS.get(probe.format) != mime:
                    raise ImageValidationError("Image MIME does not match its file format")
                width, height = probe.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise ImageValidationError("Image exceeds the 16 million pixel limit")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ImageValidationError("Animated images are not supported")
                probe.verify()

            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                mode = (
                    "RGBA"
                    if "A" in oriented.getbands() or "transparency" in oriented.info
                    else "RGB"
                )
                # A new canvas drops EXIF, ICC profiles, comments, and other
                # source metadata while retaining orientation and alpha.
                clean = Image.new(mode, oriented.size)
                clean.paste(oriented.convert(mode))
                output = _BoundedBuffer(remaining)
                clean.save(output, format="PNG")
                return output.getvalue()
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("Image data is invalid or corrupt") from exc


def normalize_images(images: Sequence[Any]) -> list[str]:
    """Decode all inputs first, then validate pixels and encode once per image."""
    if len(images) > MAX_IMAGES:
        raise ImageValidationError("Request has too many images")
    decoded: list[tuple[str, bytes]] = []
    raw_total = 0
    for image in images:
        mime, raw = _decode(image)
        raw_total += len(raw)
        if raw_total > MAX_TOTAL_IMAGE_BYTES:
            raise ImageValidationError("Images exceed the 32 MiB decoded total limit")
        decoded.append((mime, raw))

    normalized: list[str] = []
    normalized_total = 0
    for mime, raw in decoded:
        png = _normalize_png(mime, raw, MAX_TOTAL_IMAGE_BYTES - normalized_total)
        normalized_total += len(png)
        normalized.append("data:image/png;base64," + base64.b64encode(png).decode("ascii"))
    return normalized
