"""Map a filename to the document format hint Docling expects.

Mirrors the suffix→format table the SDK uses for document inputs; the hint
becomes Docling's source name and an extension fallback when byte-sniffing is
ambiguous. Unknown/absent extensions return ``None`` and let Docling sniff.
"""

_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".rtf": "rtf",
    ".odt": "odt",
}


def format_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    lowered = filename.lower()
    dot = lowered.rfind(".")
    if dot == -1:
        return None
    return _SUFFIX_TO_FORMAT.get(lowered[dot:])
