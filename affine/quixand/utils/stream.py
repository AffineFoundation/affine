from __future__ import annotations

from typing import Generator, Iterable


def iter_lines(chunks: Iterable[bytes]) -> Generator[bytes, None, None]:
	buf = bytearray()
	for chunk in chunks:
		if not chunk:
			continue
		buf.extend(chunk)
		while True:
			try:
				i = buf.index(10)  # '\n'
			except ValueError:
				break
			yield bytes(buf[: i + 1])
			del buf[: i + 1]
	if buf:
		yield bytes(buf)


