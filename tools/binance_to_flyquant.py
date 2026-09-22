#!/usr/bin/env python3
"""Download Binance USD-M archives and build a FlyQuant market-event CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import itertools
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Iterator


HEADER = [
    "exchange_ts_ns", "receive_ts_ns", "sequence", "event_type",
    "bid_px", "bid_qty", "ask_px", "ask_qty",
    "trade_px", "trade_qty", "aggressor",
]
BOOK_FIELDS = [
    "update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
    "best_ask_qty", "transaction_time", "event_time",
]
TRADE_FIELDS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


@dataclass(frozen=True)
class Event:
    timestamp_ms: int
    source_rank: int
    native_id: int
    row_number: int
    values: tuple[str, ...]

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        return (self.timestamp_ms, self.source_rank, self.native_id, self.row_number)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include a UTC offset")
    return int(parsed.timestamp() * 1000)


def archive_url(base_url: str, symbol: str, stream: str, day: str) -> str:
    filename = f"{symbol}-{stream}-{day}.zip"
    return f"{base_url.rstrip('/')}/{stream}/{symbol}/{filename}"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FlyQuant/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def ensure_archive(
    path: Path, url: str, expected_sha256: str | None, download_missing: bool
) -> None:
    if not path.is_file():
        if not download_missing:
            raise FileNotFoundError(f"missing archive: {path}")
        print(f"Downloading {url}", file=sys.stderr)
        download(url, path)
    actual = sha256_file(path)
    if expected_sha256 and actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    print(f"{actual}  {path}", file=sys.stderr)


def zip_rows(path: Path, expected_fields: list[str]) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path}, found {len(members)}")
        with archive.open(members[0]) as raw:
            text_stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.reader(text_stream)
            first = next(reader, None)
            if first is None:
                return
            header = [cell.strip() for cell in first]
            rows = reader if header == expected_fields else itertools.chain([first], reader)
            for row_number, row in enumerate(rows, start=1):
                if len(row) != len(expected_fields):
                    raise ValueError(
                        f"{path}: row {row_number} has {len(row)} columns; "
                        f"expected {len(expected_fields)}"
                    )
                yield dict(zip(expected_fields, row))


def book_events(path: Path, start_ms: int, end_ms: int) -> Iterator[Event]:
    for row_number, row in enumerate(zip_rows(path, BOOK_FIELDS), start=1):
        timestamp_ms = int(row["transaction_time"] or row["event_time"])
        if start_ms <= timestamp_ms < end_ms:
            yield Event(
                timestamp_ms, 0, int(row["update_id"]), row_number,
                (
                    "QUOTE", row["best_bid_price"], row["best_bid_qty"],
                    row["best_ask_price"], row["best_ask_qty"], "", "", "",
                ),
            )


def trade_events(path: Path, start_ms: int, end_ms: int) -> Iterator[Event]:
    for row_number, row in enumerate(zip_rows(path, TRADE_FIELDS), start=1):
        timestamp_ms = int(row["transact_time"])
        if start_ms <= timestamp_ms < end_ms:
            aggressor = "S" if row["is_buyer_maker"].lower() == "true" else "B"
            yield Event(
                timestamp_ms, 1, int(row["agg_trade_id"]), row_number,
                ("TRADE", "", "", "", "", row["price"], row["quantity"], aggressor),
            )


def convert(
    book_path: Path, trade_path: Path, output_path: Path, start_ms: int, end_ms: int
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = heapq.merge(
        book_events(book_path, start_ms, end_ms),
        trade_events(trade_path, start_ms, end_ms),
        key=lambda event: event.sort_key,
    )
    quotes = trades = 0
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(HEADER)
        for sequence, event in enumerate(merged, start=1):
            writer.writerow([event.timestamp_ms * 1_000_000, "", sequence, *event.values])
            if event.source_rank == 0:
                quotes += 1
            else:
                trades += 1
    return quotes + trades, quotes, trades


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--date", required=True, help="UTC archive date (YYYY-MM-DD)")
    result.add_argument("--symbol", default="BTCUSDT")
    result.add_argument("--start", help="inclusive ISO-8601 timestamp; defaults to date start")
    result.add_argument("--end", help="exclusive ISO-8601 timestamp; defaults to next date start")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--archive-dir", type=Path, default=Path("data/raw/binance"))
    result.add_argument(
        "--base-url", default="https://data.binance.vision/data/futures/um/daily"
    )
    result.add_argument("--book-sha256")
    result.add_argument("--trades-sha256")
    result.add_argument("--expected-output-sha256")
    result.add_argument("--no-download", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    day = date.fromisoformat(args.date)
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    start_ms = parse_utc(args.start) if args.start else int(day_start.timestamp() * 1000)
    end_ms = parse_utc(args.end) if args.end else start_ms + 86_400_000
    if end_ms <= start_ms:
        raise ValueError("--end must be after --start")

    book_path = args.archive_dir / f"{args.symbol}-bookTicker-{args.date}.zip"
    trade_path = args.archive_dir / f"{args.symbol}-aggTrades-{args.date}.zip"
    ensure_archive(
        book_path, archive_url(args.base_url, args.symbol, "bookTicker", args.date),
        args.book_sha256, not args.no_download,
    )
    ensure_archive(
        trade_path, archive_url(args.base_url, args.symbol, "aggTrades", args.date),
        args.trades_sha256, not args.no_download,
    )
    total, quotes, trades = convert(book_path, trade_path, args.output, start_ms, end_ms)
    output_sha = sha256_file(args.output)
    if args.expected_output_sha256 and output_sha.lower() != args.expected_output_sha256.lower():
        raise ValueError(
            f"output SHA-256 mismatch: expected {args.expected_output_sha256}, got {output_sha}"
        )
    print(f"events={total} quotes={quotes} trades={trades}")
    print(f"{output_sha}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
