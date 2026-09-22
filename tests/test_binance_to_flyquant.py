import csv
import importlib.util
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "binance_to_flyquant", ROOT / "tools" / "binance_to_flyquant.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_zip(path: Path, member: str, rows: list[list[str]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())


class BinanceConversionTest(unittest.TestCase):
    def test_merge_timestamp_conversion_and_aggressor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            book, trades, output = root / "book.zip", root / "trades.zip", root / "events.csv"
            write_zip(book, "book.csv", [
                MODULE.BOOK_FIELDS,
                ["10", "41999.10", "4", "41999.20", "5", "1000", "1000"],
                ["11", "42000.10", "2", "42000.20", "3", "1001", "1002"],
            ])
            write_zip(trades, "trades.csv", [
                MODULE.TRADE_FIELDS,
                ["7", "41999.15", "0.50", "19", "19", "1000", "false"],
                ["8", "42000.15", "0.25", "20", "20", "1001", "true"],
            ])

            self.assertEqual(MODULE.convert(book, trades, output, 1000, 1002), (4, 2, 2))
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["sequence"] for row in rows], ["1", "2", "3", "4"])
            self.assertEqual(
                [(row["exchange_ts_ns"], row["event_type"]) for row in rows],
                [
                    ("1000000000", "QUOTE"), ("1000000000", "TRADE"),
                    ("1001000000", "QUOTE"), ("1001000000", "TRADE"),
                ],
            )
            self.assertEqual([rows[1]["aggressor"], rows[3]["aggressor"]], ["B", "S"])
            self.assertEqual(rows[0]["receive_ts_ns"], "")

    def test_headerless_archive_and_half_open_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            book, trades, output = root / "book.zip", root / "trades.zip", root / "events.csv"
            write_zip(book, "book.csv", [
                ["1", "10", "1", "11", "1", "999", "999"],
                ["2", "12", "2", "13", "2", "1000", "1000"],
                ["3", "14", "3", "15", "3", "2000", "2000"],
            ])
            write_zip(trades, "trades.csv", [])
            self.assertEqual(MODULE.convert(book, trades, output, 1000, 2000), (1, 1, 0))

    def test_archive_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "input.zip"
            archive.write_bytes(b"not-the-expected-file")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.ensure_archive(archive, "unused", "0" * 64, False)


if __name__ == "__main__":
    unittest.main()
