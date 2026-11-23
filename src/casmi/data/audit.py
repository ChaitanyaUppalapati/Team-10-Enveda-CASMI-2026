"""Inspect actual Parquet metadata, without downloading or reading all peaks. Owner: Member 2."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def inspect_parquet(path):
    source = Path(path).expanduser().resolve()
    parquet = pq.ParquetFile(source)
    names = set(parquet.schema_arrow.names)
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "fields": [{"name": field.name, "type": str(field.type)} for field in parquet.schema_arrow],
        "native_query_id": "query_id" if "query_id" in names else ("molecule_id" if "molecule_id" in names else None),
        "native_spectrum_id": "spectrum_id" in names,
        "peak_columns": [
            name for name in ("mzs", "ms2_mzs", "intensities", "ms2_normalized_intensities") if name in names
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps([inspect_parquet(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
