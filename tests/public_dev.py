"""Compare typed Parquet results to the competition's public development answers."""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq


def verify(output, answers):
    for name in ("user_daily", "campaign_daily"):
        actual = pq.read_table(sorted((output / name).glob("*.parquet")))
        strings = {"user_id", "campaign_id", "channel", "top_category"}
        expected = csv.read_csv(answers / f"{name}.csv", convert_options=csv.ConvertOptions(
            column_types={"day": pa.date32(), **{col: pa.string() for col in strings}},
            strings_can_be_null=True,
        ))
        assert set(actual.column_names) == set(expected.column_names)
        assert actual.schema.field("day").type == pa.date32()
        actual = actual.select(expected.column_names)
        for field in expected.schema:
            datatype = actual.schema.field(field.name).type
            if pa.types.is_integer(field.type):
                assert pa.types.is_integer(datatype), (field.name, datatype)
            else:
                assert datatype == field.type, (field.name, datatype)
        keys = [(col, "ascending") for col in expected.column_names]
        assert actual.cast(expected.schema).sort_by(keys).equals(expected.sort_by(keys)), name
        print(f"PASS {output.name} {name}: {actual.num_rows} rows match the official answer")


if __name__ == "__main__":
    verify(Path(sys.argv[1]), Path(sys.argv[2]))
