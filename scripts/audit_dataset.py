
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elliptic"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES_FILE = DATA_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = DATA_DIR / "elliptic_txs_edgelist.csv"

EXPECTED_FILES = [FEATURES_FILE, CLASSES_FILE, EDGES_FILE]


def sha256_file(path):
    """Calculate a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def is_integer(value):
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


def read_csv_header(path):
    """Read the first CSV row."""
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return next(csv.reader(file))


def find_column(header, candidates):
    """Find a column by its normalised name."""
    normalised = [name.strip().lower() for name in header]

    for candidate in candidates:
        if candidate in normalised:
            return normalised.index(candidate)

    return None


def audit_dataset():
    for path in EXPECTED_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected file: {path}")

    report = {
        "files": {},
        "features": {},
        "classes": {},
        "edges": {},
        "label_counts": {},
        "timestep_counts": {},
        "validation": {},
    }

    # File metadata and hashes
    for path in EXPECTED_FILES:
        report["files"][path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }


    # Feature file: node IDs, time steps and feature columns
    node_to_step = {}
    feature_row_lengths = Counter()
    duplicate_feature_ids = 0
    invalid_feature_rows = 0
    feature_rows = 0
    first_row = read_csv_header(FEATURES_FILE)

    # The original feature CSV is expected to have no header.
    # Detect a header if its first field is not an integer.
    has_feature_header = not is_integer(first_row[0])

    with FEATURES_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.reader(file)

        if has_feature_header:
            next(reader)

        for row in reader:
            if len(row) < 2:
                invalid_feature_rows += 1
                continue

            try:
                tx_id = int(row[0])
                timestep = int(row[1])
            except ValueError:
                invalid_feature_rows += 1
                continue

            feature_rows += 1
            feature_row_lengths[len(row)] += 1

            if tx_id in node_to_step:
                duplicate_feature_ids += 1
            else:
                node_to_step[tx_id] = timestep

    report["features"] = {
        "has_header": has_feature_header,
        "rows": feature_rows,
        "unique_node_ids": len(node_to_step),
        "duplicate_node_ids": duplicate_feature_ids,
        "invalid_rows": invalid_feature_rows,
        "row_length_distribution": {
            str(length): count
            for length, count in sorted(feature_row_lengths.items())
        },
        "feature_columns_excluding_id": (
            sorted(feature_row_lengths)[-1] - 1
            if feature_row_lengths
            else 0
        ),
        "timestep_min": min(node_to_step.values(), default=None),
        "timestep_max": max(node_to_step.values(), default=None),
        "unique_timesteps": sorted(set(node_to_step.values())),
    }

    # Class file: label mapping and label counts
    first_row = read_csv_header(CLASSES_FILE)
    has_class_header = not is_integer(first_row[0])

    class_counts = Counter()
    label_by_node = {}
    duplicate_class_ids = 0
    invalid_class_rows = 0
    class_rows = 0
    class_column = None

    if has_class_header:
        class_column = find_column(first_row, ["class", "label"])
        id_column = find_column(first_row, ["txid", "tx_id", "transaction_id"])

        if class_column is None or id_column is None:
            raise ValueError(
                "Could not identify ID and class columns in classes CSV: "
                f"{first_row}"
            )
    else:
        id_column = 0
        class_column = 1

    with CLASSES_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.reader(file)

        if has_class_header:
            next(reader)

        for row in reader:
            try:
                tx_id = int(row[id_column])
                label = row[class_column].strip().lower()
            except (ValueError, IndexError):
                invalid_class_rows += 1
                continue

            class_rows += 1
            class_counts[label] += 1

            if tx_id in label_by_node:
                duplicate_class_ids += 1
            else:
                label_by_node[tx_id] = label

    report["classes"] = {
        "has_header": has_class_header,
        "rows": class_rows,
        "unique_node_ids": len(label_by_node),
        "duplicate_node_ids": duplicate_class_ids,
        "invalid_rows": invalid_class_rows,
        "label_mapping": dict(sorted(class_counts.items())),
    }

    # Per-timestep labelled / unknown counts
    counts_by_step = defaultdict(Counter)

    for tx_id, timestep in node_to_step.items():
        label = label_by_node.get(tx_id, "missing_class_record")
        counts_by_step[timestep][label] += 1

    report["timestep_counts"] = {
        str(step): dict(sorted(counts.items()))
        for step, counts in sorted(counts_by_step.items())
    }

    # Edges: endpoints, duplicates, self-loops and time steps
    first_row = read_csv_header(EDGES_FILE)
    has_edge_header = not is_integer(first_row[0])

    if has_edge_header:
        source_column = find_column(
            first_row, ["txid1", "tx_id1", "source", "source_id"]
        )
        target_column = find_column(
            first_row, ["txid2", "tx_id2", "target", "target_id"]
        )

        if source_column is None or target_column is None:
            raise ValueError(
                f"Could not identify edge columns: {first_row}"
            )
    else:
        source_column = 0
        target_column = 1

    edge_rows = 0
    invalid_edge_rows = 0
    missing_source_nodes = 0
    missing_target_nodes = 0
    self_loops = 0
    cross_timestep_edges = 0
    duplicate_edges = 0
    seen_edges = set()

    with EDGES_FILE.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.reader(file)

        if has_edge_header:
            next(reader)

        for row in reader:
            try:
                source = int(row[source_column])
                target = int(row[target_column])
            except (ValueError, IndexError):
                invalid_edge_rows += 1
                continue

            edge_rows += 1
            edge = (source, target)

            if edge in seen_edges:
                duplicate_edges += 1
            else:
                seen_edges.add(edge)

            if source == target:
                self_loops += 1

            source_step = node_to_step.get(source)
            target_step = node_to_step.get(target)

            if source_step is None:
                missing_source_nodes += 1

            if target_step is None:
                missing_target_nodes += 1

            if (
                source_step is not None
                and target_step is not None
                and source_step != target_step
            ):
                cross_timestep_edges += 1

    report["edges"] = {
        "has_header": has_edge_header,
        "rows": edge_rows,
        "unique_directed_edges": len(seen_edges),
        "duplicate_edges": duplicate_edges,
        "invalid_rows": invalid_edge_rows,
        "missing_source_nodes": missing_source_nodes,
        "missing_target_nodes": missing_target_nodes,
        "self_loops": self_loops,
        "cross_timestep_edges": cross_timestep_edges,
    }

    # ID alignment and validation summary
    feature_ids = set(node_to_step)
    class_ids = set(label_by_node)

    report["validation"] = {
        "feature_ids_missing_class_record": len(feature_ids - class_ids),
        "class_ids_missing_feature_record": len(class_ids - feature_ids),
        "all_feature_ids_unique": duplicate_feature_ids == 0,
        "all_class_ids_unique": duplicate_class_ids == 0,
        "all_edge_rows_valid": invalid_edge_rows == 0,
        "no_cross_timestep_edges": cross_timestep_edges == 0,
    }

    # Save the report locally
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "dataset_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Save a convenient CSV for later inspection and plotting.
    timestep_path = REPORT_DIR / "timestep_counts.csv"

    with timestep_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["timestep", "illicit", "licit", "unknown", "missing_class_record"]
        )

        for step, counts in sorted(counts_by_step.items()):
            writer.writerow(
                [
                    step,
                    counts.get("1", 0),
                    counts.get("2", 0),
                    counts.get("unknown", 0),
                    counts.get("missing_class_record", 0),
                ]
            )

    print(json.dumps(report, indent=2))
    print(f"\nSaved audit report: {report_path}")
    print(f"Saved timestep counts: {timestep_path}")


if __name__ == "__main__":
    audit_dataset()
