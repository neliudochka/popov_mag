from pathlib import Path
import pandas as pd
from mdd_processing.preprocessing.Data.Data import Data
from mdd_processing.preprocessing.Data.DataConfig import DataConfig
from sklearn.metrics import matthews_corrcoef, classification_report


def get_data(path_data, path_conf) -> Data:
    cnf = DataConfig.from_yaml(path_conf)
    return Data.from_file(path_data, cnf, ",")


def get_expected(data):
    return data.df[data.column_names.track_id_col]


def get_predicted(pred_data, exp_data):
    track_set = set(pred_data.df[pred_data.column_names.utc_timestamp_col])
    return exp_data.df[exp_data.column_names.utc_timestamp_col].apply(
        lambda ts: "track" if ts in track_set else "not_track"
    )


def calc_metrics(gt: Data, pred: Data):
    expected = get_expected(gt)
    predicted = get_predicted(pred, gt)

    classes = ["track", "not_track"]
    cr_dict = classification_report(
        expected,
        predicted,
        labels=classes,
        zero_division="warn",
        output_dict=True
    )

    df_cr = pd.DataFrame(cr_dict).T
    df_cr = df_cr.loc[classes, ["precision", "recall", "f1-score", "support"]]
    df_cr['mcc'] = matthews_corrcoef(expected, predicted)

    return df_cr


def evaluate_test_case(testcase_dir: Path, own_conf: Path, mg_conf: Path, gt_conf: Path):
    gt_path_file_path = testcase_dir / "ground_truth.csv"
    own_path_file_path = testcase_dir / "own.csv"
    mg_path_file_path = testcase_dir / "magdrone.csv"

    if not gt_path_file_path.exists():
        raise FileNotFoundError(f"No ground_truth.csv in {testcase_dir}")

    gt_data = get_data(gt_path_file_path, gt_conf)
    results = {}

    if own_path_file_path.exists():
        own_data = get_data(own_path_file_path, own_conf)
        results["own"] = calc_metrics(gt_data, own_data)

    if mg_path_file_path.exists():
        mg_data = get_data(mg_path_file_path, mg_conf)
        results["magdrone"] = calc_metrics(gt_data, mg_data)

    return results


def evaluate_all_test_cases(test_cases_dir: Path, own_conf: Path, mg_conf: Path, gt_conf: Path):
    results = []

    if not test_cases_dir.exists():
        raise RuntimeError(f"Test cases directory not found: {test_cases_dir}")
    for test_case_dir in sorted(test_cases_dir.iterdir()):
        if not test_case_dir.is_dir():
            continue
        test_case = test_case_dir.name
        try:
            results.append({test_case: evaluate_test_case(test_case_dir, own_conf, mg_conf, gt_conf)})
        except Exception as e:
            print(f"[SKIP] {test_case}: {e}")
            continue
        print(f"[OK] Test case {test_case} was processed")
    return results


from pathlib import Path


def parse_report_file(report_file: Path):
    """
    Parse a single report file and return a dict:
    { 'track_precision': value, 'track_recall': value, ..., 'MCC': value }
    """
    metrics = {}
    with report_file.open("r") as f:
        lines = f.read().splitlines()

    model = None
    in_table = False
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Model header
        if line.isupper() and "RESULTS" in line:
            model = line.replace(" RESULTS", "").lower()
            in_table = True
            continue

        # Table header
        if in_table and line.startswith("precision"):
            continue

        # MCC line
        if line.startswith("MCC:"):
            mcc_value = float(line.split(":")[1].strip())
            metrics[f"{model}_MCC"] = mcc_value
            in_table = False
            continue

        # Table rows
        if in_table:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_name, precision, recall, f1, support = parts
            metrics[f"{model}_{class_name}_precision"] = float(precision)
            metrics[f"{model}_{class_name}_recall"] = float(recall)
            metrics[f"{model}_{class_name}_f1-score"] = float(f1)
            metrics[f"{model}_{class_name}_support"] = float(support)

    return metrics


from pathlib import Path

def metric_sort_key(metric: str):
    if metric == "MCC":
        return (0, "")
    if metric.startswith("track_"):
        return (1, metric)
    if metric.startswith("not_track_"):
        return (2, metric)
    return (3, metric)

def generate_long_summary(results: list):
    all_rows = []

    for item in results:
        tc_name, tc_results = next(iter(item.items()))

        for algorithm, df in tc_results.items():
            algorithm = algorithm.lower()

            mcc_value = float(df["mcc"].iloc[0])
            all_rows.append({
                "Algorithm": algorithm,
                "Metric": "MCC",
                "TestCase": tc_name,
                "Value": mcc_value,
            })

            for class_name, row in df.iterrows():
                for metric in ["precision", "recall", "f1-score", "support"]:
                    all_rows.append({
                        "Algorithm": algorithm,
                        "Metric": f"{class_name}_{metric}",
                        "TestCase": tc_name,
                        "Value": float(row[metric]),
                    })

    df_long = pd.DataFrame(all_rows)
    df_long.columns = ["Algorithm", "Metric", "TestCase", "Value"]

    df_summary = df_long.pivot(
        index=["Algorithm", "Metric"],
        columns="TestCase",
        values="Value"
    )


    df_summary["Final"] = df_summary.mean(axis=1)
    cols = ["Final"] + [c for c in df_summary.columns if c != "Final"]
    df_summary = df_summary[cols]
    df_summary = df_summary.reset_index()
    df_summary = df_summary.sort_values(
        by=["Metric", "Algorithm"],
        key=lambda col: col.map(metric_sort_key) if col.name == "Metric" else col
    )
    return df_summary

def generate_short_summary(summary):
    short = summary[['Algorithm', 'Metric', 'Final']]
    keep_metrics = {
        "MCC",
        "track_precision",
        "track_recall",
        "track_f1-score",
    }
    return short[short["Metric"].isin(keep_metrics)].reset_index(drop=True)


if __name__ == "__main__":
    root_dir = Path(__file__).parent.resolve()
    tc_dir = root_dir / "Test_Data"
    report_dir = root_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    own_config = root_dir / "../../mdd_processing/preprocessing/Data/data_configs/own.yaml"
    mg_config = root_dir / "../../mdd_processing/preprocessing/Data/data_configs/magdrone.yaml"
    gt_config = root_dir / "configs/ground_truth.yaml"

    eval_results = evaluate_all_test_cases(
        tc_dir,
        own_config,
        mg_config,
        gt_config
    )

    long_summary = generate_long_summary(eval_results)
    short_summary = generate_short_summary(long_summary)

    long_path = report_dir / "long_summary.csv"
    short_path = report_dir / "summary.csv"

    print(f"[INFO] Saving long summary to: {long_path}")
    long_summary.to_csv(long_path, index=False, float_format="%.6f", decimal=",")

    print(f"[INFO] Saving short summary to: {short_path}")
    short_summary.to_csv(short_path, index=False, float_format="%.6f", decimal=",")
