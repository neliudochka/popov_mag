import logging
from pathlib import Path
from .functions import create_rows, create_combined_rows, parse_arguments


def convert_mdd_to_csv(input, save_OF=False):
    logger = logging.getLogger(__name__)

    input_path = Path(input)
    if not input_path.exists():
        logger.error(f"Incorrect input path: {input_path}")
        return

    original_format_path = input_path.with_name(input_path.stem + "_OF.csv")
    combined_format_path = input_path.with_name(input_path.stem + "_RF.csv")

    data = input_path.read_bytes()
    rows = create_rows(data)

    if save_OF:
        with open(original_format_path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(str(r) + "\n")
        logger.info(f"Output original: {original_format_path}")


    combined = create_combined_rows(rows)
    header = "Timestamp [ms]; B1x [nT]; B1y [nT]; B1z [nT]; B2x [nT]; B2y [nT]; B2z [nT]; AccX [g]; AccY [g]; AccZ [g]; Temp [Deg]; Latitude [Decimal Degrees]; Longitude [Decimal Degrees]; Altitude [m]; Satellites; Quality; GPSTime; GPSDate; GPSTime [hh:mm:ss.sss]"

    with open(combined_format_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        for row in combined:
            f.write(row.export_to_csv(";") + "\n")

    return combined_format_path

def main():
    args = parse_arguments()
    input = Path(args.input)
    # inp = Path("../files_for_test/112_cor/30/20250530_100711_MD-R3_#0112.mdd")
    # convert_mdd_to_csv(inp, save_OF=False)
