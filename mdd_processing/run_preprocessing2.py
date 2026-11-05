import argparse
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import yaml
from mdd_processing.preprocessing.main_popov import preprocess
from mdd_processing.mddConverter.main import convert_mdd_to_csv

# -----------------------------
# GLOBAL LOGGER DECLARATION
# -----------------------------
logger = logging.getLogger(__name__)

@dataclass
class FileProcessConfig:
    mdd_filepath: str | None = None
    src_filepath: str | None = None
    result_filepath: str | None = None
    delimiter: str = ","
    systeminfo_filepath: str | None = None
    datetime_specified: str | None = None
    gnss: str | None = None
    write_log: bool | None = False

    track_max_angle: int | None = None
    track_min_len: int | None = None
    fc_max_deviation: int | None = None
    use_facing: bool | None = False
    geom_offset: float | None = None
    s1_offset: float | None = None
    s2_offset: float | None = None
    alt: float | None = None
    filter_type: str | None = None
    filt_window: float | None = None
    cutoff_freq: float | None = None


def get_src(mdd_filepath, src_filepath) -> str | None:
    if not mdd_filepath:
        return src_filepath
    if src_filepath:
        logger.warning("MDD path provided. Ignoring SRC path.")
    logger.info("Decoding MDD to CSV started")
    src_filepath = convert_mdd_to_csv(mdd_filepath, save_OF=False)
    logger.info(f"Decoding MDD to CSV finished. CSV saved to {src_filepath}\n")
    return src_filepath


def validate_src(src_filepath):
    if not src_filepath:
        raise ValueError("No source CSV file specified. Cannot proceed.")

def exec_preprocessing(config, callback)-> str | None:
    fp = config.src_filepath
    logger.info(f"Preprocessing started for {fp}")
    preprocess(fp,
               config.delimiter,
               config.systeminfo_filepath,
               config.result_filepath,
               config.datetime_specified,
               config.gnss,
               config.track_max_angle,
               config.track_min_len,
               config.fc_max_deviation,
               config.use_facing,

               config.geom_offset,
               config.s1_offset,
               config.s2_offset,
               config.alt,

               config.filter_type,
               config.filt_window,
               config.cutoff_freq,
               callback)
    logger.info(f"Preprocessing finished for {fp}")


def process_single_file(config: FileProcessConfig, callback = None):
    try:
        config.src_filepath = get_src(config.mdd_filepath, config.src_filepath)
        validate_src(config.src_filepath)
        exec_preprocessing(config, callback)
    except Exception as e:
        new_error = f"File processing failed: {e}"
        logger.error(new_error)  # delete when the postup-analysis-tool handles errors correctly
        raise RuntimeError(new_error) from e


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing")

    parser.add_argument("-c", "--config", type=str, default="config_popov.yaml", help="Path to YAML config file")
    parser.add_argument("-m", "--mdd", type=str, help="Path to MDD file")
    parser.add_argument("--src", type=str, help="Path to source CSV file")
    parser.add_argument("-d", "--delimiter", type=str, help="Delimiter used in the source CSV")
    parser.add_argument("-o", "--output", type=str, help="Path to save the result CSV")
    parser.add_argument("--systeminfo_filepath", type=str,
                        help="Path to the systeminfo TXT file (magnetometer configuration file)")
    parser.add_argument("-t", "--datetime", type=str, help="Date string in format YYYYMMDD_HHMMSS.S")
    parser.add_argument("--gnss", type=str, help="Use external GNSS")
    parser.add_argument("--mdd_dir", type=str, help="Path to directory with MDD files")
    parser.add_argument("--write_log", type=bool, help="Write log to the file or not")

    parser.add_argument("--geom_offset", type=float, help="Geometric offset in meters")
    parser.add_argument("--s1_offset", type=float, help="S1 sensor offset in meters")
    parser.add_argument("--s2_offset", type=float, help="S2 sensor offset in meters")
    parser.add_argument("--alt", type=float, help="Altitude for the mission")

    parser.add_argument("--track_max_angle", type=bool, help="Max angle difference that is allowed inside the track")
    parser.add_argument("--track_min_len", type=bool, help="Min length allowed for a track")
    parser.add_argument("--fc_max_deviation", type=bool, help="Max difference in the orientation that is allowed during the track")
    parser.add_argument("--use_facing", type=bool, help="Does nothing for now")
    parser.add_argument("--filter_type", type=int, help="Type of the filter used for TFA computation")
    parser.add_argument("--filt_window", type=float, help="Size of the filter window in meters for TFA computation")
    parser.add_argument("--cutoff_freq", type=float, help="Low Pass filter cutoff frequency for TFA computation")

    return parser.parse_args()

def load_config(config_name) -> dict:
    with open(config_name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_file_config(args, config_dict) -> FileProcessConfig:
    file_config = FileProcessConfig(
        mdd_filepath=args.mdd or config_dict.get("mdd"),
        src_filepath=args.src or config_dict.get("src"),
        result_filepath=args.output or config_dict.get("output"),
        delimiter=args.delimiter or config_dict.get("delimiter", ","),
        systeminfo_filepath=args.systeminfo_filepath or config_dict.get("systeminfo_filepath"),
        datetime_specified=args.datetime or config_dict.get("datetime"),
        gnss=args.gnss or config_dict.get("gnss"),
        write_log=args.write_log or config_dict.get("write_log"),

        geom_offset=args.geom_offset or config_dict.get("geom_offset"),
        s1_offset=args.s1_offset or config_dict.get("s1_offset"),
        s2_offset=args.s2_offset or config_dict.get("s2_offset"),
        alt=args.alt or config_dict.get("alt"),

        track_max_angle=args.track_max_angle or config_dict.get("track_max_angle"),
        track_min_len=args.track_min_len or config_dict.get("track_min_len"),
        fc_max_deviation=args.fc_max_deviation or config_dict.get("fc_max_deviation"),
        use_facing=args.use_facing or config_dict.get("use_facing"),
        filter_type=args.filter_type or config_dict.get("filter_type"),
        filt_window=args.filt_window or config_dict.get("filt_window"),
        cutoff_freq=args.cutoff_freq or config_dict.get("cutoff_freq"),
    )
    return file_config


def process_mdd_directory(mdd_dir: str, base_config: FileProcessConfig):
    mdd_path = Path(mdd_dir).resolve()
    mdd_files = glob.glob(str(mdd_path / "*.mdd"))
    logger.info(f"Found {len(mdd_files)} MDD files in directory '{mdd_dir}'")

    for mdd_file in mdd_files:
        logger.info(f"Processing file: {mdd_file}")
        config_copy = FileProcessConfig(**vars(base_config))
        config_copy.mdd_filepath = mdd_file
        config_copy.src_filepath = None
        process_single_file(config_copy)


def setup_logger(write_log: bool):
    handlers = [logging.StreamHandler()]
    log_path=""
    if write_log:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))


    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers
    )

    if write_log:
        logger.info(f"Logging to: {log_path}")
    else:
        logger.info("File logging disabled via config.")


def main():
    args = parse_args()
    config_dict = load_config(args.config)
    base_config = build_file_config(args, config_dict)

    setup_logger(base_config.write_log)

    if args.mdd_dir:
        process_mdd_directory(args.mdd_dir, base_config)
    else:
        process_single_file(base_config)


if __name__ == "__main__":
    main()
