import argparse

from persona_understanding.value_measurement.measurement_utils import JobClassifier
from persona_understanding.value_measurement.values_comparison import *
from tqdm import tqdm

tqdm.pandas()


def _read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="../datasets/recruitment_data.csv",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="../datasets/recruitment_position_levels.json",
    )
    parser.add_argument("--ending_index", type=int, default=1000)

    args = parser.parse_args()

    user_profile_dataset = _read_csv(args.input_dataset)[: args.ending_index]

    job_classifier = JobClassifier()

    job_classifier = JobClassifier()
    user_profile_dataset["Position_Level"] = user_profile_dataset.progress_apply(
        lambda x: job_classifier.get_position_level(x["Job Title"]), axis=1
    )

    grouped_data = (
        user_profile_dataset.groupby("Position_Level").apply(lambda x: x.index.tolist()).to_dict()
    )

    with open(args.output_file, "w") as output_file:
        json.dump(grouped_data, output_file, indent=2)
