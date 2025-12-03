import numpy as np
import pandas as pd

def generate_fake_dataset(
    n_samples=500,
    n_x1=2,
    n_x2=2,
    noise_std=0.1,
    save_examples_path="examples.csv",
    save_labels_path="labels.csv",
    random_seed=42
):
    """
    Generate a fake dataset:
    X = [x1_1 ... x1_k, x2_1 ... x2_m]
    y1, y2 = functions of (x1 + x2)
    """

    np.random.seed(random_seed)

    # -------------------------------
    # 1) Generate x1, x2
    # -------------------------------
    X1 = np.random.uniform(low=1000, high=1300, size=(n_samples, n_x1))
    X2 = np.random.uniform(low=1200, high=1400, size=(n_samples, n_x2))

    # -------------------------------
    # 2) Define true relationships
    # -------------------------------
    # y1 = 2*x1_1 + 0.5*x2_1 + noise
    # y2 = -x1_1 + 3*x2_1 + noise

    x1_1 = X1[:, 0]
    x2_1 = X2[:, 0]

    noise = np.random.normal(0, noise_std, size=n_samples)

    y1 = 2 * x1_1 + 0.5 * x2_1 + noise
    y2 = -1 * x1_1 + 3 * x2_1 + noise

    labels = np.vstack([y1, y2]).T

    # -------------------------------
    # 3) Construct DataFrames
    # -------------------------------
    col_names_x1 = [f"x1_{i+1}" for i in range(n_x1)]
    col_names_x2 = [f"x2_{i+1}" for i in range(n_x2)]

    examples = pd.DataFrame(
        np.hstack([X1, X2]),
        columns=col_names_x1 + col_names_x2
    )

    labels_df = pd.DataFrame(
        labels,
        columns=["y1", "y2"]
    )

    # -------------------------------
    # 4) Save CSV files
    # -------------------------------
    examples.to_csv(save_examples_path, index=False)
    labels_df.to_csv(save_labels_path, index=False)

    print(f"✔ Fake dataset generated!")
    print(f"   examples: {examples.shape} -> saved to {save_examples_path}")
    print(f"   labels:   {labels_df.shape} -> saved to {save_labels_path}")

    return examples, labels_df


# Example usage
if __name__ == "__main__":
    generate_fake_dataset(
        n_samples=100,
        n_x1=2,
        n_x2=2,
        noise_std=0.05,
        save_examples_path="valid_examples_test.csv",
        save_labels_path="valid_labels_test.csv",
    )
