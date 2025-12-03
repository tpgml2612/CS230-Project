import numpy as np
import pandas as pd

def generate_rich_dataset(
    n_samples=500,
    dim_x1=100,
    dim_x2=100,
    noise_std=0.05,
    save_examples_path="examples.csv",
    save_labels_path="labels.csv",
    seed=42
):
    np.random.seed(seed)

    # ------------------------------------------
    # 1) Generate X1, X2
    # ------------------------------------------
    X1 = np.random.uniform(-3, 3, size=(n_samples, dim_x1))
    X2 = np.random.uniform(-3, 3, size=(n_samples, dim_x2))

    # ------------------------------------------
    # 2) Random coefficients for complex formulas
    # ------------------------------------------
    a = np.random.uniform(-1, 1, dim_x1)  # linear X1
    b = np.random.uniform(-0.5, 0.5, dim_x2)  # quadratic X2
    c = np.random.uniform(-0.3, 0.3, min(dim_x1, dim_x2))  # interaction

    d = np.random.uniform(-1, 1, dim_x1)  # linear X1
    e = np.random.uniform(-0.01, 0.01, dim_x2)  # cubic X2
    f = np.random.uniform(-0.5, 0.5, min(dim_x1, dim_x2))  # sin interaction

    # ------------------------------------------
    # 3) Compute y1, y2
    # ------------------------------------------
    y1_list = []
    y2_list = []

    for i in range(n_samples):
        x1 = X1[i]
        x2 = X2[i]

        # y1 = linear(X1) + quadratic(X2) + interaction(X1*X2)
        term_linear = np.sum(a * x1)
        term_quad   = np.sum(b * (x2 ** 2))
        term_inter  = np.sum(c * (x1[:len(c)] * x2[:len(c)]))

        y1 = term_linear + term_quad + term_inter

        # y2 = linear(X1) + cubic(X2) + sinusoidal(X1+X2)
        term_linear2 = np.sum(d * x1)
        term_cubic    = np.sum(e * (x2 ** 3))
        term_sin      = np.sum(f * np.sin(x1[:len(f)] + x2[:len(f)]))

        y2 = term_linear2 + term_cubic + term_sin

        # Add noise
        y1 += np.random.normal(0, noise_std)
        y2 += np.random.normal(0, noise_std)

        y1_list.append(y1)
        y2_list.append(y2)

    # ------------------------------------------
    # 4) Convert X1, X2 to string lists
    # ------------------------------------------
    X1_str = [str(list(v)) for v in X1]
    X2_str = [str(list(v)) for v in X2]

    # ------------------------------------------
    # 5) Build and save DataFrames
    # ------------------------------------------
    examples_df = pd.DataFrame({"X1": X1_str, "X2": X2_str})
    labels_df = pd.DataFrame({"y1": y1_list, "y2": y2_list})

    examples_df.to_csv(save_examples_path, index=False)
    labels_df.to_csv(save_labels_path, index=False)

    print(f"✔ Saved examples: {examples_df.shape} → {save_examples_path}")
    print(f"✔ Saved labels:   {labels_df.shape} → {save_labels_path}")

    return examples_df, labels_df


# Example run
if __name__ == "__main__":
    generate_rich_dataset(
        n_samples=200,
        dim_x1=100,
        dim_x2=100,
        noise_std=0.02,
        save_examples_path="X_examples.csv",
        save_labels_path="Y_labels.csv"
    )
