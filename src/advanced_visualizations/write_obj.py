import numpy as np
import pandas as pd


df = pd.read_csv(
    r"C:\WorkingData\Documents\2_Coding\Python\aaoca_ivus_analysis\data_eacvi\ivus\NARCO_122\Rest\PDWN7A4I_csv_files\diastolic_contours.csv",
    sep="\t",
    header=None,
    names=["contour_idx", "x", "y", "z"],
)


def reorder_contour(contour_df):
    df = contour_df.copy()
    center = df[["x", "y"]].mean()
    df["angle"] = np.arctan2(df["y"] - center.y, df["x"] - center.x)
    df = df.sort_values("angle", ascending=False).reset_index(drop=True)
    start_pos = int(df["y"].idxmax())
    df = pd.concat(
        [
            df.iloc[start_pos:].reset_index(drop=True),
            df.iloc[:start_pos].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    return df.drop(columns="angle").reset_index(drop=True)


reordered = []
for contour_idx, group in df.groupby("contour_idx"):
    c = reorder_contour(group)
    c["contour_idx"] = contour_idx
    c["point_idx"] = range(len(c))
    reordered.append(c)

reordered_df = pd.concat(reordered, ignore_index=True)


with open("generated.obj", "w") as f:
    # vertices
    for _, row in reordered_df.iterrows():
        f.write(f"v {row.x} {row.y} {row.z}\n")
    # faces: connect contour j to j+1
    contours = reordered_df["contour_idx"].unique()
    start_idxs = {
        ci: reordered_df[reordered_df.contour_idx == ci].index.tolist()
        for ci in contours
    }
    for c0, c1 in zip(contours, contours[1:]):
        idxs0 = start_idxs[c0]
        idxs1 = start_idxs[c1]
        # pairwise, and wrap the last one back to the first
        for i in range(len(idxs0)):
            i0 = idxs0[i] + 1  # OBJ is 1‑based
            i1 = idxs1[i] + 1
            i0n = idxs0[(i + 1) % len(idxs0)] + 1
            f.write(f"f {i0} {i1} {i0n}\n")
