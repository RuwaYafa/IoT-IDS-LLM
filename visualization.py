


#!/usr/bin/env python
# generate_pareto_scatter_no_labels.py
#
# 2-D Pareto chart (Accuracy vs log-latency) **without text labels** on points.
# Saves 4 000 × 3 000 px PNG → charts/pareto_accuracy_vs_latency_no_labels.png
# ───────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1. Minimal results table ────────────────────────────────────────────────
DATA = [
    # DL
    ("ANN", "DL", 0.44, 85.67), ("Autoencoder", "DL", 0.72,  3.21),
    ("CNN", "DL", 0.54, 80.18), ("DNN", "DL", 0.35, 83.71),
    ("LSTM", "DL", 0.61, 83.93), ("RNN", "DL", 0.51, 83.79),
    # ML
    ("Adaboost", "ML", 0.13, 69.78), ("BalancedSVM", "ML", 3.05, 83.35),
    ("DecisionTreeClassifier", "ML", 0.01, 81.34),
    ("ExtraTreesClassifier", "ML", 0.16, 88.35),
    ("GaussianNB", "ML", 0.01, 79.20), ("GradientBoostingClassifier", "ML", 0.03, 87.23),
    ("IsolationForestClass", "ML", 0.07, 23.30), ("KNN", "ML", 0.54, 86.56),
    ("LightGBM", "ML", 0.03, 87.63), ("LOFClass", "ML", 0.14, 21.07),
    ("LogisticRegression", "ML", 0.00, 81.43), ("MLP", "ML", 0.01, 86.43),
    ("NaiveBayes", "ML", 0.02, 76.96), ("Perceptron", "ML", 0.01, 79.69),
    ("Random Forest – Bagging", "ML", 0.06, 87.99),
    ("Random Forest – Weight", "ML", 0.07, 88.30),
    ("RandomForestClassifier", "ML", 0.05, 86.65),
    ("StackingClassifier", "ML", 2.06, 84.33),
    ("VotingClassifierModel", "ML", 1.99, 81.65), ("WeightedSVM", "ML", 2.23, 83.35),
    ("XGBoostClassifier", "ML", 0.02, 86.74),
    # LLM
    ("Mistral-7B/zeroshots", "LLM", 8243.0, 10.89),
    ("Mistral-7B/withSFT",   "LLM", 6712.0, 50.54),
    ("Llama-3.2-1B/zeroshots", "LLM", 10.07,  0.00),
    ("Llama-3.2-1B/withSFT",   "LLM", 24.08, 34.91),
    ("Llama-3.2-3B/zeroshots", "LLM", 144.0, 24.96),
    ("Llama-3.2-3B/withSFT",   "LLM", 139.0, 25.00),
]

df = pd.DataFrame(DATA, columns=["Algorithm", "Family", "PredictTime", "Accuracy"])
df["log_pred"] = np.log10(df["PredictTime"].replace(0, 1e-6))

# ─── 2. Plot ────────────────────────────────────────────────────────────────
COLORS  = {"DL": "#1f77b4", "ML": "#ff7f0e", "LLM": "#d62728"}
MARKERS = {"DL": "o",        "ML": "s",       "LLM": "^"}

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

for fam in df["Family"].unique():
    sub = df[df["Family"] == fam]
    ax.scatter(
        sub["log_pred"], sub["Accuracy"],
        c=COLORS[fam], marker=MARKERS[fam],
        label=fam, edgecolors="black", linewidths=0.4, s=70, alpha=0.85
    )

ax.set_xlabel("log10(Prediction Time  [s])", fontsize=12)
ax.set_ylabel("Accuracy  (%)", fontsize=12)
ax.set_title("Accuracy vs Inference Latency (DL · ML · LLM)", pad=12)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
ax.legend(title="Model Family")
fig.tight_layout()

out_dir = "charts"
os.makedirs(out_dir, exist_ok=True)
fig.savefig(f"{out_dir}/pareto_accuracy_vs_latency_no_labels.png", dpi=300)
print("Saved → charts/pareto_accuracy_vs_latency_no_labels.png")







# # Saves a 4 000 × 3 000 px PNG →  charts/pareto_accuracy_vs_latency.png
# # ────────────────────────────────────────────────────────────────────────────
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ────────────────────────────────────────────────────────────────────────────
# # 1.  Results table  (Algorithm, Family, Predict-time  [s], Accuracy [%])
# # ────────────────────────────────────────────────────────────────────────────
# DATA = [
#     # DL
#     ("ANN",                         "DL",     0.44, 85.67),
#     ("Autoencoder",                 "DL",     0.72,  3.21),
#     ("CNN",                         "DL",     0.54, 80.18),
#     ("DNN",                         "DL",     0.35, 83.71),
#     ("LSTM",                        "DL",     0.61, 83.93),
#     ("RNN",                         "DL",     0.51, 83.79),
#     # ML
#     ("Adaboost",                    "ML",     0.13, 69.78),
#     ("BalancedSVM",                 "ML",     3.05, 83.35),
#     ("DecisionTreeClassifier",      "ML",     0.01, 81.34),
#     ("ExtraTreesClassifier",        "ML",     0.16, 88.35),
#     ("GaussianNB",                  "ML",     0.01, 79.20),
#     ("GradientBoostingClassifier",  "ML",     0.03, 87.23),
#     ("IsolationForestClass",        "ML",     0.07, 23.30),
#     ("KNN",                         "ML",     0.54, 86.56),
#     ("LightGBM",                    "ML",     0.03, 87.63),
#     ("LOFClass",                    "ML",     0.14, 21.07),
#     ("LogisticRegression",          "ML",     0.00, 81.43),
#     ("MLP",                         "ML",     0.01, 86.43),
#     ("NaiveBayes",                  "ML",     0.02, 76.96),
#     ("Perceptron",                  "ML",     0.01, 79.69),
#     ("Random Forest – Bagging",     "ML",     0.06, 87.99),
#     ("Random Forest – Weight",      "ML",     0.07, 88.30),
#     ("RandomForestClassifier",      "ML",     0.05, 86.65),
#     ("StackingClassifier",          "ML",     2.06, 84.33),
#     ("VotingClassifierModel",       "ML",     1.99, 81.65),
#     ("WeightedSVM",                 "ML",     2.23, 83.35),
#     ("XGBoostClassifier",           "ML",     0.02, 86.74),
#     # LLM
#     ("Mistral-7B/zeroshots",        "LLM", 8243.00, 10.89),
#     ("Mistral-7B/withSFT",          "LLM", 6712.00, 50.54),
#     ("Llama-3.2-1B/zeroshots",      "LLM",   10.07,  0.00),
#     ("Llama-3.2-1B/withSFT",        "LLM",   24.08, 34.91),
#     ("Llama-3.2-3B/zeroshots",      "LLM",  144.00, 24.96),
#     ("Llama-3.2-3B/withSFT",        "LLM",  139.00, 25.00),
# ]
#
# df = pd.DataFrame(DATA, columns=["Algorithm", "Family", "PredictTime", "Accuracy"])
# df["log_pred"] = np.log10(df["PredictTime"].replace(0, 1e-6))
#
# # ────────────────────────────────────────────────────────────────────────────
# # 2.  Compute Pareto front  (higher Acc, lower log_pred)
# # ────────────────────────────────────────────────────────────────────────────
# front = []
# best_acc = -np.inf
# for i, row in df.sort_values("log_pred").iterrows():
#     if row["Accuracy"] > best_acc:
#         front.append(row["Algorithm"])
#         best_acc = row["Accuracy"]
#
# # ────────────────────────────────────────────────────────────────────────────
# # 3.  Plot
# # ────────────────────────────────────────────────────────────────────────────
# COLORS  = {"DL": "#1f77b4", "ML": "#ff7f0e", "LLM": "#d62728"}
# MARKERS = {"DL": "o",        "ML": "s",       "LLM": "^"}
#
# fig, ax = plt.subplots(figsize=(10, 7), dpi=300)   # 4 000 × 3 000 px
#
# for fam in df["Family"].unique():
#     sub = df[df["Family"] == fam]
#     ax.scatter(
#         sub["log_pred"], sub["Accuracy"],
#         c=COLORS[fam], marker=MARKERS[fam],
#         label=fam, edgecolors="black", linewidths=0.4, s=70, alpha=0.85
#     )
#
# # annotate Pareto winners
# for algo in front:
#     row = df[df["Algorithm"] == algo].iloc[0]
#     ax.annotate(
#         algo.replace("Classifier", "").replace("Random Forest – ", "RF-"),
#         xy=(row["log_pred"], row["Accuracy"]),
#         xytext=(5, 5), textcoords="offset points", fontsize=8, weight="bold"
#     )
#
# ax.set_xlabel("log10(Prediction Time  [s])", labelpad=8, fontsize=12)
# ax.set_ylabel("Accuracy  (%)", fontsize=12)
# ax.set_title("Accuracy vs. Inference Latency  (Pareto-front Highlighted)", pad=12)
# ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
# ax.legend(title="Model Family")
# fig.tight_layout()
#
# # save
# OUT_DIR = "charts"
# os.makedirs(OUT_DIR, exist_ok=True)
# out_path = os.path.join(OUT_DIR, "pareto_accuracy_vs_latency.png")
# fig.savefig(out_path, dpi=300)
# print(f"Saved high-resolution figure →  {os.path.abspath(out_path)}")
#
#
#
#
#
#




# # ──────────────────────────────────────────────────────────────────────────────
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (triggers 3-D backend)
#
# # ──────────────────────────────────────────────────────────────────────────────
# # 1.  Where to save the figure
# # ──────────────────────────────────────────────────────────────────────────────
# SAVE_DIR = "charts"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ──────────────────────────────────────────────────────────────────────────────
# # 2.  Table of results
# #     ▸ Algorithm, Family, Predict-time [s], Accuracy [%], Model-size [KB]
# # ──────────────────────────────────────────────────────────────────────────────
# data = [
#     # Deep-Learning (DL)
#     ("ANN",                         "DL",      0.44, 85.67,      204),
#     ("Autoencoder",                 "DL",      0.72,  3.21,      373),
#     ("CNN",                         "DL",      0.54, 80.18,      534),
#     ("DNN",                         "DL",      0.35, 83.71,      178),
#     ("LSTM",                        "DL",      0.61, 83.93,      712),
#     ("RNN",                         "DL",      0.51, 83.79,      102),
#
#     # Machine-Learning (ML)
#     ("Adaboost",                    "ML",      0.13, 69.78,       71),
#     ("BalancedSVM",                 "ML",      3.05, 83.35,     2113),
#     ("DecisionTreeClassifier",      "ML",      0.01, 81.34,        7),
#     ("ExtraTreesClassifier",        "ML",      0.16, 88.35,    98237),
#     ("GaussianNB",                  "ML",      0.01, 79.20,        4),
#     ("GradientBoostingClassifier",  "ML",      0.03, 87.23,      516),
#     ("IsolationForestClass",        "ML",      0.07, 23.30,     1140),
#     ("KNN",                         "ML",      0.54, 86.56,     4727),
#     ("LightGBM",                    "ML",      0.03, 87.63,     1363),
#     ("LOFClass",                    "ML",      0.14, 21.07,     8034),
#     ("LogisticRegression",          "ML",      0.00, 81.43,        3),
#     ("MLP",                         "ML",      0.01, 86.43,      270),
#     ("NaiveBayes",                  "ML",      0.02, 76.96,        3),
#     ("Perceptron",                  "ML",      0.01, 79.69,        3),
#     ("Random Forest – Bagging",     "ML",      0.06, 87.99,   109751),
#     ("Random Forest – Weight",      "ML",      0.07, 88.30,    39075),
#     ("RandomForestClassifier",      "ML",      0.05, 86.65,     3878),
#     ("StackingClassifier",          "ML",      2.06, 84.33,     2269),
#     ("VotingClassifierModel",       "ML",      1.99, 81.65,     2119),
#     ("WeightedSVM",                 "ML",      2.23, 83.35,     2113),
#     ("XGBoostClassifier",           "ML",      0.02, 86.74,      649),
#
#     # Large-Language-Models (LLM)
#     ("Mistral-7B/zeroshots",        "LLM",  8243.00, 10.89, 14848000),
#     ("Mistral-7B/withSFT",          "LLM",  6712.00, 50.54, 14848000),
#     ("Llama-3.2-1B/zeroshots",      "LLM",    10.07,  0.00,     2540),
#     ("Llama-3.2-1B/withSFT",        "LLM",    24.08, 34.91,     2540),
#     ("Llama-3.2-3B/zeroshots",      "LLM",   144.00, 24.96,     2540),
#     ("Llama-3.2-3B/withSFT",        "LLM",   139.00, 25.00,     2540),
# ]
#
# df = pd.DataFrame(data, columns=[
#     "Algorithm", "Family", "PredictTime", "Accuracy", "SizeKB"
# ])
#
# # ──────────────────────────────────────────────────────────────────────────────
# # 3.  Feature transforms
# #     • log₁₀(Predict-time), log₁₀(Size)  to tame heavy tails
# # ──────────────────────────────────────────────────────────────────────────────
# df["log_pred"] = np.log10(df["PredictTime"].replace(0, 1e-6))
# df["log_size"] = np.log10(df["SizeKB"])
#
# # ──────────────────────────────────────────────────────────────────────────────
# # 4.  3-D scatter
# # ──────────────────────────────────────────────────────────────────────────────
# fig = plt.figure(figsize=(9, 7))
# ax  = fig.add_subplot(111, projection="3d")
#
# for fam in df["Family"].unique():
#     sub = df[df["Family"] == fam]
#     ax.scatter(
#         sub["log_pred"], sub["Accuracy"], sub["log_size"],
#         label=fam, s=45
#     )
#
# # Annotate each point with the algorithm name (small font)
# for _, row in df.iterrows():
#     ax.text(row["log_pred"], row["Accuracy"], row["log_size"],
#             row["Algorithm"], fontsize=6, ha="center", va="center")
#
# ax.set_xlabel("log10(Prediction Time  [s])")
# ax.set_ylabel("Accuracy  (%)")
# ax.set_zlabel("log10(Model Size  [KB])")
# ax.set_title("Speed–Accuracy–Size Trade-off  (DL · ML · LLM)")
# ax.grid(True, linestyle="--", linewidth=0.5)
# ax.legend()
# fig.tight_layout()
#
# out_path = os.path.join(SAVE_DIR, "3d_speed_accuracy_size.png")
# fig.savefig(out_path, dpi=300)
# plt.close(fig)
#
# print(f"Figure saved →  {os.path.abspath(out_path)}")

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3-D proj)
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
#
# # ──────────────────────────────────────────────────────────────
# # 1.  Output directory
# # ──────────────────────────────────────────────────────────────
# SAVE_DIR = "charts"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ──────────────────────────────────────────────────────────────
# # 2.  Results table
# #     (Algorithm, Category, Train-time, Predict-time,
# #      Accuracy, Precision, Recall, F1, SizeKB)
# # ──────────────────────────────────────────────────────────────
# data = [
#     # ─── Deep-Learning (DL) ───────────────────────────────────
#     ("ANN",                         "DL",  28.70,     0.44, 85.67, 85.38, 85.67, 85.44,      204),
#     ("Autoencoder",                 "DL", 117.80,     0.72,  3.21,  5.44,  0.76,  1.24,      373),
#     ("CNN",                         "DL",   8.81,     0.54, 80.18, 79.86, 80.18, 79.30,      534),
#     ("DNN",                         "DL",   9.96,     0.35, 83.71, 83.61, 83.71, 82.48,      178),
#     ("LSTM",                        "DL",  36.50,     0.61, 83.93, 86.19, 83.93, 81.82,      712),
#     ("RNN",                         "DL",  24.84,     0.51, 83.79, 83.82, 83.79, 82.58,      102),
#
#     # ─── Machine-Learning (ML) ────────────────────────────────
#     ("Adaboost",                    "ML",   2.64,     0.13, 69.78, 76.69, 69.78, 67.92,       71),
#     ("BalancedSVM",                 "ML",   8.58,     3.05, 83.35, 83.35, 83.35, 81.99,     2113),
#     ("DecisionTreeClassifier",      "ML",   0.08,     0.01, 81.34, 83.30, 81.34, 81.39,        7),
#     ("ExtraTreesClassifier",        "ML",   2.62,     0.16, 88.35, 88.17, 88.35, 88.15,    98237),
#     ("GaussianNB",                  "ML",   0.04,     0.01, 79.20, 83.41, 79.20, 75.09,        4),
#     ("GradientBoostingClassifier",  "ML",  14.53,     0.03, 87.23, 87.29, 87.23, 87.24,      516),
#     ("IsolationForestClass",        "ML",   0.48,     0.07, 23.30,  5.89, 18.64,  8.95,     1140),
#     ("KNN",                         "ML",   0.02,     0.54, 86.56, 86.35, 86.56, 86.40,     4727),
#     ("LightGBM",                    "ML",   0.75,     0.03, 87.63, 87.54, 87.63, 87.55,     1363),
#     ("LOFClass",                    "ML",   3.01,     0.14, 21.07,  4.69, 16.86,  7.34,     8034),
#     ("LogisticRegression",          "ML",   0.72,     0.00, 81.43, 81.20, 81.43, 80.06,        3),
#     ("MLP",                         "ML",  41.70,     0.01, 86.43, 87.65, 86.43, 86.61,      270),
#     ("NaiveBayes",                  "ML",   0.06,     0.02, 76.96, 81.28, 76.96, 73.22,        3),
#     ("Perceptron",                  "ML",   0.12,     0.01, 79.69, 80.20, 79.69, 77.54,        3),
#     ("Random Forest – Bagging",     "ML",   3.41,     0.06, 87.99, 87.81, 87.99, 87.77,   109751),
#     ("Random Forest – Weight",      "ML",   2.37,     0.07, 88.30, 88.12, 88.30, 88.12,    39075),
#     ("RandomForestClassifier",      "ML",   1.62,     0.05, 86.65, 86.72, 86.65, 86.63,     3878),
#     ("StackingClassifier",          "ML", 174.60,     2.06, 84.33, 84.14, 84.33, 83.33,     2269),
#     ("VotingClassifierModel",       "ML",  32.72,     1.99, 81.65, 81.53, 81.65, 80.17,     2119),
#     ("WeightedSVM",                 "ML",   8.17,     2.23, 83.35, 83.35, 83.35, 81.99,     2113),
#     ("XGBoostClassifier",           "ML",   1.09,     0.02, 86.74, 86.80, 86.74, 86.73,       649),
#
#     # ─── Large-Language-Models (LLM) ──────────────────────────
#     ("Mistral-7B/zeroshots",        "LLM",   0.00,  8243.0, 10.89, 10.83,  6.22,  6.24, 14848000),
#     ("Mistral-7B/withSFT",          "LLM", 4718.1,  6712.0, 50.54, 44.48, 50.54, 43.50, 14848000),
#     ("Llama-3.2-1B/zeroshots",      "LLM",   0.00,    10.07,  0.00, 24.87,  8.03, 24.87,     2540),
#     ("Llama-3.2-1B/withSFT",        "LLM", 714.00,    24.08, 34.91, 38.15, 34.91, 24.08,     2540),
#     ("Llama-3.2-3B/zeroshots",      "LLM",   0.00,   144.0, 24.96,  6.28, 24.96, 10.03,     2540),
#     ("Llama-3.2-3B/withSFT",        "LLM",     np.nan, 139.0, 25.00,  6.27, 25.00, 10.03,     2540),
# ]
#
# df = pd.DataFrame(
#     data,
#     columns=[
#         "Algorithm", "Category", "TrainTime", "PredictTime",
#         "Accuracy", "Precision", "Recall", "F1", "SizeKB"
#     ]
# )
#
# # ──────────────────────────────────────────────────────────────
# # 3.  Feature Engineering
# #     • log₁₀-transform heavy-tailed time & size columns
# #     • replace zeros→1e-6 to avoid −∞
# # ──────────────────────────────────────────────────────────────
# for col in ["TrainTime", "PredictTime", "SizeKB"]:
#     df[col] = np.log10(df[col].replace(0, 1e-6))
#
# df = df.fillna(df.median(numeric_only=True))
#
# X = df[[
#     "Accuracy", "Precision", "Recall", "F1",
#     "TrainTime", "PredictTime", "SizeKB"
# ]]
#
# # Standardise
# X_scaled = StandardScaler().fit_transform(X)
#
# # ──────────────────────────────────────────────────────────────
# # 4.  3-D t-SNE
# # ──────────────────────────────────────────────────────────────
# tsne3 = TSNE(
#     n_components=3,
#     perplexity=5,           # good for ≈30 points
#     init="pca",
#     learning_rate="auto",
#     random_state=42
# )
# emb3 = tsne3.fit_transform(X_scaled)
#
# # ──────────────────────────────────────────────────────────────
# # 5.  3-D scatter plot
# # ──────────────────────────────────────────────────────────────
# fig = plt.figure(figsize=(9, 7))
# ax = fig.add_subplot(111, projection="3d")
#
# for cat in df["Category"].unique():
#     idx = df["Category"] == cat
#     ax.scatter(
#         emb3[idx, 0], emb3[idx, 1], emb3[idx, 2],
#         label=cat, s=40
#     )
#
# # annotate each point
# for i, name in enumerate(df["Algorithm"]):
#     ax.text(
#         emb3[i, 0], emb3[i, 1], emb3[i, 2],
#         name, fontsize=5, ha="center", va="center"
#     )
#
# ax.set_xlabel("t-SNE Dim 1")
# ax.set_ylabel("t-SNE Dim 2")
# ax.set_zlabel("t-SNE Dim 3")
# ax.set_title("3-D t-SNE Projection of Algorithm-Performance Vectors")
# ax.grid(True, linestyle="--", linewidth=0.5)
# ax.legend()
# fig.tight_layout()
#
# out_path = os.path.join(SAVE_DIR, "tsne3d_algorithms.png")
# fig.savefig(out_path, dpi=300)
# plt.close(fig)
#
# print(f"3-D t-SNE figure saved to: {os.path.abspath(out_path)}")
#


#
# #!/usr/bin/env python
# # generate_iot_tsne.py
# #
# # Visualises how DL, ML and LLM algorithms group together in performance-space
# # using t-SNE on Accuracy, Precision, Recall, F1, and log-scaled time / size.
#
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 1.  Output folder
# # ───────────────────────────────────────────────────────────────────────────────
# SAVE_DIR = "charts"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 2.  Results table  (Algorithm, Category, Train-time, Predict-time, Accuracy,
# #                     Precision, Recall, F1, Size)
# # ───────────────────────────────────────────────────────────────────────────────
# data = [
#     # Algorithm, Category, Train(s), Pred(s), Acc, Prec, Rec, F1, Size(KB)
#     ("ANN",                         "DL",    28.70,     0.44, 85.67, 85.38, 85.67, 85.44,       204),
#     ("Autoencoder",                 "DL",   117.80,     0.72,  3.21,  5.44,  0.76,  1.24,       373),
#     ("CNN",                         "DL",     8.81,     0.54, 80.18, 79.86, 80.18, 79.30,       534),
#     ("DNN",                         "DL",     9.96,     0.35, 83.71, 83.61, 83.71, 82.48,       178),
#     ("LSTM",                        "DL",    36.50,     0.61, 83.93, 86.19, 83.93, 81.82,       712),
#     ("RNN",                         "DL",    24.84,     0.51, 83.79, 83.82, 83.79, 82.58,       102),
#
#     ("Adaboost",                    "ML",     2.64,     0.13, 69.78, 76.69, 69.78, 67.92,        71),
#     ("BalancedSVM",                 "ML",     8.58,     3.05, 83.35, 83.35, 83.35, 81.99,      2113),
#     ("DecisionTreeClassifier",      "ML",     0.08,     0.01, 81.34, 83.30, 81.34, 81.39,         7),
#     ("ExtraTreesClassifier",        "ML",     2.62,     0.16, 88.35, 88.17, 88.35, 88.15,     98237),
#     ("GaussianNB",                  "ML",     0.04,     0.01, 79.20, 83.41, 79.20, 75.09,         4),
#     ("GradientBoostingClassifier",  "ML",    14.53,     0.03, 87.23, 87.29, 87.23, 87.24,       516),
#     ("IsolationForestClass",        "ML",     0.48,     0.07, 23.30,  5.89, 18.64,  8.95,      1140),
#     ("KNN",                         "ML",     0.02,     0.54, 86.56, 86.35, 86.56, 86.40,      4727),
#     ("LightGBM",                    "ML",     0.75,     0.03, 87.63, 87.54, 87.63, 87.55,      1363),
#     ("LOFClass",                    "ML",     3.01,     0.14, 21.07,  4.69, 16.86,  7.34,      8034),
#     ("LogisticRegression",          "ML",     0.72,     0.00, 81.43, 81.20, 81.43, 80.06,         3),
#     ("MLP",                         "ML",    41.70,     0.01, 86.43, 87.65, 86.43, 86.61,       270),
#     ("NaiveBayes",                  "ML",     0.06,     0.02, 76.96, 81.28, 76.96, 73.22,         3),
#     ("Perceptron",                  "ML",     0.12,     0.01, 79.69, 80.20, 79.69, 77.54,         3),
#     ("Random Forest – Bagging",     "ML",     3.41,     0.06, 87.99, 87.81, 87.99, 87.77,    109751),
#     ("Random Forest – Weight",      "ML",     2.37,     0.07, 88.30, 88.12, 88.30, 88.12,     39075),
#     ("RandomForestClassifier",      "ML",     1.62,     0.05, 86.65, 86.72, 86.65, 86.63,      3878),
#     ("StackingClassifier",          "ML",   174.60,     2.06, 84.33, 84.14, 84.33, 83.33,      2269),
#     ("VotingClassifierModel",       "ML",    32.72,     1.99, 81.65, 81.53, 81.65, 80.17,      2119),
#     ("WeightedSVM",                 "ML",     8.17,     2.23, 83.35, 83.35, 83.35, 81.99,      2113),
#     ("XGBoostClassifier",           "ML",     1.09,     0.02, 86.74, 86.80, 86.74, 86.73,       649),
#
#     ("Mistral-7B/zeroshots",        "LLM",    0.00,  8243.00, 10.89, 10.83,  6.22,  6.24, 14848000),
#     ("Mistral-7B/withSFT",          "LLM", 4718.10,  6712.00, 50.54, 44.48, 50.54, 43.50, 14848000),
#     ("Llama-3.2-1B/zeroshots",      "LLM",    0.00,    10.07,  0.00, 24.87,  8.03, 24.87,      2540),
#     ("Llama-3.2-1B/withSFT",        "LLM",  714.00,    24.08, 34.91, 38.15, 34.91, 24.08,      2540),
#     ("Llama-3.2-3B/zeroshots",      "LLM",    0.00,   144.00, 24.96,  6.28, 24.96, 10.03,      2540),
#     ("Llama-3.2-3B/withSFT",        "LLM",      np.nan, 139.00, 25.00,  6.27, 25.00, 10.03,      2540),
# ]
#
# df = pd.DataFrame(
#     data,
#     columns=[
#         "Algorithm", "Category", "TrainTime", "PredictTime",
#         "Accuracy", "Precision", "Recall", "F1", "SizeKB"
#     ]
# )
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 3.  Feature matrix for t-SNE
# #     • Log-transform heavy-tailed features (time, size) then standardise.
# # ───────────────────────────────────────────────────────────────────────────────
# feat_df = df.copy()
# for col in ["TrainTime", "PredictTime", "SizeKB"]:
#     # avoid log10(0) → -inf by replacing 0 with 1e-6
#     feat_df[col] = np.log10(feat_df[col].replace(0, 1e-6))
#
# # Fill any NaNs (e.g., “N/A” training time) with column medians
# feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
#
# # Features for t-SNE
# X = feat_df[[
#     "Accuracy", "Precision", "Recall", "F1",
#     "TrainTime", "PredictTime", "SizeKB"
# ]]
#
# # Standardise
# X_scaled = StandardScaler().fit_transform(X)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 4.  t-SNE (perplexity 5 – good for ≈30 points)
# # ───────────────────────────────────────────────────────────────────────────────
# tsne = TSNE(
#     n_components=2,
#     perplexity=5,
#     learning_rate="auto",
#     init="pca",
#     random_state=42
# )
# emb = tsne.fit_transform(X_scaled)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 5.  Scatter plot
# # ───────────────────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8, 6))
#
# for category in df["Category"].unique():
#     idx = df["Category"] == category
#     ax.scatter(
#         emb[idx, 0], emb[idx, 1],
#         label=category, alpha=0.9, edgecolors="none"
#     )
#
# # Annotate each point with algorithm name (small font)
# for i, name in enumerate(df["Algorithm"]):
#     ax.text(emb[i, 0], emb[i, 1], name, fontsize=6, ha="center", va="center")
#
# ax.set_xlabel("t-SNE Dim 1")
# ax.set_ylabel("t-SNE Dim 2")
# ax.set_title("t-SNE Projection of Algorithm-Performance Vectors")
# ax.grid(True, linestyle="--", linewidth=0.5)
# ax.legend()
# fig.tight_layout()
#
# out_path = os.path.join(SAVE_DIR, "tsne_algorithms.png")
# fig.savefig(out_path, dpi=300)
# plt.close(fig)
#
# print(f"t-SNE figure saved to:  {os.path.abspath(out_path)}")



# #!/usr/bin/env python
# # generate_iot_tsne.py
# #
# # Visualises how DL, ML and LLM algorithms group together in performance-space
# # using t-SNE on Accuracy, Precision, Recall, F1, and log-scaled time / size.
#
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 1.  Output folder
# # ───────────────────────────────────────────────────────────────────────────────
# SAVE_DIR = "charts"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 2.  Results table  (Algorithm, Category, Train-time, Predict-time, Accuracy,
# #                     Precision, Recall, F1, Size)
# # ───────────────────────────────────────────────────────────────────────────────
# data = [
#     # Algorithm, Category, Train(s), Pred(s), Acc, Prec, Rec, F1, Size(KB)
#     ("ANN",                         "DL",    28.70,     0.44, 85.67, 85.38, 85.67, 85.44,       204),
#     ("Autoencoder",                 "DL",   117.80,     0.72,  3.21,  5.44,  0.76,  1.24,       373),
#     ("CNN",                         "DL",     8.81,     0.54, 80.18, 79.86, 80.18, 79.30,       534),
#     ("DNN",                         "DL",     9.96,     0.35, 83.71, 83.61, 83.71, 82.48,       178),
#     ("LSTM",                        "DL",    36.50,     0.61, 83.93, 86.19, 83.93, 81.82,       712),
#     ("RNN",                         "DL",    24.84,     0.51, 83.79, 83.82, 83.79, 82.58,       102),
#
#     ("Adaboost",                    "ML",     2.64,     0.13, 69.78, 76.69, 69.78, 67.92,        71),
#     ("BalancedSVM",                 "ML",     8.58,     3.05, 83.35, 83.35, 83.35, 81.99,      2113),
#     ("DecisionTreeClassifier",      "ML",     0.08,     0.01, 81.34, 83.30, 81.34, 81.39,         7),
#     ("ExtraTreesClassifier",        "ML",     2.62,     0.16, 88.35, 88.17, 88.35, 88.15,     98237),
#     ("GaussianNB",                  "ML",     0.04,     0.01, 79.20, 83.41, 79.20, 75.09,         4),
#     ("GradientBoostingClassifier",  "ML",    14.53,     0.03, 87.23, 87.29, 87.23, 87.24,       516),
#     ("IsolationForestClass",        "ML",     0.48,     0.07, 23.30,  5.89, 18.64,  8.95,      1140),
#     ("KNN",                         "ML",     0.02,     0.54, 86.56, 86.35, 86.56, 86.40,      4727),
#     ("LightGBM",                    "ML",     0.75,     0.03, 87.63, 87.54, 87.63, 87.55,      1363),
#     ("LOFClass",                    "ML",     3.01,     0.14, 21.07,  4.69, 16.86,  7.34,      8034),
#     ("LogisticRegression",          "ML",     0.72,     0.00, 81.43, 81.20, 81.43, 80.06,         3),
#     ("MLP",                         "ML",    41.70,     0.01, 86.43, 87.65, 86.43, 86.61,       270),
#     ("NaiveBayes",                  "ML",     0.06,     0.02, 76.96, 81.28, 76.96, 73.22,         3),
#     ("Perceptron",                  "ML",     0.12,     0.01, 79.69, 80.20, 79.69, 77.54,         3),
#     ("Random Forest – Bagging",     "ML",     3.41,     0.06, 87.99, 87.81, 87.99, 87.77,    109751),
#     ("Random Forest – Weight",      "ML",     2.37,     0.07, 88.30, 88.12, 88.30, 88.12,     39075),
#     ("RandomForestClassifier",      "ML",     1.62,     0.05, 86.65, 86.72, 86.65, 86.63,      3878),
#     ("StackingClassifier",          "ML",   174.60,     2.06, 84.33, 84.14, 84.33, 83.33,      2269),
#     ("VotingClassifierModel",       "ML",    32.72,     1.99, 81.65, 81.53, 81.65, 80.17,      2119),
#     ("WeightedSVM",                 "ML",     8.17,     2.23, 83.35, 83.35, 83.35, 81.99,      2113),
#     ("XGBoostClassifier",           "ML",     1.09,     0.02, 86.74, 86.80, 86.74, 86.73,       649),
#
#     ("Mistral-7B/zeroshots",        "LLM",    0.00,  8243.00, 10.89, 10.83,  6.22,  6.24, 14848000),
#     ("Mistral-7B/withSFT",          "LLM", 4718.10,  6712.00, 50.54, 44.48, 50.54, 43.50, 14848000),
#     ("Llama-3.2-1B/zeroshots",      "LLM",    0.00,    10.07,  0.00, 24.87,  8.03, 24.87,      2540),
#     ("Llama-3.2-1B/withSFT",        "LLM",  714.00,    24.08, 34.91, 38.15, 34.91, 24.08,      2540),
#     ("Llama-3.2-3B/zeroshots",      "LLM",    0.00,   144.00, 24.96,  6.28, 24.96, 10.03,      2540),
#     ("Llama-3.2-3B/withSFT",        "LLM",      np.nan, 139.00, 25.00,  6.27, 25.00, 10.03,      2540),
# ]
#
# df = pd.DataFrame(
#     data,
#     columns=[
#         "Algorithm", "Category", "TrainTime", "PredictTime",
#         "Accuracy", "Precision", "Recall", "F1", "SizeKB"
#     ]
# )
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 3.  Feature matrix for t-SNE
# #     • Log-transform heavy-tailed features (time, size) then standardise.
# # ───────────────────────────────────────────────────────────────────────────────
# feat_df = df.copy()
# for col in ["TrainTime", "PredictTime", "SizeKB"]:
#     # avoid log10(0) → -inf by replacing 0 with 1e-6
#     feat_df[col] = np.log10(feat_df[col].replace(0, 1e-6))
#
# # Fill any NaNs (e.g., “N/A” training time) with column medians
# feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
#
# # Features for t-SNE
# X = feat_df[[
#     "Accuracy", "Precision", "Recall", "F1",
#     "TrainTime", "PredictTime", "SizeKB"
# ]]
#
# # Standardise
# X_scaled = StandardScaler().fit_transform(X)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 4.  t-SNE (perplexity 5 – good for ≈30 points)
# # ───────────────────────────────────────────────────────────────────────────────
# tsne = TSNE(
#     n_components=2,
#     perplexity=5,
#     learning_rate="auto",
#     init="pca",
#     random_state=42
# )
# emb = tsne.fit_transform(X_scaled)
#
# # ───────────────────────────────────────────────────────────────────────────────
# # 5.  Scatter plot
# # ───────────────────────────────────────────────────────────────────────────────
# fig, ax = plt.subplots(figsize=(8, 6))
#
# for category in df["Category"].unique():
#     idx = df["Category"] == category
#     ax.scatter(
#         emb[idx, 0], emb[idx, 1],
#         label=category, alpha=0.9, edgecolors="none"
#     )
#
# # Annotate each point with algorithm name (small font)
# for i, name in enumerate(df["Algorithm"]):
#     ax.text(emb[i, 0], emb[i, 1], name, fontsize=6, ha="center", va="center")
#
# ax.set_xlabel("t-SNE Dim 1")
# ax.set_ylabel("t-SNE Dim 2")
# ax.set_title("t-SNE Projection of Algorithm-Performance Vectors")
# ax.grid(True, linestyle="--", linewidth=0.5)
# ax.legend()
# fig.tight_layout()
#
# out_path = os.path.join(SAVE_DIR, "tsne_algorithms.png")
# fig.savefig(out_path, dpi=300)
# plt.close(fig)
#
# print(f"t-SNE figure saved to:  {os.path.abspath(out_path)}")
