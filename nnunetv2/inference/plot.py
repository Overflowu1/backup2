import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

# ---------- helper: pick first installed font ----------
def first_available(candidates):
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

# ---------- auto font selection ----------
EN_FONT = first_available([
    "Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"
])

CN_FONT = first_available([
    "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "SimHei", "Microsoft YaHei", "PingFang SC", "Arial Unicode MS"
])

print("EN_FONT =", EN_FONT)
print("CN_FONT =", CN_FONT)

plt.rcParams["font.family"] = [
    EN_FONT or "DejaVu Serif",
    CN_FONT or "DejaVu Sans",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# ---------- data ----------
# ---------- data ----------
# ---------- data ----------
methods = [
    "nnU-Net", "UNETR", "Swin-UNETR", "SegResNet",
    "STUNet", "U-Mamba",
    "SMFE-PFNet",
    "MSAC-PFNet",
    "DDPF-Net\n(本章)"
]

dsc = np.array([0.8217, 0.3879, 0.6990, 0.8549, 0.8609, 0.8557, 0.8904, 0.9113, 0.9273])
hd95 = np.array([78.2466, 182.1093, 116.1551, 171.6561, 33.2352, 32.0350, 29.9638, 28.4302, 27.6287])
acc = np.array([0.9996, 0.9981, 0.9995, 0.9997, 0.9997, 0.9997, 0.9998, 0.9999, 0.9999])
recall = np.array([0.7646, 0.6085, 0.6597, 0.8305, 0.7994, 0.8014, 0.8484, 0.9325, 0.9500])
miou = np.array([0.7054, 0.2786, 0.5564, 0.7517, 0.7572, 0.7529, 0.8070, 0.8479, 0.8675])
# ---------- plot ----------
def plot_3panel(
    save_path="three_panel.png",
    show_model_names_in_scatter=True,
    add_value_labels=False,
    bottom_gap=0.65
):
    fig = plt.figure(figsize=(12.6, 7.2), dpi=300)

    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 1.25],
        wspace=0.25,
        hspace=bottom_gap
    )

    x = np.arange(len(methods))
    w = 0.36

    # =========================
    # (a) DSC / MIoU
    # =========================
    ax_a = fig.add_subplot(gs[0, 0])

    ax_a.bar(x - w / 2, dsc, width=w, label="DSC")
    ax_a.bar(x + w / 2, miou, width=w, label="MIoU")

    ax_a.set_ylim(0, 1.0)
    ax_a.set_yticks(np.arange(0, 1.01, 0.2))

    ax_a.set_ylabel("得分")
    ax_a.set_title("(a) 整体分割性能（DSC / MIoU）", pad=10)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(methods, rotation=18, ha="right")

    ax_a.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        frameon=True,
        fontsize=9,
        borderpad=0.3,
        handlelength=1.6,
        columnspacing=1.0
    )

    if add_value_labels:
        for i in range(len(x)):
            ax_a.text(
                x[i] - w / 2,
                dsc[i] + 0.02,
                f"{dsc[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )
            ax_a.text(
                x[i] + w / 2,
                miou[i] + 0.02,
                f"{miou[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # =========================
    # (b) HD95
    # =========================
    ax_b = fig.add_subplot(gs[0, 1])

    ax_b.bar(x, hd95)

    ax_b.set_ylabel("HD95（mm，越小越好）")
    ax_b.set_title("(b) 边界误差对比（HD95）", pad=10)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(methods, rotation=18, ha="right")

    ax_b.set_ylim(0, max(200, float(hd95.max()) * 1.1))

    if add_value_labels:
        for i in range(len(x)):
            ax_b.text(
                x[i],
                hd95[i] + 3,
                f"{hd95[i]:.4f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # =========================
    # (c) Recall–Accuracy
    # =========================
    ax_c = fig.add_subplot(gs[1, :])

    ax_c.scatter(recall, acc, s=35)

    ax_c.set_title("(c) Recall–Accuracy 分布", pad=8)
    ax_c.set_xlabel("召回率（Recall）")
    ax_c.set_ylabel("准确率（Accuracy）")

    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    ax_c.set_xlim(0.60, 0.98)
    ax_c.set_ylim(0.9980, 1.0000)

    if show_model_names_in_scatter:
        label_offsets = {
            "STUNet": (-34, 10),
            "U-Mamba": (8, -14),
            "SegResNet": (-18, 10),

            "SMFE-PFNet": (-20, 10),
            "MSAC-PFNet": (-28, -14),
            "DDPF-Net\n(本章)": (8, 8),

            "nnU-Net": (8, 8),
            "UNETR": (8, 8),
            "Swin-UNETR": (8, 8),
        }

        for i, m in enumerate(methods):
            ax_c.annotate(
                m,
                (recall[i], acc[i]),
                textcoords="offset points",
                xytext=label_offsets.get(m, (6, 6)),
                ha="left",
                va="center",
                fontsize=9
            )
    # =========================
    # style
    # =========================
    for ax in (ax_a, ax_b, ax_c):
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    # 自动创建保存目录
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", save_path)


if __name__ == "__main__":
    plot_3panel(
        save_path="/mnt/data/DATA/zjyData/picture/five_1.png",
        show_model_names_in_scatter=True,
        add_value_labels=False,
        bottom_gap=0.65
    )