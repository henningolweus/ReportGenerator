from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def render_min_demo_line(out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    x = list(range(12))
    y_port = [100, 101, 103, 102, 105, 108, 110, 109, 111, 115, 114, 118]
    y_bench = [100, 100, 101, 101, 103, 104, 105, 104, 106, 108, 109, 111]

    plt.figure(figsize=(9, 4.5), dpi=200)
    plt.plot(x, y_port, label="Portfolio", linewidth=2)
    plt.plot(x, y_bench, label="Benchmark", linewidth=2)
    plt.title("Performance (Indexed = 100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def render_monthly_bars(out_path: str, months: int = 24) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.arange(months)
    port = np.random.normal(loc=0.01, scale=0.05, size=months)
    bench = port - np.random.normal(loc=0.003, scale=0.02, size=months)

    width = 0.4
    plt.figure(figsize=(10, 4), dpi=200)
    plt.bar(rng - width/2, port * 100, width=width, label="Portfolio")
    plt.bar(rng + width/2, bench * 100, width=width, label="Benchmark")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylabel("%")
    plt.title("Monthly returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def render_pie(out_path: str, labels, weights) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4), dpi=200)
    plt.pie(weights, labels=labels, autopct='%1.0f%%', startangle=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


