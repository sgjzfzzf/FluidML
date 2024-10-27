import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Draw the graph of the analysis result."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path of the analysis json.",
    )
    parser.add_argument(
        "--kernels",
        nargs="*",
        type=str,
        help="The choosen kernel name.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bar", "pie"],
        required=True,
        help="The working mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="The path of the output image.",
    )
    args: argparse.Namespace = parser.parse_args()
    table: pd.DataFrame = pd.read_json(args.input)
    table.index = ["unoptimized", "optimized"]
    if args.kernels:
        table = table[args.kernels]
    table = table.T
    if args.mode == "bar":
        ax = table.plot.bar(figsize=(18, 12))
        ax.set_xlabel("Kernel")
        ax.set_ylabel("Time (ns)")
    elif args.mode == "pie":
        ax = table.plot.pie(
            subplots=True, autopct="%1.1f%%", legend=False, figsize=(18, 12)
        )
    plt.xticks(rotation=15)
    plt.savefig(args.output)
