"""multires_cnn train entry point. Owner: Member 2."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Direct scripts work from any current directory in this source checkout.
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from common.cli import train_main  # noqa: E402


def main(argv=None):
    return train_main(HERE, "multires_cnn", argv)


if __name__ == "__main__":
    main()
