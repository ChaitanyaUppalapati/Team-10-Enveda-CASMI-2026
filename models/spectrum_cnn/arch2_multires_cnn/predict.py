"""multires_cnn predict entry point. Owner: Member 2."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Direct scripts work from any current directory in this source checkout.
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parent))

from common.cli import predict_main  # noqa: E402
from common.runner import predict  # noqa: E402, F401


def main(argv=None):
    return predict_main(HERE, "multires_cnn", argv)


if __name__ == "__main__":
    main()
