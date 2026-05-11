"""Allow running Zynthé via ``python -m zynthe``."""

import sys

from zynthe.cli import main

sys.exit(main())
