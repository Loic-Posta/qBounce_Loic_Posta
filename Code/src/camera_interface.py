"""Lower-case compatibility wrapper for CameraInterface.py.

Some modules use Python's conventional lower-case import name
`camera_interface`, while the implementation file in this project is
currently named `CameraInterface.py`. Keeping this tiny shim makes both
imports work on case-sensitive systems.
"""

from CameraInterface import *  # noqa: F401,F403
