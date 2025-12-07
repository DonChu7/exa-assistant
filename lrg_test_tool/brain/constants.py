# lrg_test_tool/brain/constants.py

KNOWN_SETUPS = [
    "xblockini",
    "tsagexastackup",
    "xrdbmsini",
    "srdbmsini",
    "tsaginit",
    "tsagnini",
]


def normalize_setup(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    # treat foo.tsc as foo
    if s.endswith(".tsc"):
        s = s[:-4]
    return s