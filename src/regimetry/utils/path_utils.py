import os

def get_project_root(marker_filename="pyproject.toml", fallback_levels=3):
    """
    Walk up the directory tree from this file to locate the project root.
    The presence of `marker_filename` (e.g., pyproject.toml or .git) is used
    to determine the root. If not found, it falls back by going up `fallback_levels`.
    """
    path = os.path.abspath(__file__)
    while True:
        parent = os.path.dirname(path)
        if os.path.isfile(os.path.join(parent, marker_filename)) or parent == path:
            return parent
        path = parent

    # fallback
    for _ in range(fallback_levels):
        path = os.path.dirname(path)
    return path
