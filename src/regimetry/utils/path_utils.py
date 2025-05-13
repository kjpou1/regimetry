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

def ensure_dir_exists(path: str, create: bool = True, verbose: bool = True, logger=None):
    if os.path.exists(path):
        return
    if create:
        os.makedirs(path, exist_ok=True)
        if verbose:
            msg = f"[PathUtils] Created directory: {path}"
            logger.info(msg) if logger else print(msg)
    else:
        raise FileNotFoundError(f"[PathUtils] Required directory does not exist: {path}")
    
def ensure_all_dirs_exist(paths: list[str], create: bool = True, verbose: bool = True, logger=None):
    """
    Ensure that all directories in a list exist.

    Args:
        paths (list[str]): List of directory paths to validate or create.
        create (bool): Whether to create directories if they don't exist.
        verbose (bool): Whether to log actions.
    """
    for path in paths:
        ensure_dir_exists(path, create=create, verbose=verbose, logger=logger)
    