import os


def mkdir(directory: str) -> None:
    """Create directory and all missing parents if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
