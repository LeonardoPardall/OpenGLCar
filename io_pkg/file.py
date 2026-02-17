import os


def get_content_of_file_project(file_path: str) -> str:
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, file_path),
        os.path.join(base, '..', file_path),
        file_path,
    ]

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            with open(candidate, 'r', encoding='utf-8') as f:
                return f.read()

    raise FileNotFoundError(file_path)
