# CLAUDE.md — Computer Vision Group

## Language & Style

- **All** comments, docstrings, README files, and commit messages must be written in **English**.
- Each Python file starts with a module-level docstring describing what the script does.
- Use `main()` as the entry point, guarded by `if __name__ == "__main__":`.

## Project Conventions

- Every project folder contains:
  - A `README.md` explaining the project, how to run it, and how it works.
  - A `requirements.txt` listing dependencies with minimum versions.
- Shared reusable code belongs in the `common/` package at the repo root.
- Import from `common/` by adding the repo root to `sys.path`:
  ```python
  import os, sys
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  from common import WebcamManager
  ```

## Auto-Commit & Push Skill

**CRITICAL — After EVERY batch of changes to project files (code, README, config), you MUST:**

1. **Review** what changed: `git status` and `git diff`.
2. **Commit** with a concise, descriptive English message summarizing the changes.
   - Use the format: `<type>: <description>` where type is one of `feat`, `fix`, `refactor`, `docs`, `chore`.
   - Example: `feat: add Hand Clap Counter project with MediaPipe clap detection`
3. **Push** to the remote: `git push`.
4. **Update the root `README.md`** if any of the following changed:
   - A new project folder was added or removed.
   - A project's description or tech stack changed.
   - The repository structure changed.
   - New shared utilities were added to `common/`.
   - Update the **Projects** table and **Shared Utilities** table accordingly.

Never leave uncommitted changes sitting in the working tree after completing a task.
