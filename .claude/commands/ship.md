Test → git add → git commit → git push a file (or files).

## Instructions

The user will provide a file path as argument: `/ship <file_path>`

Run the following steps **in order**, stopping immediately on any failure:

1. **Test** — run `python3 <file_path>`. If the file has no `__main__` block, skip this step and note it.
   - If exit code != 0: print the error and **stop**. Do NOT commit.
2. **Stage** — run `git add <file_path>`.
3. **Commit** — write a concise conventional commit message based on what changed:
   - Format: `<type>: <description>` (e.g., `feat: add VOC dataset wrapper`)
   - Run: `git commit -m "<message>"`
4. **Push** — run `git push origin <current-branch>`.
5. Report the final commit hash and push result.

## Rules

- Never use `--no-verify`.
- Never push directly to `main`.
- If the working tree is dirty with other files, stage only the specified file(s).

## Example

```
/ship dataset/voc.py
/ship model/backbone.py
```
