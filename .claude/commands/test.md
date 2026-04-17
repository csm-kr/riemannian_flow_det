Run the `__main__` block of the specified file and verify it passes.

## Instructions

The user will provide a file path as an argument: `/test <file_path>`

1. Read the file to confirm it has an `if __name__ == "__main__":` block.
2. Run `python3 <file_path>` and capture stdout/stderr.
3. Report:
   - PASS if exit code is 0 (show any printed output)
   - FAIL if exit code is non-zero (show the full error traceback)
4. Do NOT modify the file unless the user asks.

## Example

```
/test dataset/voc.py
/test model/backbone.py
```
