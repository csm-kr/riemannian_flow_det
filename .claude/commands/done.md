Mark a TODO item as complete in `docs/plan_todo.md` and show the next pending item.

## Instructions

The user will provide an item ID as argument: `/done <item_id>`

1. Read `docs/plan_todo.md`.
2. Find the line matching `<item_id>` (e.g., `1.1`, `2`, `EXP-1`).
3. Change `[ ]` to `[x]` on that line.
4. Save the file.
5. Print:
   - The line just marked done.
   - The next `[ ]` item (the next pending TODO).

If the item ID is not found, list all available item IDs so the user can pick the right one.

## Example

```
/done 1.1
/done EXP-2
```
