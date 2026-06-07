# GUI assets

Drop the neurokinematics app icon in this folder. `app_icon()` in
[`app.py`](../app.py) picks up the first matching file, in this order:

1. `neurokinematics.ico`
2. `neurokinematics.png`
3. `icon.ico`
4. `icon.png`
5. `logo.png`

Notes:

- **Windows taskbar / `.exe`:** use a multi-resolution `.ico`
  (16, 32, 48, 256 px) named `neurokinematics.ico` for the crispest result.
- **Everywhere else (window title bar, macOS, Linux):** a square `.png`
  (256×256 or 512×512) named `neurokinematics.png` works fine.
- You can ship both — the `.ico` is preferred when present.

The icon is applied to the application and the main window automatically on
launch, so just add the file here and restart the app — no code changes needed.
