cd %~dp0
pyinstaller --noconfirm --onefile --console --hidden-import "pydantic" --hidden-import "pydantic-core" --hidden-import "pydantic.deprecated.decorator"  "src\cli.py"