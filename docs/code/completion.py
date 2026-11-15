from collections.abc import Mapping
from pathlib import Path

import aiohttp_mcp_server as mcp


def prompt_completions() -> None:
    LANGUAGES = ("python", "rust", "typescript")
    FRAMEWORKS = {"python": ("aiohttp", "django", "flask"), "rust": ("actix", "axum")}

    async def code_review(language: str, framework: str = "none") -> str:
        """Review code written in the given language."""
        return f"Please review this {language} ({framework}) code."

    async def complete_language(value: str, arguments: Mapping[str, str]) -> list[str]:
        return [lang for lang in LANGUAGES if lang.startswith(value)]

    async def complete_framework(value: str, arguments: Mapping[str, str]) -> list[str]:
        # Narrowed by the already-resolved sibling argument.
        frameworks = FRAMEWORKS.get(arguments.get("language", ""), ())
        return [fw for fw in frameworks if fw.startswith(value)]

    prompt = mcp.Prompt(
        code_review,
        completions={"language": complete_language, "framework": complete_framework},
    )


def template_completions() -> None:
    PROJECT = Path("/srv/project")

    async def project_file(path: str) -> str:
        """A file from the project directory."""
        return (PROJECT / path).read_text()

    async def complete_path(value: str, arguments: Mapping[str, str]) -> mcp.Completion:
        matches = sorted(p.name for p in PROJECT.glob(f"{value}*"))
        return mcp.Completion(matches[:10], total=len(matches), has_more=len(matches) > 10)

    resource = mcp.Resource(
        project_file, "file:///{+path}", completions={"path": complete_path}
    )
