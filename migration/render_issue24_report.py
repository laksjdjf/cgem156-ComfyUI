"""Render issue #24 V3 migration report from structured registry."""

from __future__ import annotations

from issue24_v3_registry import (
    ACCEPTANCE_CHECKLIST,
    NODE_MIGRATION_RECORDS,
    P0_PRIORITY_TOPICS,
    P1_PRIORITY_TOPICS,
    P2_PRIORITY_TOPICS,
    P3_PRIORITY_TOPICS,
    count_by_difficulty,
)


def _print_section(title: str) -> None:
    print(f"\n## {title}")


def _print_bullets(items: tuple[str, ...]) -> None:
    for item in items:
        print(f"- {item}")


def main() -> None:
    counts = count_by_difficulty()

    _print_section("全体サマリー")
    print(f"- Node total: {len(NODE_MIGRATION_RECORDS)}")
    print(f"- Difficulty A/B/C: {counts['A']}/{counts['B']}/{counts['C']}")
    print("- Registration entrypoint: /tmp/workspace/laksjdjf/cgem156-ComfyUI/__init__.py")

    _print_section("ノード一覧")
    print("| Node | File | Difficulty | V1 Components | Migration Notes |")
    print("|---|---|---|---|---|")
    for r in NODE_MIGRATION_RECORDS:
        components = ", ".join(r.v1_components)
        notes = "; ".join(r.migration_notes) if r.migration_notes else "-"
        print(f"| {r.node_name} | {r.file_path} | {r.difficulty} | {components} | {notes} |")

    _print_section("優先度付き移行プラン")
    print("### P0")
    _print_bullets(P0_PRIORITY_TOPICS)
    print("### P1")
    _print_bullets(P1_PRIORITY_TOPICS)
    print("### P2")
    _print_bullets(P2_PRIORITY_TOPICS)
    print("### P3")
    _print_bullets(P3_PRIORITY_TOPICS)

    _print_section("受け入れ基準")
    _print_bullets(ACCEPTANCE_CHECKLIST)


if __name__ == "__main__":
    main()
