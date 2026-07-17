"""Hierarchical site maps rendered as plain HTML (no JavaScript).

Crawled pages keep parent/child links, so the map is a real tree:
an index of crawled sites, a collapsible tree per site, and a per-page
view with breadcrumbs, siblings, and children for navigation.
"""
from __future__ import annotations

import html as html_lib

from webdocs.database import Database, PageRecord

_STYLE = """
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; }
  a { color: #0969da; text-decoration: none; } a:hover { text-decoration: underline; }
  ul { line-height: 1.7; }
  .crumbs { color: #57606a; font-size: 0.9rem; margin-bottom: 1rem; }
  .card { border: 1px solid #d0d7de; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; color: #57606a; }
  pre { white-space: pre-wrap; background: #f6f8fa; padding: 1rem; border-radius: 8px; }
</style>
"""


def _esc(value: str) -> str:
    return html_lib.escape(value, quote=True)


def _page_link(page: PageRecord) -> str:
    return f'<a href="/map/page/{_esc(page.id)}">{_esc(page.title or page.url)}</a>'


def render_index(db: Database) -> str:
    roots = db.root_pages()
    items = "".join(
        f"<li>{_page_link(root)} <small>({_esc(root.domain)})</small> "
        f'&mdash; <a href="/map/site/{_esc(root.id)}">tree</a></li>'
        for root in roots
    )
    body = f"<ul>{items}</ul>" if items else "<p>No sites crawled yet. POST /fetch_url to start.</p>"
    return f"<!doctype html><title>Crawled sites</title>{_STYLE}<h1>Crawled sites</h1>{body}"


def _render_subtree(db: Database, page: PageRecord) -> str:
    children = db.children_of(page.id)
    inner = "".join(f"<li>{_render_subtree(db, child)}</li>" for child in children)
    tree = f"<ul>{inner}</ul>" if inner else ""
    return f"{_page_link(page)}{tree}"


def render_site_tree(db: Database, root_id: str) -> str | None:
    root = db.get_page(root_id)
    if root is None:
        return None
    return (
        f"<!doctype html><title>Site map: {_esc(root.domain)}</title>{_STYLE}"
        f'<p class="crumbs"><a href="/map">All sites</a> / {_esc(root.domain)}</p>'
        f"<h1>Site map &mdash; {_esc(root.title or root.url)}</h1><ul><li>{_render_subtree(db, root)}</li></ul>"
    )


def _breadcrumbs(db: Database, page: PageRecord) -> list[PageRecord]:
    trail: list[PageRecord] = []
    current: PageRecord | None = page
    while current is not None and len(trail) < 50:
        trail.append(current)
        current = db.get_page(current.parent_id) if current.parent_id else None
    return list(reversed(trail))


def render_page(db: Database, page_id: str) -> str | None:
    page = db.get_page(page_id)
    if page is None:
        return None
    trail = _breadcrumbs(db, page)
    crumbs = " / ".join(_page_link(p) if p.id != page.id else _esc(p.title or p.url) for p in trail)

    siblings = [s for s in (db.children_of(page.parent_id) if page.parent_id else []) if s.id != page.id]
    children = db.children_of(page.id)

    def _section(title: str, pages: list[PageRecord]) -> str:
        if not pages:
            return ""
        items = "".join(f"<li>{_page_link(p)}</li>" for p in pages)
        return f'<div class="card"><h2>{title}</h2><ul>{items}</ul></div>'

    excerpt = _esc(page.text[:1500]) + ("&hellip;" if len(page.text) > 1500 else "")
    return (
        f"<!doctype html><title>{_esc(page.title or page.url)}</title>{_STYLE}"
        f'<p class="crumbs"><a href="/map">All sites</a> / {crumbs}</p>'
        f"<h1>{_esc(page.title or page.url)}</h1>"
        f'<p><a href="{_esc(page.url)}">{_esc(page.url)}</a> &middot; '
        f'<a href="/map/page/{_esc(page.id)}/raw">raw text</a></p>'
        f"<pre>{excerpt}</pre>"
        f"{_section('Children', children)}{_section('Siblings', siblings)}"
    )
