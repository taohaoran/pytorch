#!/usr/bin/env python3
"""build-docs-html.py — 部署时把目录下 Markdown 默认渲染为 HTML 文档页，并强制生成导览 index.md。

解决痛点：GitHub Pages 纯静态托管 .md 时浏览器直接显示源码、阅读体验差；站点缺少能覆盖全部文档的导览页。
本脚本在部署前统一完成两件事（默认开启）：
  1) md 默认转 html：递归渲染部署目录下所有 .md → 同目录同名 .html（.md 保留为源），
     文档间相对链接 .md → .html 自动改写；指向 .html 的图片引用改写为「打开交互式图」按钮。
  2) 导览文件（强制）：在部署目录根生成 index.md，链接全部 .md 与全部 .html
     （md 条目指向其渲染后的 .html，独立 html 页直接链接），并一并渲染为 index.html 作为站点根入口；
     保证任何文档都可从导览到达，无孤儿页面。

用法:
  python3 build-docs-html.py <site_dir> [--site-name "站点名"] [--group-label "core=核心|ai=智能"]
依赖: pip install markdown

行为:
  - 生成导览 index.md（按顶层目录分组，链接全部 md 与 html）并渲染为 index.html
  - 递归渲染 site_dir 下所有 .md → 同目录同名 .html
  - 每页顶部生成「← 站点首页」面包屑与所属板块标签（按一级子目录分组）
"""
import argparse
import re
import sys
from pathlib import Path

import markdown

DOC_CSS = """
  :root {
    --bg: #f5f7fa; --card: #ffffff; --ink: #1c2733; --ink-2: #5a6b7c;
    --line: #e3e8ee; --accent: #1f5f8b; --accent-soft: #e8f1f8;
    --code-bg: #f0f3f6; --quote: #eef4f9; --code-ink: #0f1a24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.75;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 32px 24px 64px; }
  nav.topnav {
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px;
    font-size: 13px; color: var(--ink-2);
  }
  nav.topnav a { color: var(--accent); text-decoration: none; font-weight: 600; }
  nav.topnav a:hover { text-decoration: underline; }
  nav.topnav .crumb {
    background: var(--card); border: 1px solid var(--line); border-radius: 999px;
    padding: 3px 12px; font-size: 12px; font-weight: 600; color: var(--ink-2);
  }
  article.doc {
    background: var(--card); border: 1px solid var(--line); border-radius: 12px;
    padding: 36px 44px;
  }
  article.doc h1 { font-size: 27px; line-height: 1.4; letter-spacing: -0.01em; padding-bottom: 14px; border-bottom: 1px solid var(--line); margin-bottom: 26px; }
  article.doc h2 { font-size: 20px; margin: 36px 0 14px; padding-bottom: 8px; border-bottom: 1px solid #eef1f5; }
  article.doc h3 { font-size: 16.5px; margin: 26px 0 10px; }
  article.doc h4 { font-size: 15px; margin: 20px 0 8px; }
  article.doc p { margin: 12px 0; }
  article.doc ul, article.doc ol { margin: 12px 0; padding-left: 26px; }
  article.doc li { margin: 5px 0; }
  article.doc a { color: var(--accent); text-decoration: none; }
  article.doc a:hover { text-decoration: underline; }
  article.doc table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; display: block; overflow-x: auto; }
  article.doc th { background: #f0f4f8; text-align: left; padding: 9px 13px; border: 1px solid var(--line); font-weight: 600; white-space: nowrap; }
  article.doc td { padding: 9px 13px; border: 1px solid var(--line); vertical-align: top; }
  article.doc tr:nth-child(even) td { background: #fafbfc; }
  article.doc code { background: var(--code-bg); padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; }
  article.doc pre { background: var(--code-ink); color: #e6edf3; padding: 16px 18px; border-radius: 8px; overflow-x: auto; margin: 16px 0; line-height: 1.6; }
  article.doc pre code { background: none; color: inherit; padding: 0; }
  article.doc blockquote { border-left: 4px solid #b9cddd; background: var(--quote); margin: 16px 0; padding: 10px 16px; color: var(--ink-2); border-radius: 0 8px 8px 0; }
  article.doc blockquote p { margin: 4px 0; }
  article.doc img { max-width: 100%; border-radius: 8px; border: 1px solid var(--line); }
  article.doc hr { border: none; border-top: 1px solid var(--line); margin: 28px 0; }
  a.diagram-link {
    display: inline-block; margin: 4px 0;
    background: var(--accent-soft); border: 1px solid #cfe0ee; color: var(--accent);
    border-radius: 8px; padding: 7px 16px; font-size: 13.5px; font-weight: 600;
    text-decoration: none !important; transition: background 0.15s, border-color 0.15s;
  }
  a.diagram-link:hover { background: #dbeaf5; border-color: var(--accent); }
  a.diagram-link::before { content: "打开交互式图："; font-weight: 400; }
  footer.doc-foot { margin-top: 24px; font-size: 12.5px; color: var(--ink-2); text-align: center; }
  @media (max-width: 640px) {
    .wrap { padding: 20px 12px 40px; }
    article.doc { padding: 24px 18px; }
    article.doc h1 { font-size: 22px; }
  }
"""


def parse_group_labels(spec: str) -> dict:
    """--group-label 'core=核心|ai=AI' → {'core':'核心','ai':'AI'}"""
    labels = {}
    if not spec:
        return labels
    for pair in spec.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            labels[k.strip()] = v.strip()
    return labels


def extract_title(md_text: str) -> str:
    m = re.search(r"^#\s+(.+?)\s*$", md_text, re.MULTILINE)
    return m.group(1).strip() if m else "文档"


def preprocess(md_text: str) -> str:
    """把指向 .html 的图片引用改为交互图链接按钮。"""
    def repl(m):
        alt, url = m.group(1), m.group(2)
        return f'<a class="diagram-link" href="{url}" target="_blank" rel="noopener">{alt}</a>'
    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.html)([^)]*)\)", repl, md_text)


def rewrite_links(html: str) -> str:
    """把渲染后指向 .md 的相对链接改写为 .html。"""
    def repl(m):
        anchor = m.group(2) or ""
        return f'href="{m.group(1)}.html{anchor}"'
    return re.sub(r'href="([^"]+)\.md(#.*)?"', repl, html)


def render_one(md_path: Path, root: Path, site_name: str, group_labels: dict) -> Path:
    md_text = md_path.read_text(encoding="utf-8")
    title = extract_title(md_text)
    body = markdown.markdown(
        preprocess(md_text),
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        output_format="html5",
    )
    body = rewrite_links(body)

    rel = md_path.relative_to(root)
    group = rel.parts[0] if len(rel.parts) > 1 else ""
    back = "index.html" if not group else "../index.html"
    group_label = group_labels.get(group, group) if group else ""

    crumb = f'<span class="crumb">{group_label}</span>' if group_label else ""
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} · {site_name}</title>
<style>{DOC_CSS}</style>
</head>
<body>
<div class="wrap">
  <nav class="topnav">
    <a href="{back}">← {site_name}</a>
    {crumb}
  </nav>
  <article class="doc">
{body}
  </article>
  <footer class="doc-foot">{site_name} · 渲染文档</footer>
</div>
</body>
</html>
"""
    out = md_path.with_suffix(".html")
    out.write_text(page, encoding="utf-8")
    return out


def collect_docs(root: Path):
    """返回 (全部 .md 列表, 独立 .html 列表)。
    排除导览自身 index.md/index.html；.md 渲染出的同名 .html 不算独立 html。"""
    mds = sorted(p for p in root.rglob("*.md") if p.name != "index.md")
    rendered = {p.with_suffix(".html") for p in mds}
    htmls = sorted(
        p for p in root.rglob("*.html")
        if p.name != "index.html" and p not in rendered
    )
    return mds, htmls


def generate_nav(root: Path, site_name: str) -> Path:
    """生成导览 index.md：链接全部 .md 与全部 .html（md 条目指向其渲染后的 .html）。"""
    mds, htmls = collect_docs(root)
    lines = [f"# {site_name}", ""]
    lines.append("> 本站导览：覆盖全部文档页面（Markdown 源文件部署时已渲染为 HTML）。")
    lines.append("")
    groups: dict = {}
    for p in list(mds) + list(htmls):
        rel = p.relative_to(root)
        g = rel.parts[0] if len(rel.parts) > 1 else "."
        groups.setdefault(g, []).append(p)
    for g in sorted(groups, key=lambda x: (x != ".", x)):
        title = "顶层" if g == "." else g
        lines.append(f"## {title}")
        for p in sorted(groups[g], key=lambda x: x.relative_to(root).as_posix()):
            rel = p.relative_to(root)
            display = str(rel.with_suffix("")).replace("/", " / ")
            href = rel.as_posix()
            if p.suffix == ".md":
                href = rel.with_suffix(".html").as_posix()
            lines.append(f"- [{display}]({href})")
        lines.append("")
    lines.append(f"共 {len(mds)} 个 Markdown 文档、{len(htmls)} 个独立 HTML 页面。")
    nav = root / "index.md"
    nav.write_text("\n".join(lines), encoding="utf-8")
    return nav


def main():
    ap = argparse.ArgumentParser(description="Markdown 渲染为 HTML 并生成导览 index（GitHub Pages 部署时默认执行）")
    ap.add_argument("docs_dir", type=Path, help="站点目录（默认当前目录）", nargs="?", default=Path("."))
    ap.add_argument("--site-name", default="架构文档", help="站点名（用于导览标题/面包屑/页脚/页面标题）")
    ap.add_argument("--group-label", default="", help='一级子目录分组标签，如 "core=核心|ai=智能"')
    args = ap.parse_args()

    root = args.docs_dir.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2
    group_labels = parse_group_labels(args.group_label)

    nav = generate_nav(root, args.site_name)
    print(f"nav: {nav.relative_to(root)}")

    mds = sorted(root.rglob("*.md"))  # 含导览 index.md
    for md in mds:
        out = render_one(md, root, args.site_name, group_labels)
        print(f"built: {out.relative_to(root)}")
    print(f"done: {len(mds)} markdown files rendered (incl. nav index)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
