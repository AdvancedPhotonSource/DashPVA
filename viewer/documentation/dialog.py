#!/usr/bin/env python3
"""
Reusable Documentation Dialog (viewer/documentation/dialog.py)

Provides a simple dialog to display viewer-specific documentation.
- Prefers rendering local HTML files (index.html) using QWebEngineView if available
- Falls back to QTextBrowser with basic HTML rendering
- Optionally converts Markdown (README.md) to HTML if the 'markdown' package is installed
- Auto-discovers documentation paths relative to the viewer's module file

Expected doc locations per viewer module:
  <viewer_module_dir>/doc/index.html  (preferred)
  <viewer_module_dir>/doc/README.md   (fallback, converted to HTML if possible)

Viewers may override discovery by setting an attribute 'doc_path' on their window
(e.g., self.doc_path = "/absolute/or/relative/path/to/index.html").
"""

from pathlib import Path
import inspect

from PyQt5.QtWidgets import QDialog, QVBoxLayout
from PyQt5.QtCore import QUrl

# Try to use QWebEngineView for full HTML support (CSS/JS). Fallback to QTextBrowser.
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView  # type: ignore
    WEBENGINE_AVAILABLE = True
except Exception:
    from PyQt5.QtWidgets import QTextBrowser
    QWebEngineView = None  # type: ignore
    WEBENGINE_AVAILABLE = False

# Optional markdown support
try:
    import markdown  # type: ignore
    MARKDOWN_AVAILABLE = True
except Exception:
    markdown = None  # type: ignore
    MARKDOWN_AVAILABLE = False


class DocumentationDialog(QDialog):
    """Dialog to render documentation content (HTML or converted Markdown)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documentation")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        if WEBENGINE_AVAILABLE:
            self.view = QWebEngineView(self)
        else:
            # Basic HTML rendering (no JS/CSS external files) using QTextBrowser
            self.view = QTextBrowser(self)  # type: ignore
        layout.addWidget(self.view)

    def set_content_html(self, html: str) -> None:
        """Render raw HTML in the dialog."""
        if WEBENGINE_AVAILABLE:
            self.view.setHtml(html)  # QWebEngineView
        else:
            # QTextBrowser
            self.view.setHtml(html)

    def load_html_file(self, file_path: str) -> None:
        """Load and render a local HTML file by path."""
        p = Path(file_path)
        if WEBENGINE_AVAILABLE:
            # Load file via URL for QWebEngineView
            url = QUrl.fromLocalFile(str(p.resolve()))
            self.view.setUrl(url)
        else:
            # Read file and render HTML as text
            try:
                html = p.read_text(encoding="utf-8")
            except Exception:
                html = f"<html><body><h3>Unable to read HTML file:</h3><pre>{p}</pre></body></html>"
            self.view.setHtml(html)

    @staticmethod
    def _discover_doc_candidates_for(viewer_widget) -> dict:
        """
        Return a dict of candidate documentation sources for a viewer widget:
          {
            'index_html': Path | None,
            'readme_md': Path | None,
          }
        """
        # Allow explicit override via attribute on the viewer
        try:
            override = getattr(viewer_widget, 'doc_path', None)
            if override:
                override_path = Path(override)
                # If override points to a directory, assume index.html inside
                if override_path.is_dir():
                    override_path = override_path / 'index.html'
                return {
                    'index_html': override_path if override_path.suffix.lower() == '.html' else None,
                    'readme_md': override_path if override_path.suffix.lower() in ('.md', '.markdown') else None,
                }
        except Exception:
            pass

        # Discover relative to the viewer's class/module file
        try:
            mod_file = inspect.getfile(viewer_widget.__class__)
        except Exception:
            mod_file = None
        index_html = None
        readme_md = None
        if mod_file:
            mod_dir = Path(mod_file).parent
            doc_dir = mod_dir / 'doc'
            idx = doc_dir / 'index.html'
            md = doc_dir / 'README.md'
            if idx.exists():
                index_html = idx
            if md.exists():
                readme_md = md
        return {'index_html': index_html, 'readme_md': readme_md}

    def open_for_viewer(self, viewer_widget) -> None:
        """
        Auto-discover documentation next to the viewer module and display it.
        Preference order:
          1. <module>/doc/index.html (full HTML)
          2. <module>/doc/README.md (converted to HTML if markdown available)
          3. Fallback placeholder HTML
        """
        candidates = self._discover_doc_candidates_for(viewer_widget)
        idx = candidates.get('index_html')
        md = candidates.get('readme_md')

        if idx is not None:
            self.load_html_file(str(idx))
            return

        if md is not None:
            try:
                md_text = Path(md).read_text(encoding='utf-8')
            except Exception:
                md_text = f"# Documentation\n\nUnable to read file: {md}"
            if MARKDOWN_AVAILABLE:
                html = markdown.markdown(md_text, extensions=[
                    'fenced_code', 'tables', 'toc'
                ])
            else:
                # Minimal conversion: wrap in <pre> if markdown not available
                html = f"<pre style=\"white-space: pre-wrap; font-family: monospace;\">{md_text}</pre>"
            # Basic styling
            styled_html = (
                "<html><head><style>"
                "body{font-family:Arial, sans-serif; padding:12px;}"
                "pre{background:#f8f9fa; padding:8px; border-radius:4px;}"
                "code{background:#f1f3f5; padding:2px 4px; border-radius:3px;}"
                "h1,h2,h3{color:#2c3e50;}"
                "a{color:#0069d9;}"
                "</style></head><body>" + html + "</body></html>"
            )
            self.set_content_html(styled_html)
            return

        # Fallback: show a helpful placeholder
        viewer_name = getattr(viewer_widget, 'viewer_name', viewer_widget.__class__.__name__)
        placeholder = f"""
        <html>
          <head><style>
            body{{font-family: Arial, sans-serif; padding: 16px;}}
            .title{{font-size: 22px; font-weight: bold; margin-bottom: 6px;}}
            .subtitle{{color:#555; margin-bottom: 12px;}}
            code{{background:#f5f5f5; padding:2px 4px; border-radius:4px;}}
            .hint{{margin-top: 12px; font-size: 12px; color: #777;}}
          </style></head>
          <body>
            <div class="title">{viewer_name} Documentation</div>
            <div class="subtitle">No documentation found.</div>
            <p>Create one of the following files next to the viewer module:</p>
            <ul>
              <li><code>doc/index.html</code> (preferred)</li>
              <li><code>doc/README.md</code> (fallback, converted to HTML)</li>
            </ul>
            <div class="hint">Tip: Put files under the module's directory, e.g., <code>viewer/workbench/doc/index.html</code></div>
          </body>
        </html>
        """
        self.set_content_html(placeholder)
