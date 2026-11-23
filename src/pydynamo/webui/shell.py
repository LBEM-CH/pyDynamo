"""The Qt window of the dcp GUI: a QWebEngineView on the local server (pydynamo.webui.server).

The profile is off the record: nothing is written to ~/.local/share or ~/.cache. On remote desktops (VNC, xrdp)
Chromium's GPU paths fail and fall back to software rendering after noisy retries; --disable-gpu starts there
directly (a form-based UI needs no GPU). PYDYNAMO_WEBUI_GPU=1 keeps the GPU paths.
Ctrl+= / Ctrl+- / Ctrl+0 zoom the whole interface (large or small remote-desktop screens).
"""
import os
import sys

TITLE = "pyDynamo dcp (Dynamo 1.1.514 project manager)"


def webengine_problem():
    """None if Qt WebEngine can be imported, else the reason."""
    try:
        import PySide6.QtWebEngineWidgets  # noqa: F401
    except Exception as e:
        return ("Qt WebEngine is not available (%s). Install the GUI section of requirements/pydynamo-lock.txt "
                "(PySide6-Addons)." % e)
    return None


def run_window(server, title=TITLE, snap=None, script=None, timeout_s=60, settle_ms=1500):
    """Show the view of `server` in a Qt window until it is closed. Returns 0.

    snap: save the window as PNG after the page has loaded (and `settle_ms` after `script`, a JS snippet, ran), then
    quit; used by tests and screenshots with QT_QPA_PLATFORM=offscreen."""
    if not os.environ.get("PYDYNAMO_WEBUI_GPU"):
        flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        if "--disable-gpu" not in flags:
            os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (flags + " --disable-gpu").strip()
    from PySide6.QtCore import QTimer, QUrl
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("pyDynamo dcp")
    profile = QWebEngineProfile(app)                     # no storage name: off the record

    class Page(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line, source):
            if os.environ.get("PYDYNAMO_WEBUI_DEBUG") or level == QWebEnginePage.ErrorMessageLevel:
                print("[dcp view] %s (%s:%d)" % (message, source.rsplit("/", 1)[-1], line), file=sys.stderr)

    view = QWebEngineView()
    page = Page(profile, view)
    view.setPage(page)
    # the copy buttons: let the page write the clipboard (Qt WebEngine denies it by default)
    from PySide6.QtWebEngineCore import QWebEngineSettings
    page.settings().setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
    try:
        from PySide6.QtWebEngineCore import QWebEnginePermission

        def grant(permission):
            if permission.permissionType() == QWebEnginePermission.PermissionType.ClipboardReadWrite:
                permission.grant()
            else:
                permission.deny()
        page.permissionRequested.connect(grant)
    except (ImportError, AttributeError):                   # Qt < 6.8: the setting above is enough
        pass
    win = QMainWindow()
    win.setWindowTitle("%s   %s" % (title, os.getcwd()))
    win.setCentralWidget(view)
    win.resize(1440, 900)
    for keys, f in (("Ctrl+=", 1.1), ("Ctrl++", 1.1), ("Ctrl+-", 1 / 1.1)):
        QShortcut(QKeySequence(keys), win, lambda f=f: view.setZoomFactor(max(0.5, min(3.0, view.zoomFactor() * f))))
    QShortcut(QKeySequence("Ctrl+0"), win, lambda: view.setZoomFactor(1.0))

    if snap:
        state = {"waited": 0}

        def ready():
            view.page().runJavaScript("window.dcpReady === true", 0, after_ready)

        def after_ready(ok):
            if not ok and state["waited"] < timeout_s * 5:
                state["waited"] += 1
                QTimer.singleShot(200, ready)
                return
            if script:
                view.page().runJavaScript(script, 0, lambda _r: QTimer.singleShot(settle_ms, grab))
            else:
                QTimer.singleShot(500, grab)

        def grab():
            win.grab().save(snap)
            print("snapshot", snap)
            app.quit()
        view.loadFinished.connect(lambda _ok: QTimer.singleShot(300, ready))
    view.load(QUrl(server.start_url))
    win.show()
    app.exec()
    server.shutdown()
    return 0
