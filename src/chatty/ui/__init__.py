"""Textual UI for chatty.

Module Structure
----------------
- app.py: ChatApp class, bindings, and main entry point
- widgets.py: ChatInput, MessageWidget, ChatLog, StatusBar
- modals.py: FileInputModal, SessionBrowserModal, ModelPickerModal, ModelInputModal
- footer.py: ChattyFooter (custom footer with controlled keybinding display)
- app.tcss: Stylesheet for widget layout and colors
"""

from chatty.ui.app import ChatApp, main
from chatty.ui.footer import ChattyFooter

__all__ = ["ChatApp", "ChattyFooter", "main"]
