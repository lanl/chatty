"""Textual UI for chatty.

Module Structure
----------------
- app.py: ChatApp class, bindings, and main entry point
- widgets.py: ChatInput, MessageWidget, ChatLog, StatusBar
- modals.py: FileInputModal, SessionBrowserModal, ModelPickerModal, ModelInputModal
- app.tcss: Stylesheet for widget layout and colors
"""

from chatty.ui.app import ChatApp, main

__all__ = ["ChatApp", "main"]
