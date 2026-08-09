"""Shared console-logging configuration.

Logs go to the console only. They appear in:
VS Code terminal,
local Streamlit server terminal, 
Streamlit Community Cloud log view
"""
import logging


def configure_logging() -> None:
    """Configure project console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
