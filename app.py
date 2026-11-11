"""Entry point for the web demo, also usable as a Hugging Face Space app file.

    python app.py            # same as: meeting-notes serve
"""

import os

from meeting_notes.webapp import launch

if __name__ == "__main__":
    launch(host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"), port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
