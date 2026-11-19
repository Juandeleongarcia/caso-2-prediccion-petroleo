"""Loopback development server. Never a production Internet server."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsgiref.simple_server import make_server
from web.backend import app
if __name__ == '__main__':
    print('Brent Forecast: http://127.0.0.1:8765', flush=True)
    make_server('127.0.0.1', 8765, app).serve_forever()
