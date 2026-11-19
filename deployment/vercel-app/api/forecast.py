"""Vercel loads this WSGI callable; never start Waitress or a socket server here."""
from web.vercel_runtime import app
