#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entry point for the FastAPI application.
Imports and runs the modular app from the app package.
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)