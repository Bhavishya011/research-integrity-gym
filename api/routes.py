"""
routes.py — documents the API surface for reference.

All routes are registered via FastAPI decorators in app.py.
This module exists as a reference map and for documentation generation.
Import it to get the route table without executing any route logic.
"""

ROUTE_TABLE = [
    {"method": "GET",  "path": "/health",   "tag": "system",     "description": "Liveness probe"},
    {"method": "POST", "path": "/reset",    "tag": "openenv",    "description": "Start new episode (supports task1-5 including task5_fda_approval)"},
    {"method": "POST", "path": "/step",     "tag": "openenv",    "description": "Execute action (incl. submit_fda_verdict for Task 5), returns obs/reward/done/info"},
    {"method": "GET",  "path": "/state",    "tag": "openenv",    "description": "Current episode state (no ground truth)"},
    {"method": "GET",  "path": "/tasks",    "tag": "openenv",    "description": "List all 5 tasks with action schemas"},
    {"method": "POST", "path": "/grader",   "tag": "evaluation", "description": "Run grader on a completed episode (all 5 tasks supported)"},
    {"method": "POST", "path": "/baseline", "tag": "evaluation", "description": "Trigger baseline script, return scores"},
]
