"""
Thin Calendar API v3 wrapper, matching blogger_client.py's shape.

Usage:
    from calendar_client import list_upcoming, create_event

    for e in list_upcoming():
        print(e["start"], e["summary"])
"""

from datetime import datetime, timezone

from google_client import get_calendar_service


def list_upcoming(max_results: int = 10, calendar_id: str = "primary") -> list[dict]:
    service = get_calendar_service()
    now = datetime.now(timezone.utc).isoformat()
    resp = service.events().list(
        calendarId=calendar_id, timeMin=now, maxResults=max_results,
        singleEvents=True, orderBy="startTime",
    ).execute()
    out = []
    for e in resp.get("items", []):
        start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
        out.append({
            "id": e["id"],
            "summary": e.get("summary", "(no title)"),
            "start": start,
            "location": e.get("location", ""),
            "htmlLink": e.get("htmlLink", ""),
        })
    return out


def create_event(summary: str, start_iso: str, end_iso: str,
                  description: str = "", calendar_id: str = "primary") -> dict:
    """start_iso/end_iso are RFC3339 datetimes, e.g. '2030-01-15T14:00:00-05:00'."""
    service = get_calendar_service()
    body = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }
    return service.events().insert(calendarId=calendar_id, body=body).execute()
