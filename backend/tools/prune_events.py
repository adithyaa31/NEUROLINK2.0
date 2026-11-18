"""Prune old local analytics events with an explicit UTC cutoff."""
import argparse
from datetime import datetime

from backend.app.database import delete_events_before, initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="UTC cutoff in ISO format, e.g. 2026-01-01T00:00:00")
    args = parser.parse_args()
    cutoff = datetime.fromisoformat(args.before).strftime("%Y-%m-%d %H:%M:%S")
    initialize()
    print(f"Deleted {delete_events_before(cutoff)} event rows older than {cutoff} UTC.")


if __name__ == "__main__":
    main()
