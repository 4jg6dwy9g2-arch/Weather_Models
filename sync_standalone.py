#!/usr/bin/env python3
"""
Standalone sync script for weather model data.
Delegates to app.run_master_sync() to keep behavior consistent.

Usage:
    python sync_standalone.py [--force] [--init-hour HOUR]

Options:
    --force         Force refresh even if data already exists
    --init-hour     Specific init hour (0, 6, 12, or 18). If not specified, uses latest available.
"""

import sys
import argparse
import logging

from app import run_master_sync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Standalone sync script for weather model data'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh even if data already exists'
    )
    parser.add_argument(
        '--init-hour',
        type=int,
        choices=[0, 6, 12, 18],
        help='Specific init hour (0, 6, 12, or 18). If not specified, uses latest available.'
    )

    args = parser.parse_args()

    try:
        results = run_master_sync(force=args.force, init_hour=args.init_hour)
        logger.info(results)
        return 0 if results.get("success", False) else 1
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
