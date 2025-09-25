"""
Legacy module - now imports from the new organized structure
For backwards compatibility, re-exports the main predictor class
"""
from src.predictor import FPLMatchPredictor, main

# Re-export for backwards compatibility
__all__ = ['FPLMatchPredictor', 'main']

if __name__ == "__main__":
    main()
