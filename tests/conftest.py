import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from hypothesis import settings  # noqa: E402

settings.register_profile("default", deadline=None, max_examples=60)
settings.register_profile("thorough", deadline=None, max_examples=600)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
