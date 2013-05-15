import os
import sys
from recommender_utils import get_recommendations
from recommender_utils import get_suggestions

_basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IDENTIFIER_FILE = _basedir + '/data/ASTkeywords.set'
try:
    IDENTIFIERS = open(IDENTIFIER_FILE).read().strip().split('\n')
except:
    IDENTIFIERS = []

