import os
import sys

def getArgvKeyValye(key, default = None):
    try:
        k = sys.argv.index(key)
        return sys.argv[k+1]
    except ValueError:
        return default

def isKeyPresented(key):
    try:
        sys.argv.index(key)
        return True
    except ValueError:
        return False