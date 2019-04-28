import os
import sys

ct_folder = 'all2' # folder with all computer tomography images
cancer_folder = 'cancer' # folder with cancerous tomography images

def getArgvKeyValue(key, default = None):
    try:
        k = sys.argv.index(key)
        return sys.argv[k+1]
    except ValueError:
        return default

def isArgvKeyPresented(key):
    try:
        sys.argv.index(key)
        return True
    except ValueError:
        return False