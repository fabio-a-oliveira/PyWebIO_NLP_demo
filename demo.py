# =============================================================================
# Imports
# =============================================================================

#import pandas as pd
#import tensorflow as tf
#import requests
#import re
#from os import chdir, getcwd, listdir, mkdir
#from os.path import join
#import pkg_resources
#import pickle

#import googletrans

from pywebio.input import input, select, radio
from pywebio.output import put_text, put_markdown
from scripts import *


# =============================================================================
# Call to web_interface() function    
# =============================================================================
    
if __name__ == '__main__':      
    
    put_markdown('## NLP applied to aviation regulations')
    put_markdown('### Demonstrations using 14 CFR Part 121 and RBAC 121')
    put_text('This series of demos showcases the use of some Natural Language Processing (NLP) techniques to aviation regulations. Choose your favorite and enjoy!')
    put_markdown('---')
    
    for _ in range(100):
    
        choose_demonstration()