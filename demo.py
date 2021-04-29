# ====================================================================================================================================
# Imports
# ====================================================================================================================================

from pywebio import start_server
from scripts import *
import argparse

# ====================================================================================================================================
# Call to main() function    
# ====================================================================================================================================

if __name__ == '__main__':      
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    parser.add_argument("-l", "--locale", type=str, default="remote")
    args = parser.parse_args()
    
    if args.locale == "local":
        main()
    else:
        start_server(main, port=args.port)