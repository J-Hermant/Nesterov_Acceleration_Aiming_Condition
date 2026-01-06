import os
print(os.environ.get("HTTPS_PROXY"))
print(os.environ.get("HTTP_PROXY"))
import urllib.request
urllib.request.urlopen("https://www.google.com")