import gdown
import sys
import os.path

#1KVo4Z1vThfHI732Asg-OeIYTISwV1kpe

if len(sys.argv) < 3:
	print("pass id, filename as argv")
	exit(0)

if os.path.isfile(sys.argv[2]):
	print("file exist")
	exit(0)
	
url = 'https://drive.google.com/uc?id='+sys.argv[1]
gdown.download(url, sys.argv[2], quiet=False)