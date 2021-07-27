# Run this script after installing:
#   Visual Studio Build Tools for C++ with Visual C++ tools for CMake
#   Visual Studio Code
#   Python 3.8 and below

import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy==1.19.5'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'tensorflow==2.4.1'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'keras==2.4.3'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pandas==1.1.5'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'imutils==0.5.4'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'cmake'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'dlib==19.22.0'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pika==1.2.0'])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'opencv-python==4.5.1.48'])
