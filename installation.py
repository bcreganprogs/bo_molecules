import subprocess

with open('requirements.txt', 'r') as file:
    for line in file:
        package = line.strip()
        subprocess.run(['pip', 'install', package])