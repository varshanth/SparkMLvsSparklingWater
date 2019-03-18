import re
from sys import argv

if len(argv) < 3:
    print('Insufficient number of args')

orig_log_file = argv[1]
new_log_file = argv[2]

with open(orig_log_file, 'r') as f:
    content = f.read()
look_for = re.compile('EXPERIMENT (.*): [-]{0,4}(.*)[-]{0,4}')
filtered_content = look_for.findall(content)
with open(new_log_file, 'w') as f:
    for event in filtered_content:
        f.write('{0}\n'.format(event))

