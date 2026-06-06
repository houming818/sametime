import os

with open('/home/nio/log/holds/SameTime/experiments/train_massive_l1.py', 'r') as f:
    content = f.read()

# Fix the split logic. \\t was used which escapes to string '\t' instead of tab character
content = content.replace("split('\\\\t')", "split('\\t')")

with open('/home/nio/log/holds/SameTime/experiments/train_massive_l1.py', 'w') as f:
    f.write(content)
