import os

blog_path = "/home/nio/log/blogs/www.grepcode.cn/src/spr/s2-g-04-phase3-l2-scaling-beam.md"

with open(blog_path, 'r') as f:
    content = f.read()

# Only add frontmatter if not already present
if not content.startswith("---"):
    # Strip the title and date from the top to replace with proper frontmatter
    content = content.replace("# SPR Phase 3: Scaling L2 Decoder & Breaking the Alignment Ceiling\n\n*Date: 2026-06-05*\n\n", "")
    
    frontmatter = """---
title: "[SPR-S2-G-04] Phase 3 L2 Scaling & Breaking the Alignment Ceiling"
date: 2026-06-05
draft: false
tags: ["SPR", "Semantic Prefix Routing", "L2", "Beam Search", "Scaling"]
description: Documenting the scaling of the L2 Decoder to 512d, breaking the Oracle 1-NN BLEU ceiling, and mitigating repetition loops using Beam Search and repetition penalties.
---

# Phase 3 L2 Scaling & Breaking the Alignment Ceiling

"""
    content = frontmatter + content
    
    with open(blog_path, 'w') as f:
        f.write(content)
        print("Frontmatter added.")
else:
    print("Frontmatter already exists.")
