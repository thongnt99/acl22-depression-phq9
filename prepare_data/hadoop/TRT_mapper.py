#!/usr/bin/env python
import json
import sys
askreddit_user_ids = set([line.strip() for line in open("askreddit_user_ids.txt", "r")])
for line in sys.stdin:
    try:
        post = json.loads(line)
    except:
        continue
    if isinstance(post, dict) and "author" in post and not post["author"] in ["[deleted]", "[removed]"]:
        user_id = post["author"]
        if user_id in askreddit_user_ids:
            print(user_id)
