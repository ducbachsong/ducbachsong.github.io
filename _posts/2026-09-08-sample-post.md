---
title: "Sample Blog Post"
date: 2026-09-08
permalink: /blog/2026/09/08/sample-post/
tags:
  - sample
  - notes
---

The first paragraph shows up as the excerpt on the `/blog/` list page, so make
it say something on its own.

## Writing a new post

Create a file in `_posts/` named `YYYY-MM-DD-short-title.md`. The date in the
filename sets the publish date and the sort order, so it is not optional.

Regular markdown works throughout — headings, lists, links, images, and code:

```python
print("hello")
```

Images go in `images/` and are referenced as `/images/name.png`. Files you want
people to download go in `files/` instead.

## Notes

Delete this file once you have a real post. Posts dated in the future are still
built, because `future: true` is set in `_config.yml`.
