---
layout: single
title: posts
permalink: /posts/
---

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}  ({{ post.date | date: "%Y-%m-%d" }}) </a>
    </li>
  {% endfor %}
</ul>
