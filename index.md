---
layout: default
title: Home
notitle: true
---

<ul>
{% for post in site.posts %}
<li>
  <span>
    <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
  </span>
  <br>
  <span>
    {{ post.date | date: "%b %-d, %Y" }}
  </span>
</li>
{% endfor %}
</ul>
