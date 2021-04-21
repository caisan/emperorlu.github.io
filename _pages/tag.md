---
layout: default
title: tags
permalink: /tags/
---

<link rel="stylesheet" href="/assets/css/responsive.css">
<link rel="stylesheet" href="/assets/css/index.css">

<div id="main" role="main">
{% include sidebar.html %}
<section class="container content">
    <div class="columns">
        <div class="column two-fourths">
            <article class="article-content markdown-body">
                <section class="container posts-content">
                    {% assign sorted_tags = site.tags | sort %}
                    {% for category in sorted_tags %}
                    <h3>{{ category | first }}</h3>
                    <ol class="posts-list" id="{{ category[0] }}">
                        {% for post in category.last %}
                        <li class="posts-list-item">
                            <span class="posts-list-meta">{{ post.date | date:"%Y-%m-%d" }}</span> <a class="posts-list-name" href="{{ post.url }}">{{ post.title }}</a>
                        </li>
                        {% endfor %}
                    </ol>
                    {% endfor %}
                </section>
                <!-- /section.content -->
            </article>
        </div>
        <div class="column one-fourth">
            {% include sidebar-tags-nav.html %}
        </div>
    </div>
</section>
</div>


