---
classes: wide
title: How I prepared for (and passed!) the Google coding interviews
tags: python algorithms
header:
    teaser: "/assets/posts/algos_studying/charts.png"
excerpt_separator: <!--more-->
---

Update: I wrote about how my internship experience went [here]({% post_url 2022-11-01-google-internship %}).
<!--more-->

I received great news that I did well on my Google internship technical interviews!
I have zero formal computer science training so I am particularly proud of making it through, and I wanted to share my experience.

## My disdain for Leetcode

Leetcode.com is no doubt _the_ site that everyone recommends for studying Data Structures and Algorithms (DSA) interview questions.
But when using it to prepare, I had two main issues with it:

1) Leetcode is overwhelming for beginners (me)

There's thousands of questions of varying difficulty (and quality...) and it's difficult to study with an a-la-carte style question bank. They do provide curated lists, and while the example questions in these lists cover all the material you'll need, the overall sequence and difficulty progression is sub-optimal and your learning effectively becomes unstructured. If you're starting with zero DSA knowledge, then Leetcode is not the way to go.

2) Leetcode's metrics make no sense

In fact, the metrics that Leetcode tracks are arguably useless when it comes to coding interviews. Ranking submitted-answer execution speed and memory deludes people into thinking that being 99th percentile in execution speed automatically makes their algorithm better than others. This is evidenced by the countless solutions that users share in question discussion forums, citing their execution performance percentile without even understanding the algorithmic big-O complexity (i.e. what actually matters).

## My study solution

Simply doing more problems wasn't helping me learn fundamental concepts. It was disheartening returning to a problem after a few days only to realize I had forgotten how to solve it. I only had a couple weeks to learn the massive list of DSA "must-knows" that Google sends to help you prepare for the interview, so I couldn't afford to waste time and effort like that. What I really needed was:

1) a reduced number of questions, arranged in logical sequence to help build conceptual understanding (e.g. linked lists -> trees -> graphs, etc.)

2) questions stripped down to the bare minimum to understand their fundamental concept, facilitating generalizability to harder word problems similar to real interview questions

3) a way to **honestly assess** and record my confidence with different problems as I studied, allowing me to do targeted practice with spaced repetition

Through sheer luck, the [Youtube algorithm recommended me this 5.5h video on dynamic programming](https://youtu.be/oBt53YbR9Kk) while I was studying. I watched the entire thing because I enjoyed the teaching style so much (I don't even know Javascript!). In fact, the instructor, Alvin Zablan, has a bunch of free videos on Youtube covering other DSA topics too like graphs, trees, and recursion. I watched all of them while following along using Python. When I found out that he recently started his own online algorithms course, [structy.net](https://structy.net), I checked out a few of the free videos, then immediately bought a month of subscription. It was ~$30 (for reference, so is Leetcode Premium); I used it for 12 days prior to my interview and learned almost every single required "must-know" topic on the Google interview preparation guide. (This isn't an ad lol I genuinely think it's great.)

The questions were ordered with concepts building upon each other sequentially (#1 on my list above); and most of the questions were bare-bones (#2 on my list), allowing you to more easily spot the concept when they came up in wordier problems. The only downsides with the platform were that dynamic programming only covered using recursion and not tabulation, and the exhaustive exhaustion problems (e.g. backtracking) were a little scarce. In the end though these were minor details that didn't affect my interview preparedness. The one thing that the platform did lack was a better way to learn via spaced repetition (#3 on my list). The last 30 or so questions on the platform are randomly ordered and should help in recalling/practicing possibly-forgotten concepts from earlier in the course, but I think it can be improved. As far as I'm aware, there are no platforms out there that execute the idea of effective scheduled studying well, so this motivated me to roll my own solution to track my study progress.

## My study dashboard

I had 12 days (Nov 6th-18th) to study all Google interview material, but also had regular grad-school research responsibilities to juggle during the work day, so I needed to create an efficient way to systematically learn new concepts each day while simultaneously reviewing old concepts through spaced repetition. In [Google Sheets, I made a database](/assets/posts/algos_studying/google_dsa_prep.pdf) of all 100ish questions offered by Structy.

<div style="text-align:center"><img src="/assets/posts/algos_studying/spreadsheet.png" style="width:30em"/></div>

After completing a problem, I would punch in: the unique question ID; the date on which I completed it; and most importantly, my confidence in completing it from a scale of 1-5. The main sheet would display each unique question's most recent completion date, most recent confidence score, and the total number of times that I completed it. Keeping track of repetitions was important because I wanted to make sure I would repeat a concept at least every couple days to prevent myself from forgetting it. I colour-coded all Structy's DSA topics (arrays, dynamic programming, etc.), and used cell-formatting to apply a heatmap to low-confidence questions, and questions that haven't been completed in a while. If I had my own online algos course, I would make these stats absolutely private to each user -- there's no benefit in competing with others on how confident you are or how many times you've practiced a problem.

<div style="text-align:center"><img src="/assets/posts/algos_studying/charts.png" style="width:35em"/></div>

Finally, I created some charts showing the number of questions completed, and my overall confidence in each subject.  I think **this** is a proof-of-principle concept for what all online algo courses and question banks should provide. The course by itself was great but its effectiveness was magnified 10x with systematic, targeted studying. Seeing my strengths and weaknesses in real-time was the key to efficient learning.

There just wasn't enough time to learn _everything_ that Google says I should know. But in the end I was definitely prepared for both multi-part questions asked during my interviews.

Now I'm onto the Project Search round where you're supposed to patiently wait for a team to fish your application out of the large pool of candidates to see if you'd be a good fit. This stage seems to depend on whether the skills you already possess (coding languages, domain experience, etc.) match the requirements of specific teams on the intern projects they have available. I'd be really stoked to land a match, but it's out of my control at this point. At the very least, I'm happy that all the effort I put into the part that _was_ under my control paid off and that I made it this far.
