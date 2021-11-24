---
classes: wide
title: How I prepared for (and passed!) the Google coding interviews
image: /assets/images/.png
tags: python algorithms
header:
    teaser: "/assets/images/.png"
excerpt_separator: <!--more-->
---
An idea for all online algos courses
<!--more-->

I received great news that I did well on my Google internship technical phone interviews!
I have zero formal computer science education so this is a particularly proud moment for me, and I wanted to share my experience.

## I needed an alternative to Leetcode

Leetcode.com is no doubt _the_ site that everyone recommends for studying Data Structures and Algorithms (DSA) interview questions.
There are a lot of reasons why I dislike Leetcode, but the main two qualms I have with it are:

### 1) Leetcode is overwhelming for beginners (me)

There's literally thousands of questions of varying difficulty (and quality...) and it is difficult to study with an a-la-carte style question bank.
They do provide curated lists (e.g. "Learn Dynamic Programming in 14 days"; "Google/Facebook/etc. Interview Questions"), and while the example questions in these lists cover all the material you'll need, the overall sequence and difficulty progression is sub-optimal and your learning effectively becomes unstructured.
If you are starting from no DSA knowledge, then Leetcode is not the way to go.
IMO, you only need ~100 questions -- with most of them being easy -- to actually be prepared.

### 2) Leetcode's metrics make no sense

In fact, the metrics that Leetcode tracks are arguably useless when it comes to coding interviews.
Ranking submitted-answer execution speed and memory deludes people into thinking that being 99th percentile in execution speed automatically makes their algorithm better than others'.
This is evidenced by the countless solutions that users share in question discussion forums, citing their execution speed percentile without even understanding the algorithmic big-Oh complexity (i.e. what actually matters).

## My study solution

Simply doing more problems wasn't helping me learn fundamental concepts.
It was disheartening to come back to a problem after a few days and realize that I had forgotten how to solve it.
I only had about two weeks to learn a massive list of DSA "must-knows" for the Google technical phone interviews.
What I really needed was:

1) a reduced number of questions, arranged in logical sequence (e.g. linked lists -> trees -> graphs, etc.)
2) questions stripped down to the bare minimum to understand their fundamental concept, facilitating generalizability to harder word problems similar to real interview questions
3) a way to _honestly_ assess my expertise and confidence in different classes of problems as I studied

Just by chance, the [Youtube algorithm recommended me this 5.5h video on dynamic programming](https://youtu.be/oBt53YbR9Kk).
I watched the entire thing because I enjoyed the teaching style so much (I don't even know Javascript!).
In fact, the instructor (Alvin Zablan) has a bunch of videos similar to that on Youtube covering other DSA topics too like graphs, and recursion.
I watched all of them while following along using Python.
When I found out that he recently started his own online algorithms course, [structy.net](structy.net), I checked out a few of the free videos and then immediately bought a month of subscription.
It was ~$30; I used it for 12 days prior to my interview and learned almost every single required "must-know" topic on the Google interview preparation guide.
(This isn't an ad lol it really is great.)

The Structy questions were ordered such that concepts built upon each other sequentially (#1 of my list above);and most of the questions were bare-bones (#2 of my list), allowing you to more easily spot the concept when they came up in wordier problems.
The only downsides with the platform were that dynamic programming only covered using recursion and not tabulation, and the exhaustive exhaustion problems (AKA backtracking) were a little scarce.
In the end though these were minor details and didn't affect my interview preparedness.

The one thing that the platform did lack was a better way to learn via spaced repetition.
The last 30 or so questions on the platform are randomly ordered and should help in recalling/practicing possibly-forgotten concepts from earlier in the course, but I think this can be improved.
There are no platforms out there that do this well, so this motivated me to create my own dashboard to track my study progress.

### My study dashboard

I had 12 days (Nov 6th-18th) to study all Google interview material, but also had general grad-school research responsibilities to juggle, so I needed to create an efficient way to systematically learn new concepts each day while simultaneously reviewing old concepts through spaced repetition.
In Google Sheets, I made a database of all the nontrivial questions offered by Structy.
Whenever I completed a problem, I would punch in: the question number/name; the date on which I completed it, the difficulty level of the problem; and most importantly, my confidence in completing it from a scale of 1-5.
On the main sheet, each unique question would display its most recent completion date, most recent confidence score, and the total number of times that I completed it.
Keeping track of repetitions was important because I wanted to make sure I would repeat a concept at least every couple days to prevent myself from forgetting it.
I colour-coded all Structy's DSA topics (arrays, dynamic programming, etc.), and used cell-formatting to apply a heatmap to low-confidence questions, and questions that haven't been completed in a while.

![png](/assets/posts/algos_studying/.png)

I created some charts showing the number of questions completed, and my overall confidence in each subject.
I also used a couple pivot tables to show this data over time as well.
*This* is pretty much a proof-of-principle concept for what all online algo courses and question banks should provide.
The course by itself was great but its effectiveness was magnified 10x by using my dashboard for targeted studying.
Even though I had done all questions at least once, I didn't really become confident or proficient at their underlying concepts until I practiced and repeated them.

![png](/assets/posts/algos_studying/.png)

## General interview thoughts

