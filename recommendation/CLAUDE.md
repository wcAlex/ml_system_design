# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository contains multiple different recommendation systems, so please create a sub directory for each recommendation project.

## Project

This repository contains system design work for several ML systems. For example, mini-YouTube video recommendation systems, event recommendation systems, video / image search system, Harmful content detection system, 

I'm learning Machine learning system design on recommendation systems, I want to learn the state of art design, key techinical insights, thought process, trade-off and model choices through building this project.

Also, I need to prepare ML system design interview, please also make sure the design and code covers the most frequent asked questions and key points (such as bottleneck, technical challenges, maintain challenges, scaling challanges) in the ML system interviews on recommendation topics.  

Make sure each project has following components:
1. Clear ML objectives, 
2. System input & output, 
3. Model choices (and why), explain each model algorithm and architecture
4. Data preparation (engineering), 
5. Feature engineering,  
6. Model development, (at least two good options)
7. Evaluation 
8. Model Hosting / Inference
9. End-to-End systems (mini version)
10. Challenges and talking points (common failing points)
11. Summary for the overall project (architecture, components, data flow, scaling, etc.)
12. Scaling, both at model size and inferece or data process

It will be great you can create separate doc for each section above.

Please make this an iterative process, prompt/ask me on important questions, so I could also learn from the design process by knowing what kind key questions I could ask in the interview.

Start from first principles and iterate as needed.

create virtual environment for each project.

About model choice, mention LLM model approach if it's popular and effective

For each pahse/components, list at least 2 or 3 options with trade off, so I'm not only get the state-of-the-art approach but also understand the thinking behind, then I can reason during interview. 

If there are similiar example/approach from Google, Youtube, Amazon, prioritize their system approach as samples.

## Current focus

Project One: Build a video recommendation system (like YouTube), objective is to increase user engagement. 
all content is in "video_recommendation" folder



