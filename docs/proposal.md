---
layout: default
title:  Proposal
---

## Summary of the Project
---
In our game, the agent will spawn in a world with a floor made of different colors of wool. The user is able to type in commands (e.g. “go to the purple block”). The goal of our agent is to navigate the game world and complete the requests as quickly as possible without touching any incorrect color blocks. For a challenge, the agent may come across mobs and the user can incorporate them into the commands (e.g. “hit the pig, feed the dog”). The input will be a text command in the Minecraft textbox, and the output will result in our agent executing all the instructions as efficiently as possible. Possible third party applications that will be used in this project include PyTorch and Tensorflow for NLP.

## AI/ML Algorithms
---
We haven’t decided exactly which natural language processing framework to use but PyTorch or TensorFlow are very likely to be used because of their ability to tokenize, parse, and tag text input, and we will utilize reinforcement learning through q-learning for to get the agent to complete the task as quickly as possible.

## Evaluation Plan
---

### Quantitative Analysis
Two metrics we will utilize are the average success rate on a given map and the command completion time, which will depend on the following two tasks. First, the program must be able to process the natural language commands given by the user. Our metric will be the average rate of success in translating natural language to a list of commands, executable by the agent. Our baseline is being able to understand key, single word commands (like “purple”, “pig”, “move”). In the beginning, we expect to be able to only process those words, but by the end we should be able to ask simple sentence queries, such as “go to the purple block.” The data we will evaluate on will be lists of commands, sorted in varying complexities. Second, is the agent’s actual execution of the commands and navigation around the map. Again, we will measure the success rate of each command on different maps as well as the time to complete each. Our baseline will be the success rate and time to complete a greedy approach to traversing the map and reaching the goal. Our approach should improve this significantly as the agent informs itself about the map, hopefully improving time while maintaining a high success rate. We will evaluate this on a set of predetermined maps that the agent will run on.

### Qualitative Analysis
The NLP sanity cases will look similar to the baseline: it must be able to translate single words into executable commands. For the agent execution sanity cases, it must be able to successfully navigate maps with only the correct color block and small maps with only 4 blocks (with different colors). To visualize our algorithms, we will create flow diagrams to map out the logic and use debugging tools to track the output. Our moonshot case would be if the agent we train could on average perform better than a human attempting that same challenge.

![Proposal Diagram](proposal_img/diagram1.jpg "Diagram for Proposal" | width = 256)

## Appointment with the Instructor
---
1/21/2021 @ 12:15pm

















