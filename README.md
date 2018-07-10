# A Machine Learning Deepdive

I'm following along with the [deeplearning.ai Courses on Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/home/info), but I'm worried that I won't learn enough by just filling in the blanks in a Jupyter Notebook.

To make sure I actually understand the fundamentals, I'll be creating iterations of the same Neural Network, first starting from scratch for all mathematical operations, and only after that will I incorporate Numpy to simplify the code and improve performance.

For the likely one other person who's ever seen my personal GitHub repo, you'll be able to follow along and see how the project evolves. I might also incorporate some kind of deployment tool learning at the same time so you can see performance metrics across the different versions, stuff like that. We'll seeeeee!

## Journal

I like to take notes as I reason through programming exercises, so I'll keep a reverse chronological journal here.

#### Entry 2, 7/9/2018
Christmas vacation is a great time to play around with fun stuff but life certainly has a way of happening. For the first time in a while I actually feel like taking this on again, so I'm blowing away everything I've written and starting fresh. While Coursera classes are helpful in introducing the concepts, they babysit you way too much with pre-written code. I need to get a feel for coding like a data scientist from the ground up.

In that spirit I'm also an engineer, and tested code is important to me. I'm going to try to figure out ways to TDD this code if at all possible. I expect when trying to figure out the shape of the problem it's going to be a crap shoot, but as I get more of an intuition for how to write this kind of code, finding the seams for testing will get easier.

#### Entry 1, 12/28/2017
It took quite a few false starts, but I'm finally back to basics. In other words, I created a linear regression machine that can judge the input set with 70% accuracy in about 16 seconds, up to 72% after about 2 minutes.

This is important because linear regression is the bedrock for all of the other techniques, and a specific variant of the more general concept of a neural net. I'm hoping that I can trivially extend the code I already have to handle `count(hidden_nodes) > 0`