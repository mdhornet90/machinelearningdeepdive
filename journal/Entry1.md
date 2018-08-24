### 12/28/2017
It took quite a few false starts, but I'm finally back to basics. In other words, I created a linear regression machine that can judge the input set with 70% accuracy in about 16 seconds, up to 72% after about 2 minutes.

This is important because linear regression is the bedrock for all of the other techniques, and a specific variant of the more general concept of a neural net. I'm hoping that I can trivially extend the code I already have to handle `count(hidden_nodes) > 0`