### 12/30/2018

Finally! I was able to train a model to predict doubling a number. The big issue that was tripping me up was assigning one weight to each input (for 1000 inputs, 1000 weights). That is obviously not a scalable solution as a model grows to the millions of inputs, and it also intuitively does not make sense. I knew that the model is ultimately supposed to look something like y = 2x + 0.

#### Training
This should hopefully reinforce what I understand of the training loop:

1. Attempt a prediction, get the differences from expected values
2. The cost is the average sum of those squared differences
3. Updates to the weight is the derivative of that cost function with respect to the weight variable. This derived function ultimately includes an X term, which makes sense because the input is directly multiplied by this weight. Note that this value can be huge
4. Update to the bias is the derivative of that cost function with respect to the bias variable. This derived function does NOT include an x term because the bias is an independent scalar. Note that this value can be huge
5. Update the weight and bias with their respective updates multiplied by a learning rate. The right learning rate seems to be critical because if it's too large it will cause the (already large) update values to ultimately bounce up out of the global minimum N-dimensional "bowl". This will be observed by the result of the cost function increasing after a run rather than decreasing. If you do detect an increasing cost (even just after one run), make the learning rate smaller and re-run.

One thing I noticed was that for this particular example, the cost never reduced below a certain threshold, so I had to introduce a fallback "fall out of loop after 10k iterations" variable. I would guess that for this particular problem the threshold was too low, but I'll see if that pattern holds for future problems.