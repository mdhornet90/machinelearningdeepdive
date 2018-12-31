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

#### Predicting with Test Data
Predicting itself was pretty straightforward: use the weight and the bias that was derived from the training.

The more interesting part was using R^2 to see how the predictions are correlated. I'm not sure I understand all of it fully, but it requires a couple of parts:

- Start by taking the mean of all expected outputs of the test data
- For each expected test output, get the difference of the output against the mean and square it. Sum all of those values together.
- Separately, get the residual sum of squares of the predicted outputs against the expected outputs.
- Divide the RSS by that total sum of squares, subtract that quotient from 1. The closer to 1 that value is (and hence the closer that quotient is to 0), the more is explained by the model.

If I had to interpret this subjectively, I would guess that the total sum of squares is establishing a baseline for how much expected variance there is between all of the outputs - all of those outputs are independent from one another, so it's supposed to be the background noise inherent in the problem?

In general, the more important value is the RSS, from what I've researched. The smaller that number is, the smaller the ratio between it at the total and hence the closer to zero. A large RSS means the predicated values are skewing wildly from the actual values and don't really explain much. This is not to say that a high R^2 value necessarily means that your model is good, just that it fits the test data you've fed it.

#### Next up
I'm going to attempt to flip this somewhat and train it to recognize a constant value added to each number. I would expect the weight to be 1 and the bias to be 10.

Hmm, so it didn't exactly go as planned. I had to turn the learning rate waaaaay up and set the weight to 1. Perhaps these models are too simple to effectively use regression and I should just jump in to something more complex. Another thing to note is that the R^2 value was exceptionally high despite the predictions being way off, so it's indeed true that R^2 is not a reliable indicator of a good fit.