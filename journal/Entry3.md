### 7/10/2017
I want to strip down the complexity of all of this stuff and start with a bare minimum example, so my first attempt will _just_ be linear regression against a set of numbers to determine if they're even or odd? Is this an interesting problem? Probably not, but I need to start somewhere to build intution.

Still, looks like I'll have to start from scratch to understand the notation/vocabulary:

- I have a `parity_train` file under `datasets`. It has a number on the left, and its parity on the right. Therefore I have a vector of training examples at left (`x`), and a set of classifications on the right (`y`)

- This is _not_ attempting to classify a cat with a 64x64 picture. This is one number mapped to one result. Therefore this problem doesn't even involve a 2D matrix, it's really just a literal array of training examples. That means the shape of `X` is (`1`, `m`). That means `Y` is also (`1`, `m`) 

- This _also_ means that the weights are no represented by a vector either, but a single number. The shape ought to be (`1`, `2`) (the bias and a single weight)

- I know that the intermediate value for a particular example is z = w<sup>T</sup>x + b. Because w, is a single value, this should reduce down to everyone's favorite algebra equation, z = wx + b (y = mx + b for those somehow slower than I am)

So what's the general algorithm I need to re-learn?

1. Map inputs to an array, `X`.
2. For each `x` in `X`, calculate `z` from the linear eq. Run the result through the sigmoid function.
    - From what I remember of the sigmoid function, it both places a bound on all values passing through it to be between 0 and 1, and it pretty much forces the algorithm to make _some_ kind of choice about the data, since it very rapidly moves to y-hat = [0 or 1] on either side of z = 0.
3. Compare the value calculated to the actual value using the loss function.
4. Average the deviation of the bias + weight together
5. Iterate over the values until the total loss has been minimized to an acceptable degree.