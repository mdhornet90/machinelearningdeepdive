### 1/4/2019

Talked with my buddy Stuart who does this for a living, asked him some ultra-newb questions. I noticed that my quadratic equation wasn't working as I expected, and he pointed out that linear regression can't really express that concept because of the x^2 term. He also pointed me toward some good datasets to use for testing machine learning algorithms, so I think I'm going to use one of those, which is closer to a real world example without being as complicated as image recognition.

I think first I want to build some intuition visually so I can reasonably guess how my model is going to perform when I set it up mathematically

### 1/5/2019

Managed to plot all of the different aspects of the wine compared to its quality. Just by visual inspection, I can see some general trends. By far the strongest correlation visually is the density metric. This is obviously something I can probe directly once I get these models up and running, but a denser wine may have a correlation between residual sugar? Alcohol is less dense than water, but something like simple syrup has a higher density. My hypothesis is that this means the higher the ratio of residual sugar to alcohol (which should tweak the density score up), the better the wine quality. In that vein, I would also hazard a guess that the correlation between acids and RS is pretty high, seeing as too much of either trait makes for some gross wine. Time to figure out how to correlate this data using linear regression...

One thing I'm concerned about is the plotting - visually inspecting the spreadsheet didn't yield as strong a correlation as what I saw in the plots, so maybe I used matplotlib wrong?

Ok, yes I did. Values are showing up all over the place in the list.
Hm, or not. Plotting just that still seems to give me the same values. We'll see what happens...

I clearly don't know how to use matplotlib. I decided to just chart it in apple numbers and the correlations became a lot clearer. There actually seems to be an association with decreasing density and a correlation with increasing alcohol levels, meaning boozier and less sweet is perceived as better quality. Additionally, the level of sulphates appears to be positively correlated. These trends should shake out clearly during linear regression.