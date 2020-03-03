OnionOrNot
----------

A machine learning model trained to classify news headlines as fake (from the satirical newspaper [The Onion](http://theonion.com)) or real.

The raw dataset used to train this model was provided by Luke Feilberg, and can be found [here](https://github.com/lukefeilberg/onion).

The trained model (model.h5) achieves ~87% accuracy on a test set. You can test it on new headlines by using `python3 predict.py "News Headline"` 

# Example

```
> python3 predict.py "Study: Majority Of Americans Not Prepared For When Sun Engulfs Earth In 7.5 Billion Years"`
[[0.03900201 0.960998  ]]
```
Indicating a 96% probability that the headline is from The Onion.

```
> `python3 predict.py "Apple will pay up to $500 million to settle lawsuit over slowing down older iPhones"`
[[0.7632308  0.23676917]]
```

This headline is from CNN - the model is somewhat confident (76%) that the article is real.
