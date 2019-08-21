# uTensor is misclassifying...

Here is an example of the same classification with both TF and uTensor. uTensor misclassifies this significantly for the first example...

We're using [trained.pb](trained.pb) here. It's a simple MLP.

## Classify with TensorFlow

```
$ python3 classify.py raw_data.txt

[
    [0.1487136036157608,0.07502711564302444,0.07752762734889984,0.6987316608428955],
    [0.9997518658638,9.205692913383245e-05,7.743923197267577e-05,7.861117774154991e-05]]
```

## Classify with uTensor

Turn into uTensor model:

```
$ utensor-cli convert trained.pb --output-nodes=y_pred/Softmax -m fw/utensor-model
```

Compile and run the `fw` folder:

```
Raw data:
output_neurons 4
0: 0.532946
1: 0.180336
2: 0.112851
3: 0.173868
Empty data:
output_neurons 4
0: 0.999745
1: 0.000097
2: 0.000077
3: 0.000080
```

## Conclusion

So the second one (all empty) looks very similar. The first one is completely wrong.