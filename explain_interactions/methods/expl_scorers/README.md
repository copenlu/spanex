A scorer class has this structure:

```python
class Scorer:
    """
    Given an input and an Explanation, produce scores.
    """

    def __init__(self, handler: Handler, **kwargs):
        self.handler = handler
        self.tokenizer = self.handler.tokenizer

    def run(self, instance_explanations: List[Tuple[Instance, InstanceExplanation]]) -> List[InstanceExplanation]:
        """
        produce scores from
        :param instance_explanations:
        :return:
        """
        pass
```

The goal of a scorer method is to take a list of (instance, instance_explanation) tuple and produce a list of instance_explanation objects. It takes a handler during init so that it can process these instances as it sees fit.    