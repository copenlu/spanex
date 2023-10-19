We want to make sure that the model results can be somewhat re-produced in our dataset. We should have _almost_ the same performance **on the entire dataset** when the models are tested through [train.py](../../explain_interactions/models/train.py) and [predict-nli-hf.py](../../explain_interactions/scripts/predict-nli-hf.py).

Note that to keep things simple, the models are trained and tested on data points which when tokenized has a length <= 512.

Here are the performance comparisons (all f1 scores)

#### SNLI


|                    |original|full data|annotation dataset|
|--------------------|--------|---------|------------------|
| bert-base-cased    |NA|90.19|90.91|
| bert-base-uncased  |NA|90.11|91.16|
| bert-large-cased   |NA|91.16|91.86|
| bert-large-uncased |NA|91.26|92.48|
| roberta-base       |NA|89.86|90.65|
| roberta-large      |NA|89.83|90.55|

#### Fever

|               |original|full data| annotation dataset |
|---------------|--------|---------|--------------------|
|bert-base-cased|86.21|86.18| **68.88**          |
|bert-base-uncased|86.00|85.95| 83.13              |
|bert-large-cased|87.19|87.20| **46.79**          |
|bert-large-uncased|87.56|**83.00**| 83.00              |
|roberta-base|88.23|**83.43**| 81.13              |
|roberta-large|88.55|87.87| 85.99              |


