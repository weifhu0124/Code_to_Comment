# Code_to_Comment
We proposed a Sequence to Sequence model with attention that generates comments given Python code.  

## IMPORTANT
The model we proposed does not show promising performance. A more complicated model is needed for such task.

## Model Diagram
We use a standard Seq2Seq model with attention in the middle.
![Alt text](./imgs/model.png?raw=true "Model")

The code is transfered using Structure-based traversal on Abstract Syntax Trees. The idea is taken from: https://xin-xia.github.io/publication/icpc182.pdf

## Performance
The first table shows the cross entropy loss on validation set for Seq2Seq model with vs without attention.
![Alt text](./imgs/result1.png?raw=true "Result 1")

The second table shows some example of generated comments.
![Alt text](./imgs/result2.png?raw=true "Result 2")
