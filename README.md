
# The Layman's AI Framework

Artificial Intelligence is a subject of rapid development and learning it with interest and understanding is crucial. But, due to complexity and steep learning curve of this subject makes it difficult for begginers to learn it, especially school students.

I believe that A.I is a practical subject and cannot be taught through definitions only or gamified tools which do provide an overview of how A.I works but not the actual picture.

**The Layman's AI Framework** is a framework designed specially for high school students to create A.I models following basic definitions.

```
What is a neural network ?

> A neural network is structure containing different types of layers namely input, hidden and output layer. These layer contains nodes or neurons that are interconnected to the neuron of the next layer. 
```

This is the basic definition of a **Neural Network**. And based on this definition you can create a neural network using this framework

```
from layman import Network

net = Network()
net.create_layer(64, "input")
net.create_layer(32, "hidden")
net.create_layer(10, "output")
net.adjust_network()
```

DONE !! Nothing fancy nor advance math. 

Along with this the framework comes with an interactive software visualising this whole process from network creation to each training stage.
