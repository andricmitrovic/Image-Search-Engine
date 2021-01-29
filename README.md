# Image-Search-Engine
Image Search Engine implementation in PyTorch from scratch.<br><br>

# Training
To train our model run `main.py`.
To change training parameters, change argumuments in constructor of class Utils when instancing an object in `main.py`. </br></br>

<table>
    <tr>
        <th>Parameter</th>
        <th>Default value</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>EPOCHS</td>
        <td>500</td>
        <td>number of epochs for training</td>
    </tr>
    <tr>
        <td>batchSize</td>
        <td>256</td>
        <td>size of batch during training</td>
    </tr>
    <tr>
        <td>learning_rate</td>
        <td>learning rate for training</td>
        <td>0.0005</td>
    </tr>
    <tr>
        <td>optim</td>
        <td>Adam</td>
        <td>optimizer used for training (SGD or Adam)</td>
    </tr>
    <tr>
        <td>weightDecay</td>
        <td>None</td>
        <td>used with SGD optimizer</td>
    </tr>
    <tr>
        <td>momentum</td>
        <td>None</td>
        <td>used with SGD optimizer</td>
    </tr>
    <tr>
        <td>lastLayerActivation</td>
        <td>Sigmoid</td>
        <td>last layer activation function</td>
    </tr>
</table></br> 


# Testing

To test our model run `usemodel.py`, and set `path` parameter to where the model is saved.<br><br>

# Implementation 

Refer to [Image-Search-Engine.pdf](Image-Search-Engine.pdf)  