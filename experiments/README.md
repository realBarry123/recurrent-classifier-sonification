# Experiments
A series of informal experiments to test the robustness of the sonification method on models with different variables. 
## Noise
### Method
I train instances of the model on image datasets with added Gaussian noise. The model is instantiated as: 
```py
model = RClassifier(
    t=16, 
    z_size=33, 
    conv_channels=2, 
    activation="softsign"
)
```
Each model is trained for 8 epochs using the Adam optimizer with `lr=0.0005`. Noise is added to the FashionMNIST dataset with the following torchvision transform: 

```py
transforms_v2.GaussianNoise(mean=0.0, sigma=std, clip=True)
```

I compare the sonifications of models with `std = 0`, `0.1`, and `0.7` respectively. I sonify the hidden layer `z` (1 second per iteration) with linear stereo panning and no interpolation. 

### Results

