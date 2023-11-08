# nams
Source code for the ICC'23 paper : [Interpreting Neural Min-Sum Decoders](https://ieeexplore.ieee.org/abstract/document/10279074)

<ins> Abstract</ins>: In decoding linear block codes, it was shown that noticeable reliability gains can be achieved by introducing learnable parameters to the Belief Propagation (BP) decoder. 
Despite the success of these methods, there are two key open problems. The first is the lack of interpretation of the learned weights, and the other is the lack of analysis for non-AWGN channels. 
In this work, we aim to bridge this gap by providing insights into the weights learned and their connection to the structure of the underlying code. 
We show that the weights are heavily influenced by the distribution of short cycles in the code.
We next look at the performance of these decoders in non-AWGN channels, both synthetic and over-the-air channels, and study the complexity vs. performance trade-offs, demonstrating that increasing the number of parameters helps significantly in complex channels. 
Finally, we show that the decoders with learned weights achieve higher reliability than those with weights optimized analytically under the Gaussian approximation. 

This repository includes:
- A PyTorch implementation of augmenting min-sum decoder for linear block codes.
- A framework to generate LTE channel simulations for EPA/EVA/ETU channels

All Python libraries required can be installed using:
```
pip3 install -r requirements.txt
```
## Generating LTE data

To generate the LTE data, run the following command from MATLAB terminal:
```
generate_lte_data
```

## Training Neural Min-Sum decoder

Models are saved at data_files/saved_models. To train the model from scratch, run the following command, after setting appropriate parameters:
```
python nams_train.py
```

## Testing Neural Min-Sum decoder

Results are saved at data_files/ber_data. To test the model using saved weights, run the following command, after setting appropriate parameters:
```
python nams_test.py
```

