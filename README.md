# nams
Interpreting Neural Min-Sum Decoders : [Manuscript](https://drive.google.com/file/d/1Dn2Nb5ohBhQowof_sSdoMs4E5L9wBnIr/view?usp=share_link)

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

