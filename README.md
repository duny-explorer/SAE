# sae_recsys

Sparse autoencoders for interpretation and control of sequential recommenders.

Run streamlit ui
```sh
streamlit run streamlit_ui/main_page.py 
```

Run SAE training
```sh
python train_sae.py --config-name=l1_sae_ml-20m
```

Run transformer training
```sh
python train_transformer.py --config-name=gpt_ml-20m
```

Run data splitting
```sh
python data_split.py --config-name=data_split_ml-20m
```