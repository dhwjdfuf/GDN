# GDN

Code implementation for : [Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)

I studied the code at https://github.com/d-ailin/GDN and tried to improve it myself.

Trained model parameters will be saved at 'model.pt'

This code is for cpu.

You can set hyperparameters by modifying config={} at line 94 of main.py.

# how to run, at anaconda

conda create --name pyg python=3.6

conda activate pyg

conda install pytorch==1.5.1 cudatoolkit=10.2 -c pytorch

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html

pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html

pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html

pip install torch-geometric==1.5.0

pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html


python main.py


