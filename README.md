# SAGEPHOS: SAGE BIO-COUPLED AND AUGMENTED FUSION FOR PHOSPHORYLATION SITE DETECTION

This repository contains the open-source implementation of the paper "[SAGEPHOS: SAGE BIO-COUPLED AND AUGMENTED FUSION FOR PHOSPHORYLATION SITE DETECTION]([SAGEPhos: Sage Bio-Coupled and Augmented Fusion for Phosphorylation Site Detection | OpenReview](https://openreview.net/forum?id=hLwcNSFhC2))". SAGEPhos introduces a Bio-Coupled Modal Fusion method, distilling essential kinase sequence information to refine task-oriented local substrate feature space, creating a shared semantic space that captures crucial kinase-substrate interaction patterns. Within the substrate’s intra-modality domain, it focuses on Bio-Augmented Fusion, emphasizing 2D local sequence information while selectively incorporating 3D spatial information from predicted structures to complement the sequence space.
![image-20250204112156169](/Users/zhangjingjie/Library/Application Support/typora-user-images/image-20250204112156169.png)

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/ZhangJJ26/SAGEPhos.git
   cd SAGEPhos
   ```

2. Install dependencies:

   (1) You can directly use the `environment.yml` file to set up the environment:

   ```bash
   conda env create -f environment.yml
   conda activate SAGEPhos
   ```

   (2) Alternatively, you can manually install the environment using the following commands:

   ```bash
   conda create -n SAGEPhos python=3.8
   conda activate SAGEPhos
   conda install torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # Here, CUDA 12.1 is used as an example. Please install the corresponding version of PyTorch based on your specific CUDA version.
   conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg
   conda install easydict pyyaml -c conda-forge
   conda install scipy joblib wandb
   pip install git+https://github.com/facebookresearch/esm.git
   pip install ninja rdkit-pypi scikit-learn h5py atom3d
   ```

3. Download the pretrained model:

   We provide a pretrained model for SAGEPhos. [Download it here](https://github.com/ZhangJJ26/SAGEPhos/releases/tag/v1.0.0) and place it in the `checkpoint` directory.

   ```sh
   mkdir checkpoint
   # After downloading
   tar -zcvf checkpoint.tar.gz
   # Move checkpoint.pth to the checkpoint directory.
   ```

## Usage

### Training

By default, we use 2 NVIDIA A40 GPUs for training. You can adjust the batch size according to your GPU memory.

```sh
python script/downstream.py -c config/phos/esm_gearnet_parallel.yaml --ckpt null
```

- `--ckpt`: Specify the path to the model checkpoint. For training, `ckpt` can be set to `null`.

### Testing

For testing, you can run the following command:

```sh
python script/downstream.py -c config/phos/esm_gearnet_parallel.yaml --ckpt ${checkpoint_path}
```

- `--ckpt`: Specify the path to the model checkpoint (default is `checkpoint/checkpoint.pth`).

## Contact

If you have any questions, please feel free to contact the authors.

- Jingjie Zhang (1155224008@link.cuhk.edu.hk)
