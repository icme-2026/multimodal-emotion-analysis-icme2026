# C-Mani: Conflict-aware Semi-supervised Multimodal Sentiment Analysis against Pseudo-label Sycophancy
This is the anonymous implementation of our paper "C-Mani: Conflict-aware Semi-supervised Multimodal Sentiment Analysis against Pseudo-label Sycophancy" (Paper under review for ICME 2026).

## Abstract
Multimodal Sentiment Analysis (MSA) faces challenges in label-scarce scenarios, where existing semi-supervised methods easily suffer from "pseudo-label sycophancy"—over-relying on high-confidence but incorrect pseudo-labels from one modality on cross-modal conflict samples. We propose C-Mani, a conflict-aware semi-supervised framework that integrates curriculum pseudo-label filtering, cross-modal consistency weighting, and decoupled representation regularization to mitigate this issue. Extensive experiments show C-Mani outperforms state-of-the-art methods on three benchmark datasets, especially in extremely low-label settings.

## Prerequisites 
  - Python 3.8.10                                                                                                                                                 
  - torch 1.11.0+cu113                                                                                                                                          
  - torchvision 0.12.0+cu113                                                                                                                                    
  - torchtext 0.12.0                                                                                                                                            
  - transformers 4.19.2                                                                                                                                         
  - numpy 1.22.4                                                                                                                                                
  - pandas 2.0.3                                                                                                                                                
  - scikit-learn 1.0.2   

### Installation
pip install -r requirements.txt

## Dataset
We evaluate on three mainstream image-text sentiment datasets: MVSA-Single, MVSA-Multiple, and Twitter (merged from Twitter-15/17).


1.For the relabeled MVSA dataset, please request access from the corresponding authors of the original MVSA dataset, and preprocess it according to the format specified in the data directory and train.json template. 
2. For the Twitter datasets, please refer to the original paper. After obtaining them, preprocess the data according to the format in the `data` and `train.json` files of the relabeled MVSA dataset.

## Training (Example: MVSA-Single, 51 labels)
```bash
python main.py \
  --save_dir ./saved_models \
  --save_name mvsa_single_n51 \
  --epoch 160 \
  --num_train_iter 256 \
  --num_labels 51 \
  --uratio 2 \
  --T 0.7 \
  --p_cutoff 0.92 \
  --p_cutoff_end 0.98 \
  --p_rampup_ratio 0.8 \
  --noise_th 0.45 \
  --ulb_loss_ratio 0.06 \
  --dynamic_th 0.9 \
  --threshold 0.97 \
  --label_filter_min 0.92 \
  --gate_clean_start 12 \
  --optim AdamW \
  --lr 3e-5 \
  --warmup_ratio 0.1 \
  --dropout 0.2 \
  --train_data_dir datasets/MVSA_Single \
  --test_data_dir datasets/MVSA_Single \
  --patience 40 \
  --ulb_rampup_ratio 0.8
```

### Key Hyperparameters
- --num_labels: Number of labeled samples (default: 600)
- --lr: Learning rate (default: 5e-5)
- --batch_size: Labeled batch size (unlabeled batch size is 4× labeled batch size, default: 2)
- --threshold: Pseudo-label confidence threshold (default: 0.95)
- --ema_momentum: EMA momentum for teacher model (default: 0.999)
- --num_epochs: Total training epochs (default: 150)


## Notes
- Experiments are conducted on NVIDIA RTX 4090. Reduce --batch_size for GPUs with smaller memory.
- Model weights are saved automatically during training (best performance on validation set).
- Fixed random seeds ensure reproducibility.
