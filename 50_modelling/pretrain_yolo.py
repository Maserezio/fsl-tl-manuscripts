# pretrain_and_finetune_yolov11.py
# Full Python Script: Pretraining YOLOv11 with DINOv3 using LightlyTrain + Fine-tuning

# Step 0: Install required packages before running:
# pip install lightly-train ultralytics

import lightly_train
from ultralytics import YOLO

def pretrain_yolov11_with_dinov3(unlabeled_data_dir, output_dir):
    """
    Pretrain YOLOv11 on unlabeled data using DINOv3 backbone via LightlyTrain.
    
    Parameters:
        unlabeled_data_dir (str): Path to unlabeled images directory
        output_dir (str): Directory where pretraining outputs (checkpoints/logs) are saved
    """
    print(f"Starting self-supervised pretraining for YOLOv11 with DINOv3 on data: {unlabeled_data_dir}")
    
    lightly_train.pretrain(
        out=output_dir,                        # Output directory for checkpoints and logs
        data=unlabeled_data_dir,               # Path to unlabeled image directory
        model="ultralytics/yolov8n.yaml",     # YOLOv11 model (YOLOv8 used as placeholder)
        epochs=5,
        batch_size=16,
        overwrite=True
    )
    
    print(f"Pretraining completed. Outputs saved to: {output_dir}")


def fine_tune_yolov11(pretrained_weights_path, labeled_train_data_path, labeled_val_data_path, epochs=50):
    """
    Fine-tune the pretrained YOLOv11 model on labeled data.
    
    Parameters:
        pretrained_weights_path (str): Path to YOLOv11 pretrained weights from LightlyTrain
        labeled_train_data_path (str): Path or dataset config for supervised training dataset
        labeled_val_data_path (str): Path or dataset config for supervised validation dataset
        epochs (int): Number of fine-tuning epochs
    """
    print(f"Loading pretrained model from {pretrained_weights_path} for fine-tuning.")
    
    model = YOLO(pretrained_weights_path)
    
    print(f"Starting fine-tuning for {epochs} epochs on labeled dataset.")
    model.train(
        data={
            'train': labeled_train_data_path,
            'val': labeled_val_data_path,
            'nc': 80,       # number of classes, adjust if needed
            'names': []     # JSON list of class names, optional
        },
        epochs=epochs
    )
    
    print("Fine-tuning complete. Model checkpoint and results are saved by Ultralytics YOLO automatically.")


if __name__ == "__main__":
    # Paths
    unlabeled_data_dir = "../../../cbad_prepr/"
    pretrain_output_dir = "../runs/pretrain/"
    
    # Step 1: Pretrain YOLOv11 with self-supervised DINOv3 backbone
    pretrain_yolov11_with_dinov3(unlabeled_data_dir, pretrain_output_dir)
    
    # Step 2: Fine-tune with labeled data (uncomment and set paths if available)
    # labeled_train_data = "path/to/labeled/train/dataset"
    # labeled_val_data = "path/to/labeled/val/dataset"
    # pretrained_weights_file = f"{pretrain_output_dir}/exported_models/exported_last.pt"
    # fine_tune_yolov11(pretrained_weights_file, labeled_train_data, labeled_val_data, epochs=50)
