import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import copy
import sys
import datetime
import shutil  

RESULTS_DIR = "results_phase2"
os.makedirs(RESULTS_DIR, exist_ok=True)

class Logger:
    def __init__(self, num_labeled_samples):
        self.terminal = sys.stdout
        
        self.filename = os.path.join(RESULTS_DIR, f"results_{num_labeled_samples}_samples.txt")
        
        self.log_file = open(self.filename, 'w')
        print(f"Logging results to: {self.filename}")
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()

def analyze_labels(csv_file):
    print("\n=== Analyzing Label Distribution (Including 'Unknown') ===")
    
    try:
        df = pd.read_csv(csv_file, header=0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        print("Attempting to fix CSV format...")
        try:
            df = pd.read_csv(csv_file, header=None)
            if len(df.columns) == 1 and ',' in df.iloc[0, 0]:
                df = pd.DataFrame([x.split(',', 1) for x in df[0].tolist()], 
                                columns=['image_path', 'label'])
            else:
                df = df.iloc[:, :2]
                df.columns = ['image_path', 'label']
        except Exception as e2:
            print(f"Still having issues with CSV: {e2}")
            print("Please check the format of your CSV file.")
            raise
    
    print("First few rows of the CSV:")
    print(df.head())
    
    total_labeled = len(df)
    print(f"Total number of labeled samples (including 'unknown'): {total_labeled}")
    
    all_labeled_paths = set(df['image_path'].apply(os.path.basename))

    label_counts = Counter(df['label'])
    print("\nLabel Distribution (Including 'Unknown'):")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} images ({count/len(df)*100:.1f}%)")
    
    plt.figure(figsize=(12, 6))
    labels, counts = zip(*label_counts.most_common())
    plt.bar(labels, counts)
    plt.title('Label Distribution (Including "Unknown")')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_filename = os.path.join(RESULTS_DIR, f"label_distribution_{total_labeled}_samples.png")
    plt.savefig(plot_filename)
    print(f"\nLabel distribution plot saved as '{plot_filename}'")
    
    return df, all_labeled_paths, total_labeled

class LabeledImageDataset(Dataset):
    def __init__(self, dataframe, source_dir, transform=None):
        self.dataframe = dataframe
        self.source_dir = source_dir
        self.transform = transform
        
        unique_labels = sorted(dataframe['label'].unique())
        self.class_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.classes = unique_labels
        
        print(f"Dataset created with {len(dataframe)} images and {len(unique_labels)} classes")
        print(f"Class mapping: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]['image_path']
        label_name = self.dataframe.iloc[idx]['label']
        label = self.class_to_idx[label_name]
        
        img_path = os.path.join(self.source_dir, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            blank = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), color='black')
            return blank, label

def prepare_data_for_training(df, source_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    train_dfs = []
    val_dfs = []
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        n_samples = len(label_df)
        
        if n_samples == 0:
            continue
            
        label_df = label_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_size = max(1, int(0.8 * n_samples))  
        
        train_dfs.append(label_df.iloc[:train_size])
        
        if train_size < n_samples:
            val_dfs.append(label_df.iloc[train_size:])
        else:
            val_dfs.append(label_df.iloc[:1])
    
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    train_label_counts = Counter(train_df['label'])
    print("\nTraining set label distribution:")
    for label, count in train_label_counts.most_common():
        print(f"  {label}: {count} images")
    
    train_dataset = LabeledImageDataset(train_df, source_dir, transform=data_transforms['train'])
    val_dataset = LabeledImageDataset(val_df, source_dir, transform=data_transforms['val'])

    return train_dataset, val_dataset

def train_model(train_dataset, val_dataset, total_labeled, num_epochs=15):
    print("\n=== Training Model ===")
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"Class names: {class_names}")
    print(f"Number of classes (including 'unknown'): {num_classes}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    def train_model_internal(model, criterion, optimizer, scheduler, num_epochs=15):
        since = time.time()
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        print("\nTraining Progress:")
        print("-" * 70)
        print(f"{'Epoch':^10}|{'Train Loss':^15}|{'Train Acc':^15}|{'Val Loss':^15}|{'Val Acc':^15}")
        print("-" * 70)
        
        for epoch in range(num_epochs):
            epoch_results = {"train": {}, "val": {}}
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() 
                else:
                    model.eval() 
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                epoch_results[phase]["loss"] = epoch_loss
                epoch_results[phase]["acc"] = epoch_acc
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            
            print(f"{epoch+1:^10}|{epoch_results['train']['loss']:^15.4f}|"
                  f"{epoch_results['train']['acc']:^15.4f}|{epoch_results['val']['loss']:^15.4f}|"
                  f"{epoch_results['val']['acc']:^15.4f}")
        
        print("-" * 70)
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        model.load_state_dict(best_model_wts)
        return model, best_acc
    
    model, best_val_acc = train_model_internal(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    model_filename = os.path.join(RESULTS_DIR, f"model_{total_labeled}_samples.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")
    
    return model, class_names, val_dataset.transform, best_val_acc

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            blank = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), color='black')
            return blank, img_path

def find_unlabeled_images(source_dir, labeled_paths):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    unlabeled_images = []
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(root, filename)
                if os.path.basename(full_path) not in labeled_paths:
                    unlabeled_images.append(full_path)
    
    return unlabeled_images

def perform_inference(model, class_names, transform, source_dir, labeled_paths, total_labeled):
    print("\n=== Performing Inference on Unlabeled Images ===")
    
    unlabeled_images = find_unlabeled_images(source_dir, labeled_paths)
    
    print(f"Found {len(unlabeled_images)} unlabeled images for inference")
    
    if not unlabeled_images:
        print("No unlabeled images found. Skipping inference.")
        return pd.DataFrame(), 0
    
    unlabeled_dataset = UnlabeledImageDataset(unlabeled_images, transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    
    with torch.no_grad():
        for inputs, paths in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_scores, preds = torch.max(probabilities, 1)
            
            for i in range(len(paths)):
                results.append({
                    'image_path': os.path.basename(paths[i]),
                    'predicted_class': class_names[preds[i]],
                    'confidence': confidence_scores[i].item(),
                    'full_path': paths[i]  
                })

    results_df = pd.DataFrame(results)
    avg_confidence = results_df['confidence'].mean() if not results_df.empty else 0
    print(f"Average confidence on unlabeled images: {avg_confidence:.4f}")

    confidence_file = os.path.join(RESULTS_DIR, 'confidence_tracking.txt')
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(confidence_file, 'a') as f:
        f.write(f"{timestamp}, {total_labeled} samples, {len(results_df)} unlabeled, Avg confidence: {avg_confidence:.4f}\n")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_predictions_csv = os.path.join(RESULTS_DIR, f"all_predictions_{total_labeled}_samples.csv")
    results_df.to_csv(all_predictions_csv, index=False)

    high_conf_df = results_df[results_df['confidence'] >= 0.85].copy()
    print(f"Adding {len(high_conf_df)} high-confidence predictions to auto-labeled set")

    if not high_conf_df.empty:
        auto_label_csv = os.path.join(RESULTS_DIR, "auto_labeled.csv")
        high_conf_df = high_conf_df[['image_path', 'predicted_class']].rename(
            columns={'predicted_class': 'label'}
        )

        if os.path.exists(auto_label_csv):
            high_conf_df.to_csv(auto_label_csv, mode='a', index=False, header=False)
        else:
            high_conf_df.to_csv(auto_label_csv, index=False, header=True)

    return results_df, avg_confidence

def empty_directory(directory_path):
    if os.path.exists(directory_path):
        print(f"Emptying directory: {directory_path}")
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)

def extract_samples_by_confidence(results_df, total_labeled, most_confident_threshold=0.85, least_confident_count=100):
    print("\n=== Extracting Samples by Confidence ===")
    
    if results_df.empty:
        print("No prediction results available. Skipping extraction.")
        return None, None, None, None

    most_confident = results_df[
        (results_df['confidence'] >= most_confident_threshold) &
        (results_df['predicted_class'].str.lower() != 'unknown')
    ].copy()

    print(f"\nFound {len(most_confident)} confident (≥ {most_confident_threshold:.2f}) non-'unknown' predictions.")

    confidence_check_dir = os.path.join(RESULTS_DIR, "most_confident_by_class")
    empty_directory(confidence_check_dir)

    copied_confident_count = 0
    for _, row in most_confident.iterrows():
        label = row['predicted_class']
        label_dir = os.path.join(confidence_check_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        src_path = row.get('full_path') or os.path.join('./data/selected_images', row['image_path'])
        dst_path = os.path.join(label_dir, os.path.basename(src_path))

        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                copied_confident_count += 1
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")
        else:
            print(f"Source not found: {src_path}")

    print(f"Copied {copied_confident_count} confident samples to '{confidence_check_dir}'")

    confident_csv = os.path.join(RESULTS_DIR, f"most_confident_{total_labeled}_samples.csv")
    most_confident.to_csv(confident_csv, index=False)
    print(f"Saved confident sample list to '{confident_csv}'")

    sorted_df = results_df.sort_values('confidence', ascending=True)
    least_confident = sorted_df.head(min(least_confident_count, len(sorted_df)))

    least_confident_dir = os.path.join(RESULTS_DIR, "samples_to_label")
    empty_directory(least_confident_dir)

    copied_count = 0
    for _, row in least_confident.iterrows():
        src_path = row.get('full_path') or os.path.join('./data/selected_images', row['image_path'])
        dst_path = os.path.join(least_confident_dir, os.path.basename(src_path))

        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        else:
            print(f"Source file not found: {src_path}")

    print(f"Copied {copied_count} least confident samples to '{least_confident_dir}'")

    return most_confident, least_confident, least_confident_dir, copied_count

def main(use_auto_labeled=False, add_confident_to_training=False):
    labels_csv = 'filtered_output_labels.csv'
    source_image_dir = './data/selected_images'
    
    confidence_file = os.path.join(RESULTS_DIR, 'confidence_tracking.txt')
    if not os.path.exists(confidence_file):
        with open(confidence_file, 'w') as f:
            f.write("Timestamp, Total Labeled, Unlabeled Images, Average Confidence\n")
    
    df, labeled_paths, total_labeled = analyze_labels(labels_csv)

    if use_auto_labeled:
        auto_label_csv = os.path.join(RESULTS_DIR, "auto_labeled.csv")
        if os.path.exists(auto_label_csv):
            print("\n=== Including Auto-Labeled Samples ===")
            auto_df = pd.read_csv(auto_label_csv)
            auto_df = auto_df.dropna(subset=['image_path', 'label'])
            print(f"Loaded {len(auto_df)} auto-labeled samples")
            auto_df = auto_df[~auto_df['image_path'].isin(df['image_path'])]
            df = pd.concat([df, auto_df], ignore_index=True)
            labeled_paths.update(auto_df['image_path'].apply(os.path.basename).tolist())
            total_labeled = len(df)
            print(f"New total labeled samples (with auto-labeled): {total_labeled}")

    if add_confident_to_training:
        print("\n=== Including Confident Samples from Folder ===")
        confident_dir = os.path.join(RESULTS_DIR, "most_confident_by_class")
        new_rows = []

        if os.path.isdir(confident_dir):
            for label in os.listdir(confident_dir):
                label_path = os.path.join(confident_dir, label)
                if not os.path.isdir(label_path):
                    continue
                for filename in os.listdir(label_path):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                        continue
                    if filename not in labeled_paths:
                        new_rows.append((filename, label))

            if new_rows:
                confident_df = pd.DataFrame(new_rows, columns=["image_path", "label"])
                print(f"Adding {len(confident_df)} confident samples to training set from folder structure.")
                df = pd.concat([df, confident_df], ignore_index=True)
                labeled_paths.update(confident_df['image_path'].tolist())

                with open('filtered_output_labels.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    for row in new_rows:
                        writer.writerow(row)

                total_labeled = len(df)
            else:
                print("No new confident samples found in folder.")
        else:
            print(f"Folder '{confident_dir}' not found. Skipping confident sample inclusion.")

    sys.stdout = Logger(total_labeled)
    print(f"\n=== Starting Transfer Learning Pipeline with {total_labeled} labeled samples (Including 'Unknown') ===")

    train_dataset, val_dataset = prepare_data_for_training(df, source_image_dir)

    model, class_names, transform, best_val_acc = train_model(train_dataset, val_dataset, total_labeled)

    results_df, avg_confidence = perform_inference(model, class_names, transform, source_image_dir, labeled_paths, total_labeled)

    most_confident, least_confident, least_confident_dir, _ = extract_samples_by_confidence(results_df, total_labeled)

    if add_confident_to_training and most_confident is not None:
        print("\n=== Including Confident Predictions in Training Set ===")
        confident_df = most_confident[['image_path', 'predicted_class']].rename(columns={'predicted_class': 'label'})
        confident_df = confident_df.dropna(subset=['image_path', 'label'])
        confident_df = confident_df[~confident_df['label'].str.lower().eq('unknown')]
        confident_df = confident_df[~confident_df['image_path'].isin(df['image_path'])]
        df = pd.concat([df, confident_df], ignore_index=True)
        labeled_paths.update(confident_df['image_path'].apply(os.path.basename).tolist())
        print(f"Added {len(confident_df)} confident samples to labeled set.")
    
    print("\n=== Process Complete ===")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"All results saved in '{RESULTS_DIR}'")

    if least_confident_dir:
        print(f"Manually label images in: '{least_confident_dir}'")
        print(f"Average confidence on unlabeled images: {avg_confidence:.4f}")
    else:
        print("No unlabeled images were found for inference.")

    print("\n=== Saving Final Model and Data ===")
    
    final_model_path = os.path.join(RESULTS_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    final_data_path = os.path.join(RESULTS_DIR, "final_labeled_data.csv")
    df.to_csv(final_data_path, index=False)
    print(f"Final labeled dataset saved to: {final_data_path}")

    final_predictions_path = os.path.join(RESULTS_DIR, "final_predictions.csv")
    results_df.to_csv(final_predictions_path, index=False)
    print(f"Final inference predictions saved to: {final_predictions_path}")

    if isinstance(sys.stdout, Logger):
        log_filename = sys.stdout.filename
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        print(f"Results logged to: {log_filename}")
        print(f"Confidence history tracked in '{confidence_file}'")


if __name__ == "__main__":
    use_auto = "--include" in sys.argv
    add_confident = "--add-confident" in sys.argv
    main(use_auto_labeled=use_auto, add_confident_to_training=add_confident)
