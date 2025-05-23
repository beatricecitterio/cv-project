import os
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
import json

class Logger:
    def __init__(self, filename="results_phase3_log.txt"):
        self.terminal = sys.stdout
        
        os.makedirs("results_phase3", exist_ok=True)
        
        self.filename = os.path.join("results_phase3", filename)
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
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
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
    
    train_dataset = LabeledImageDataset(train_df, source_dir, transform=data_transforms['train'])
    val_dataset = LabeledImageDataset(val_df, source_dir, transform=data_transforms['val'])
    
    return train_dataset, val_dataset, data_transforms

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print("\nTraining Progress:")
    print("-" * 70)
    print(f"{'Epoch':^10}|{'Train Loss':^15}|{'Train Acc':^15}|{'Val Loss':^15}|{'Val Acc':^15}")
    print("-" * 70)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
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
            epoch_results[phase]["acc"] = epoch_acc.item()
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
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
    return model, best_acc, history

def get_model_architectures(num_classes):
    # architecture 1: ResNet18
    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad = False
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, num_classes)
    
    # architecture 2: MobileNetV2
    mobilenet = models.mobilenet_v2(pretrained=True)
    for param in mobilenet.parameters():
        param.requires_grad = False
    num_ftrs = mobilenet.classifier[1].in_features
    mobilenet.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # architecture 3: DenseNet121
    densenet = models.densenet121(pretrained=True)
    for param in densenet.parameters():
        param.requires_grad = False
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, num_classes)
    
    return {
        "resnet18": resnet18,
        "mobilenet_v2": mobilenet,
        "densenet121": densenet
    }

def train_all_models(train_dataset, val_dataset, num_classes):  
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    models_dict = get_model_architectures(num_classes)
    
    num_epochs = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trained_models = {}
    model_accuracies = {}
    all_histories = {}
    
    for model_name, model in models_dict.items():
        print(f"\n=== Training {model_name} ===")
        
        criterion = nn.CrossEntropyLoss()
        
        if model_name == "resnet18":
            optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif model_name == "mobilenet_v2":
            optimizer = optim.SGD(model.classifier[1].parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif model_name == "densenet121":
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        trained_model, best_val_acc, history = train_model(
            model, criterion, optimizer, scheduler, 
            dataloaders, dataset_sizes, num_epochs=num_epochs
        )
        
        trained_models[model_name] = trained_model
        model_accuracies[model_name] = best_val_acc.item()
        all_histories[model_name] = history
        
        model_filename = os.path.join("results_phase3", f"model_{model_name}.pth")
        torch.save(trained_model.state_dict(), model_filename)
        print(f"Model saved as '{model_filename}'")
    
    print("\nModel Validation Accuracies:")
    for model_name, acc in model_accuracies.items():
        print(f"  {model_name}: {acc:.4f}")
    
    return trained_models, train_dataset.classes

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

def find_all_images(source_dir):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    all_images = []
    
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(root, filename)
                all_images.append(full_path)
    
    return all_images

def perform_ensemble_inference(trained_models, class_names, transform, source_dir):
    print("\n=== Performing Ensemble Inference ===")
    
    all_images = find_all_images(source_dir)
    print(f"Found {len(all_images)} images for inference")
    
    if not all_images:
        print("No images found. Skipping inference.")
        return pd.DataFrame()
    
    dataset = UnlabeledImageDataset(all_images, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name, model in trained_models.items():
        model.eval()
        model.to(device)
    
    all_results = {model_name: [] for model_name in trained_models.keys()}
    
    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            
            for model_name, model in trained_models.items():
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence_scores, preds = torch.max(probabilities, 1)
                
                for i in range(len(paths)):
                    all_results[model_name].append({
                        'image_path': paths[i],
                        'predicted_class': class_names[preds[i]],
                        'confidence': confidence_scores[i].item()
                    })
    
    model_dfs = {}
    for model_name, results in all_results.items():
        model_dfs[model_name] = pd.DataFrame(results)
        
        csv_path = os.path.join("results_phase3", f"predictions_{model_name}.csv")
        model_dfs[model_name].to_csv(csv_path, index=False)
        print(f"Saved predictions for {model_name} to {csv_path}")
    
    consensus_results = find_consensus_predictions(model_dfs, class_names)
    
    csv_path = os.path.join("results_phase3", "consensus_predictions.csv")
    consensus_results.to_csv(csv_path, index=False)
    print(f"Saved consensus predictions to {csv_path}")
    
    return consensus_results

def find_consensus_predictions(model_dfs, class_names):
    print("\n=== Finding Consensus Predictions ===")
    
    model_names = list(model_dfs.keys())
    first_model = model_names[0]
    image_paths = model_dfs[first_model]['image_path'].tolist()
    
    consensus_results = []
    
    for path in image_paths:
        predictions = {}
        confidences = {}
        
        for model_name in model_names:
            df = model_dfs[model_name]
            row = df[df['image_path'] == path].iloc[0]
            predictions[model_name] = row['predicted_class']
            confidences[model_name] = row['confidence']
        
        all_preds = list(predictions.values())
        consensus = all(pred == all_preds[0] for pred in all_preds)
        
        if consensus:
            avg_confidence = sum(confidences.values()) / len(confidences)
            consensus_results.append({
                'image_path': path,
                'predicted_class': all_preds[0],
                'average_confidence': avg_confidence,
                'confidence_details': confidences
            })
    
    consensus_df = pd.DataFrame(consensus_results)
    
    if not consensus_df.empty:
        consensus_df['basename'] = consensus_df['image_path'].apply(os.path.basename)
    
    print(f"Found {len(consensus_results)} consensus predictions out of {len(image_paths)} images")
    print(f"Consensus rate: {len(consensus_results) / len(image_paths) * 100:.2f}%")
    
    if not consensus_df.empty:
        consensus_distribution = Counter(consensus_df['predicted_class'])
        print("\nDistribution of Consensus Predictions:")
        for label, count in consensus_distribution.most_common():
            print(f"  {label}: {count} images ({count/len(consensus_df)*100:.1f}%)")
    
    return consensus_df

def copy_consensus_images(consensus_df, dest_dir="results_phase3/consensus_images"):   
    if consensus_df.empty:
        print("No consensus predictions. Skipping copy operation.")
        return
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for class_name in consensus_df['predicted_class'].unique():
        os.makedirs(os.path.join(dest_dir, class_name), exist_ok=True)
    
    copied_count = 0
    for _, row in consensus_df.iterrows():
        src_path = row['image_path']
        if os.path.exists(src_path):
            class_dir = os.path.join(dest_dir, row['predicted_class'])
            dst_path = os.path.join(class_dir, os.path.basename(src_path))
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
    
    print(f"Copied {copied_count} consensus-predicted images to {dest_dir}")
    return dest_dir

def create_consensus_labels_csv(consensus_df, output_file="results_phase3/consensus_labels.csv"):
    if consensus_df.empty:
        print("No consensus predictions. Skipping CSV creation.")
        return
    
    labels_df = consensus_df[['basename', 'predicted_class']].copy()
    labels_df.columns = ['image_path', 'label']
    
    labels_df.to_csv(output_file, index=False)
    print(f"Created consensus labels CSV at {output_file}")

def main():
    os.makedirs("results_phase3", exist_ok=True)
    
    sys.stdout = Logger()
    
    print("=== Phase 3: Ensemble Model Fine-Tuning and Consensus Inference ===")
    print(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    labels_csv = 'final_labeled_data.csv'
    source_image_dir = './data/selected_images'
    df = pd.read_csv(labels_csv, header=0)

    df_known = df[df['label'].str.lower() != 'unknown']

    train_dataset, val_dataset, data_transforms = prepare_data_for_training(df_known, source_image_dir)
    
    trained_models, class_names = train_all_models(train_dataset, val_dataset, len(train_dataset.classes))
    
    consensus_results = perform_ensemble_inference(
        trained_models, class_names, data_transforms['val'], source_image_dir)
    
    consensus_dir = copy_consensus_images(consensus_results)
    
    create_consensus_labels_csv(consensus_results)
    
    print("\n=== Process Complete ===")
    print(f"Found {len(consensus_results)} consensus predictions")
    print(f"Results saved to the 'results_phase3' directory")
    
    class_mapping = {i: class_name for i, class_name in enumerate(class_names)}
    with open("results_phase3/class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=4)
    
    if isinstance(sys.stdout, Logger):
        log_filename = sys.stdout.filename
        sys.stdout.close()
        sys.stdout = sys.__stdout__  
        print(f"Results logged to: {log_filename}")

if __name__ == "__main__":
    main()