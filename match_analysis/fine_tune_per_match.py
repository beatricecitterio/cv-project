import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import time
import copy
import sys


''' This code takes a model and fine tunes it on a specific match then applies it to other matches to see 
how well it generalizes'''

# --- 1. Setup ---
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
selected_labels = pd.read_csv('selected_labels.csv')
selected_labels = selected_labels.rename(columns={'Image Number': 'image_id'})

# drop the columns that we don't want to predict
columns_to_drop = ['Ball', 'referee Image', 'Gate', 'Red Card', 'Penalty Image', 'yellow card', 'Start/Restar the game', 'The joy of the players']
for col in columns_to_drop:
    if col in selected_labels.columns:
        selected_labels = selected_labels.drop(columns=[col])
        print(f"Dropped column: {col}")
    else:
        print(f"Column {col} not found in selected_labels")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. Dataset creation ---
class FootballDataset(Dataset):
    def __init__(self, df, image_dir, label_cols, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.label_cols = label_cols
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(row[self.label_cols].values.astype(np.float32))
        match = row['label'] 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, match

# --- 3. Train model ---
def train_model(df, image_dir, label_columns, trained_on='rou-alb'):
    train_match = trained_on  
    train_df = df[df['label'] == train_match].reset_index(drop=True)
    
    if len(train_df) < 50:  
        print(f"Not enough samples for match {train_match} (only {len(train_df)})")
        return None, None
    
    print(f"Training on match: {train_match} with {len(train_df)} samples")
    
    train_dataset = FootballDataset(train_df, image_dir, label_columns, train_transform)
    
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(label_columns))
    model = model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_indices)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss = val_loss / len(val_indices)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    model.load_state_dict(best_model_wts)
    print(f"Finished training on match {train_match}")
   
    model_path = f'resnet18_{train_match}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, model_path

# --- 4. Evaluate model ---
def evaluate_on_all_matches(model, df, image_dir, label_columns, trained_on):
    model.eval()
    
    match_counts = df['label'].value_counts()
    valid_matches = match_counts[match_counts >= 100].index.tolist()
    valid_matches = [m for m in valid_matches if m != 'unknown']
    
    print(f"Evaluating on {len(valid_matches) - 1} matches (excluding training match): {[m for m in valid_matches if m != trained_on]}")
    
    results = {match: {} for match in valid_matches if match != trained_on}
    
    for match in valid_matches:
        if match == trained_on:
            continue

        print(f"Evaluating on {match}")
            
        match_df = df[df['label'] == match].reset_index(drop=True)
        print(f"  - {len(match_df)} samples")
        
        match_dataset = FootballDataset(match_df, image_dir, label_columns, val_transform)
        match_loader = DataLoader(match_dataset, batch_size=BATCH_SIZE)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, _ in match_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs) > 0.5
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        for i, class_name in enumerate(label_columns):
            class_preds = all_preds[:, i]
            class_labels = all_labels[:, i]
            
            if np.sum(class_labels) > 0:  
                accuracy = accuracy_score(class_labels, class_preds)
                results[match][class_name] = accuracy * 100  
            else:
                results[match][class_name] = np.nan
                
        overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten()) * 100
        results[match]['Overall'] = overall_acc
    
    return results

# --- 5. Visualize ---
def visualize_results(results, trained_on):
    results_df = pd.DataFrame(results).T
    
    if 'Overall' in results_df.columns:
        column_order = ['Overall'] + sorted([col for col in results_df.columns if col != 'Overall'], 
                                         key=lambda x: results_df[x].mean(), 
                                         reverse=True)
        results_df = results_df[column_order]
    
    formatted_df = results_df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].map(lambda x: f'{x:.2f}%' if not pd.isna(x) else 'N/A')
    
    print("\nAccuracy Results for model trained on", trained_on)
    print(formatted_df)
    
    formatted_df.to_csv(f'generalization_results_{trained_on}.csv')
    
    print(f"\n\nGeneralization Table for Model trained on {trained_on}")
    print("=" * 120)
    
    clean_df = results_df

    # Select the display columns we want in the final table (for clean formatting)
    display_columns = [
        'Stadium View', 
        'Free Kick',
        'Generic Moment'
    ]
    
    valid_columns = [col for col in display_columns if col in clean_df.columns]
    
    if len(valid_columns) == 0:
        print("None of the specified columns found in results. Available columns:", clean_df.columns.tolist())
        valid_columns = clean_df.columns.tolist()
    
    display_df = clean_df[valid_columns].copy()
    
    for col in display_df.columns:
        display_df[col] = display_df[col].map(lambda x: f'{x:.2f}%' if not pd.isna(x) else 'N/A')
    
    model_accuracy = clean_df[valid_columns].mean()
    model_accuracy_formatted = {col: f'{val:.2f}%' for col, val in model_accuracy.items()}
    
    print(display_df)
    
    print("-" * 120)
    print("| Model Accuracy |", end=" ")
    for col in valid_columns:
        print(f"| {model_accuracy_formatted[col]} |", end=" ")
    print()
    print("=" * 120)
    
    with open(f'generalization_table_{trained_on}.txt', 'w') as f:
        f.write(f"GENERALIZATION RESULTS FOR MODEL TRAINED ON {trained_on}\n")
        f.write("=" * 120 + "\n")
        f.write(display_df.to_string() + "\n")
        f.write("-" * 120 + "\n")
        f.write("| Model Accuracy |")
        for col in valid_columns:
            f.write(f" | {model_accuracy_formatted[col]} |")
        f.write("\n")
        f.write("=" * 120 + "\n")

def main(trained_on='rou-alb'):
    df = pd.read_csv('final_labeled_data.csv')
    print("Label distribution in training match:")

    match_counts = df['label'].value_counts()
    valid_matches = match_counts[match_counts >= 100].index.tolist()
    valid_matches = [m for m in valid_matches if m != 'unknown']
    
    if trained_on not in valid_matches:
        print(f"Error: Training match '{trained_on}' not found or has fewer than 100 samples")
        print(f"Available valid matches: {valid_matches}")
        return
    
    df_filtered = df[df['label'].isin(valid_matches)].reset_index(drop=True)
    
    print(f"Original data: {len(df)} samples")
    print(f"After filtering: {len(df_filtered)} samples")
    print(f"Matches with ≥100 samples (excluding 'unknown'): {valid_matches}")
    
    if 'image_id' not in df_filtered.columns:
        df_filtered['image_id'] = df_filtered['image_path'].str.extract(r'(\d+)').astype(int)
    
    label_columns = [col for col in selected_labels.columns if col != 'image_id']
    print(f"Using label columns: {label_columns}")
    
    df_with_labels = pd.merge(df_filtered, selected_labels, on='image_id', how='inner')
    print(f"Combined dataset has {len(df_with_labels)} samples with {len(df_with_labels['label'].unique())} unique matches")
    train_df = df_with_labels[df_with_labels['label'] == trained_on]
    for col in label_columns:
        positive_count = train_df[col].sum()
        total_count = len(train_df)
        print(f"{col}: {positive_count}/{total_count} ({positive_count/total_count*100:.2f}%)")

    image_dir = 'data/selected_images'  
    
    model_path = f'resnet18_{trained_on}_model.pth'
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, len(label_columns))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)
    else:
        print(f"No existing model found for {trained_on}, training a new one")
        model, model_path = train_model(df_with_labels, image_dir, label_columns, trained_on)
        if model is None:
            print(f"Failed to train model for {trained_on}")
            return
    
    results = evaluate_on_all_matches(model, df_with_labels, image_dir, label_columns, trained_on)
    
    visualize_results(results, trained_on)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        trained_on = sys.argv[1]
    else:
        trained_on = 'rou-alb'
        
    print(f"Training and evaluating model on match: {trained_on}")
    main(trained_on)