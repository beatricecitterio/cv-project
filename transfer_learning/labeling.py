import os
import random
import shutil
import csv
import sys
from PIL import Image

def get_previously_sampled_images(output_file):
    previously_sampled = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  
            for row in reader:
                if len(row) >= 1:
                    img_path = row[0]
                    previously_sampled.add(img_path)
    
    return previously_sampled

def sample_images(source_dir, sample_dir, num_samples=10, previously_sampled=None):
    if previously_sampled is None:
        previously_sampled = set()
    
    os.makedirs(sample_dir, exist_ok=True)
    
    for filename in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for filename in os.listdir(source_dir):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            full_path = os.path.join(source_dir, filename)
            if full_path not in previously_sampled:
                image_files.append(full_path)
    
    if not image_files:
        print(f"Error: No new unseen images found in {source_dir}")
        print(f"All {len(previously_sampled)} images have been previously sampled.")
        sys.exit(1)
    
    num_to_sample = min(num_samples, len(image_files))
    sampled_images = random.sample(image_files, num_to_sample)
    
    print(f"Found {len(image_files)} unseen images out of which {num_to_sample} will be sampled")
    
    sampled_paths = []
    path_mapping = {}  
    
    for i, img_path in enumerate(sampled_images):
        original_ext = os.path.splitext(img_path)[1]
        seq_filename = f"image_{i+1:03d}{original_ext}"
        
        dest_path = os.path.join(sample_dir, seq_filename)
        shutil.copy2(img_path, dest_path)
        sampled_paths.append(dest_path)
        path_mapping[dest_path] = img_path 
        print(f"Copied: {os.path.basename(img_path)} → {seq_filename}")
    
    print(f"\nSuccessfully sampled {len(sampled_paths)} new images to {sample_dir}")
    return sampled_paths, path_mapping

class TerminalImageLabeler:
    def __init__(self, image_dir, output_file, path_mapping=None):
        self.image_dir = image_dir
        self.output_file = output_file
        self.path_mapping = path_mapping or {}  
        self.image_files = []
        self.current_image_index = 0
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        for filename in os.listdir(image_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                full_path = os.path.join(image_dir, filename)
                self.image_files.append(full_path)
        
        self.image_files.sort()
        
        if not self.image_files:
            print("Error: No images found in the specified directory!")
            sys.exit(1)
            
        print(f"Ready to label {len(self.image_files)} sampled images in sequential order.")
    
    def display_image_info(self, image_path):
        try:
            img = Image.open(image_path)
            width, height = img.size
            filename = os.path.basename(image_path)
            
            print("\n" + "="*50)
            print(f"Image {self.current_image_index + 1} of {len(self.image_files)}")
            print(f"Filename: {filename}")
            print(f"Size: {width}x{height} pixels")
            
            if image_path in self.path_mapping:
                original_filename = os.path.basename(self.path_mapping[image_path])
                print(f"Original filename: {original_filename}")
            
            print("="*50)
            
            print(f"\nTo view this image, check: {image_path}")
            print(f"Or navigate to the sample folder and open: {filename}")
            print("="*50)
        except Exception as e:
            print(f"Error reading image: {str(e)}")
    
    def append_labels_to_file(self, new_labels):
        file_exists = os.path.exists(self.output_file)
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow(['image_path', 'label'])
            
            for img_path, label in new_labels.items():
                source_path_full = self.path_mapping.get(img_path, img_path)
            
                source_path = os.path.basename(source_path_full)
                writer.writerow([source_path, label])
        
        print(f"Labels appended to {self.output_file}")
    
    def run(self):
        print("\nWelcome to Terminal Image Labeler!")
        print("Processing images in sequential order.")
        print("Type 'q' to quit, 'prev' for previous image, 'skip' to skip current image.")
        
        self.current_image_index = 0
        new_labels = {} 
        
        while self.current_image_index < len(self.image_files):
            current_image = self.image_files[self.current_image_index]
            
            self.display_image_info(current_image)
            
            label = input("\nEnter label for this image: ").strip()
            
            if label.lower() == 'q':
                print("Saving and exiting...")
                if new_labels:
                    self.append_labels_to_file(new_labels)
                return
            
            elif label.lower() == 'prev' and self.current_image_index > 0:
                self.current_image_index -= 1
                continue
            
            elif label.lower() == 'skip':
                print("Skipping this image...")
                self.current_image_index += 1
                continue
            
            elif label:
                new_labels[current_image] = label
                print(f"Label '{label}' saved for {os.path.basename(current_image)}")
                
                self.current_image_index += 1
                
                if len(new_labels) % 3 == 0:
                    self.append_labels_to_file(new_labels)
                    new_labels = {} 
            
            else:
                print("Please enter a valid label or command.")
            
        if new_labels:
            self.append_labels_to_file(new_labels)
        print("\nAll images have been processed!")

def main():
    if len(sys.argv) < 3:
        print("Usage: python labeling.py <source_image_directory> <output_csv_file> [number_of_samples]")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    num_samples = 100
    if len(sys.argv) > 3:
        try:
            num_samples = int(sys.argv[3])
        except ValueError:
            print("Error: Number of samples must be an integer")
            sys.exit(1)
    
    if not os.path.isdir(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    previously_sampled = get_previously_sampled_images(output_file)
    if previously_sampled:
        print(f"Found {len(previously_sampled)} previously sampled images in the output file.")
    
    sample_dir_name = "sampled_images_to_label"
    sample_dir = os.path.join(os.path.dirname(source_dir), sample_dir_name)
    
    print(f"Sampling {num_samples} NEW images from {source_dir} to {sample_dir}...")
    sampled_images, path_mapping = sample_images(source_dir, sample_dir, num_samples, previously_sampled)
    
    print(f"\nStarting labeling process for {len(sampled_images)} sampled images...")
    labeler = TerminalImageLabeler(sample_dir, output_file, path_mapping)
    labeler.run()
    
    print("\nLabeling completed!")
    print(f"Sampled images are in: {sample_dir}")
    print(f"Labels are saved to: {output_file}")

if __name__ == "__main__":
    main()