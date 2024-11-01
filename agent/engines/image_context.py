import torch
from typing import List, Dict, Union
from pathlib import Path
import time
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor, 
    CLIPModel,
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer
)
import numpy as np
from dataclasses import dataclass
from torchvision import transforms
import json

@dataclass
class ImageAnalysisResult:
    """Data class to store the results of image analysis"""
    description: str
    tags: List[str]
    emotions: Dict[str, float]
    objects: List[Dict[str, Union[str, float]]]
    style_attributes: Dict[str, float]
    technical_metadata: Dict[str, Union[str, int, float]]

class ImageContextAnalyzer:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = "./model_cache"
    ):
        """
        Initialize the image analyzer with multiple models for comprehensive analysis.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            cache_dir: Directory to cache downloaded models
        """
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        print("Loading models...")
        
        # Image captioning model (BLIP)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
        
        # Visual feature extraction (CLIP)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # Detailed image description (Git-large-COCO)
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.git_model = VisionEncoderDecoderModel.from_pretrained("microsoft/git-large-coco").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/git-large-coco")
        
        # Image normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _generate_caption(self, image: Image.Image) -> str:
        """Generate a basic caption for the image"""
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        output = self.blip_model.generate(**inputs, max_length=50)
        return self.blip_processor.decode(output[0], skip_special_tokens=True)

    def _extract_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract visual features using CLIP"""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        return features.cpu().detach().numpy()

    def _analyze_emotions(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze emotional content of the image"""
        emotions = ["happy", "sad", "peaceful", "energetic", "mysterious", "romantic"]
        inputs = self.clip_processor(
            text=emotions,
            images=None,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        text_features = self.clip_model.get_text_features(**inputs)
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(features).to(self.device),
            text_features
        )
        
        return dict(zip(emotions, similarity.cpu().detach().numpy().tolist()))

    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Union[str, float]]]:
        """Detect and describe objects in the image"""
        inputs = self.vit_processor(image, return_tensors="pt").to(self.device)
        outputs = self.git_model.generate(
            pixel_values=inputs.pixel_values,
            max_length=50,
            num_beams=4
        )
        
        objects_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse the objects text into structured data
        objects = []
        for obj in objects_text.split(","):
            obj = obj.strip()
            if obj:
                objects.append({"name": obj, "confidence": 0.0})  # Confidence placeholder
        return objects

    def _analyze_style(self, features: np.ndarray) -> Dict[str, float]:
        """Analyze artistic and stylistic attributes"""
        style_attributes = [
            "photographic", "artistic", "abstract", "realistic",
            "vintage", "modern", "minimalist", "detailed"
        ]
        
        inputs = self.clip_processor(
            text=style_attributes,
            images=None,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        text_features = self.clip_model.get_text_features(**inputs)
        similarity = torch.nn.functional.cosine_similarity(
            torch.tensor(features).to(self.device),
            text_features
        )
        
        return dict(zip(style_attributes, similarity.cpu().detach().numpy().tolist()))

    def analyze_image(self, image_path: Union[str, Path]) -> ImageAnalysisResult:
        """
        Analyze an image and generate comprehensive textual context.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            ImageAnalysisResult containing various aspects of the image analysis
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Extract basic technical metadata
            technical_metadata = {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "format": image.format,
                "mode": image.mode
            }
            
            # Generate basic caption
            description = self._generate_caption(image)
            
            # Extract visual features
            features = self._extract_features(image)
            
            # Analyze emotions
            emotions = self._analyze_emotions(features)
            
            # Detect objects
            objects = self._detect_objects(image)
            
            # Analyze style
            style_attributes = self._analyze_style(features)
            
            # Generate tags based on top emotions, objects, and styles
            tags = (
                [k for k, v in emotions.items() if v > 0.5] +
                [obj["name"] for obj in objects[:5]] +
                [k for k, v in style_attributes.items() if v > 0.5]
            )
            
            return ImageAnalysisResult(
                description=description,
                tags=list(set(tags)),  # Remove duplicates
                emotions=emotions,
                objects=objects,
                style_attributes=style_attributes,
                technical_metadata=technical_metadata
            )
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise

    def save_analysis(self, result: ImageAnalysisResult, output_path: Union[str, Path]):
        """Save the analysis results to a JSON file"""
        output_path = Path(output_path)
        with output_path.open('w') as f:
            json.dump(result.__dict__, f, indent=2)

def generate_context_from_images(
    image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    max_images: int = 5
) -> str:
    """
    Analyze multiple images and generate a combined context for AI consumption.
    
    Args:
        image_paths: List of paths to image files
        output_dir: Directory to save individual analysis results
        max_images: Maximum number of images to process
        
    Returns:
        str: Combined context from all analyzed images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    analyzer = ImageContextAnalyzer()
    
    all_results = []
    for img_path in image_paths[:max_images]:
        try:
            # Analyze image
            result = analyzer.analyze_image(img_path)
            
            # Save individual analysis
            output_path = output_dir / f"analysis_{Path(img_path).stem}.json"
            analyzer.save_analysis(result, output_path)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Generate combined context
    combined_context = "Image Context Summary:\n\n"
    
    for i, result in enumerate(all_results, 1):
        combined_context += f"Image {i}:\n"
        combined_context += f"Description: {result.description}\n"
        combined_context += f"Key emotions: {', '.join(k for k, v in result.emotions.items() if v > 0.5)}\n"
        combined_context += f"Main objects: {', '.join(obj['name'] for obj in result.objects[:3])}\n"
        combined_context += f"Style: {', '.join(k for k, v in result.style_attributes.items() if v > 0.5)}\n"
        combined_context += f"Tags: {', '.join(result.tags)}\n\n"
    
    return combined_context

# Example usage
if __name__ == "__main__":
    # Example inputs
    image_paths = [
        Path("./images/sample1.jpg"),
        Path("./images/sample2.jpg")
    ]
    output_dir = Path("./analysis_results")
    
    try:
        context = generate_context_from_images(
            image_paths,
            output_dir
        )
        print("Generated Context:")
        print(context)
        
        # Save the combined context
        with open(output_dir / "combined_context.txt", "w") as f:
            f.write(context)
            
    except Exception as e:
        print(f"Failed to generate context: {str(e)}")
