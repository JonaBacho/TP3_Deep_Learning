import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

# ==================== PART 3: Exercise 4 - Neural Style Transfer ====================

def load_and_process_image(image_path, target_size=(512, 512)):
    """Charge et prétraite une image pour VGG16"""
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return img

def deprocess_image(processed_img):
    """Convertit l'image prétraitée vers format visualisable"""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    
    # Retirer le preprocessing VGG16
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # BGR → RGB
    x = x[:, :, ::-1]
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def create_extractor(model, style_layers, content_layers):
    """Crée un modèle extracteur de features"""
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

def gram_matrix(input_tensor):
    """Calcule la matrice de Gram pour la style loss"""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def compute_style_loss(style_outputs, target_style_outputs):
    """Calcule la style loss entre les features"""
    style_loss = 0
    for style_output, target_style_output in zip(style_outputs, target_style_outputs):
        style_gram = gram_matrix(style_output)
        target_gram = gram_matrix(target_style_output)
        style_loss += tf.reduce_mean(tf.square(style_gram - target_gram))
    return style_loss

def compute_content_loss(content_outputs, target_content_outputs):
    """Calcule la content loss entre les features"""
    content_loss = 0
    for content_output, target_content_output in zip(content_outputs, target_content_outputs):
        content_loss += tf.reduce_mean(tf.square(content_output - target_content_output))
    return content_loss

def compute_total_loss(outputs, content_targets, style_targets, 
                      num_content_layers, num_style_layers,
                      content_weight=1e3, style_weight=1e-2):
    """Calcule la loss totale (content + style)"""
    style_outputs = outputs[:num_style_layers]
    content_outputs = outputs[num_style_layers:]
    
    style_loss = compute_style_loss(style_outputs, style_targets)
    content_loss = compute_content_loss(content_outputs, content_targets)
    
    style_loss *= style_weight / num_style_layers
    content_loss *= content_weight / num_content_layers
    
    total_loss = style_loss + content_loss
    return total_loss, style_loss, content_loss

@tf.function
def train_step(image, extractor, content_targets, style_targets,
               num_content_layers, num_style_layers,
               content_weight, style_weight, optimizer):
    """Une étape d'optimisation du style transfer"""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss, style_loss, content_loss = compute_total_loss(
            outputs, content_targets, style_targets,
            num_content_layers, num_style_layers,
            content_weight, style_weight
        )
    
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, -1.0, 1.0))
    
    return loss, style_loss, content_loss

def exercise_4_style_transfer(content_image_path, style_image_path, 
                              output_dir="style_transfer_results",
                              epochs=10, steps_per_epoch=100):
    """Exercise 4: Neural Style Transfer avec VGG16"""
    mlflow.set_experiment("TP3-Exercise4-StyleTransfer")
    
    # Créer répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    with mlflow.start_run(run_name="Ex4_StyleTransfer_VGG16"):
        mlflow.log_param("exercise", "4")
        mlflow.log_param("model_type", "vgg16_style_transfer")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("steps_per_epoch", steps_per_epoch)
        mlflow.log_param("content_image", os.path.basename(content_image_path))
        mlflow.log_param("style_image", os.path.basename(style_image_path))
        
        # Charger les images
        print("Loading images...")
        content_image = load_and_process_image(content_image_path)
        style_image = load_and_process_image(style_image_path)
        
        # Charger VGG16 pré-entraîné
        print("Loading VGG16 model...")
        vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # Définir les couches pour extraction
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                       'block4_conv1', 'block5_conv1']
        
        mlflow.log_param("content_layers", str(content_layers))
        mlflow.log_param("style_layers", str(style_layers))
        
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        
        # Créer l'extracteur
        extractor = create_extractor(vgg, style_layers, content_layers)
        
        # Extraire features des images cibles
        print("Extracting target features...")
        style_targets = extractor(style_image)[:num_style_layers]
        content_targets = extractor(content_image)[num_style_layers:]
        
        # Initialiser l'image générée avec l'image de contenu
        generated_image = tf.Variable(content_image, dtype=tf.float32)
        
        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.02)
        
        # Poids des losses
        content_weight = 1e3
        style_weight = 1e-2
        
        mlflow.log_param("content_weight", content_weight)
        mlflow.log_param("style_weight", style_weight)
        mlflow.log_param("learning_rate", 0.02)
        
        print("\n" + "="*60)
        print("EXERCISE 4: Neural Style Transfer")
        print("="*60)
        
        # Optimisation
        step = 0
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_style_loss = 0
            epoch_content_loss = 0
            
            for i in range(steps_per_epoch):
                loss, style_loss, content_loss = train_step(
                    generated_image, extractor, content_targets, style_targets,
                    num_content_layers, num_style_layers,
                    content_weight, style_weight, optimizer
                )
                
                epoch_loss += loss.numpy()
                epoch_style_loss += style_loss.numpy()
                epoch_content_loss += content_loss.numpy()
                
                step += 1
            
            # Moyennes des losses
            avg_loss = epoch_loss / steps_per_epoch
            avg_style_loss = epoch_style_loss / steps_per_epoch
            avg_content_loss = epoch_content_loss / steps_per_epoch
            
            # Log des métriques
            mlflow.log_metric("total_loss", avg_loss, step=epoch)
            mlflow.log_metric("style_loss", avg_style_loss, step=epoch)
            mlflow.log_metric("content_loss", avg_content_loss, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f} "
                  f"(Style: {avg_style_loss:.2f}, Content: {avg_content_loss:.2f})")
            
            # Sauvegarder image intermédiaire tous les 2 epochs
            if (epoch + 1) % 2 == 0:
                result_img = deprocess_image(generated_image.numpy())
                save_path = os.path.join(output_dir, f"result_epoch_{epoch+1}.png")
                Image.fromarray(result_img).save(save_path)
                mlflow.log_artifact(save_path)
        
        # Image finale
        print("\nSaving final result...")
        final_result = deprocess_image(generated_image.numpy())
        final_path = os.path.join(output_dir, "final_result.png")
        Image.fromarray(final_result).save(final_path)
        mlflow.log_artifact(final_path)
        
        # Sauvegarder aussi les images originales
        content_orig = deprocess_image(content_image)
        style_orig = deprocess_image(style_image)
        
        Image.fromarray(content_orig).save(os.path.join(output_dir, "content_original.png"))
        Image.fromarray(style_orig).save(os.path.join(output_dir, "style_original.png"))
        
        mlflow.log_artifact(os.path.join(output_dir, "content_original.png"))
        mlflow.log_artifact(os.path.join(output_dir, "style_original.png"))
        
        # Créer une visualisation comparative
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(content_orig)
        axes[0].set_title("Content Image")
        axes[0].axis('off')
        
        axes[1].imshow(style_orig)
        axes[1].set_title("Style Image")
        axes[1].axis('off')
        
        axes[2].imshow(final_result)
        axes[2].set_title("Generated Image")
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        mlflow.log_artifact(comparison_path)
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 4 Completed!")
        print(f"{'='*60}")
        print(f"Final Loss: {avg_loss:.2f}")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*60}\n")

# ==================== MAIN ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python style_transfer.py <content_image_path> <style_image_path>")
        print("\nExample:")
        print("  python style_transfer.py images/content.jpg images/style.jpg")
        sys.exit(1)
    
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    
    if not os.path.exists(content_path):
        print(f"Error: Content image not found: {content_path}")
        sys.exit(1)
    
    if not os.path.exists(style_path):
        print(f"Error: Style image not found: {style_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TP3: Neural Style Transfer")
    print("="*70 + "\n")
    
    exercise_4_style_transfer(
        content_image_path=content_path,
        style_image_path=style_path,
        epochs=10,
        steps_per_epoch=100
    )