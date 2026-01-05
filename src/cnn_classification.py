import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

# ==================== PART 1: Data Preparation ====================
def load_and_preprocess_cifar10():
    """Charge et prétraite les données CIFAR-10"""
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    NUM_CLASSES = 10
    INPUT_SHAPE = x_train.shape[1:]  # (32, 32, 3)
    
    # Normaliser les pixels [0, 255] → [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convertir les labels en One-Hot Encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)
    
    print(f"Input data shape: {INPUT_SHAPE}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES

# ==================== PART 2: Exercise 1 - Basic CNN ====================
def build_basic_cnn(input_shape, num_classes):
    """Construit un CNN classique avec Conv2D + MaxPooling"""
    model = keras.Sequential([
        # Convolutional Layer 1: 32 filtres, 3x3, ReLU
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                           input_shape=input_shape),
        # Max Pooling Layer 1: 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Convolutional Layer 2: 64 filtres, 3x3, ReLU
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Max Pooling Layer 2: 2x2
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten pour transition vers Dense layers
        keras.layers.Flatten(),
        
        # Dense Layer 1: 512 units, ReLU
        keras.layers.Dense(512, activation='relu'),
        # Output Layer: num_classes units, Softmax
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def exercise_1_basic_cnn():
    """Exercise 1: Entraînement d'un CNN classique sur CIFAR-10"""
    mlflow.set_experiment("TP3-Exercise1-BasicCNN")
    
    # Charger les données
    (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES = load_and_preprocess_cifar10()
    
    with mlflow.start_run(run_name="Ex1_BasicCNN_CIFAR10"):
        mlflow.log_param("exercise", "1")
        mlflow.log_param("model_type", "basic_cnn")
        mlflow.log_param("dataset", "cifar10")
        mlflow.log_param("input_shape", str(INPUT_SHAPE))
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("validation_split", 0.1)
        
        # Architecture details
        mlflow.log_param("conv_layers", 2)
        mlflow.log_param("filters_layer1", 32)
        mlflow.log_param("filters_layer2", 64)
        mlflow.log_param("dense_units", 512)
        
        # Construire le modèle
        model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Afficher l'architecture
        model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Entraînement
        print("\n" + "="*60)
        print("EXERCISE 1: Training Basic CNN on CIFAR-10")
        print("="*60)
        
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=10,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Log des métriques par epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Évaluation sur test set
        print("\nEvaluating on test set...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Prédictions pour métriques détaillées
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Sauvegarder le rapport
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")
        
        # Logger le modèle
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=MLflowConfig.MODEL_NAME
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 1 Completed!")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"F1 Score: {report['weighted avg']['f1-score']:.4f}")
        print(f"{'='*60}\n")

# ==================== PART 2: Exercise 2 - Residual Networks ====================
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """Bloc résiduel simplifié avec skip connection"""
    # Main path (chemin principal)
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride,
                           padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)
    
    # Skip connection path
    if stride > 1 or x.shape[-1] != filters:
        # Adapter les dimensions si nécessaire
        x = keras.layers.Conv2D(filters, (1, 1), strides=stride)(x)
    
    # Addition du skip path avec le main path
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    
    return z

def build_resnet_cnn(input_shape, num_classes):
    """Construit un CNN avec blocs résiduels"""
    input_layer = keras.Input(shape=input_shape)
    
    # Initial Conv layer
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2)  # Réduction de dimension
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)  # Réduction de dimension
    x = residual_block(x, 128)
    
    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model

def exercise_2_resnet():
    """Exercise 2: CNN avec blocs résiduels (ResNet)"""
    mlflow.set_experiment("TP3-Exercise2-ResNet")
    
    # Charger les données
    (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES = load_and_preprocess_cifar10()
    
    with mlflow.start_run(run_name="Ex2_ResNet_CIFAR10"):
        mlflow.log_param("exercise", "2")
        mlflow.log_param("model_type", "resnet_custom")
        mlflow.log_param("dataset", "cifar10")
        mlflow.log_param("residual_blocks", 5)
        mlflow.log_param("epochs", 15)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("validation_split", 0.1)
        mlflow.log_param("dropout_rate", 0.5)
        
        # Construire le modèle
        model = build_resnet_cnn(INPUT_SHAPE, NUM_CLASSES)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Entraînement
        print("\n" + "="*60)
        print("EXERCISE 2: Training ResNet CNN on CIFAR-10")
        print("="*60)
        
        history = model.fit(
            x_train, y_train,
            batch_size=64,
            epochs=15,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Log des métriques
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Évaluation
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Métriques détaillées
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        mlflow.log_metric("precision", report['weighted avg']['precision'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])
        
        # Logger le modèle
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            registered_model_name=f"{MLflowConfig.MODEL_NAME}-resnet"
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Exercise 2 Completed!")
        print(f"{'='*60}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"F1 Score: {report['weighted avg']['f1-score']:.4f}")
        print(f"{'='*60}\n")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TP3: Convolutional Neural Networks - CIFAR-10")
    print("="*70 + "\n")
    
    # Exercise 1: Basic CNN
    print("Executing Exercise 1: Basic CNN Architecture...")
    exercise_1_basic_cnn()
    
    # Exercise 2: ResNet
    print("\nExecuting Exercise 2: ResNet Architecture...")
    exercise_2_resnet()
    
    print("\n" + "="*70)
    print("✓ All TP3 CNN exercises completed successfully!")
    print("Check MLflow UI for detailed results")
    print("="*70 + "\n")