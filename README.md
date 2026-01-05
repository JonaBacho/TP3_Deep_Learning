# 🧠 TP3: Convolutional Neural Networks - MLOps Pipeline

[![TP3 CNN Training](https://github.com/<YOUR-USERNAME>/<YOUR-REPO>/actions/workflows/tp3-cnn-training.yml/badge.svg)](https://github.com/<YOUR-USERNAME>/<YOUR-REPO>/actions/workflows/tp3-cnn-training.yml)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue.svg)](https://mlflow.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**École Nationale Supérieure Polytechnique de Yaoundé**  
Département de Génie Informatique - 5GI  
Instructeurs: Louis Fippo Fitime, Claude Tinku, Kerolle Sonfack

Ce projet implémente le **TP3 sur les réseaux de neurones convolutifs (CNN)** avec un pipeline MLOps complet. Il couvre la classification d'images avec CIFAR-10, les architectures ResNet avec skip connections, et le style transfer neuronal avec VGG16.

---

## 🎯 Objectifs d'Apprentissage

- **Comprendre** les principes fondamentaux de convolution et pooling
- **Construire** et entraîner un CNN pour la classification d'images
- **Intégrer** des blocs résiduels (ResNets) pour des réseaux profonds
- **Appliquer** les CNNs au transfert de style neuronal
- **Automatiser** l'entraînement et le tracking via GitHub Actions et MLflow

---

## 📁 Structure du Projet

```text
.
├── config/
│   └── mlflow_config.py              # Configuration MLflow centralisée
├── src/
│   ├── cnn_classification.py         # Exercises 1-2: CNNs sur CIFAR-10
│   ├── style_transfer.py             # Exercise 4: Neural Style Transfer
│   ├── app.py                        # API Flask (optionnel)
│   ├── auto_promote.py               # Promotion automatique des modèles
│   └── promote_model.py              # Gestion manuelle des stages
├── images/                           # Images pour style transfer
│   ├── content/
│   └── style/
├── tests/
│   └── test_model.py                 # Tests unitaires
├── .github/workflows/
│   ├── tp3-cnn-training.yml          # Pipeline principal CNN
│   └── deploy.yml                    # Déploiement API (optionnel)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Exercices du TP3

### Part 1: Concepts Fondamentaux (Théorie)

**Questions théoriques à répondre dans le rapport**:
- Rôle du filtre (kernel) et du stride dans une convolution
- Différence entre Max Pooling et Average Pooling
- Transformation des features spatiales en vecteurs pour Dense layers
- Problème résolu par les skip connections dans ResNets

### Part 2: Implémentation CNN

#### Exercise 1: Classic CNN Architecture
Architecture CNN classique sur **CIFAR-10** (32×32 couleur, 10 classes).

**Architecture**:
```
Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → 
Flatten → Dense(512) → Dense(10, softmax)
```

**Expérience MLflow**: `TP3-Exercise1-BasicCNN`

**Métriques**:
- `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
- `test_accuracy`, `precision`, `recall`, `f1_score`

#### Exercise 2: Residual Networks (ResNets)
CNN avec blocs résiduels et skip connections.

**Architecture**:
```
Conv2D(32) → ResBlock(32) → ResBlock(64, stride=2) → 
ResBlock(64) → ResBlock(128, stride=2) → ResBlock(128) →
GlobalAvgPooling → Dense(256) → Dropout → Dense(10)
```

**Expérience MLflow**: `TP3-Exercise2-ResNet`

**Question clé**: Expliquer l'avantage d'ajouter l'input `x` à l'output du chemin convolutif.

### Part 3: Applications Avancées

#### Exercise 3: Recognition & Detection (Concepts)
**À rechercher et expliquer dans le rapport**:

1. **Segmentation (U-Net)**:
   - Output d'un modèle de segmentation vs classification
   - Rôle des upsampling steps dans U-Net

2. **Object Detection**:
   - Concept des Bounding Boxes
   - Prédiction de position (x, y, w, h) en plus de la classe

#### Exercise 4: Neural Style Transfer
Transfert de style avec **VGG16** pré-entraîné.

**Expérience MLflow**: `TP3-Exercise4-StyleTransfer`

**Principe**:
- **Content Loss**: Préserve le contenu de l'image source
- **Style Loss**: Capture le style via Gram matrices
- **Optimisation**: Minimise `content_loss + style_loss`

**Couches utilisées**:
- Content: `block5_conv2`
- Style: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`, `block5_conv1`

---

## 🛠️ Installation

### Prérequis
- Python 3.10+
- Accès à un serveur MLflow
- GitHub repository avec Actions activées

### Installation Locale

```bash
# Cloner le projet
git clone <votre-repo-url>
cd TP3_CNN_Vision

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Configurer environnement
cp .env.example .env
# Éditer .env avec vos credentials MLflow
```

---

## 🔌 Utilisation

### Exécution via GitHub Actions (Recommandé)

```bash
# 1. Push sur main → déclenche automatiquement
git add .
git commit -m "feat: run TP3 CNN experiments"
git push origin main

# 2. Ou exécution manuelle depuis GitHub UI
# Actions → "TP3 - CNN Training & Style Transfer" → Run workflow
# Choisir: all, exercise1, exercise2, ou exercise4
```

**Durée d'exécution**:
- Exercise 1 (Basic CNN): ~15-20 min
- Exercise 2 (ResNet): ~20-25 min
- Total parallèle: ~25-30 min

### Exécution Locale

#### Exercise 1 & 2: CNN Classification

```bash
# Tous les exercices CNN
python src/cnn_classification.py

# Exercice spécifique
python -c "from src.cnn_classification import exercise_1_basic_cnn; exercise_1_basic_cnn()"
python -c "from src.cnn_classification import exercise_2_resnet; exercise_2_resnet()"
```

#### Exercise 4: Style Transfer

```bash
# Préparer vos images
mkdir -p images/content images/style

# Placer vos images dans ces dossiers
# Puis exécuter:
python src/style_transfer.py images/content/photo.jpg images/style/painting.jpg

# Résultats dans: style_transfer_results/
```

---

## 🤖 Pipeline CI/CD

### 1. TP3 - CNN Training (`tp3-cnn-training.yml`)

**Déclenchement**:
- Push sur `main`/`dev`
- Modifications des fichiers CNN
- Workflow manual dispatch

**Jobs**:
1. **exercise1-basic-cnn** (15-20 min)
   - Entraînement CNN classique sur CIFAR-10
   - Expérience MLflow: `TP3-Exercise1-BasicCNN`

2. **exercise2-resnet** (20-25 min)
   - Entraînement ResNet sur CIFAR-10
   - Expérience MLflow: `TP3-Exercise2-ResNet`

3. **promote-best-model**
   - Compare les performances
   - Promeut le meilleur modèle en Production

4. **summary**
   - Génère rapport consolidé
   - Artifacts: logs + summary markdown

**Note**: Exercise 4 (Style Transfer) nécessite des images donc s'exécute manuellement en local.

---

## 📊 Visualisation des Résultats

### Dans MLflow UI

Accédez à votre serveur MLflow:

```bash
# Expériences créées:
- TP3-Exercise1-BasicCNN        (1 run)
- TP3-Exercise2-ResNet          (1 run)
- TP3-Exercise4-StyleTransfer   (1+ runs, local)
```

**Métriques trackées**:

**Classification (Ex 1-2)**:
- `train_loss`, `train_accuracy` (par epoch)
- `val_loss`, `val_accuracy` (par epoch)
- `test_accuracy`, `precision`, `recall`, `f1_score` (final)

**Style Transfer (Ex 4)**:
- `total_loss`, `content_loss`, `style_loss` (par epoch)
- Images: content, style, résultat à chaque epoch

### Dans GitHub Actions

```
Actions → TP3 - CNN Training → Latest run
→ Artifacts: 
  - exercise1-logs
  - exercise2-logs
  - tp3-summary-report
→ Summary: Résumé consolidé
```

---

## ⚙️ Configuration

### Variables d'Environnement (`.env`)

```bash
# MLflow Tracking
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# Model Registry
MODEL_NAME=cifar10-cnn-classifier

# Promotion Thresholds
MIN_ACCURACY=0.70
MIN_F1_SCORE=0.68

# S3/MinIO (si utilisé)
MLFLOW_S3_ENDPOINT_URL=https://your-s3-endpoint
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Secrets GitHub

**Settings → Secrets and variables → Actions**:

```
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_S3_ENDPOINT_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
MIN_ACCURACY=0.70
MIN_F1_SCORE=0.68
```

---

## 🔌 API REST (Optionnel)

Servir les modèles CNN entraînés via Flask.

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de santé |
| GET | `/model/info` | Infos sur le modèle CNN |
| POST | `/predict` | Prédiction sur image 32×32 |
| POST | `/model/reload` | Rechargement depuis MLflow |

### Exemple `/predict`

```json
{
  "image": [[...], [...], ...],  // 32x32x3 array
  "flatten": false
}
```

### Déploiement

```bash
# Local
python -m src.app

# Docker
docker build -t cifar10-cnn-app .
docker run -p 5000:5000 --env-file .env cifar10-cnn-app
```

---

## 📈 Résultats Attendus

### Exercise 1: Basic CNN
```
Dataset: CIFAR-10 (50k train, 10k test)
Architecture: Conv2D → MaxPool → Conv2D → MaxPool → Dense
Test Accuracy: ~70-75%
Training Time: ~15 min (10 epochs)
```

### Exercise 2: ResNet
```
Architecture: 5 Residual Blocks avec skip connections
Test Accuracy: ~75-80% (meilleur que Basic CNN)
Training Time: ~20 min (15 epochs)
Avantage: Meilleure convergence et gradient flow
```

### Exercise 4: Style Transfer
```
Modèle: VGG16 pré-entraîné
Content Weight: 1e3
Style Weight: 1e-2
Optimization: 10 epochs × 100 steps
Résultat: Image avec contenu préservé + style appliqué
```

---

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/test_model.py

# Avec couverture
pytest --cov=src tests/

# Test d'une image CIFAR-10
python -c "
from src.cnn_classification import load_and_preprocess_cifar10, build_basic_cnn
(x_train, y_train), (x_test, y_test), shape, classes = load_and_preprocess_cifar10()
model = build_basic_cnn(shape, classes)
print('Model built successfully!')
print(model.summary())
"
```

---

## 🐛 Troubleshooting

### Erreur: CIFAR-10 Download Failed
```bash
# Télécharger manuellement
python -c "from tensorflow import keras; keras.datasets.cifar10.load_data()"
```

### Erreur: Out of Memory (GPU)
```python
# Limiter l'utilisation GPU
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
```

### Style Transfer trop lent
```bash
# Réduire epochs et steps
python src/style_transfer.py content.jpg style.jpg --epochs 5 --steps 50

# Ou réduire la taille des images (256x256 au lieu de 512x512)
```

---

## 📚 Documentation & Ressources

- [Énoncé TP3 (PDF)](./TP3_DL_5GI_2025_EN.pdf)
- [TensorFlow CNN Guide](https://www.tensorflow.org/tutorials/images/cnn)
- [ResNet Paper (2015)](https://arxiv.org/abs/1512.03385)
- [Neural Style Transfer Paper](https://arxiv.org/abs/1508.06576)
- [U-Net Paper (Segmentation)](https://arxiv.org/abs/1505.04597)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 👥 Auteurs

**ENSPY - Université de Yaoundé I** 
FOMEKONG TAMDJI JONATHAN BACHELARD 21P021 
Département de Génie Informatique - Promotion 5GI 2025

**Instructeurs**:
- Louis Fippo Fitime - louis.fippo@univ-yaounde1.cm

---

## ⚖️ Licence

Distribué sous la licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.