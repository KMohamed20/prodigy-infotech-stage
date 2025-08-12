# 🚀 TOUTES LES TÂCHES COMPLÈTES - STAGE PRODIGY INFOTECH
## Khalid Ag Mohamed Aly - Machine Learning Intern

---

## 📋 TÂCHE OBLIGATOIRE - Amélioration Profil LinkedIn

### ✅ Actions Requises
1. **Post de l'Offer Letter**
   - Partager une capture de ta lettre d'offre
   - Message : "Ravi d'annoncer que j'ai rejoint Prodigy InfoTech en tant que Machine Learning Intern ! Hâte de contribuer à des projets innovants en IA. #MachineLearning #Internship #ProdigyInfoTech"

2. **Mise à jour du profil**
   - **Titre :** "Machine Learning Intern at Prodigy InfoTech"
   - **Expérience :** Ajouter "Prodigy InfoTech" comme entreprise actuelle
   - **Section À propos :** Mentionner tes compétences ML et objectifs

3. **Articles à lire et appliquer**
   - Optimisation du résumé professionnel
   - Amélioration de la section expérience
   - Ajout de compétences techniques pertinentes

### 🎯 Livrable
- Profil LinkedIn optimisé
- Post initial publié
- Screenshots des améliorations

---

## 🌐 TÂCHE 2 - Développement Web

### **WD Task 01 - Landing Page Responsive**
**Repo:** `PRODIGY_WD_01`

#### Spécifications
- **Menu de navigation interactif**
- Changement de couleur au scroll/hover
- Position fixe, visible sur toutes les pages
- HTML structure + CSS styling + JavaScript interactivité

#### Code Complet - Landing Page Moderne

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TechSolutions - Landing Page</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
        }

        /* Navigation fixe */
        .navbar {
            position: fixed;
            top: 0;
            width: 100%;
            padding: 1rem 5%;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            z-index: 1000;
            transition: all 0.3s ease;
        }

        .navbar.scrolled {
            background: rgba(74, 144, 226, 0.95);
            box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
        }

        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }

        .logo {
            font-size: 1.8rem;
            font-weight: bold;
            color: #4a90e2;
            transition: color 0.3s ease;
        }

        .navbar.scrolled .logo {
            color: white;
        }

        .nav-menu {
            display: flex;
            list-style: none;
            gap: 2rem;
        }

        .nav-item a {
            text-decoration: none;
            color: #333;
            font-weight: 500;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            border-radius: 5px;
        }

        .nav-item a:hover {
            background: #4a90e2;
            color: white;
            transform: translateY(-2px);
        }

        .navbar.scrolled .nav-item a {
            color: white;
        }

        .navbar.scrolled .nav-item a:hover {
            background: rgba(255, 255, 255, 0.2);
        }

        /* Section Hero */
        .hero {
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: white;
        }

        .hero-content h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
            animation: fadeInUp 1s ease;
        }

        .hero-content p {
            font-size: 1.2rem;
            margin-bottom: 2rem;
            animation: fadeInUp 1s 0.3s ease both;
        }

        .cta-button {
            display: inline-block;
            padding: 1rem 2rem;
            background: #ff6b6b;
            color: white;
            text-decoration: none;
            border-radius: 50px;
            font-weight: bold;
            transition: all 0.3s ease;
            animation: fadeInUp 1s 0.6s ease both;
        }

        .cta-button:hover {
            background: #ff5252;
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(255, 107, 107, 0.4);
        }

        /* Sections */
        section {
            padding: 5rem 5%;
            max-width: 1200px;
            margin: 0 auto;
        }

        .section-title {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 3rem;
            color: #333;
        }

        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }

        .feature-card {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            text-align: center;
        }

        .feature-card:hover {
            transform: translateY(-10px);
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .nav-menu {
                flex-direction: column;
                position: absolute;
                top: 100%;
                left: 0;
                width: 100%;
                background: rgba(255, 255, 255, 0.95);
                padding: 1rem;
                display: none;
            }

            .hero-content h1 {
                font-size: 2.5rem;
            }

            .navbar {
                padding: 1rem 2%;
            }
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar" id="navbar">
        <div class="nav-container">
            <div class="logo">TechSolutions</div>
            <ul class="nav-menu">
                <li class="nav-item"><a href="#home">Accueil</a></li>
                <li class="nav-item"><a href="#about">À propos</a></li>
                <li class="nav-item"><a href="#services">Services</a></li>
                <li class="nav-item"><a href="#contact">Contact</a></li>
            </ul>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero" id="home">
        <div class="hero-content">
            <h1>Solutions Tech Innovantes</h1>
            <p>Nous créons des expériences numériques exceptionnelles qui transforment votre business</p>
            <a href="#services" class="cta-button">Découvrir nos services</a>
        </div>
    </section>

    <!-- Services Section -->
    <section id="services">
        <h2 class="section-title">Nos Services</h2>
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <h3>Développement Web</h3>
                <p>Sites web modernes et responsive, optimisés pour la performance</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📱</div>
                <h3>Applications Mobile</h3>
                <p>Apps natives et cross-platform pour iOS et Android</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3>Intelligence Artificielle</h3>
                <p>Solutions ML et IA pour automatiser vos processus métier</p>
            </div>
        </div>
    </section>

    <script>
        // Navigation scroll effect
        window.addEventListener('scroll', function() {
            const navbar = document.getElementById('navbar');
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });

        // Smooth scrolling pour les liens de navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    </script>
</body>
</html>
```

---

## 🤖 TÂCHE 3 - Machine Learning

### **ML Task 01 - Analyse Prédictive des Prix**
**Repo:** `PRODIGY_ML_01`

#### Projet Complet - Prédiction Prix Immobilier

```python
# house_price_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def create_dataset(self, n_samples=1000):
        """Crée un dataset synthétique réaliste"""
        np.random.seed(42)
        
        data = {
            'surface': np.random.normal(120, 40, n_samples),
            'chambres': np.random.poisson(3, n_samples) + 1,
            'salle_bain': np.random.poisson(1.5, n_samples) + 1,
            'age': np.random.exponential(15, n_samples),
            'garage': np.random.binomial(1, 0.7, n_samples),
            'quartier_score': np.random.normal(7, 2, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calcul du prix réaliste
        prix = (
            df['surface'] * 2000 +
            df['chambres'] * 15000 +
            df['salle_bain'] * 10000 -
            df['age'] * 1000 +
            df['garage'] * 20000 +
            df['quartier_score'] * 5000 +
            np.random.normal(0, 15000, n_samples)
        )
        
        df['prix'] = np.clip(prix, 80000, 800000)
        
        return df
    
    def train_model(self, df):
        """Entraîne le modèle de prédiction"""
        X = df.drop('prix', axis=1)
        y = df['prix']
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entraînement Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Prédictions et évaluation
        y_pred = self.model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"🎯 Performance du modèle:")
        print(f"RMSE: {rmse:,.0f}€")
        print(f"R²: {r2:.3f}")
        
        return X_test, y_test, y_pred
    
    def predict_price(self, features):
        """Prédit le prix pour de nouvelles caractéristiques"""
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]
    
    def plot_results(self, y_test, y_pred):
        """Visualise les résultats"""
        plt.figure(figsize=(12, 4))
        
        # Graphique 1: Prédictions vs Réalité
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Prix Réel (€)')
        plt.ylabel('Prix Prédit (€)')
        plt.title('Prédictions vs Réalité')
        
        # Graphique 2: Importance des features
        plt.subplot(1, 2, 2)
        feature_names = ['Surface', 'Chambres', 'SDB', 'Âge', 'Garage', 'Quartier']
        importance = self.model.feature_importances_
        plt.barh(feature_names, importance)
        plt.xlabel('Importance')
        plt.title('Importance des Caractéristiques')
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# Exécution principale
if __name__ == "__main__":
    print("🏠 SYSTÈME DE PRÉDICTION PRIX IMMOBILIER")
    print("=" * 50)
    
    predictor = HousePricePredictor()
    
    # 1. Création du dataset
    df = predictor.create_dataset()
    print(f"✅ Dataset créé: {len(df)} propriétés")
    print(f"Prix moyen: {df['prix'].mean():,.0f}€")
    
    # 2. Entraînement
    X_test, y_test, y_pred = predictor.train_model(df)
    
    # 3. Visualisation
    predictor.plot_results(y_test, y_pred)
    
    # 4. Exemple de prédiction
    print("\n🔮 EXEMPLE DE PRÉDICTION:")
    exemple = [150, 4, 2, 10, 1, 8.5]  # surface, chambres, sdb, âge, garage, quartier
    prix_predit = predictor.predict_price(exemple)
    print(f"Maison 150m², 4 ch, 2 sdb, 10 ans, garage, bon quartier")
    print(f"Prix prédit: {prix_predit:,.0f}€")
```

### **ML Task 02 - Classification d'Images**
**Repo:** `PRODIGY_ML_02`

```python
# image_classifier.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ImageClassifier:
    def __init__(self, num_classes=10, img_height=32, img_width=32):
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def load_data(self):
        """Charge le dataset CIFAR-10"""
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Normalisation des pixels
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Conversion des labels en categorical
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        print(f"✅ Data loaded: Train {x_train.shape}, Test {x_test.shape}")
        
        return x_train, y_train, x_test, y_test
    
    def build_model(self):
        """Construit le modèle CNN"""
        self.model = keras.Sequential([
            # Couche d'entrée
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(self.img_height, self.img_width, 3)),
            layers.MaxPooling2D((2, 2)),
            
            # Couches cachées
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Couches de classification
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modèle CNN créé")
        return self.model.summary()
    
    def train_model(self, x_train, y_train, x_test, y_test, epochs=20):
        """Entraîne le modèle"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.001
        )
        
        # Entraînement
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("✅ Entraînement terminé")
        
    def evaluate_model(self, x_test, y_test):
        """Évalue le modèle"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"🎯 Précision sur le test: {test_acc:.3f}")
        
        # Prédictions pour la matrice de confusion
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        return y_true_classes, y_pred_classes
    
    def plot_training_history(self):
        """Visualise l'historique d'entraînement"""
        if self.history is None:
            print("❌ Aucun historique d'entraînement disponible")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Précision
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Précision du Modèle')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Précision')
        axes[0].legend()
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Loss du Modèle')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Affiche la matrice de confusion"""
        class_names = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                      'chien', 'grenouille', 'cheval', 'bateau', 'camion']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# Exécution principale
if __name__ == "__main__":
    print("🖼️ CLASSIFICATEUR D'IMAGES CNN")
    print("=" * 50)
    
    # Initialisation
    classifier = ImageClassifier()
    
    # Chargement des données
    x_train, y_train, x_test, y_test = classifier.load_data()
    
    # Construction du modèle
    classifier.build_model()
    
    # Entraînement
    print("🚀 Début de l'entraînement...")
    classifier.train_model(x_train, y_train, x_test, y_test, epochs=10)
    
    # Évaluation
    y_true, y_pred = classifier.evaluate_model(x_test, y_test)
    
    # Visualisations
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_true, y_pred)
    
    # Rapport de classification
    class_names = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                  'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    print("\n📊 RAPPORT DE CLASSIFICATION:")
    print(classification_report(y_true, y_pred, target_names=class_names))
```

---

## 📊 TÂCHE 4 - Data Science

### **DS Task 01 - Dashboard Analytique**
**Repo:** `PRODIGY_DS_01`

```python
# dashboard_analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

class BusinessDashboard:
    def __init__(self):
        self.data = self.generate_sample_data()
    
    def generate_sample_data(self):
        """Génère des données business réalistes"""
        np.random.seed(42)
        
        # Génération de 12 mois de données
        dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
        n_days = len(dates)
        
        # Tendances saisonnières
        trend = np.linspace(100000, 150000, n_days)
        seasonality = 20000 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        noise = np.random.normal(0, 5000, n_days)
        
        data = {
            'date': dates,
            'ventes': trend + seasonality + noise,
            'region': np.random.choice(['Nord', 'Sud', 'Est', 'Ouest'], n_days),
            'produit': np.random.choice(['ProductA', 'ProductB', 'ProductC'], n_days),
            'clients': np.random.poisson(50, n_days),
            'cout_acquisition': np.random.normal(25, 5, n_days)
        }
        
        df = pd.DataFrame(data)
        df['profit'] = df['ventes'] - (df['clients'] * df['cout_acquisition'])
        df['mois'] = df['date'].dt.strftime('%Y-%m')
        
        return df
    
    def create_dashboard(self):
        """Crée le dashboard Streamlit"""
        st.set_page_config(
            page_title="Dashboard Analytics",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Dashboard Analytics Business")
        st.markdown("---")
        
        # Sidebar pour les filtres
        st.sidebar.header("🎛️ Filtres")
        
        # Filtres
        region_filter = st.sidebar.multiselect(
            "Sélectionner les régions:",
            self.data['region'].unique(),
            default=self.data['region'].unique()
        )
        
        produit_filter = st.sidebar.multiselect(
            "Sélectionner les produits:",
            self.data['produit'].unique(),
            default=self.data['produit'].unique()
        )
        
        # Filtrage des données
        df_filtered = self.data[
            (self.data['region'].isin(region_filter)) &
            (self.data['produit'].isin(produit_filter))
        ]
        
        # KPI Cards
        self.create_kpi_cards(df_filtered)
        
        # Graphiques principaux
        col1, col2 = st.columns(2)
        
        with col1:
            self.create_sales_trend(df_filtered)
            
        with col2:
            self.create_region_performance(df_filtered)
        
        # Graphiques secondaires
        col3, col4 = st.columns(2)
        
        with col3:
            self.create_product_analysis(df_filtered)
            
        with col4:
            self.create_correlation_heatmap(df_filtered)
    
    def create_kpi_cards(self, df):
        """Crée les cartes KPI"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_ventes = df['ventes'].sum()
            st.metric(
                "💰 Ventes Totales",
                f"{total_ventes:,.0f}€",
                f"+{np.random.uniform(5, 15):.1f}%"
            )
        
        with col2:
            total_clients = df['clients'].sum()
            st.metric(
                "👥 Clients Totaux",
                f"{total_clients:,}",
                f"+{np.random.uniform(-5, 10):.1f}%"
            )
        
        with col3:
            profit_moyen = df['profit'].mean()
            st.metric(
                "📈 Profit Moyen",
                f"{profit_moyen:,.0f}€",
                f"+{np.random.uniform(0, 20):.1f}%"
            )
        
        with col4:
            roi = (df['profit'].sum() / df['ventes'].sum()) * 100
            st.metric(
                "🎯 ROI",
                f"{roi:.1f}%",
                f"+{np.random.uniform(-2, 8):.1f}%"
            )
    
    def create_sales_trend(self, df):
        """Graphique de tendance des ventes"""
        st.subheader("📈 Évolution des Ventes")
        
        monthly_sales = df.groupby('mois')['ventes'].sum().reset_index()
        
        fig = px.line(
            monthly_sales,
            x='mois',
            y='ventes',
            title="Ventes Mensuelles",
            color_discrete_sequence=['#1f77b4']
        )
        
        fig.update_layout(
            xaxis_title="Mois",
            yaxis_title="Ventes (€)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_region_performance(self, df):
        """Graphique performance par région"""
        st.subheader("🗺️ Performance par Région")
        
        region_performance = df.groupby('region').agg({
            'ventes': 'sum',
            'profit': 'sum',
            'clients': 'sum'
        }).reset_index()
        
        fig = px.bar(
            region_performance,
            x='region',
            y='ventes',
            color='profit',
            title="Ventes et Profit par Région",
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_product_analysis(self, df):
        """Analyse des produits"""
        st.subheader("🛍️ Analyse des Produits")
        
        product_sales = df.groupby('produit')['ventes'].sum().reset_index()
        
        fig = px.pie(
            product_sales,
            values='ventes',
            names='produit',
            title="Répartition des Ventes par Produit"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_correlation_heatmap(self, df):
        """Heatmap des corrélations"""
        st.subheader("🔥 Corrélations")
        
        # Sélection des colonnes numériques
        numeric_cols = ['ventes', 'clients', 'cout_acquisition', 'profit']
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax
        )
        
        st.pyplot(fig)

# Script principal pour lancer le dashboard
if __name__ == "__main__":
    dashboard = BusinessDashboard()
    dashboard.create_dashboard()
```

**Commande pour lancer :** `streamlit run dashboard_analytics.py`

---

## 📱 TÂCHE 5 - Développement Android

### **AD Task 01 - Application Todo**
**Repo:** `PRODIGY_AD_01`

#### MainActivity.java
```java
package com.prodigy.todoapp;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements TodoAdapter.OnTaskClickListener {
    
    private RecyclerView recyclerView;
    private TodoAdapter adapter;
    private List<TodoItem> todoList;
    private EditText editTextTask;
    private FloatingActionButton fabAdd;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initViews();
        setupRecyclerView();
        setupClickListeners();
    }
    
    private void initViews() {
        recyclerView = findViewById(R.id.recyclerViewTodos);
        editTextTask = findViewById(R.id.editTextTask);
        fabAdd = findViewById(R.id.fabAdd);
        
        todoList = new ArrayList<>();
        // Ajout de quelques tâches d'exemple
        todoList.add(new TodoItem("Terminer projet ML", false));
        todoList.add(new TodoItem("Réviser les algorithmes", true));
        todoList.add(new TodoItem("Préparer présentation", false));
    }
    
    private void setupRecyclerView() {
        adapter = new TodoAdapter(todoList, this);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));
        recyclerView.setAdapter(adapter);
    }
    
    private void setupClickListeners() {
        fabAdd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                addNewTask();
            }
        });
    }
    
    private void addNewTask() {
        String taskText = editTextTask.getText().toString().trim();
        
        if (taskText.isEmpty()) {
            Toast.makeText(this, "Veuillez saisir une tâche", Toast.LENGTH_SHORT).show();
            return;
        }
        
        TodoItem newTask = new TodoItem(taskText, false);
        todoList.add(0, newTask); // Ajouter en première position
        adapter.notifyItemInserted(0);
        
        editTextTask.setText("");
        recyclerView.scrollToPosition(0);
        
        Toast.makeText(this, "Tâche ajoutée ✅", Toast.LENGTH_SHORT).show();
    }
    
    @Override
    public void onTaskClick(int position) {
        TodoItem task = todoList.get(position);
        task.setCompleted(!task.isCompleted());
        adapter.notifyItemChanged(position);
        
        String message = task.isCompleted() ? "Tâche complétée ✅" : "Tâche réactivée 🔄";
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }
    
    @Override
    public void onTaskDelete(int position) {
        todoList.remove(position);
        adapter.notifyItemRemoved(position);
        Toast.makeText(this, "Tâche supprimée 🗑️", Toast.LENGTH_SHORT).show();
    }
}
```

#### TodoAdapter.java
```java
package com.prodigy.todoapp;

import android.graphics.Paint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.ImageButton;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;

public class TodoAdapter extends RecyclerView.Adapter<TodoAdapter.TodoViewHolder> {
    
    private List<TodoItem> todoList;
    private OnTaskClickListener listener;
    
    public interface OnTaskClickListener {
        void onTaskClick(int position);
        void onTaskDelete(int position);
    }
    
    public TodoAdapter(List<TodoItem> todoList, OnTaskClickListener listener) {
        this.todoList = todoList;
        this.listener = listener;
    }
    
    @NonNull
    @Override
    public TodoViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_todo, parent, false);
        return new TodoViewHolder(view);
    }
    
    @Override
    public void onBindViewHolder(@NonNull TodoViewHolder holder, int position) {
        TodoItem currentItem = todoList.get(position);
        
        holder.textViewTask.setText(currentItem.getTask());
        holder.checkBoxCompleted.setChecked(currentItem.isCompleted());
        
        // Style pour les tâches complétées
        if (currentItem.isCompleted()) {
            holder.textViewTask.setPaintFlags(
                holder.textViewTask.getPaintFlags() | Paint.STRIKE_THRU_TEXT_FLAG
            );
            holder.textViewTask.setAlpha(0.6f);
        } else {
            holder.textViewTask.setPaintFlags(
                holder.textViewTask.getPaintFlags() & (~Paint.STRIKE_THRU_TEXT_FLAG)
            );
            holder.textViewTask.setAlpha(1.0f);
        }
        
        // Click listeners
        holder.checkBoxCompleted.setOnClickListener(v -> {
            if (listener != null) {
                listener.onTaskClick(position);
            }
        });
        
        holder.buttonDelete.setOnClickListener(v -> {
            if (listener != null) {
                listener.onTaskDelete(position);
            }
        });
    }
    
    @Override
    public int getItemCount() {
        return todoList.size();
    }
    
    static class TodoViewHolder extends RecyclerView.ViewHolder {
        TextView textViewTask;
        CheckBox checkBoxCompleted;
        ImageButton buttonDelete;
        
        public TodoViewHolder(@NonNull View itemView) {
            super(itemView);
            textViewTask = itemView.findViewById(R.id.textViewTask);
            checkBoxCompleted = itemView.findViewById(R.id.checkBoxCompleted);
            buttonDelete = itemView.findViewById(R.id.buttonDelete);
        }
    }
}
```

#### TodoItem.java
```java
package com.prodigy.todoapp;

public class TodoItem {
    private String task;
    private boolean completed;
    
    public TodoItem(String task, boolean completed) {
        this.task = task;
        this.completed = completed;
    }
    
    // Getters and Setters
    public String getTask() { return task; }
    public void setTask(String task) { this.task = task; }
    
    public boolean isCompleted() { return completed; }
    public void setCompleted(boolean completed) { this.completed = completed; }
}
```

---

## 💻 TÂCHE 6 - Développement Logiciel

### **SW Task 01 - Application Desktop avec JavaFX**
**Repo:** `PRODIGY_SW_01`

```java
// TaskManagerApp.java
package com.prodigy.taskmanager;

import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.*;
import javafx.stage.Stage;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TaskManagerApp extends Application {
    
    private ObservableList<Task> tasks = FXCollections.observableArrayList();
    private ListView<Task> taskListView;
    private TextField titleField;
    private TextArea descriptionArea;
    private ComboBox<Priority> priorityComboBox;
    
    public static void main(String[] args) {
        launch(args);
    }
    
    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("📋 Task Manager Pro - Prodigy InfoTech");
        
        // Layout principal
        BorderPane root = new BorderPane();
        
        // Panel gauche - Formulaire d'ajout
        VBox leftPanel = createInputPanel();
        leftPanel.setPrefWidth(300);
        leftPanel.setPadding(new Insets(20));
        leftPanel.setStyle("-fx-background-color: #f8f9fa;");
        
        // Panel central - Liste des tâches
        VBox centerPanel = createTaskListPanel();
        centerPanel.setPadding(new Insets(20));
        
        // Panel droit - Statistiques
        VBox rightPanel = createStatsPanel();
        rightPanel.setPrefWidth(250);
        rightPanel.setPadding(new Insets(20));
        rightPanel.setStyle("-fx-background-color: #e9ecef;");
        
        root.setLeft(leftPanel);
        root.setCenter(centerPanel);
        root.setRight(rightPanel);
        
        // Ajout de tâches d'exemple
        addSampleTasks();
        
        Scene scene = new Scene(root, 1200, 800);
        scene.getStylesheets().add(getClass().getResource("styles.css").toExternalForm());
        
        primaryStage.setScene(scene);
        primaryStage.show();
    }
    
    private VBox createInputPanel() {
        VBox panel = new VBox(15);
        
        // Titre
        Label titleLabel = new Label("➕ Nouvelle Tâche");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Champs de saisie
        titleField = new TextField();
        titleField.setPromptText("Titre de la tâche");
        
        descriptionArea = new TextArea();
        descriptionArea.setPromptText("Description détaillée...");
        descriptionArea.setPrefRowCount(4);
        
        priorityComboBox = new ComboBox<>();
        priorityComboBox.getItems().addAll(Priority.values());
        priorityComboBox.setValue(Priority.MOYENNE);
        
        // Boutons
        Button addButton = new Button("✅ Ajouter Tâche");
        addButton.setStyle("-fx-background-color: #28a745; -fx-text-fill: white; -fx-font-weight: bold;");
        addButton.setOnAction(e -> addTask());
        
        Button clearButton = new Button("🗑️ Effacer");
        clearButton.setOnAction(e -> clearForm());
        
        HBox buttonBox = new HBox(10, addButton, clearButton);
        
        panel.getChildren().addAll(
            titleLabel,
            new Label("Titre:"), titleField,
            new Label("Description:"), descriptionArea,
            new Label("Priorité:"), priorityComboBox,
            buttonBox
        );
        
        return panel;
    }
    
    private VBox createTaskListPanel() {
        VBox panel = new VBox(15);
        
        Label titleLabel = new Label("📝 Mes Tâches");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Configuration ListView
        taskListView = new ListView<>(tasks);
        taskListView.setCellFactory(lv -> new TaskCell());
        taskListView.setPrefHeight(600);
        
        // Barre d'outils
        HBox toolbar = new HBox(10);
        
        Button completeButton = new Button("✅ Marquer Terminée");
        completeButton.setOnAction(e -> markTaskComplete());
        
        Button deleteButton = new Button("🗑️ Supprimer");
        deleteButton.setStyle("-fx-background-color: #dc3545; -fx-text-fill: white;");
        deleteButton.setOnAction(e -> deleteTask());
        
        ComboBox<String> filterCombo = new ComboBox<>();
        filterCombo.getItems().addAll("Toutes", "En cours", "Terminées", "Haute priorité");
        filterCombo.setValue("Toutes");
        filterCombo.setOnAction(e -> filterTasks(filterCombo.getValue()));
        
        toolbar.getChildren().addAll(completeButton, deleteButton, new Label("Filtrer:"), filterCombo);
        
        panel.getChildren().addAll(titleLabel, toolbar, taskListView);
        
        return panel;
    }
    
    private VBox createStatsPanel() {
        VBox panel = new VBox(15);
        
        Label titleLabel = new Label("📊 Statistiques");
        titleLabel.setStyle("-fx-font-size: 18px; -fx-font-weight: bold;");
        
        // Cartes de statistiques
        VBox totalCard = createStatCard("Total", "0", "#007bff");
        VBox completedCard = createStatCard("Terminées", "0", "#28a745");
        VBox pendingCard = createStatCard("En cours", "0", "#ffc107");
        VBox highPriorityCard = createStatCard("Haute priorité", "0", "#dc3545");
        
        panel.getChildren().addAll(titleLabel, totalCard, completedCard, pendingCard, highPriorityCard);
        
        return panel;
    }
    
    private VBox createStatCard(String title, String value, String color) {
        VBox card = new VBox(5);
        card.setStyle(String.format(
            "-fx-background-color: white; -fx-padding: 15; -fx-background-radius: 10; " +
            "-fx-border-color: %s; -fx-border-width: 2; -fx-border-radius: 10;", color
        ));
        
        Label titleLabel = new Label(title);
        titleLabel.setStyle("-fx-font-size: 12px; -fx-text-fill: #666;");
        
        Label valueLabel = new Label(value);
        valueLabel.setStyle(String.format("-fx-font-size: 24px; -fx-font-weight: bold; -fx-text-fill: %s;", color));
        
        card.getChildren().addAll(titleLabel, valueLabel);
        
        return card;
    }
    
    private void addTask() {
        String title = titleField.getText().trim();
        String description = descriptionArea.getText().trim();
        Priority priority = priorityComboBox.getValue();
        
        if (title.isEmpty()) {
            showAlert("Erreur", "Le titre est obligatoire!");
            return;
        }
        
        Task newTask = new Task(title, description, priority);
        tasks.add(newTask);
        
        clearForm();
        updateStats();
        
        showAlert("Succès", "Tâche ajoutée avec succès! ✅");
    }
    
    private void clearForm() {
        titleField.clear();
        descriptionArea.clear();
        priorityComboBox.setValue(Priority.MOYENNE);
    }
    
    private void markTaskComplete() {
        Task selected = taskListView.getSelectionModel().getSelectedItem();
        if (selected != null) {
            selected.setCompleted(!selected.isCompleted());
            taskListView.refresh();
            updateStats();
        }
    }
    
    private void deleteTask() {
        Task selected = taskListView.getSelectionModel().getSelectedItem();
        if (selected != null) {
            tasks.remove(selected);
            updateStats();
        }
    }
    
    private void filterTasks(String filter) {
        // Implementation du filtrage
        ObservableList<Task> filteredTasks = FXCollections.observableArrayList();
        
        for (Task task : tasks) {
            switch (filter) {
                case "Toutes":
                    filteredTasks.add(task);
                    break;
                case "En cours":
                    if (!task.isCompleted()) filteredTasks.add(task);
                    break;
                case "Terminées":
                    if (task.isCompleted()) filteredTasks.add(task);
                    break;
                case "Haute priorité":
                    if (task.getPriority() == Priority.HAUTE) filteredTasks.add(task);
                    break;
            }
        }
        
        taskListView.setItems(filteredTasks);
    }
    
    private void updateStats() {
        // Mise à jour des statistiques (simplified)
        System.out.println("Stats updated: " + tasks.size() + " tasks total");
    }
    
    private void addSampleTasks() {
        tasks.addAll(
            new Task("Terminer projet ML", "Finaliser le modèle de prédiction", Priority.HAUTE),
            new Task("Réviser algorithmes", "Revoir les concepts de base", Priority.MOYENNE),
            new Task("Préparer présentation", "Slides pour la démonstration finale", Priority.HAUTE)
        );
    }
    
    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
    
    // Classes internes
    enum Priority {
        BASSE("🟢 Basse"), MOYENNE("🟡 Moyenne"), HAUTE("🔴 Haute");
        
        private final String displayName;
        
        Priority(String displayName) {
            this.displayName = displayName;
        }
        
        @Override
        public String toString() {
            return displayName;
        }
    }
    
    static class Task {
        private String title;
        private String description;
        private Priority priority;
        private boolean completed;
        private LocalDateTime createdAt;
        
        public Task(String title, String description, Priority priority) {
            this.title = title;
            this.description = description;
            this.priority = priority;
            this.completed = false;
            this.createdAt = LocalDateTime.now();
        }
        
        // Getters and Setters
        public String getTitle() { return title; }
        public String getDescription() { return description; }
        public Priority getPriority() { return priority; }
        public boolean isCompleted() { return completed; }
        public void setCompleted(boolean completed) { this.completed = completed; }
        public LocalDateTime getCreatedAt() { return createdAt; }
        
        @Override
        public String toString() {
            String status = completed ? "✅" : "⏳";
            return String.format("%s %s %s - %s", 
                status, priority.toString(), title, 
                createdAt.format(DateTimeFormatter.ofPattern("dd/MM HH:mm"))
            );
        }
    }
    
    static class TaskCell extends ListCell<Task> {
        @Override
        protected void updateItem(Task task, boolean empty) {
            super.updateItem(task, empty);
            
            if (empty || task == null) {
                setText(null);
                setStyle("");
            } else {
                setText(task.toString());
                
                if (task.isCompleted()) {
                    setStyle("-fx-background-color: #d4edda; -fx-text-fill: #155724;");
                } else if (task.getPriority() == Priority.HAUTE) {
                    setStyle("-fx-background-color: #f8d7da; -fx-text-fill: #721c24;");
                } else {
                    setStyle("");
                }
            }
        }
    }
}
```

---

## 🔒 TÂCHE 7 - Cybersécurité

### **CY Task 01 - Analyseur de Sécurité Web**
**Repo:** `PRODIGY_CY_01`

```python
# security_analyzer.py
import requests
import ssl
import socket
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime
import hashlib
import re

class WebSecurityAnalyzer:
    def __init__(self, target_url):
        self.target_url = target_url
        self.parsed_url = urlparse(target_url)
        self.vulnerabilities = []
        self.security_score = 100
        
    def analyze_security(self):
        """Lance une analyse complète de sécurité"""
        print(f"🔍 ANALYSE SÉCURISÉE DE: {self.target_url}")
        print("=" * 60)
        
        try:
            # Tests de sécurité
            self.check_https()
            self.check_security_headers()
            self.check_ssl_certificate()
            self.check_common_vulnerabilities()
            self.scan_open_ports()
            
            # Génération du rapport
            self.generate_report()
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse: {e}")
    
    def check_https(self):
        """Vérifie l'utilisation de HTTPS"""
        print("🔒 Vérification HTTPS...")
        
        if self.parsed_url.scheme != 'https':
            self.add_vulnerability(
                "HTTP Non Sécurisé",
                "Le site utilise HTTP au lieu de HTTPS",
                "HAUTE",
                "Implémenter HTTPS avec un certificat SSL valide"
            )
        else:
            print("✅ HTTPS activé")
    
    def check_security_headers(self):
        """Vérifie les en-têtes de sécurité"""
        print("🛡️ Vérification des en-têtes de sécurité...")
        
        try:
            response = requests.get(self.target_url, timeout=10)
            headers = response.headers
            
            # En-têtes de sécurité essentiels
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,
                'Content-Security-Policy': None,
                'Referrer-Policy': None
            }
            
            for header, expected in security_headers.items():
                if header not in headers:
                    self.add_vulnerability(
                        f"En-tête manquant: {header}",
                        f"L'en-tête de sécurité {header} n'est pas présent",
                        "MOYENNE",
                        f"Ajouter l'en-tête {header} dans la configuration serveur"
                    )
                else:
                    print(f"✅ {header}: {headers[header]}")
                    
        except requests.RequestException as e:
            print(f"❌ Erreur lors de la vérification des en-têtes: {e}")
    
    def check_ssl_certificate(self):
        """Vérifie le certificat SSL"""
        print("📜 Vérification du certificat SSL...")
        
        if self.parsed_url.scheme != 'https':
            return
            
        try:
            hostname = self.parsed_url.hostname
            port = self.parsed_url.port or 443
            
            # Création du contexte SSL
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    # Vérification de la validité
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.now()).days
                    
                    if days_until_expiry < 30:
                        self.add_vulnerability(
                            "Certificat SSL expirant",
                            f"Le certificat expire dans {days_until_expiry} jours",
                            "HAUTE",
                            "Renouveler le certificat SSL avant expiration"
                        )
                    else:
                        print(f"✅ Certificat SSL valide (expire dans {days_until_expiry} jours)")
                        
        except Exception as e:
            self.add_vulnerability(
                "Problème certificat SSL",
                f"Erreur lors de la vérification: {e}",
                "HAUTE",
                "Vérifier la configuration SSL du serveur"
            )
    
    def check_common_vulnerabilities(self):
        """Vérifie les vulnérabilités communes"""
        print("🎯 Recherche de vulnérabilités communes...")
        
        # Test de divulgation d'information
        common_files = [
            '/.git/HEAD',
            '/admin',
            '/wp-admin/',
            '/.env',
            '/config.php',
            '/phpinfo.php',
            '/server-status',
            '/robots.txt'
        ]
        
        for file_path in common_files:
            test_url = urljoin(self.target_url, file_path)
            try:
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    if file_path in ['/.git/HEAD', '/.env', '/config.php']:
                        self.add_vulnerability(
                            f"Fichier sensible exposé: {file_path}",
                            f"Le fichier {file_path} est accessible publiquement",
                            "CRITIQUE",
                            f"Bloquer l'accès au fichier {file_path}"
                        )
                    else:
                        print(f"ℹ️ Fichier trouvé: {file_path}")
                        
            except requests.RequestException:
                continue
    
    def scan_open_ports(self):
        """Scan basique des ports ouverts"""
        print("🔍 Scan des ports communs...")
        
        hostname = self.parsed_url.hostname
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432]
        
        open_ports = []
        
        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            try:
                result = sock.connect_ex((hostname, port))
                if result == 0:
                    open_ports.append(port)
                sock.close()
            except:
                continue
        
        if open_ports:
            print(f"🔓 Ports ouverts détectés: {open_ports}")
            
            # Ports potentiellement dangereux
            dangerous_ports = [21, 23, 25, 3306, 5432]
            exposed_dangerous = [p for p in open_ports if p in dangerous_ports]
            
            if exposed_dangerous:
                self.add_vulnerability(
                    "Ports sensibles exposés",
                    f"Ports potentiellement dangereux ouverts: {exposed_dangerous}",
                    "HAUTE",
                    "Fermer les ports non nécessaires et utiliser un firewall"
                )
    
    def add_vulnerability(self, title, description, severity, recommendation):
        """Ajoute une vulnérabilité détectée"""
        vulnerability = {
            'title': title,
            'description': description,
            'severity': severity,
            'recommendation': recommendation,
            'detected_at': datetime.now().isoformat()
        }
        
        self.vulnerabilities.append(vulnerability)
        
        # Réduction du score de sécurité
        severity_scores = {'CRITIQUE': 25, 'HAUTE': 15, 'MOYENNE': 10, 'BASSE': 5}
        self.security_score -= severity_scores.get(severity, 5)
        
        print(f"⚠️ {severity}: {title}")
    
    def generate_report(self):
        """Génère un rapport complet de sécurité"""
        print("\n" + "=" * 60)
        print("📋 RAPPORT DE SÉCURITÉ")
        print("=" * 60)
        
        print(f"🎯 URL analysée: {self.target_url}")
        print(f"📅 Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🏆 Score de sécurité: {max(0, self.security_score)}/100")
        
        if self.security_score >= 80:
            print("✅ Niveau de sécurité: EXCELLENT")
        elif self.security_score >= 60:
            print("⚠️ Niveau de sécurité: BON")
        elif self.security_score >= 40:
            print("🔶 Niveau de sécurité: MOYEN")
        else:
            print("❌ Niveau de sécurité: CRITIQUE")
        
        print(f"\n🎯 Vulnérabilités détectées: {len(self.vulnerabilities)}")
        
        # Détail des vulnérabilités
        if self.vulnerabilities:
            print("\n⚠️ VULNÉRABILITÉS DÉTECTÉES:")
            print("-" * 40)
            
            for i, vuln in enumerate(self.vulnerabilities, 1):
                severity_icons = {
                    'CRITIQUE': '🔴',
                    'HAUTE': '🟠', 
                    'MOYENNE': '🟡',
                    'BASSE': '🟢'
                }
                
                print(f"\n{i}. {severity_icons.get(vuln['severity'], '⚪')} {vuln['title']}")
                print(f"   Sévérité: {vuln['severity']}")
                print(f"   Description: {vuln['description']}")
                print(f"   Recommandation: {vuln['recommendation']}")
        
        # Sauvegarde du rapport
        self.save_report_json()
        
        print(f"\n✅ Rapport sauvegardé: security_report_{self.parsed_url.hostname}.json")
        print("\n🔒 Analyse terminée!")
    
    def save_report_json(self):
        """Sauvegarde le rapport en JSON"""
        report = {
            'target_url': self.target_url,
            'analysis_date': datetime.now().isoformat(),
            'security_score': max(0, self.security_score),
            'vulnerabilities_count': len(self.vulnerabilities),
            'vulnerabilities': self.vulnerabilities
        }
        
        filename = f"security_report_{self.parsed_url.hostname}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

# Exemple d'utilisation
if __name__ == "__main__":
    # Analyse de sécurité d'un site web
    analyzer = WebSecurityAnalyzer("https://example.com")
    analyzer.analyze_security()
```

### **CY Task 02 - Générateur de Mots de Passe Sécurisés**
**Repo:** `PRODIGY_CY_02`

```python
# password_security.py
import secrets
import string
import hashlib
import re
from typing import List, Dict
import json
from datetime import datetime

class PasswordSecurityTool:
    def __init__(self):
        self.common_passwords = self.load_common_passwords()
        self.password_history = []
    
    def load_common_passwords(self) -> List[str]:
        """Charge une liste de mots de passe communs"""
        # Top mots de passe faibles (pour la démonstration)
        return [
            "123456", "password", "123456789", "12345678", "12345",
            "1234567", "1234567890", "qwerty", "abc123", "million",
            "000000", "1234", "iloveyou", "aaron431", "password1",
            "qqww1122", "123", "omgpop", "123321", "654321"
        ]
    
    def generate_secure_password(self, length: int = 16, 
                                include_symbols: bool = True,
                                exclude_ambiguous: bool = True) -> Dict:
        """Génère un mot de passe sécurisé"""
        
        if length < 8:
            raise ValueError("La longueur minimale doit être de 8 caractères")
        
        # Définition des caractères
        lowercase = string.ascii_lowercase
        uppercase = string.ascii_uppercase
        digits = string.digits
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Exclusion des caractères ambigus si demandé
        if exclude_ambiguous:
            lowercase = lowercase.replace('l', '').replace('o', '')
            uppercase = uppercase.replace('I', '').replace('O', '')
            digits = digits.replace('0', '').replace('1', '')
            symbols = symbols.replace('|', '').replace('l', '')
        
        # Construction du jeu de caractères
        all_chars = lowercase + uppercase + digits
        if include_symbols:
            all_chars += symbols
        
        # Génération du mot de passe avec garantie de diversité
        password = []
        
        # Au moins un caractère de chaque type
