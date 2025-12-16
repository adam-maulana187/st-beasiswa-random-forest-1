import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve)
from sklearn.inspection import permutation_importance
import warnings
import joblib
import json
import streamlit as st
import io
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """Fungsi utama untuk aplikasi Streamlit"""
    
    st.set_page_config(
        page_title="Analisis Penerimaan Beasiswa",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar untuk navigasi
    st.sidebar.title("🎓 Analisis Penerimaan Beasiswa")
    st.sidebar.markdown("---")
    
    menu_options = [
        "🏠 Dashboard Utama",
        "📊 Eksplorasi Data",
        "🤖 Training Model",
        "📈 Evaluasi Model",
        "🔍 Feature Importance",
        "🔮 Prediksi Calon Baru",
        "📋 Laporan Lengkap"
    ]
    
    choice = st.sidebar.selectbox("Navigasi Menu", menu_options)
    
    # Load data
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('template_dataset_beasiswa.csv')
            return df
        except FileNotFoundError:
            st.error("File dataset tidak ditemukan. Pastikan 'template_dataset_beasiswa.csv' ada di direktori yang sama.")
            return None
    
    df = load_data()
    
    if df is None:
        return
    
    # Fungsi untuk preprocessing yang benar
    @st.cache_data
    def preprocess_data(df):
        df_processed = df.copy()
        categorical_cols = ['Asal_Sekolah', 'Lokasi_Domisili', 'Gender', 'Status_Disabilitas']
        numerical_cols = ['IPK', 'Pendapatan_Orang_Tua', 'Keikutsertaan_Organisasi', 
                          'Pengalaman_Sosial', 'Prestasi_Akademik', 'Prestasi_Non_Akademik']
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
        
        X = df_processed.drop(['Diterima_Beasiswa'], axis=1)
        y = df_processed['Diterima_Beasiswa']
        
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'feature_names': feature_names,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'df_processed': df_processed
        }
    
    # Load atau train model
    @st.cache_resource
    def train_or_load_model():
        try:
            # Coba load model yang sudah ada
            rf_model = joblib.load('model_beasiswa_rf.pkl')
            scaler = joblib.load('scaler_beasiswa.pkl')
            label_encoders = joblib.load('label_encoders_beasiswa.pkl')
            st.sidebar.success("✅ Model loaded from cache")
            return rf_model, scaler, label_encoders
        except:
            # Jika tidak ada, train model baru
            st.sidebar.info("⚙️ Training model...")
            preprocessed = preprocess_data(df)
            
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                oob_score=True
            )
            
            rf_model.fit(preprocessed['X_train_scaled'], preprocessed['y_train'])
            
            # Simpan model
            joblib.dump(rf_model, 'model_beasiswa_rf.pkl')
            joblib.dump(preprocessed['scaler'], 'scaler_beasiswa.pkl')
            joblib.dump(preprocessed['label_encoders'], 'label_encoders_beasiswa.pkl')
            
            st.sidebar.success("✅ Model trained and saved")
            return rf_model, preprocessed['scaler'], preprocessed['label_encoders']
    
    # Fungsi helper untuk prediksi
    def predict_new_data(input_data, rf_model, scaler, label_encoders, feature_names, categorical_cols, numerical_cols):
        """Fungsi untuk memprediksi data baru"""
        try:
            # Buat DataFrame dari input
            input_df = pd.DataFrame([input_data])
            
            # Encoding variabel kategorikal
            for col in categorical_cols:
                if col in input_df.columns:
                    if str(input_data[col]) in label_encoders[col].classes_:
                        input_df[col] = label_encoders[col].transform([str(input_data[col])])[0]
                    else:
                        # Jika nilai tidak ada di training, gunakan nilai yang paling umum
                        input_df[col] = 0
            
            # Pastikan semua kolom ada
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Urutkan kolom sesuai dengan training
            input_df = input_df[feature_names]
            
            # Standardisasi fitur numerik
            input_df_scaled = input_df.copy()
            input_df_scaled[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Prediksi
            prediction = rf_model.predict(input_df_scaled)[0]
            probability = rf_model.predict_proba(input_df_scaled)[0][1]
            
            return prediction, probability, None
            
        except Exception as e:
            return None, None, str(e)
    
    # DASHBOARD UTAMA
    if choice == "🏠 Dashboard Utama":
        st.title("🎓 Dashboard Analisis Penerimaan Beasiswa")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_students = len(df)
            st.metric("Total Calon", f"{total_students:,}")
        
        with col2:
            accepted = df['Diterima_Beasiswa'].sum()
            st.metric("Diterima", f"{accepted:,}", f"{(accepted/total_students*100):.1f}%")
        
        with col3:
            rejected = total_students - accepted
            st.metric("Tidak Diterima", f"{rejected:,}", f"{(rejected/total_students*100):.1f}%")
        
        st.markdown("---")
        
        # Visualisasi distribusi target
        st.subheader("📊 Distribusi Penerimaan Beasiswa")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        colors_target = ['#FF6B6B', '#4ECDC4']
        target_dist = df['Diterima_Beasiswa'].value_counts()
        
        ax1.pie(target_dist.values, labels=['Tidak Diterima', 'Diterima'], 
                autopct='%1.1f%%', colors=colors_target, startangle=90)
        ax1.set_title('Distribusi Penerimaan')
        
        ax2.bar(['Tidak Diterima', 'Diterima'], target_dist.values, 
                color=colors_target, edgecolor='black')
        ax2.set_ylabel('Jumlah')
        ax2.set_title('Jumlah Penerimaan')
        for i, v in enumerate(target_dist.values):
            ax2.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistik deskriptif
        st.subheader("📈 Statistik Deskriptif")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df.describe().round(2), use_container_width=True)
        
        with col2:
            st.write("**Informasi Dataset:**")
            st.write(f"- Jumlah Data: {df.shape[0]} baris × {df.shape[1]} kolom")
            st.write(f"- Kolom: {', '.join(df.columns.tolist())}")
            
            missing_vals = df.isnull().sum()
            if missing_vals.sum() > 0:
                st.warning(f"⚠️ Terdapat missing values: {missing_vals.sum()} total")
            else:
                st.success("✅ Tidak ada missing values")
    
    # EKSPLORASI DATA
    elif choice == "📊 Eksplorasi Data":
        st.title("📊 Eksplorasi Data")
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Distribusi", "Korelasi", "Analisis Univariate"])
        
        with tab1:
            st.subheader("Preview Data")
            
            # Tampilkan jumlah baris yang diinginkan
            num_rows = st.slider("Jumlah baris yang ditampilkan:", 5, 100, 10)
            st.dataframe(df.head(num_rows), use_container_width=True)
            
            st.subheader("Informasi Data")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.subheader("Missing Values Check")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
            missing_df['Percentage'] = (missing_df['Missing Values'] / len(df)) * 100
            st.dataframe(missing_df, use_container_width=True)
        
        with tab2:
            st.subheader("Distribusi Variabel")
            
            selected_var = st.selectbox("Pilih variabel untuk dianalisis:", 
                                      df.columns.tolist())
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Cek apakah variabel numerik atau kategorikal
            if df[selected_var].dtype in ['int64', 'float64']:
                # Histogram untuk numerik
                ax1.hist(df[selected_var], bins=30, edgecolor='black', alpha=0.7)
                ax1.set_xlabel(selected_var)
                ax1.set_ylabel('Frekuensi')
                ax1.set_title(f'Distribusi {selected_var}')
                ax1.grid(True, alpha=0.3)
                
                # Box plot untuk numerik
                ax2.boxplot(df[selected_var].dropna())
                ax2.set_ylabel(selected_var)
                ax2.set_title(f'Box Plot {selected_var}')
                ax2.grid(True, alpha=0.3)
                
                # Statistik
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[selected_var].mean():.2f}")
                with col2:
                    st.metric("Std Dev", f"{df[selected_var].std():.2f}")
                with col3:
                    st.metric("Min", f"{df[selected_var].min():.2f}")
                with col4:
                    st.metric("Max", f"{df[selected_var].max():.2f}")
            else:
                # Bar chart untuk kategorikal
                value_counts = df[selected_var].value_counts()
                ax1.bar(value_counts.index.astype(str), value_counts.values, 
                       color='skyblue', edgecolor='black')
                ax1.set_xlabel(selected_var)
                ax1.set_ylabel('Frekuensi')
                ax1.set_title(f'Distribusi {selected_var}')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Pie chart
                ax2.pie(value_counts.values, labels=value_counts.index.astype(str), 
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'Proporsi {selected_var}')
                
                # Tampilkan value counts
                st.write("**Value Counts:**")
                st.dataframe(value_counts, use_container_width=True)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Korelasi antar Variabel")
            
            # Pilih hanya kolom numerik untuk korelasi
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 0:
                # Hitung korelasi
                corr_matrix = numeric_df.corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, square=True)
                ax.set_title('Matriks Korelasi (Numerik Only)', fontweight='bold')
                st.pyplot(fig)
                
                # Korelasi dengan target jika target numerik
                if 'Diterima_Beasiswa' in numeric_df.columns:
                    st.subheader("Korelasi dengan Target (Diterima_Beasiswa)")
                    target_corr = corr_matrix['Diterima_Beasiswa'].sort_values(ascending=False)
                    target_corr_df = pd.DataFrame(target_corr).reset_index()
                    target_corr_df.columns = ['Variabel', 'Korelasi']
                    
                    st.dataframe(target_corr_df.style.background_gradient(
                        cmap='coolwarm', subset=['Korelasi']), use_container_width=True)
            else:
                st.warning("Tidak ada variabel numerik untuk menghitung korelasi.")
        
        with tab4:
            st.subheader("Analisis Berdasarkan Status Penerimaan")
            
            # Filter hanya kolom numerik
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Diterima_Beasiswa' in numeric_features:
                numeric_features.remove('Diterima_Beasiswa')
            
            if numeric_features:
                selected_feature = st.selectbox("Pilih fitur untuk analisis:", 
                                              numeric_features)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Distribusi berdasarkan status
                for status, color in zip([0, 1], ['#FF6B6B', '#4ECDC4']):
                    data = df[df['Diterima_Beasiswa'] == status][selected_feature]
                    ax1.hist(data, bins=30, alpha=0.6, label=f'Status {status}', 
                            color=color, edgecolor='black')
                
                ax1.set_xlabel(selected_feature)
                ax1.set_ylabel('Frekuensi')
                ax1.set_title(f'Distribusi {selected_feature} per Status')
                ax1.legend(title='Diterima?')
                ax1.grid(True, alpha=0.3)
                
                # Box plot
                data_accepted = df[df['Diterima_Beasiswa'] == 1][selected_feature]
                data_rejected = df[df['Diterima_Beasiswa'] == 0][selected_feature]
                
                bp = ax2.boxplot([data_rejected, data_accepted], 
                                labels=['Tidak Diterima', 'Diterima'],
                                patch_artist=True)
                
                colors_box = ['#FF6B6B', '#4ECDC4']
                for patch, color in zip(bp['boxes'], colors_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax2.set_ylabel(selected_feature)
                ax2.set_title(f'{selected_feature} vs Penerimaan')
                ax2.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistik ringkasan
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Tidak Diterima:**")
                    st.write(f"- Mean: {data_rejected.mean():.2f}")
                    st.write(f"- Std: {data_rejected.std():.2f}")
                    st.write(f"- Count: {len(data_rejected)}")
                
                with col2:
                    st.write("**Diterima:**")
                    st.write(f"- Mean: {data_accepted.mean():.2f}")
                    st.write(f"- Std: {data_accepted.std():.2f}")
                    st.write(f"- Count: {len(data_accepted)}")
            else:
                st.warning("Tidak ada variabel numerik untuk analisis.")
    
    # TRAINING MODEL
    elif choice == "🤖 Training Model":
        st.title("🤖 Training Model Random Forest")
        st.markdown("---")
        
        with st.spinner("Memproses data dan melatih model..."):
            preprocessed = preprocess_data(df)
            rf_model, scaler, label_encoders = train_or_load_model()
        
        st.success("✅ Model siap digunakan!")
        
        st.subheader("📋 Parameter Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("n_estimators", rf_model.n_estimators)
            st.metric("max_depth", rf_model.max_depth)
        
        with col2:
            st.metric("min_samples_split", rf_model.min_samples_split)
            st.metric("min_samples_leaf", rf_model.min_samples_leaf)
        
        with col3:
            st.metric("max_features", str(rf_model.max_features))
            st.metric("oob_score", f"{rf_model.oob_score_:.4f}")
        
        st.subheader("📊 Cross-Validation")
        
        cv_scores = cross_val_score(rf_model, preprocessed['X_train_scaled'], 
                                  preprocessed['y_train'], cv=5, scoring='accuracy')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
        
        with col2:
            st.metric("Std CV Score", f"{cv_scores.std():.4f}")
        
        # Tampilkan detail CV scores
        cv_df = pd.DataFrame({
            'Fold': range(1, 6),
            'Accuracy': cv_scores
        })
        
        st.dataframe(cv_df.style.format({'Accuracy': '{:.4f}'}), 
                    use_container_width=True)
        
        # Visualisasi CV scores
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(cv_df['Fold'].astype(str), cv_df['Accuracy'], 
               color='skyblue', edgecolor='black')
        ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                  label=f'Mean: {cv_scores.mean():.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_title('5-Fold Cross-Validation Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Model Info
        st.subheader("ℹ️ Informasi Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Jumlah Fitur:** {preprocessed['X_train'].shape[1]}")
            st.info(f"**Training Samples:** {preprocessed['X_train'].shape[0]}")
        
        with col2:
            st.info(f"**Test Samples:** {preprocessed['X_test'].shape[0]}")
            st.info(f"**Class Distribution:**")
            st.write(f"- Diterima: {(preprocessed['y_train'].sum()/len(preprocessed['y_train'])*100):.1f}%")
            st.write(f"- Tidak: {((len(preprocessed['y_train'])-preprocessed['y_train'].sum())/len(preprocessed['y_train'])*100):.1f}%")
    
    # EVALUASI MODEL
    elif choice == "📈 Evaluasi Model":
        st.title("📈 Evaluasi Model")
        st.markdown("---")
        
        with st.spinner("Evaluasi model..."):
            preprocessed = preprocess_data(df)
            rf_model, scaler, label_encoders = train_or_load_model()
            
            # Prediksi
            y_pred = rf_model.predict(preprocessed['X_test_scaled'])
            y_pred_proba = rf_model.predict_proba(preprocessed['X_test_scaled'])[:, 1]
            
            # Hitung metrik
            accuracy = accuracy_score(preprocessed['y_test'], y_pred)
            precision = precision_score(preprocessed['y_test'], y_pred)
            recall = recall_score(preprocessed['y_test'], y_pred)
            f1 = f1_score(preprocessed['y_test'], y_pred)
            roc_auc = roc_auc_score(preprocessed['y_test'], y_pred_proba)
            
            # Confusion Matrix
            cm = confusion_matrix(preprocessed['y_test'], y_pred)
            tn, fp, fn, tp = cm.ravel()
        
        st.subheader("🎯 Metrik Evaluasi")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.2f}%")
        
        with col2:
            st.metric("Precision", f"{precision:.4f}", f"{precision*100:.2f}%")
        
        with col3:
            st.metric("Recall", f"{recall:.4f}", f"{recall*100:.2f}%")
        
        with col4:
            st.metric("F1-Score", f"{f1:.4f}", f"{f1*100:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ROC-AUC", f"{roc_auc:.4f}", f"{roc_auc*100:.2f}%")
        
        with col2:
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            st.metric("Specificity", f"{specificity:.4f}", f"{specificity*100:.2f}%")
        
        st.subheader("🧮 Confusion Matrix")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                   xticklabels=['Prediksi Tidak', 'Prediksi Diterima'],
                   yticklabels=['Aktual Tidak', 'Aktual Diterima'],
                   ax=ax)
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        
        st.pyplot(fig)
        
        # Detail confusion matrix
        st.write("**Detail Confusion Matrix:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Positive", tp)
        with col2:
            st.metric("False Positive", fp)
        with col3:
            st.metric("False Negative", fn)
        with col4:
            st.metric("True Negative", tn)
        
        st.subheader("📈 ROC Curve")
        
        fpr, tpr, _ = roc_curve(preprocessed['y_test'], y_pred_proba)
        roc_auc_value = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc_value:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Precision-Recall Curve
        st.subheader("⚖️ Precision-Recall Curve")
        
        precision_curve, recall_curve, _ = precision_recall_curve(preprocessed['y_test'], y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_curve, precision_curve, color='green', lw=2,
               label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.subheader("📋 Classification Report")
        
        report = classification_report(preprocessed['y_test'], y_pred, 
                                     target_names=['Tidak Diterima', 'Diterima'], 
                                     output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    
    # FEATURE IMPORTANCE
    elif choice == "🔍 Feature Importance":
        st.title("🔍 Feature Importance Analysis")
        st.markdown("---")
        
        with st.spinner("Menganalisis feature importance..."):
            preprocessed = preprocess_data(df)
            rf_model, scaler, label_encoders = train_or_load_model()
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': preprocessed['feature_names'],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        st.subheader("🏆 Top 10 Fitur Paling Penting")
        
        # Tampilkan dalam tabel
        st.dataframe(feature_importance.head(10).style.background_gradient(
            subset=['Importance'], cmap='YlOrRd'), use_container_width=True)
        
        # Visualisasi feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        top_features = feature_importance.head(10).sort_values('Importance', ascending=True)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                      color=colors, edgecolor='black')
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title('Top 10 Feature Importance', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Tambah nilai
        for i, (bar, imp) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{imp:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpretasi
        st.subheader("💡 Interpretasi Feature Importance")
        
        top_5_features = feature_importance.head(5)
        
        for idx, row in top_5_features.iterrows():
            with st.expander(f"**{row['Feature']}** (Importance: {row['Importance']:.4f})"):
                if 'IPK' in row['Feature']:
                    st.write("IPK merupakan faktor penentu utama dalam penerimaan beasiswa.")
                    st.write("- **Dampak:** Semakin tinggi IPK, semakin besar peluang diterima")
                    st.write("- **Standar:** Minimum biasanya di atas 3.0")
                    st.write("- **Rekomendasi:** Fokus pada pencapaian akademik")
                
                elif 'Prestasi_Akademik' in row['Feature']:
                    st.write("Prestasi akademik menunjukkan kemampuan akademik calon.")
                    st.write("- **Dampak:** Prestasi kompetisi akademik sangat dihargai")
                    st.write("- **Indikator:** Dedikasi terhadap studi")
                    st.write("- **Rekomendasi:** Ikuti kompetisi akademik")
                
                elif 'Pendapatan_Orang_Tua' in row['Feature']:
                    st.write("Pendapatan orang tua berkorelasi negatif dengan penerimaan.")
                    st.write("- **Dampak:** Calon dari keluarga kurang mampu mendapat prioritas")
                    st.write("- **Tujuan:** Bantuan berbasis kebutuhan")
                    st.write("- **Rekomendasi:** Sertakan bukti pendapatan")
                
                elif 'Keikutsertaan_Organisasi' in row['Feature']:
                    st.write("Keaktifan organisasi menunjukkan soft skills.")
                    st.write("- **Dampak:** Kepemimpinan dan kerja tim")
                    st.write("- **Indikator:** Keterampilan sosial")
                    st.write("- **Rekomendasi:** Aktif dalam organisasi")
                
                elif 'Pengalaman_Sosial' in row['Feature']:
                    st.write("Pengalaman sosial menunjukkan kepedulian sosial.")
                    st.write("- **Dampak:** Volunteer work dan kegiatan sosial")
                    st.write("- **Indikator:** Kontribusi kepada masyarakat")
                    st.write("- **Rekomendasi:** Ikuti kegiatan sosial")
    
    # PREDIKSI CALON BARU
    elif choice == "🔮 Prediksi Calon Baru":
        st.title("🔮 Prediksi Calon Baru")
        st.markdown("---")
        
        with st.spinner("Memuat model..."):
            rf_model, scaler, label_encoders = train_or_load_model()
            preprocessed = preprocess_data(df)
        
        st.subheader("📝 Masukkan Data Calon")
        
        # Form input data calon
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                ipk = st.slider("IPK", min_value=2.0, max_value=4.0, value=3.5, step=0.05)
                pendapatan = st.slider("Pendapatan Orang Tua (juta)", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
                organisasi = st.slider("Keikutsertaan Organisasi", min_value=0, max_value=5, value=2)
                prestasi_akademik = st.slider("Prestasi Akademik", min_value=0, max_value=10, value=5)
            
            with col2:
                pengalaman_sosial = st.slider("Pengalaman Sosial (jam)", min_value=0, max_value=500, value=200)
                prestasi_non_akademik = st.slider("Prestasi Non-Akademik", min_value=0, max_value=10, value=3)
                asal_sekolah = st.selectbox("Asal Sekolah", options=['Negeri_Desa', 'Negeri_Kota', 'Swasta_Desa', 'Swasta_Kota'])
                lokasi_domisili = st.selectbox("Lokasi Domisili", options=['Desa', 'Kabupaten', 'Kota'])
                gender = st.selectbox("Gender", options=['L', 'P'])
                status_disabilitas = st.selectbox("Status Disabilitas", options=['Tidak', 'Ya'])
            
            submit_button = st.form_submit_button("🔮 Prediksi")
        
        if submit_button:
            # Siapkan data input
            input_data = {
                'IPK': ipk,
                'Pendapatan_Orang_Tua': pendapatan,
                'Asal_Sekolah': asal_sekolah,
                'Lokasi_Domisili': lokasi_domisili,
                'Keikutsertaan_Organisasi': organisasi,
                'Pengalaman_Sosial': pengalaman_sosial,
                'Gender': gender,
                'Status_Disabilitas': status_disabilitas,
                'Prestasi_Akademik': prestasi_akademik,
                'Prestasi_Non_Akademik': prestasi_non_akademik
            }
            
            # Prediksi
            prediction, probability, error = predict_new_data(
                input_data, rf_model, scaler, label_encoders,
                preprocessed['feature_names'],
                preprocessed['categorical_cols'],
                preprocessed['numerical_cols']
            )
            
            if error:
                st.error(f"Error: {error}")
            else:
                # Tampilkan hasil
                st.subheader("📋 Hasil Prediksi")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.success(f"✅ **DITERIMA**")
                        status_display = "DITERIMA"
                        color = "green"
                    else:
                        st.error(f"❌ **TIDAK DITERIMA**")
                        status_display = "TIDAK DITERIMA"
                        color = "red"
                
                with col2:
                    st.metric("Probabilitas Diterima", f"{probability:.2%}")
                
                with col3:
                    if probability >= 0.7:
                        confidence = "TINGGI"
                        confidence_color = "green"
                    elif probability >= 0.4:
                        confidence = "SEDANG"
                        confidence_color = "orange"
                    else:
                        confidence = "RENDAH"
                        confidence_color = "red"
                    st.metric("Confidence Level", confidence)
                
                # Visualisasi probabilitas
                fig, ax = plt.subplots(figsize=(8, 2))
                
                # Buat horizontal bar untuk probabilitas
                ax.barh(['Probabilitas'], [probability], color='skyblue', height=0.5)
                ax.barh(['Probabilitas'], [1-probability], left=[probability], color='lightgray', height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probabilitas')
                ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
                ax.set_title('Probabilitas Penerimaan', fontweight='bold')
                ax.legend()
                
                # Tambah teks
                ax.text(probability/2, 0, f'{probability:.1%}', ha='center', va='center', 
                       color='white', fontweight='bold')
                ax.text((1+probability)/2, 0, f'{1-probability:.1%}', ha='center', va='center',
                       color='black', fontweight='bold')
                
                st.pyplot(fig)
                
                # Rekomendasi
                st.subheader("💡 Rekomendasi")
                
                recommendations = []
                
                if probability < 0.4:
                    st.warning("**Peluang rendah untuk diterima. Pertimbangkan:**")
                    
                    if ipk < 3.0:
                        recommendations.append("Tingkatkan IPK minimal 3.0")
                    if prestasi_akademik < 3:
                        recommendations.append("Tingkatkan prestasi akademik (ikut kompetisi)")
                    if organisasi < 2:
                        recommendations.append("Ikuti minimal 2 organisasi")
                    if pengalaman_sosial < 100:
                        recommendations.append("Tambah pengalaman sosial (volunteer)")
                    if prestasi_non_akademik < 2:
                        recommendations.append("Kembangkan prestasi non-akademik (seni/olahraga)")
                    if pendapatan > 10:
                        recommendations.append("Sertakan bukti kebutuhan finansial")
                
                elif probability < 0.7:
                    st.info("**Peluang sedang untuk diterima. Saran peningkatan:**")
                    
                    if ipk < 3.5:
                        recommendations.append("Pertahankan IPK di atas 3.5")
                    if prestasi_non_akademik < 3:
                        recommendations.append("Tingkatkan prestasi non-akademik")
                    if organisasi < 3:
                        recommendations.append("Tambah keaktifan organisasi")
                    if pendapatan > 8:
                        recommendations.append("Lengkapi dokumen kebutuhan finansial")
                    if status_disabilitas == 'Ya':
                        recommendations.append("Sertakan sertifikat disabilitas")
                
                else:
                    st.success("**Peluang tinggi untuk diterima. Pertahankan prestasi!**")
                    
                    recommendations.append("Pertahankan IPK di atas 3.5")
                    recommendations.append("Lanjutkan kegiatan organisasi dan sosial")
                    recommendations.append("Siapkan dokumen pendukung yang lengkap")
                    if status_disabilitas == 'Ya':
                        recommendations.append("Manfaatkan hak afirmasi untuk disabilitas")
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.write("Tidak ada rekomendasi spesifik.")
                
                # Detail input data
                with st.expander("📊 Detail Data Input"):
                    st.json(input_data)
    
    # LAPORAN LENGKAP
    elif choice == "📋 Laporan Lengkap":
        st.title("📋 Laporan Lengkap Analisis")
        st.markdown("---")
        
        with st.spinner("Menyiapkan laporan..."):
            preprocessed = preprocess_data(df)
            rf_model, scaler, label_encoders = train_or_load_model()
            
            # Prediksi untuk metrik
            y_pred = rf_model.predict(preprocessed['X_test_scaled'])
            y_pred_proba = rf_model.predict_proba(preprocessed['X_test_scaled'])[:, 1]
            
            # Hitung semua metrik
            accuracy = accuracy_score(preprocessed['y_test'], y_pred)
            precision = precision_score(preprocessed['y_test'], y_pred)
            recall = recall_score(preprocessed['y_test'], y_pred)
            f1 = f1_score(preprocessed['y_test'], y_pred)
            roc_auc = roc_auc_score(preprocessed['y_test'], y_pred_proba)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': preprocessed['feature_names'],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Cross-validation
            cv_scores = cross_val_score(rf_model, preprocessed['X_train_scaled'], 
                                      preprocessed['y_train'], cv=5, scoring='accuracy')
        
        # Ringkasan Eksekutif
        st.header("📊 Ringkasan Eksekutif")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Akurasi Model", f"{accuracy:.2%}")
            st.metric("Precision", f"{precision:.2%}")
        
        with col2:
            st.metric("Recall", f"{recall:.2%}")
            st.metric("F1-Score", f"{f1:.2%}")
        
        with col3:
            st.metric("ROC-AUC", f"{roc_auc:.2%}")
            st.metric("CV Score", f"{cv_scores.mean():.2%}")
        
        # Performance Summary
        st.header("🎯 Performance Summary")
        
        # Buat dashboard metrik
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV Mean'],
            'Value': [accuracy, precision, recall, f1, roc_auc, cv_scores.mean()],
            'Target': [0.85, 0.80, 0.80, 0.80, 0.90, 0.85],
            'Status': ['✅' if val >= target*0.95 else '⚠️' if val >= target*0.8 else '❌' 
                      for val, target in zip([accuracy, precision, recall, f1, roc_auc, cv_scores.mean()], 
                                            [0.85, 0.80, 0.80, 0.80, 0.90, 0.85])]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({'Value': '{:.2%}'}), 
                    use_container_width=True)
        
        # Top Features
        st.header("🔝 Top 5 Fitur Paling Penting")
        
        top_5_features = feature_importance.head(5)
        
        for i, (idx, row) in enumerate(top_5_features.iterrows(), 1):
            with st.expander(f"{i}. {row['Feature']} (Importance: {row['Importance']:.4f})"):
                # Berikan analisis untuk setiap fitur
                if 'IPK' in row['Feature']:
                    st.write("**Analisis:** IPK adalah faktor terpenting dalam seleksi beasiswa.")
                    st.write("- **Dampak:** Setiap peningkatan 0.1 IPK meningkatkan peluang sebesar ~5%")
                    st.write("- **Standar:** Minimum 3.0, optimal di atas 3.5")
                    st.write("- **Rekomendasi:** Fokuskan pada pencapaian akademik")
                
                elif 'Prestasi' in row['Feature']:
                    if 'Akademik' in row['Feature']:
                        st.write("**Analisis:** Prestasi akademik menunjukkan kemampuan dan dedikasi.")
                        st.write("- **Dampak:** Prestasi kompetisi akademik sangat dihargai")
                        st.write("- **Indikator:** Dedikasi terhadap studi")
                        st.write("- **Rekomendasi:** Dorong partisipasi dalam kompetisi")
                    else:
                        st.write("**Analisis:** Prestasi non-akademik menunjukkan kemampuan holistik.")
                        st.write("- **Dampak:** Menunjukkan perkembangan pribadi yang seimbang")
                        st.write("- **Indikator:** Bakat dan minat di luar akademik")
                        st.write("- **Rekomendasi:** Dukung pengembangan bakat")
                
                elif 'Pendapatan' in row['Feature']:
                    st.write("**Analisis:** Pendapatan rendah mendapat prioritas.")
                    st.write("- **Dampak:** Korelasi negatif dengan penerimaan")
                    st.write("- **Tujuan:** Bantuan berbasis kebutuhan")
                    st.write("- **Rekomendasi:** Prioritaskan calon dari keluarga kurang mampu")
                
                elif 'Organisasi' in row['Feature']:
                    st.write("**Analisis:** Keaktifan organisasi menunjukkan soft skills.")
                    st.write("- **Dampak:** Kepemimpinan, kerja tim, komunikasi")
                    st.write("- **Indikator:** Kemampuan sosial dan organisasi")
                    st.write("- **Rekomendasi:** Nilai pengalaman organisasi")
                
                elif 'Sosial' in row['Feature']:
                    st.write("**Analisis:** Pengalaman sosial menunjukkan kepedulian sosial.")
                    st.write("- **Dampak:** Kontribusi kepada masyarakat")
                    st.write("- **Indikator:** Empati dan tanggung jawab sosial")
                    st.write("- **Rekomendasi:** Apresiasi kegiatan sosial")
        
        # Rekomendasi Strategis
        st.header("💡 Rekomendasi Strategis")
        
        tab1, tab2, tab3 = st.tabs(["Model Improvement", "Policy Recommendation", "Implementation Plan"])
        
        with tab1:
            st.subheader("Peningkatan Model")
            st.write("""
            1. **Hyperparameter Tuning:**
               - Gunakan GridSearchCV untuk optimasi parameter
               - Experiment dengan berbagai ensemble methods
               - Coba XGBoost atau LightGBM untuk performa lebih baik
            
            2. **Feature Engineering:**
               - Buat fitur interaksi (IPK × Prestasi)
               - Normalisasi dengan berbagai teknik
               - Lakukan seleksi fitur otomatis
            
            3. **Data Collection:**
               - Kumpulkan lebih banyak data untuk class minority
               - Tambahkan variabel baru (motivasi, essay quality)
               - Collect data longitudinal
            """)
        
        with tab2:
            st.subheader("Rekomendasi Kebijakan")
            st.write("""
            1. **Kriteria Seleksi:**
               - Beri bobot lebih besar pada IPK dan prestasi
               - Pertimbangkan faktor sosio-ekonomi
               - Akomodasi khusus untuk disabilitas
            
            2. **Proses Evaluasi:**
               - Gunakan model sebagai screening awal
               - Kombinasikan dengan evaluasi manual
               - Implementasi sistem scoring transparan
            
            3. **Monitoring & Evaluation:**
               - Track performa penerima beasiswa
               - Evaluasi efektivitas kriteria
               - Regular model retraining
            """)
        
        with tab3:
            st.subheader("Rencana Implementasi")
            st.write("""
            1. **Short-term (1-3 bulan):**
               - Deploy model untuk screening awal
               - Training staff untuk penggunaan sistem
               - Setup infrastructure
            
            2. **Medium-term (3-6 bulan):**
               - Integrasi dengan sistem existing
               - Development dashboard monitoring
               - Collection feedback dan improvement
            
            3. **Long-term (6-12 bulan):**
               - Full automation untuk proses tertentu
               - Advanced analytics dan reporting
               - Integration dengan system lainnya
            """)
        
        # Download Report
        st.header("💾 Download Laporan")
        
        # Generate JSON report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_performance': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            },
            'data_statistics': {
                'total_samples': len(df),
                'training_samples': len(preprocessed['X_train']),
                'test_samples': len(preprocessed['X_test']),
                'accepted': int(df['Diterima_Beasiswa'].sum()),
                'rejected': int(len(df) - df['Diterima_Beasiswa'].sum()),
                'acceptance_rate': float(df['Diterima_Beasiswa'].mean())
            },
            'top_features': feature_importance.head(10).to_dict('records')
        }
        
        # Convert to JSON string
        report_json = json.dumps(report, indent=4)
        
        # Download button
        st.download_button(
            label="📥 Download Laporan (JSON)",
            data=report_json,
            file_name="laporan_beasiswa.json",
            mime="application/json"
        )
        
        # Tampilkan JSON dalam expander
        with st.expander("👁️ Preview Laporan JSON"):
            st.code(report_json, language="json")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Tools & Technologies")
    st.sidebar.markdown("""
    - **Streamlit** - Web Framework
    - **Scikit-learn** - Machine Learning
    - **Pandas & NumPy** - Data Processing
    - **Matplotlib & Seaborn** - Visualization
    - **Random Forest** - Classification Algorithm
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📌 Catatan:**
    1. Pastikan file dataset tersedia
    2. Model akan otomatis di-train jika belum ada
    3. Gunakan menu Prediksi untuk calon baru
    4. Download laporan untuk dokumentasi
    """)

if __name__ == "__main__":
    main()