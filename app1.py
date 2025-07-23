import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman & Styling (Tidak Berubah) ---
st.set_page_config(
    page_title="Dashboard Analisis Kinerja Kurir",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .cluster-info { background-color: #e8f4f8; padding: 1rem; border-left: 4px solid #1f77b4; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📦 Dashboard Analisis Kinerja Kurir</h1>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV data kurir", type=['csv'],
    help="Pastikan file CSV memiliki kolom seperti 'courier', 'delivery_status', 'delivery_duration', dll."
)

# --- Fungsi Backend (Tidak Berubah) ---
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        datetime_columns = ['request_pickup', 'requested_pickup', 'real_pickup', 'manifested_at', 'final_date']
        for col in datetime_columns:
            if col in df.columns: df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'real_pickup' in df.columns and 'final_date' in df.columns:
            df['delivery_duration'] = (df['final_date'] - df['real_pickup']).dt.total_seconds() / 3600
        else: df['delivery_duration'] = np.random.uniform(1, 48, len(df))
        required_columns = ['courier', 'delivery_duration', 'delivery_status']
        for col in required_columns:
            if col not in df.columns:
                if col == 'delivery_status': df[col] = np.random.choice(['DELIVERED', 'FAILED'], len(df), p=[0.8, 0.2])
                elif col == 'courier': df[col] = [f'Courier_{i%20}' for i in range(len(df))]
        if 'total_complain' not in df.columns: df['total_complain'] = np.random.uniform(0, 5, len(df))
        if 'ship_cost' not in df.columns: df['ship_cost'] = np.random.uniform(10000, 50000, len(df))
        if 'solved_percent' not in df.columns: df['solved_percent'] = np.random.uniform(0.5, 1.0, len(df))
        return df
    except Exception as e:
        st.error(f"Error memuat data: {str(e)}")
        return None

def normalize_delivery_status(status):
    if pd.isna(status): return 'UNKNOWN'
    status_str = str(status).strip().upper()
    success_values = ['DELIVERED', 'SUCCESS', 'SUKSES', 'COMPLETE', 'SELESAI', 'DITERIMA']
    failed_values = ['FAILED', 'GAGAL', 'CANCEL', 'RETURN', 'PENDING', 'LOST', 'RUSAK']
    if any(s in status_str for s in success_values): return 'DELIVERED'
    if any(s in status_str for s in failed_values): return 'FAILED'
    return 'DELIVERED'

# --- [SINKRONISASI] Fungsi Clustering Utama ---
@st.cache_data
def perform_analysis(df):
    try:
        df_clean = df.dropna(subset=['courier', 'delivery_duration', 'delivery_status'])
        df_clean = df_clean[df_clean['delivery_duration'] > 0]
        df_clean['delivery_status_normalized'] = df_clean['delivery_status'].apply(normalize_delivery_status)

        courier_aggregated = []
        for courier in df_clean['courier'].unique():
            courier_data = df_clean[df_clean['courier'] == courier]
            total_deliveries = len(courier_data)
            if total_deliveries == 0: continue
            successful_deliveries = (courier_data['delivery_status_normalized'] == 'DELIVERED').sum()
            courier_metrics = {
                'courier': courier, 'total_deliveries': total_deliveries,
                'successful_deliveries': successful_deliveries,
                'failed_deliveries': total_deliveries - successful_deliveries,
                'success_rate': successful_deliveries / total_deliveries,
                'avg_delivery_duration': courier_data['delivery_duration'].mean(),
                'avg_complain': courier_data['total_complain'].mean(),
                'avg_ship_cost': courier_data['ship_cost'].mean()
            }
            courier_aggregated.append(courier_metrics)

        courier_performance = pd.DataFrame(courier_aggregated)
        min_deliveries = 5
        courier_performance = courier_performance[courier_performance['total_deliveries'] >= min_deliveries].reset_index(drop=True)

        if len(courier_performance) < 3:
            st.error(f"Data tidak cukup setelah filter. Hanya {len(courier_performance)} kurir (minimal 3).")
            return None

        features_for_clustering = ['avg_delivery_duration', 'success_rate', 'avg_complain', 'avg_ship_cost']
        X = courier_performance[features_for_clustering].fillna(0)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(X)

        # --- Logika Penentuan K (Sama seperti Colab) ---
        n_samples = len(features_scaled)
        max_k_allowed = n_samples - 1
        desired_k = [3, 4, 5, 6, 7]
        K_to_test = [k for k in desired_k if k <= max_k_allowed]

        if not K_to_test:
            st.error(f"Jumlah kurir ({n_samples}) tidak cukup untuk K yang diinginkan.")
            return None

        inertias, silhouette_scores, dbi_scores = [], [], []
        for k in K_to_test:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, labels))
            dbi_scores.append(davies_bouldin_score(features_scaled, labels))

        eval_metrics = pd.DataFrame({
            'K': K_to_test, 'Inertia': inertias,
            'Silhouette Score': silhouette_scores, 'Davies-Bouldin Index': dbi_scores
        })

        optimal_k = K_to_test[np.argmax(silhouette_scores)]

        # --- Clustering Final ---
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        courier_performance['cluster'] = kmeans.fit_predict(features_scaled)
        
        # --- Perhitungan Skor Performa ---
        max_duration = courier_performance['avg_delivery_duration'].max()
        max_complain = courier_performance['avg_complain'].max()
        if max_duration == 0: max_duration = 1
        if max_complain == 0: max_complain = 1
        courier_performance['performance_score'] = (
            courier_performance['success_rate'] * 0.4 +
            (1 - courier_performance['avg_delivery_duration'] / max_duration) * 0.3 +
            (1 - courier_performance['avg_complain'] / max_complain) * 0.3
        )

        return courier_performance, eval_metrics, optimal_k, features_scaled

    except Exception as e:
        st.error(f"Terjadi error saat analisis: {str(e)}")
        return None

# --- Main Content & Tampilan Dashboard ---
if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    if df is not None:
        # Show data info
        st.subheader("📋 Info Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baris", len(df))
        with col2:
            st.metric("Total Kolom", len(df.columns))
        analysis_results = perform_analysis(df)
        
        if analysis_results:
            courier_performance, eval_metrics, optimal_k, features_scaled = analysis_results
            
            # --- Overview Metrics (Tidak Berubah) ---
            st.subheader("📊 Overview Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Kurir Dianalisis", len(courier_performance))
            with col2: st.metric("Jumlah Cluster Optimal", optimal_k)
            with col3: st.metric("Rata-rata Success Rate", f"{courier_performance['success_rate'].mean():.1%}")
            with col4: st.metric("Rata-rata Durasi", f"{courier_performance['avg_delivery_duration'].mean():.1f} jam")

            # --- Definisi Tabs (Dengan Tab Baru) ---
            tab_k, tab_cluster, tab_performer, tab_detail, tab_data, tab_individu = st.tabs([
                "⚙️ Penentuan K Optimal", "📈 Analisis Clustering", "🏆 Top Performers", 
                "📊 Detail Cluster", "📋 Tabel Data", "🔍 Analisis Individu"
            ])

            # --- [TAB BARU] Penentuan K Optimal ---
            with tab_k:
                st.header("⚙️ Proses Penentuan Jumlah Cluster (K) Optimal")
                st.markdown("Untuk menemukan pengelompokan yang paling bermakna, kami mengevaluasi beberapa jumlah cluster (K) menggunakan tiga metode standar:")
                st.markdown("""
                - **Elbow Method (Inertia)**: Mencari titik 'siku' di mana penurunan nilai tidak lagi signifikan.
                - **Silhouette Score**: Mengukur seberapa mirip sebuah objek dengan clusternya sendiri dibandingkan dengan cluster lain. **Nilai tertinggi (mendekati 1) adalah yang terbaik.**
                - **Davies-Bouldin Index**: Mengukur rasio antara kepadatan dalam cluster dengan pemisahan antar cluster. **Nilai terendah (mendekati 0) adalah yang terbaik.**
                """)

                # Grafik Perbandingan
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eval_metrics['K'], y=eval_metrics['Inertia'], name='Inertia (Elbow)', yaxis='y1'))
                fig.add_trace(go.Scatter(x=eval_metrics['K'], y=eval_metrics['Silhouette Score'], name='Silhouette Score', yaxis='y2'))
                fig.add_trace(go.Scatter(x=eval_metrics['K'], y=eval_metrics['Davies-Bouldin Index'], name='Davies-Bouldin Index', yaxis='y2'))
                
                fig.update_layout(
                    title_text="Perbandingan Metrik Evaluasi Clustering",
                    yaxis=dict(title="Inertia (WCSS)"),
                    yaxis2=dict(title="Skor (Silhouette & DBI)", overlaying='y', side='right', range=[0,1]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Tabel Perbandingan
                st.subheader("Tabel Perbandingan Metrik")
                st.dataframe(eval_metrics.set_index('K').round(3))
                st.success(f"Berdasarkan **Silhouette Score tertinggi**, nilai **K = {optimal_k}** dipilih sebagai jumlah cluster optimal untuk analisis lebih lanjut.")

            # --- Tab Analisis Clustering ---
            with tab_cluster:
                st.header("📈 Analisis Hasil Clustering")
                col1, col2 = st.columns([2, 1])
                with col1:
                    pca = PCA(n_components=2).fit_transform(features_scaled)
                    fig_pca = px.scatter(
                        x=pca[:, 0], y=pca[:, 1], color=courier_performance['cluster'].astype(str),
                        title="Visualisasi Clustering (PCA)", labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                        hover_data={'courier': courier_performance['courier']}
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                with col2:
                    cluster_counts = courier_performance['cluster'].value_counts().sort_index()
                    fig_pie = px.pie(values=cluster_counts.values, names=[f'Cluster {i}' for i in cluster_counts.index], title="Distribusi Kurir per Cluster")
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.subheader("Karakteristik per Cluster (Heatmap)")
                features_for_heatmap = ['avg_delivery_duration', 'success_rate', 'avg_complain', 'avg_ship_cost']
                cluster_summary = courier_performance.groupby('cluster')[features_for_heatmap].mean()
                fig_heatmap = px.imshow(cluster_summary.T, labels=dict(x="Cluster", y="Fitur", color="Rata-rata"), color_continuous_scale="RdYlBu_r")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # --- Sisa Tabs (Logika Tidak Berubah) ---
            with tab_performer:
                st.header("🏆 Top & Bottom Performers")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🥇 Top 5 Kurir Terbaik")
                    for _, courier in courier_performance.nlargest(5, 'performance_score').iterrows():
                        st.markdown(f"""<div class="metric-card"><strong>{courier['courier']}</strong> (Cluster {courier['cluster']})<br>
                                        Score: {courier['performance_score']:.3f} | Rate: {courier['success_rate']:.1%} | Durasi: {courier['avg_delivery_duration']:.1f} jam</div>""", unsafe_allow_html=True)
                with col2:
                    st.markdown("### 🚨 Bottom 5 Kurir")
                    for _, courier in courier_performance.nsmallest(5, 'performance_score').iterrows():
                        st.markdown(f"""<div class="metric-card"><strong>{courier['courier']}</strong> (Cluster {courier['cluster']})<br>
                                        Score: {courier['performance_score']:.3f} | Rate: {courier['success_rate']:.1%} | Durasi: {courier['avg_delivery_duration']:.1f} jam</div>""", unsafe_allow_html=True)

            with tab_detail:
                st.header("📊 Detail Karakteristik per Cluster")
               # === [PERBAIKAN] MENAMPILKAN NAMA KURIR DI SETIAP CLUSTER ===
                for cluster in sorted(courier_performance['cluster'].unique()):
                    cluster_data = courier_performance[courier_performance['cluster'] == cluster]
                    list_of_couriers = cluster_data['courier'].tolist()

                    st.markdown(f"""
                    <div class="cluster-info">
                        <h4>Cluster {cluster} ({len(cluster_data)} kurir)</h4>
                        <ul>
                            <li>Rata-rata durasi: {cluster_data['avg_delivery_duration'].mean():.2f} jam</li>
                            <li>Rata-rata success rate: {cluster_data['success_rate'].mean():.1%}</li>
                            <li>Rata-rata komplain: {cluster_data['avg_complain'].mean():.2f}</li>
                            <li>Rata-rata biaya: Rp {cluster_data['avg_ship_cost'].mean():,.0f}</li>
                        </ul>
                        <strong>Anggota Cluster:</strong> {', '.join(list_of_couriers)}
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab_data:
                st.header("📋 Tabel Data Hasil Analisis")
                st.dataframe(courier_performance.round(3), use_container_width=True)
                csv = courier_performance.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data CSV", csv, "hasil_analisis_kurir.csv", "text/csv")

            with tab_individu:
                st.header("🔍 Analisis Individual Kurir")
                selected_courier = st.selectbox("Pilih Kurir", options=courier_performance['courier'].sort_values().tolist())
                if selected_courier:
                    courier_data = courier_performance[courier_performance['courier'] == selected_courier].iloc[0]
                    st.markdown(f"#### Detail untuk **{selected_courier}** (Bagian dari Cluster **{courier_data['cluster']}**)")
                    col1, col2 = st.columns(2)
                    with col1:
                        categories = ['Success Rate', 'Speed', 'Low Complaints', 'Cost Efficiency']
                        def normalize(value, series):
                            min_val, max_val = series.min(), series.max()
                            if max_val == min_val: return 0.5
                            return (value - min_val) / (max_val - min_val)
                        normalized_vals = [
                            normalize(courier_data['success_rate'], courier_performance['success_rate']),
                            1 - normalize(courier_data['avg_delivery_duration'], courier_performance['avg_delivery_duration']),
                            1 - normalize(courier_data['avg_complain'], courier_performance['avg_complain']),
                            1 - normalize(courier_data['avg_ship_cost'], courier_performance['avg_ship_cost'])
                        ]
                        fig_radar = go.Figure(data=go.Scatterpolar(r=normalized_vals, theta=categories, fill='toself'))
                        fig_radar.update_layout(title=f"Performance Radar - {selected_courier}", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                        st.plotly_chart(fig_radar, use_container_width=True)
                    with col2:
                        cluster_avg = courier_performance[courier_performance['cluster'] == courier_data['cluster']].mean(numeric_only=True)
                        comparison_data = pd.DataFrame({
                            'Metric': ['Success Rate', 'Durasi (jam)', 'Komplain'],
                            'Kurir': [courier_data['success_rate'], courier_data['avg_delivery_duration'], courier_data['avg_complain']],
                            'Rata-rata Cluster': [cluster_avg['success_rate'], cluster_avg['avg_delivery_duration'], cluster_avg['avg_complain']]
                        }).melt(id_vars='Metric', var_name='Type', value_name='Value')
                        fig_comparison = px.bar(comparison_data, x='Metric', y='Value', color='Type', barmode='group', title=f"Perbandingan vs Rata-rata Cluster {courier_data['cluster']}")
                        st.plotly_chart(fig_comparison, use_container_width=True)

else:
    st.info("👆 Silakan upload file CSV data kurir untuk memulai analisis.")
    st.subheader("Contoh Format Data yang Diharapkan")
    st.dataframe(pd.DataFrame({
         'courier': ['Courier_1', 'Courier_2', 'Courier_3'],
        'delivery_status': ['DELIVERED', 'DELIVERED', 'RETURNED'],
        'delivery_duration': [24.5, 18.2, 36.8],
        'total_complain': [1, 0, 3],
        'ship_cost': [25000, 30000, 20000],
        'real_pickup': ['2024-01-01 09:00:00', '2024-01-01 10:00:00', '2024-01-01 11:00:00'],
        'final_date': ['2024-01-02 09:30:00', '2024-01-01 18:12:00', '2024-01-02 23:48:00']
    }))

    st.markdown("""
    **Kolom yang diperlukan:**
    - `courier`: Nama kurir
    - `delivery_status`: Status pengiriman (DELIVERED/FAILED)
    - `delivery_duration`: Durasi pengiriman dalam jam (opsional, akan dihitung otomatis)
    - `total_complain`: Total komplain (opsional)
    - `ship_cost`: Biaya pengiriman (opsional)
    - `real_pickup`: Tanggal pickup (opsional)
    - `final_date`: Tanggal pengiriman selesai (opsional)
    """)

st.markdown("---")
st.markdown("*Dashboard dibuat untuk analisis kinerja kurir menggunakan Machine Learning Clustering*")