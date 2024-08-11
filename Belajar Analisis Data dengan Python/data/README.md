## Setup Environment
# Installasi dependensi
pip install numpy pandas matplotlib seaborn plotly scikit-learn streamlit

# Mengunduh dataset
!gdown https://drive.google.com/uc?id=1RaBmV6Q6FYWU4HWZs80Suqd7KQC34diQ
# Mengekstrak dataset
!unzip Bike-sharing-dataset.zip
# Menghapus file zip setelah diekstrak
!rm Bike-sharing-dataset.zip

# Menjalankan aplikasi Streamlit
streamlit run dashboard.py