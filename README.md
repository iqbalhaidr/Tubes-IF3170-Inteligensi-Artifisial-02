# Tugas Besar 2 IF3170 Intelegensi Artifisial

Tugas ini mengimplementasikan algoritma pembelajaran mesin (**Decision Tree Learning, Logistic Regression, dan Support Vector Machine**) yang dibangun dari awal (*from scratch*) dan dibandingkan dengan implementasi pustaka `scikit-learn`. Proyek ini dikerjakan untuk memenuhi Tugas Besar 2 mata kuliah IF3170 Inteligensi Artifisial.

## Algoritma yang Diimplementasikan:
1.  **Decision Tree Learning (C4.5)**: Menggunakan *Gain Ratio* (*Information Gain*/ *Split Information*) untuk pemilihan *feature*, menangani data kontinu & kategorikal, serta fitur *Post-Pruning*.
2.  **Logistic Regression**: Menggunakan optimasi *Mini-batch Gradient Descent* dan strategi *One-vs-One* untuk klasifikasi multiclass.
3.  **Support Vector Machine (SVM)**: Menggunakan algoritma *Sequential Minimal Optimization* (SMO), Kernel RBF, dan strategi *One-vs-One*.

## Persyaratan Awal
Tugas ini menggunakan Python3. Pastikan telah menginstall library berikut:
* pandas
* numpy
* matplotlib
* scikit-learn
* imbalanced-learn
* cvxopt
* jupyter
* ipykernel

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn cvxopt ipykernel jupyter
```

## Cara Menjalankan
### Cara 1 (Direkomendasikan)
1. Jalankan pada Google Colab dengan link notebook berikut (make a copy dan pastikan dataset sudah di upload): https://colab.research.google.com/drive/1AnWVb8H_OFPmaRxuefh7ow-Tg2q3tSrf?usp=sharing

### Cara 2
1. Clone repository ini
```bash
git clone https://github.com/iqbalhaidr/Tubes_AI_2.git
```
2. Pindah ke directory src
```bash
cd path/to/repository/src/
```
3. Instalasi library persyaratan awal (direkomendasikan dengan venv)
4. Buka file .ipynb, pilih kernel venv Python Environment
5. Jalankan semua (Run All). Pilih kernel 

## Kelompok 16 - LimaSerangkAI
|   NIM    |                  Nama                  |
| :------: | :------------------------------------: |
| 13523023 |           Muhammad Aufa Farabi         |
| 13523025 |       Joel Hotlan Haris Siahaan        |
| 13523030 |              Julius Arthur             |
| 13523051 |      Ferdinand Gabe Tua Sinaga         |
| 13523111 |         Muhammad Iqbal Haidar          |
