# Tugas Akhir

- Nama: Bayu Adjie Sidharta
- NIP: 05111940000172
- Judul Tugas Akhir: Prediksi Tingkat Partisipasi Pemilihan Umum dengan Pendekatan Regresi menggunakan Kombinasi Fitur Demografi dan Opini Masyarakat

## Abstrak

Partisipasi dalam pemilihan umum merupakan aspek penting bagi pelaksanaan demokrasi. Namun, kenyataannya, tingkat partisipasi pemilih sering kali berbeda-beda antara satu daerah dengan daerah lainnya. Kesenjangan partisipasi yang tinggi antar daerah dapat menghasilkan hasil pemilihan yang tidak mewakili kepentingan seluruh rakyat. Memastikan akses yang merata pada pemilu dapat meningkatkan partisipasi masyarakat dalam mengambil keputusan secara inkusif dan merupaka upaya penting dalam menjaga sistem politik yang adil (SDG 16).  

Untuk mengatasi masalah ini, survey biasa dilakukan untuk mengukur partisipasi dalam suatu daerah. Namun, survey memerlukan waktu yang lama dan biaya yang cukup substansial. Diluar dari kedua kekurangan tersebut, bias sampling dan bias jawaban juga dapat terjadi saat pelaksanaan survey. Oleh karena itu, peneliti akan merancang sebuah model pembelajaran mesin yang dapat memprediksi tingkat partisipasi di suatu daerah dengan menggunakan kombinasi data demografi dan opini publik sebagai alternatif dari penggunaan survey.  

Usulan model prediksi tingkat partisipasi pada penelitian ini akan dievaluasi menggunakan data di demografis kabupaten kota di Pulau Jawa dengan mempertimbangkan banyaknya data opini masyarakat melalui media Twitter. Terdapat beberapa sumber data yaitu data partisipasi pemilih (OpenData KPU), data demografi (BPS) tempat pemilih serta data opini masyarakat (Twitter). Analisa sentimen dilakukan terhadap data opini menggunakan model pemrosesan bahasa alami IndoBertTweet sebelum ditransformasi berdasarkan kabupaten kota.  

Berbagai teknik seleksi fitur akan diuji untuk mendapatkan performa subset fitur yang terbaik. Hasil seleksi akan dijadikan acuan untuk memilih model regresi berdasarkan performa model. Penyetelan ulang model dilakukan guna mencari akurasi terbaik yang dapat dihasilkan. Hasil akhir dari penelitian ini merupakan sebuah model yang dapat memprediksi partisipasi pemilu masyarakat yang dapat menjadi alternatif dari penggunaan survey.  

## Deskripsi

Folder - folder pada repository merepresentasikan sebuah tahapan dalam penelitian. Urutan folder tersebut adalah sebagai berikut:

1. Data Collection
2. Sentiment Analysis
3. Preprocessing
4. Modelling
5. Feature Selection

## Referensi dan Sumber Data

[Referrences](https://docs.google.com/spreadsheets/d/1sEzx6QILsXO-0RqY3hZLDVjcJU6wDW-8Kp3CWpWAepg/edit?usp=sharing)
[Tables and Data](https://docs.google.com/spreadsheets/d/1rmpPuEMTv5ql0cMH9ZMNXI9wAqmswI2d967pbCMM3bA/edit?usp=sharing)
