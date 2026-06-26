# Pemodelan Komputasi Sistem Kamera Digital

Repositori ini memuat kode sumber dan laporan akhir untuk proyek pemodelan komputasi tujuh subsistem fisis pada kamera mirrorless dan action camera digital. Simulasi ini mencakup berbagai fenomena fisika, mulai dari manajemen termal pada sensor gambar, perilaku medan elektromagnetik pada lensa, hingga penangkapan elektron secara kuantum.

## Pembagian Tugas Anggota

Proyek ini dikerjakan oleh Kelompok 5 dengan pembagian fokus simulasi sebagai berikut:
* Aisyah Nur Khairan: Bertanggung jawab atas Kasus 1 (Konduksi Panas 3D) dan Kasus 6 (Elektrodinamika Maxwell 3D FDTD).
* Adinda Yulia Putri Rachmat: Bertanggung jawab atas Kasus 4 (Magnetostatika Aktuator VCM) dan Kasus 5 (Propagasi dan Difraksi Lensa Tipis).
* Sandy Fauzi Amrulloh: Bertanggung jawab atas Kasus 2 (Elektrostatika Silindris Kapasitor Tabung Flash), Kasus 3 (Metode Finite Volume pada Tetesan Kristal Cair PDLC), dan Kasus 7 (Persamaan Schrodinger 2D Fotodioda).

## Dokumentasi Program

Kode sumber untuk seluruh simulasi dieksekusi melalui Python dan diorganisasikan dalam bentuk Jupyter Notebook serta direktori laporan LaTeX. Seluruh program telah dioptimalkan secara matematis dan komputasional.

### Fitur Utama dan Optimasi
1. Akselerasi Numba JIT: Proses iterasi berat pada metode beda hingga dan rakitan matriks diakselerasi menggunakan kompilasi Just In Time untuk mendekati kecepatan eksekusi bahasa C.
2. Matriks Sparse SciPy: Penyelesaian sistem persamaan linear skala besar menggunakan format matriks sparse (CSR/CSC) dan diselesaikan dengan faktorisasi LU (SuperLU) untuk efisiensi memori yang maksimal.
3. Vektorisasi NumPy: Operasi array multidimensi diproses secara paralel tingkat instruksi menggunakan fitur vektorisasi bawaan NumPy.
4. Visualisasi Komprehensif: Setiap kasus menghasilkan keluaran berupa plot proyeksi permukaan 3D, plot kontur 2D, irisan penampang silang, serta animasi pergerakan gelombang yang disimpan dalam format video.

### Rincian Kasus Simulasi
* Kasus 1 (Konduksi Panas Sensor CMOS): Menggunakan metode beda hingga Forward Time Central Space (FTCS) 3D untuk memodelkan disipasi panas akibat perekaman video beresolusi 4K.
* Kasus 2 (Potensial Kapasitor Tabung Flash): Menggunakan metode Finite Difference Frequency Domain (FDFD) konservatif pada koordinat silinder untuk mengevaluasi medan listrik statis dan efek tepi (fringing field) pada ujung gulungan elektroda kapasitor tegangan tinggi sistem flash, sebagai dasar penentuan ketebalan isolator anti dadal listrik.
* Kasus 3 (Medan Tetesan Kristal Cair PDLC): Menggunakan metode Finite Volume untuk menyelesaikan diskontinuitas permitivitas antarmuka antara tetesan kristal cair dan matriks polimer pada ND filter elektronik (Polymer Dispersed Liquid Crystal).
* Kasus 4 (Aktuator VCM Fokus): Menganalisis medan magnetostatik tunak pada aktuator koil menggunakan pendekatan persamaan Poisson untuk potensial vektor magnetik koordinat silinder.
* Kasus 5 (Propagasi Lensa Tipis): Menyimulasikan prinsip optika gelombang untuk memantau efek perambatan cahaya menembus medium optik.
* Kasus 6 (Gelombang Maxwell 3D Lensa): Menggunakan metode Finite Difference Time Domain (FDTD) dengan kisi bersilang Yee leapfrog untuk mensimulasikan perambatan pulsa medan elektromagnetik melalui lensa bikonveks secara murni tanpa pendekatan optika geometris.
* Kasus 7 (Efisiensi Kuantum Piksel CMOS): Menyimulasikan persamaan Schrodinger bergantung waktu menggunakan algoritma Crank-Nicolson dua dimensi yang uniter dan stabil untuk menganalisis probabilitas tangkapan elektron fotoeksitasi.

## Cara Penggunaan
Untuk menjalankan simulasi, pastikan lingkungan Python telah terpasang beserta dependensi utama yaitu NumPy, SciPy, Matplotlib, dan Numba. Anda dapat membuka Jupyter Notebook yang disediakan atau menjalankan kode sumber Python secara langsung dari direktori laporan. Hasil luaran gambar dan video akan otomatis diperbarui saat program dieksekusi.
