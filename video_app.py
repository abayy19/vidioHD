"""
AI Video Enhancer & Filter
============================
Aplikasi web berbasis Streamlit untuk meningkatkan kualitas video (HD Enhancer)
dan menerapkan berbagai filter estetik (termasuk gaya iPhone) ke video.

Cara kerja singkat:
    Video diproses frame-per-frame menggunakan OpenCV. Setiap frame dibaca,
    diberi efek (enhance atau filter), lalu ditulis ulang ke file video baru.

Struktur file:
- Bagian 1: FRAME PROCESSING FUNCTIONS (logika murni per-frame, numpy/OpenCV,
            tidak menyentuh Streamlit)
- Bagian 2: VIDEO I/O FUNCTIONS (baca info video & proses seluruh video)
- Bagian 3: STREAMLIT UI FUNCTIONS (tampilan & interaksi pengguna)
- Bagian 4: MAIN APP (menyatukan semuanya)

Catatan penting:
- File output TIDAK menyertakan audio (OpenCV VideoWriter tidak mendukung
  audio). Lihat TODO di bagian process_video() untuk cara menggabungkan
  audio asli kembali menggunakan moviepy.
- Proses video jauh lebih berat daripada gambar. Video panjang/resolusi
  tinggi akan butuh waktu lebih lama untuk diproses.

Cara menjalankan:
    streamlit run video_app.py
"""

import os
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import streamlit as st


# =====================================================================
# BAGIAN 1: FRAME PROCESSING FUNCTIONS
# =====================================================================
# Semua fungsi di sini menerima & mengembalikan frame dalam format
# numpy array BGR (format asli OpenCV), agar pemrosesan video tetap cepat
# tanpa bolak-balik konversi ke PIL di setiap frame.


def enhance_hd_frame(
    frame: np.ndarray,
    sharpness: float = 1.5,
    contrast: float = 1.2,
    brightness: float = 1.0,
    saturation: float = 1.1,
    upscale_factor: float = 1.0,
    denoise: bool = False,
) -> np.ndarray:
    """
    Mode 1 - AI HD Enhancer untuk video (versi awal menggunakan OpenCV).

    Meningkatkan ketajaman (unsharp masking), kontras, kecerahan, saturasi,
    dan (opsional) memperbesar resolusi frame.

    ------------------------------------------------------------------
    TODO (PENGEMBANGAN SELANJUTNYA - AI Upscaler Sungguhan untuk Video):
    ------------------------------------------------------------------
    Untuk hasil upscaling video berbasis AI sungguhan, fungsi ini bisa
    diganti dengan pemanggilan model seperti:

    1) Real-ESRGAN Video (via Replicate API), per-frame atau per-clip:

        import replicate
        output = replicate.run(
            "nightmareai/real-esrgan:42fed1c4...",
            input={"image": frame_bytes, "scale": 2}
        )

    2) OpenCV DNN Super Resolution (model .pb, contoh EDSR/ESPCN),
       dipanggil per-frame di dalam loop pemrosesan video:

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("EDSR_x2.pb")
        sr.setModel("edsr", 2)
        frame = sr.upsample(frame)

    Catatan: upscaling AI per-frame untuk video SANGAT lambat karena
    dilakukan berulang untuk setiap frame. Pertimbangkan memproses hanya
    beberapa frame per detik (frame skipping) atau menjalankan di GPU.
    ------------------------------------------------------------------
    """
    result = frame

    # Kurangi noise (opsional) sebelum sharpening
    if denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 7, 7, 7, 21)

    # Perbesar resolusi frame (jika diminta)
    if upscale_factor and upscale_factor > 1.0:
        new_w = int(result.shape[1] * upscale_factor)
        new_h = int(result.shape[0] * upscale_factor)
        result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Sharpening menggunakan teknik unsharp mask
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=3)
    result = cv2.addWeighted(result, 1 + (sharpness - 1), blurred, -(sharpness - 1), 0)

    # Kontras & kecerahan: pixel_baru = pixel * kontras + (kecerahan-1)*255
    result = cv2.convertScaleAbs(result, alpha=contrast, beta=(brightness - 1) * 60)

    # Saturasi warna (dilakukan lewat ruang warna HSV)
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def filter_grayscale(frame: np.ndarray) -> np.ndarray:
    """Ubah frame menjadi hitam-putih."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_sepia(frame: np.ndarray) -> np.ndarray:
    """Terapkan efek sepia menggunakan matriks transformasi warna."""
    arr = frame.astype(np.float64)
    # Matriks sepia dalam urutan BGR (karena OpenCV pakai BGR)
    sepia_matrix = np.array(
        [
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393],
        ]
    )
    sepia = arr @ sepia_matrix.T
    return np.clip(sepia, 0, 255).astype(np.uint8)


def filter_pencil_sketch(frame: np.ndarray) -> np.ndarray:
    """Ubah frame menjadi sketsa pensil (dodge blending)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def filter_blur(frame: np.ndarray, radius: int = 5) -> np.ndarray:
    """Terapkan Gaussian Blur. Radius otomatis dijadikan ganjil."""
    k = radius * 2 + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def filter_vintage_warm(frame: np.ndarray) -> np.ndarray:
    """Efek vintage/warm + vignette ringan."""
    arr = frame.astype(np.float64)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.15, 0, 255)  # Red channel (BGR index 2)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.05, 0, 255)  # Green
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.85, 0, 255)  # Blue channel (BGR index 0)

    h, w = arr.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    vignette = 1 - np.clip(np.sqrt(xv**2 + yv**2) * 0.7, 0, 1)
    vignette = np.dstack([vignette] * 3)

    arr = arr * (0.6 + 0.4 * vignette)
    result = np.clip(arr, 0, 255).astype(np.uint8)
    result = cv2.convertScaleAbs(result, alpha=0.9, beta=10)  # kontras turun, brightness naik dikit
    return result


# ---------------------------------------------------------------------
# Filter bergaya iPhone (sama seperti versi gambar, disesuaikan ke OpenCV)
# ---------------------------------------------------------------------


def _adjust_temperature_tint(frame: np.ndarray, warm: float = 0.0, cool: float = 0.0) -> np.ndarray:
    """Helper: geser temperatur warna frame (BGR)."""
    arr = frame.astype(np.float64)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 + warm * 0.15 - cool * 0.05), 0, 255)  # Red
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + cool * 0.15 - warm * 0.05), 0, 255)  # Blue
    return arr.astype(np.uint8)


def filter_iphone_vivid(frame: np.ndarray) -> np.ndarray:
    """iPhone 'Vivid': saturasi & kontras dinaikkan."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result = cv2.convertScaleAbs(result, alpha=1.15, beta=8)
    return result


def filter_iphone_vivid_warm(frame: np.ndarray) -> np.ndarray:
    return _adjust_temperature_tint(filter_iphone_vivid(frame), warm=1.0)


def filter_iphone_vivid_cool(frame: np.ndarray) -> np.ndarray:
    return _adjust_temperature_tint(filter_iphone_vivid(frame), cool=1.0)


def filter_iphone_dramatic(frame: np.ndarray) -> np.ndarray:
    """iPhone 'Dramatic': kontras tinggi, saturasi ditekan, vignette gelap."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.9, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result = cv2.convertScaleAbs(result, alpha=1.3, beta=-10)

    h, w = result.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    vignette = 1 - np.clip(np.sqrt(xv**2 + yv**2) * 0.8, 0, 1)
    vignette = np.dstack([vignette] * 3)

    arr = result.astype(np.float64) * (0.55 + 0.45 * vignette)
    return np.clip(arr, 0, 255).astype(np.uint8)


def filter_iphone_dramatic_warm(frame: np.ndarray) -> np.ndarray:
    return _adjust_temperature_tint(filter_iphone_dramatic(frame), warm=1.0)


def filter_iphone_dramatic_cool(frame: np.ndarray) -> np.ndarray:
    return _adjust_temperature_tint(filter_iphone_dramatic(frame), cool=1.0)


def filter_iphone_mono(frame: np.ndarray) -> np.ndarray:
    """iPhone 'Mono': hitam-putih kontras tegas."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_iphone_silvertone(frame: np.ndarray) -> np.ndarray:
    """iPhone 'Silvertone': hitam-putih terang & lembut."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=0.95, beta=15)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def filter_iphone_noir(frame: np.ndarray) -> np.ndarray:
    """iPhone 'Noir': hitam-putih gelap, kontras sangat tinggi, sinematik."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=-15)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# Kamus pemetaan nama filter (untuk dropdown) -> fungsi pemrosesnya
FILTER_FUNCTIONS = {
    "Grayscale": filter_grayscale,
    "Sepia": filter_sepia,
    "Pencil Sketch": filter_pencil_sketch,
    "Blur": filter_blur,
    "Vintage/Warm": filter_vintage_warm,
    "iPhone - Vivid": filter_iphone_vivid,
    "iPhone - Vivid Warm": filter_iphone_vivid_warm,
    "iPhone - Vivid Cool": filter_iphone_vivid_cool,
    "iPhone - Dramatic": filter_iphone_dramatic,
    "iPhone - Dramatic Warm": filter_iphone_dramatic_warm,
    "iPhone - Dramatic Cool": filter_iphone_dramatic_cool,
    "iPhone - Mono": filter_iphone_mono,
    "iPhone - Silvertone": filter_iphone_silvertone,
    "iPhone - Noir": filter_iphone_noir,
}


def apply_filter(frame: np.ndarray, filter_name: str, **kwargs) -> np.ndarray:
    """Terapkan filter berdasarkan nama yang dipilih pengguna dari dropdown."""
    func = FILTER_FUNCTIONS.get(filter_name)
    if func is None:
        return frame
    if filter_name == "Blur":
        return func(frame, radius=kwargs.get("blur_radius", 5))
    return func(frame)


# =====================================================================
# BAGIAN 2: VIDEO I/O FUNCTIONS
# =====================================================================


def ffmpeg_available() -> bool:
    """Cek apakah binary FFmpeg tersedia di server (diinstal via packages.txt)."""
    return shutil.which("ffmpeg") is not None


def normalize_video_with_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Konversi video apa pun (termasuk .mov dari iPhone dengan codec HEVC/H.265
    dan metadata rotasi) menjadi file .mp4 berformat H.264 standar yang bisa
    dibaca dengan andal oleh OpenCV.

    FFmpeg otomatis "membakar" rotasi metadata ke dalam piksel video saat
    transcoding, sehingga masalah video iPhone portrait yang muncul miring/
    landscape saat dibaca OpenCV langsung dapat teratasi.

    Mengembalikan True jika konversi berhasil, False jika FFmpeg tidak
    tersedia atau proses konversi gagal (dalam hal ini, aplikasi akan
    mencoba membaca file asli secara langsung sebagai fallback).
    """
    if not ffmpeg_available():
        return False

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-an",  # Buang audio di tahap ini (tetap tidak dipakai OpenCV)
                output_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> dict:
    """Ambil metadata video: fps, jumlah frame, lebar, tinggi, durasi (detik)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def process_video(
    input_path: str,
    output_path: str,
    process_frame_fn,
    output_scale: float = 1.0,
    progress_callback=None,
) -> None:
    """
    Baca video dari input_path, terapkan process_frame_fn ke setiap frame,
    lalu tulis hasilnya ke output_path.

    Parameter:
        process_frame_fn   : fungsi yang menerima 1 frame (np.ndarray BGR)
                              dan mengembalikan 1 frame hasil olahan.
        output_scale       : faktor tambahan untuk menyesuaikan ukuran
                              VideoWriter jika process_frame_fn mengubah
                              resolusi (misalnya saat upscale_factor > 1).
        progress_callback   : fungsi callback(current_frame, total_frame)
                              dipanggil setiap frame selesai diproses,
                              berguna untuk update progress bar di UI.

    ------------------------------------------------------------------
    TODO (Mempertahankan Audio Asli):
    ------------------------------------------------------------------
    OpenCV VideoWriter TIDAK menyertakan audio. Jika audio perlu
    dipertahankan, gabungkan audio asli ke video hasil proses dengan
    moviepy setelah fungsi ini selesai dijalankan, contoh:

        from moviepy.editor import VideoFileClip
        original_clip = VideoFileClip(input_path)
        processed_clip = VideoFileClip(output_path)
        final_clip = processed_clip.set_audio(original_clip.audio)
        final_clip.write_videofile("output_with_audio.mp4", codec="libx264")

    (Butuh tambahan dependensi: moviepy dan ffmpeg)
    ------------------------------------------------------------------
    """
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(input_path)

    out_width = int(info["width"] * output_scale)
    out_height = int(info["height"] * output_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, info["fps"], (out_width, out_height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame_fn(frame)

        # Pastikan ukuran frame hasil sesuai dengan ukuran VideoWriter
        if (processed.shape[1], processed.shape[0]) != (out_width, out_height):
            processed = cv2.resize(processed, (out_width, out_height))

        writer.write(processed)
        frame_idx += 1

        if progress_callback:
            progress_callback(frame_idx, info["frame_count"])

    cap.release()
    writer.release()


# =====================================================================
# BAGIAN 3: STREAMLIT UI FUNCTIONS
# =====================================================================


def setup_page():
    """Konfigurasi dasar halaman Streamlit."""
    st.set_page_config(
        page_title="AI Video Enhancer & Filter",
        page_icon="🎬",
        layout="wide",
    )
    st.title("🎬 AI Video Enhancer & Filter")
    st.caption(
        "Tingkatkan kualitas video Anda atau terapkan filter estetik (termasuk gaya iPhone). "
        "Mendukung format MP4, MOV (termasuk video iPhone/HEVC), AVI, dan MKV."
    )
    st.warning(
        "⚠️ Video hasil proses saat ini **tidak menyertakan audio** "
        "(keterbatasan OpenCV VideoWriter). Lihat komentar TODO di kode "
        "untuk cara menggabungkan audio kembali menggunakan moviepy.",
        icon="⚠️",
    )


def render_sidebar():
    """Render menu sidebar: upload video, pemilihan mode, dan parameter."""
    st.sidebar.header("⚙️ Pengaturan")

    uploaded_file = st.sidebar.file_uploader(
        "Unggah Video", type=["mp4", "mov", "avi", "mkv"]
    )

    mode = st.sidebar.radio(
        "Pilih Mode",
        options=["AI HD Enhancer", "Filter Estetik"],
        help="Pilih mode pemrosesan video yang ingin digunakan.",
    )

    settings = {"mode": mode}

    if mode == "AI HD Enhancer":
        st.sidebar.subheader("🔧 Parameter HD Enhancer")
        settings["sharpness"] = st.sidebar.slider("Sharpness (Ketajaman)", 1.0, 3.0, 1.5, 0.1)
        settings["contrast"] = st.sidebar.slider("Contrast (Kontras)", 0.5, 2.0, 1.2, 0.1)
        settings["brightness"] = st.sidebar.slider("Brightness (Kecerahan)", 0.5, 2.0, 1.0, 0.1)
        settings["saturation"] = st.sidebar.slider("Color Saturation (Saturasi)", 0.0, 2.0, 1.1, 0.1)
        settings["upscale_factor"] = st.sidebar.select_slider(
            "Upscale Resolution",
            options=[1.0, 1.5, 2.0],
            value=1.0,
            help="Semakin besar, semakin lama proses render video.",
        )
        settings["denoise"] = st.sidebar.checkbox("Kurangi Noise (Denoise)", value=False)
    else:
        st.sidebar.subheader("🎨 Pilih Filter")
        settings["filter_name"] = st.sidebar.selectbox(
            "Jenis Filter", options=list(FILTER_FUNCTIONS.keys())
        )
        if settings["filter_name"] == "Blur":
            settings["blur_radius"] = st.sidebar.slider("Blur Radius", 1, 25, 5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Performa")
    settings["max_duration"] = st.sidebar.number_input(
        "Batas Durasi Diproses (detik, 0 = seluruh video)",
        min_value=0,
        value=0,
        step=5,
        help="Batasi durasi video yang diproses agar lebih cepat, terutama saat mencoba-coba parameter.",
    )

    return uploaded_file, settings


def build_frame_processor(settings: dict):
    """Bangun fungsi pemroses-frame tunggal berdasarkan mode & pengaturan pengguna."""
    if settings["mode"] == "AI HD Enhancer":
        def _process(frame):
            return enhance_hd_frame(
                frame,
                sharpness=settings["sharpness"],
                contrast=settings["contrast"],
                brightness=settings["brightness"],
                saturation=settings["saturation"],
                upscale_factor=settings["upscale_factor"],
                denoise=settings["denoise"],
            )
        return _process, settings["upscale_factor"]
    else:
        def _process(frame):
            return apply_filter(
                frame,
                settings["filter_name"],
                blur_radius=settings.get("blur_radius", 5),
            )
        return _process, 1.0


def render_video_preview(original_path: str, output_path: str):
    """Tampilkan video 'Sebelum' dan 'Sesudah' bersebelahan."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎞️ Sebelum")
        st.video(original_path)
    with col2:
        st.subheader("✨ Sesudah")
        st.video(output_path)


def render_download_button(output_path: str):
    """Tombol untuk mengunduh video hasil edit."""
    with open(output_path, "rb") as f:
        video_bytes = f.read()
    st.download_button(
        label="⬇️ Unduh Video Hasil Edit",
        data=video_bytes,
        file_name="hasil_edit.mp4",
        mime="video/mp4",
        use_container_width=True,
    )


def render_empty_state():
    """Tampilan ketika belum ada video yang diunggah."""
    st.info("👈 Silakan unggah video terlebih dahulu melalui menu di sidebar untuk memulai.")
    st.markdown(
        """
        **Fitur yang tersedia:**
        - 🔍 **AI HD Enhancer** — tingkatkan ketajaman, kontras, dan resolusi video.
        - 🎨 **Filter Estetik** — Grayscale, Sepia, Pencil Sketch, Blur, Vintage/Warm,
          serta grup filter bergaya **iPhone** (Vivid, Dramatic, Mono, Silvertone, Noir, dst).
        """
    )


# =====================================================================
# BAGIAN 4: MAIN APP
# =====================================================================


def main():
    setup_page()
    uploaded_file, settings = render_sidebar()

    if uploaded_file is None:
        render_empty_state()
        return

    # Simpan file yang diunggah ke file sementara (VideoCapture butuh path file, bukan buffer)
    temp_dir = tempfile.mkdtemp()
    raw_input_path = os.path.join(temp_dir, "raw_input" + os.path.splitext(uploaded_file.name)[1])
    input_path = os.path.join(temp_dir, "normalized_input.mp4")
    output_path = os.path.join(temp_dir, "output_video.mp4")

    with open(raw_input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Normalisasi video via FFmpeg agar format MOV/HEVC dari iPhone (dan
    # rotasi metadata-nya) terbaca dengan benar oleh OpenCV.
    with st.spinner("Menyiapkan video (menormalkan format & rotasi)..."):
        converted = normalize_video_with_ffmpeg(raw_input_path, input_path)

    if not converted:
        # Fallback: FFmpeg tidak tersedia atau konversi gagal -> coba file asli langsung.
        st.warning(
            "⚠️ FFmpeg tidak tersedia/gagal melakukan normalisasi. Mencoba membaca "
            "file asli secara langsung — untuk video iPhone (.mov/HEVC), ini bisa "
            "gagal atau menghasilkan orientasi yang salah. Pastikan `packages.txt` "
            "berisi `ffmpeg` sudah ada di repo Anda.",
            icon="⚠️",
        )
        input_path = raw_input_path

    info = get_video_info(input_path)
    if info["frame_count"] == 0 or info["width"] == 0:
        st.error(
            "❌ Video tidak dapat dibaca. Format file mungkin tidak didukung. "
            "Coba unggah dalam format MP4 (H.264), atau pastikan FFmpeg terinstal "
            "di server (lihat `packages.txt`)."
        )
        return
    st.markdown(
        f"**Info Video:** {info['width']}x{info['height']} px · "
        f"{info['fps']:.1f} FPS · {info['frame_count']} frame · "
        f"~{info['duration']:.1f} detik"
    )

    process_btn = st.button("🚀 Proses Video", type="primary", use_container_width=True)

    if process_btn:
        # Batasi jumlah frame yang diproses jika pengguna mengatur batas durasi
        max_duration = settings.get("max_duration", 0)
        if max_duration and max_duration > 0:
            max_frames = int(max_duration * info["fps"])
        else:
            max_frames = info["frame_count"]

        frame_processor, scale = build_frame_processor(settings)

        progress_bar = st.progress(0, text="Memulai proses video...")
        start_time = time.time()

        def _update_progress(current, total):
            total_capped = min(total, max_frames) if max_frames else total
            pct = min(current / total_capped, 1.0) if total_capped else 0
            elapsed = time.time() - start_time
            progress_bar.progress(
                pct, text=f"Memproses frame {current}/{total_capped} ({elapsed:.1f}s)"
            )

        # Proses video dengan batas frame (jika diatur) melalui wrapper sederhana
        _process_video_with_limit(
            input_path=input_path,
            output_path=output_path,
            process_frame_fn=frame_processor,
            output_scale=scale,
            max_frames=max_frames,
            progress_callback=_update_progress,
        )

        progress_bar.progress(1.0, text="Selesai!")
        st.success(f"✅ Video berhasil diproses dalam {time.time() - start_time:.1f} detik.")

        render_video_preview(input_path, output_path)
        st.markdown("---")
        render_download_button(output_path)


def _process_video_with_limit(
    input_path, output_path, process_frame_fn, output_scale, max_frames, progress_callback
):
    """Wrapper di sekitar process_video() yang menghormati batas jumlah frame (max_frames)."""
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(input_path)

    out_width = int(info["width"] * output_scale)
    out_height = int(info["height"] * output_scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, info["fps"], (out_width, out_height))

    frame_idx = 0
    limit = max_frames if max_frames else info["frame_count"]

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame_fn(frame)
        if (processed.shape[1], processed.shape[0]) != (out_width, out_height):
            processed = cv2.resize(processed, (out_width, out_height))

        writer.write(processed)
        frame_idx += 1

        if progress_callback:
            progress_callback(frame_idx, info["frame_count"])

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
