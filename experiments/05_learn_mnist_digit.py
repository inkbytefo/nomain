import brian2 as b2
import matplotlib
matplotlib.use('Agg')  # GUI olmayan backend
import matplotlib.pyplot as plt
import numpy as np
from psinet.io.encoders import image_to_poisson_rates, create_input_layer
from psinet.network.hierarchy import SimpleHierarchy

# Test görüntüsü oluşturma fonksiyonları

def create_digit_zero():
    """Basit bir '0' rakamı benzeri test görüntüsü oluşturur"""
    image = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            dist = np.sqrt((i - 14)**2 + (j - 14)**2)
            if 8 <= dist <= 12:
                image[i, j] = 255
    return image.astype(np.uint8)


def create_digit_one():
    """Basit bir '1' rakamı benzeri test görüntüsü oluşturur"""
    image = np.zeros((28, 28), dtype=np.uint8)
    image[:, 14:17] = 255  # Orta sütunda kalın bir dikey çizgi
    image[4:8, 12:20] = 255  # Üst başlık
    return image

print("🧠 PSINet - Çoklu Desen Öğrenme ve Ayrım Testi Başlatılıyor...")

# Brian2'nin uyarılarını azalt
b2.prefs.codegen.target = 'numpy'

# --- 1. Veri ve Ağ Kurulumu ---
print("📸 Test görüntüleri oluşturuluyor...")
image_zero = create_digit_zero()
image_one = create_digit_one()

rates_zero = image_to_poisson_rates(image_zero, max_rate=120*b2.Hz)
rates_one = image_to_poisson_rates(image_one, max_rate=120*b2.Hz)

# Başlangıçta sessizlik
input_layer = create_input_layer(np.zeros_like(rates_zero))

# Öğrenme AKTİF olarak hiyerarşiyi oluştur
print("\n🎯 Öğrenme aktif hiyerarşi oluşturuluyor...")
network_hierarchy = SimpleHierarchy(input_layer, num_excitatory=100, num_inhibitory=25, enable_learning=True)

# --- 2. İzleyicileri Ayarlama ---
print("📊 İzleyiciler ayarlanıyor...")
l1_exc_monitor = b2.SpikeMonitor(network_hierarchy.layer1.excitatory_neurons.group)

# --- 3. Dinamik Girdi Döngüsü ile Simülasyonu Çalıştırma ---
show_time = 1.0 * b2.second
rest_time = 100 * b2.ms
cycles = 5

net = network_hierarchy.build_network(l1_exc_monitor)

zero_windows = []
one_windows = []

print(f"\n⏱️  Döngü: {cycles}x [ '0' ({show_time}), dinlen ({rest_time}), '1' ({show_time}), dinlen ({rest_time}) ]")
current_t = 0 * b2.second
for c in range(cycles):
    # '0' göster
    network_hierarchy.input_layer.rates = rates_zero
    net.run(show_time, report='text')
    zero_windows.append((current_t, current_t + show_time))
    current_t += show_time

    # Dinlen
    network_hierarchy.input_layer.rates = 0 * b2.Hz
    net.run(rest_time)
    current_t += rest_time

    # '1' göster
    network_hierarchy.input_layer.rates = rates_one
    net.run(show_time, report='text')
    one_windows.append((current_t, current_t + show_time))
    current_t += show_time

    # Dinlen
    network_hierarchy.input_layer.rates = 0 * b2.Hz
    net.run(rest_time)
    current_t += rest_time

print("🎉 Simülasyon tamamlandı! PSINet çoklu desenlerle eğitildi!")

# --- 4. Sonuçları Görselleştirme ve Analiz ---
print("📈 Sonuçlar görselleştiriliyor ve analiz ediliyor...")
N = 100  # L1 uyarıcı nöron sayısı
counts_zero = np.zeros(N, dtype=int)
counts_one = np.zeros(N, dtype=int)

# Yardımcı: Belirli zaman aralığındaki spike'ları say
spike_t = l1_exc_monitor.t
spike_i = l1_exc_monitor.i

for t0, t1 in zero_windows:
    mask = (spike_t >= t0) & (spike_t < t1)
    if np.any(mask):
        idx, cnt = np.unique(spike_i[mask], return_counts=True)
        counts_zero[idx] += cnt

for t0, t1 in one_windows:
    mask = (spike_t >= t0) & (spike_t < t1)
    if np.any(mask):
        idx, cnt = np.unique(spike_i[mask], return_counts=True)
        counts_one[idx] += cnt

preference = counts_zero - counts_one  # >0 ise '0' tercihi, <0 ise '1' tercihi

# Uzmanlaşmış nöron gruplarını belirle (en belirgin 50'şer)
zero_specialists = np.argsort(-preference)[:50]
one_specialists = np.argsort(preference)[:50]

# Özet metrikler
total_spikes = len(spike_t)
active_neurons = len(np.unique(spike_i))
zero_total = counts_zero.sum()
one_total = counts_one.sum()

print("\n📊 ÖĞRENME ANALİZİ:")
print("=" * 50)
print(f"🔥 Toplam L1 ateşleme: {total_spikes:,}")
print(f"🧠 Aktif nöron sayısı: {active_neurons}/{N} (%{active_neurons/N*100:.1f})")
print(f"'0' pencerelerinde ateşleme toplamı: {zero_total}")
print(f"'1' pencerelerinde ateşleme toplamı: {one_total}")
print(f"'0' uzmanları (ilk 5): {zero_specialists[:5].tolist()}")
print(f"'1' uzmanları (ilk 5): {one_specialists[:5].tolist()}")

# Görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Test görüntüleri
axes[0, 0].imshow(image_zero, cmap='gray')
axes[0, 0].set_title('Girdi Örneği: "0"')
axes[0, 0].axis('off')
axes[0, 1].imshow(image_one, cmap='gray')
axes[0, 1].set_title('Girdi Örneği: "1"')
axes[0, 1].axis('off')

# L1 raster
axes[1, 0].plot(spike_t / b2.ms, spike_i, '.k', markersize=1)
axes[1, 0].set_title('L1 Uyarıcı Nöron Ateşlemeleri (Tüm Süre)')
axes[1, 0].set_xlabel('Zaman (ms)')
axes[1, 0].set_ylabel('Nöron İndisi')
axes[1, 0].grid(True, alpha=0.3)

# Tercih/uzmanlaşma grafiği
colors = np.where(preference >= 0, 'tab:red', 'tab:blue')
axes[1, 1].bar(np.arange(N), preference, color=colors, width=0.9)
axes[1, 1].set_title('Nöron Bazlı Tercih (Pozitif: "0", Negatif: "1")')
axes[1, 1].set_xlabel('Nöron İndisi')
axes[1, 1].set_ylabel('Tercih Skoru (spike_0 - spike_1)')
axes[1, 1].axhline(0, color='k', linewidth=0.8)

plt.tight_layout()
plt.suptitle('🎯 PSINet Çoklu-Desen Öğrenme ve Ayrım (L1 Uzmanlaşma)', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('multi_digit_learning_results.png', dpi=150, bbox_inches='tight')
print("Grafik 'multi_digit_learning_results.png' dosyasına kaydedildi.")