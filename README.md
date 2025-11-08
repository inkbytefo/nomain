## Developer: inkbytefo
## Modified: 2025-11-08

# PSINet - Biologically Inspired Neural Network Framework

🧠 **PSINet** (Plasticity-based Spiking Intelligence Network), biyolojik beyin işleyişinden ilham alan, spike-timing dependent plasticity (STDP) tabanlı bir yapay sinir ağı framework'üdür.

## 🎯 Proje Hedefi

PSINet, geleneksel yapay sinir ağlarının aksine, gerçek nöronların çalışma prensiplerini taklit eder:
- **Spike-based İletişim**: Nöronlar binary spike'lar ile iletişim kurar
- **Temporal Dynamics**: Zamansal dinamikler ve timing kritik öneme sahiptir  
- **STDP Öğrenme**: "Birlikte ateşleyen nöronlar birlikte bağlanır" prensibi
- **Winner-Take-All**: Rekabetçi öğrenme mekanizmaları
- **Hiyerarşik İşleme**: Kortikal sütun benzeri yapılar

## 🏗️ Mimari

```
PSINet/
├── psinet/              # Ana kütüphane
│   ├── core/            # Temel bileşenler
│   │   ├── neuron.py    # BionicNeuron sınıfı
│   │   ├── synapse.py   # BionicSynapse (STDP öğrenme)
│   │   └── learning_rules.py # Öğrenme algoritmaları
│   ├── network/         # Ağ yapıları
│   │   ├── column.py    # BionicColumn (Winner-Take-All)
│   │   └── hierarchy.py # Hiyerarşik ağ yapıları
│   ├── io/              # Girdi/Çıktı işleme
│   │   ├── encoders.py  # Görüntü → Spike dönüştürücüler
│   │   └── loaders.py   # MNIST veri yükleme
│   └── simulation/      # Simülasyon motoru
│       └── simulator.py # Ana Simulator sınıfı
├── configs/             # Konfigürasyon dosyaları
│   ├── mnist_deep_hierarchy.yaml
│   ├── mnist_deep_performance.yaml
│   └── README.md
├── experiments/         # Simülasyon çalıştırıcı
│   └── run_simulation.py
└── outputs/             # Simülasyon sonuçları
```

## 🚀 Özellikler

### ✅ Tamamlanan Bileşenler

- **BionicNeuron**: Leaky Integrate-and-Fire modeli ile gerçekçi nöron davranışı
- **BionicSynapse**: STDP tabanlı öğrenme ile adaptif bağlantılar
- **BionicColumn**: Winner-Take-All mekanizması ile rekabetçi öğrenme
- **Hierarchy**: Çok katmanlı hiyerarşik ağ yapıları
- **Simulator**: YAML tabanlı konfigürasyon ile tam simülasyon kontrolü
- **Görsel Kodlama**: Statik görüntüleri spike dizilerine dönüştürme
- **L2 Selectivity Analysis**: Derin katman nöron seçicilik analizi

### 🎯 Test Edilen Yetenekler

1. **Nöron Dinamikleri**: Gerçekçi ateşleme davranışları
2. **STDP Öğrenme**: Zamansal korelasyon tabanlı öğrenme
3. **Winner-Take-All**: Gürültüden sinyal ayırma
4. **Görsel İşleme**: MNIST rakamlarını spike dizilerine dönüştürme
5. **Derin Hiyerarşi**: Çok katmanlı öğrenme ve analiz

## 🧪 Simülasyonlar

### Konfigürasyon Tabanlı Simülasyon Sistemi

PSINet artık YAML konfigürasyon dosyaları üzerinden çalışan modern bir simülasyon sistemine sahiptir:

#### 1. Temel Hiyerarşi Simülasyonu (`configs/mnist_deep_hierarchy.yaml`)
- **Cihaz**: Runtime (hızlı test için)
- **Süre**: 3 döngü, 250ms per rakam
- **Ağ**: 2 katman (L1: 100 nöron, L2: 50 nöron)
- **Amaç**: Temel öğrenme dinamiklerini test etme

#### 2. Performans Simülasyonu (`configs/mnist_deep_performance.yaml`)
- **Cihaz**: cpp_standalone (yüksek performans)
- **Süre**: 20 döngü, 200ms per rakam
- **Ağ**: 2 katman (L1: 100 nöron, L2: 50 nöron)
- **STDP**: Tam konfigurasyonlu zaman sabitleri
- **Amaç**: Uzun süreli bilimsel deneyler

### Simülasyon Çalıştırma

```bash
# Temel hiyerarşi simülasyonu
cd experiments
python run_simulation.py ../configs/mnist_deep_hierarchy.yaml

# Performans simülasyonu (cpp_standalone)
python run_simulation.py ../configs/mnist_deep_performance.yaml
```

### Çıktılar

Her simülasyon şu dosyaları üretir:
- **`final_plot.png`**: L1/L2 spike raster'ları, ağırlık dinamikleri, L2 seçicilik analizi
- **`raw_data.npz`**: Tüm spike zamanları, ağırlık verileri, analiz pencereleri
- **Log dosyaları**: Detaylı simülasyon ilerlemesi

## 📊 L2 Nöron Seçicilik Analizi

Yeni eklenen özellik ile L2 katmanındaki her nöronun rakam tercihleri analiz edilir:

- **Bar Chart**: Her L2 nöronunun tercih ettiği rakam (renk kodlu)
- **İstatistikler**: Rakam uzmanı nöron dağılımı
- **Görselleştirme**: tab10 colormap ile rakam-renk eşleştirme

Bu analiz, derin öğrenmenin nöron düzeyinde nasıl özelleştiğini gösterir.

## 🛠️ Kurulum

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# Projeyi klonla
git clone https://github.com/inkbytefo/PSINet.git
cd PSINet

# Simülasyon çalıştır
cd experiments
python run_simulation.py ../configs/mnist_deep_hierarchy.yaml
```

## 🔬 Kullanım Örneği

### YAML Konfigürasyonu ile Simülasyon

```yaml
# configs/mnist_deep_performance.yaml
run_id: mnist_deep_performance_v1

simulation_params:
  brian2_device: cpp_standalone
  duration_per_pattern_ms: 200
  silence_period_ms: 100
  cycles: 20
  present_all_digits: true

network_params:
  layers:
    - name: L1
      num_excitatory: 100
      num_inhibitory: 25
      enable_lateral_inhibition: true
      lateral_strength: 0.2
    - name: L2
      num_excitatory: 50
      num_inhibitory: 12
      enable_lateral_inhibition: true
      lateral_strength: 0.3

connections_params:
  inp_l1:
    w_max: 0.3
    a_plus: 0.01
    a_minus: -0.01
    tau_plus_ms: 20.0
    tau_minus_ms: 20.0
  l1_l2:
    w_max: 0.5
    a_plus: 0.01
    a_minus: -0.01
    tau_plus_ms: 20.0
    tau_minus_ms: 20.0
```

### Programatik Kullanım

```python
from psinet.simulation.simulator import Simulator

# Simülasyon oluştur ve çalıştır
sim = Simulator('configs/mnist_deep_performance.yaml')
sim.build()
sim.run()
sim.save_results()
```

## 🎯 Gelecek Planları

- [x] **Çok Katmanlı Hiyerarşi**: Derin kortikal ağ yapıları ✅
- [x] **L2 Selectivity Analysis**: Nöron uzmanlık analizi ✅
- [x] **Performance Optimization**: cpp_standalone desteği ✅
- [ ] **Dikkat Mekanizması**: Odaklanma ve filtreleme
- [ ] **Hafıza Sistemleri**: Hippocampus benzeri yapılar  
- [ ] **Desen Tanıma**: Karmaşık görsel desen öğrenme
- [ ] **Reinforcement Learning**: Ödül tabanlı öğrenme

## 📚 Teorik Temeller

PSINet, aşağıdaki nörobiyoloji prensiplerini uygular:

- **Hebb Kuralı**: "Cells that fire together, wire together"
- **Spike-Timing Dependent Plasticity (STDP)**: Zamansal korelasyon öğrenme
- **Lateral Inhibition**: Rekabetçi dinamikler
- **Cortical Columns**: Modüler işleme birimleri
- **Hierarchical Processing**: Aşamalı bilgi soyutlama

## 🤝 Katkıda Bulunma

PSINet açık kaynak bir projedir. Katkılarınızı bekliyoruz!

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

## 🙏 Teşekkürler

- **Brian2**: Spiking neural network simülasyonu
- **NumPy & Matplotlib**: Bilimsel hesaplama ve görselleştirme
- **MNIST**: Test veri seti

---

**PSINet - Beynin sırlarını çözmek için bir adım** 🧠✨