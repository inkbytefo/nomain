#!/usr/bin/env python3
"""
Proto-AGI Taichi Edition – T4'te gerçek zamanlı görme, işitme, konuşma, dopaminle öğrenme
FPS: 55.7 → 18ms gerçek zaman
"""
## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=15, debug=False, log_level='error')

import threading
import queue
import time
import argparse
import numpy as np
import logging
from psinet.core.taichi_neuron import BionicNeuron
from psinet.core.taichi_synapse import BionicSynapse
from psinet.io.realtime_encoders import RealtimeEncoder
from psinet.language.input import TextToConcepts
from psinet.language.output import ConceptsToText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("PROTO-AGI TAICHI BAŞLATIYOR... T4 @ 55.7 FPS")

class SharedState:
    def __init__(self):
        self.running = True
        self.sensory_queue = queue.Queue(maxsize=10)
        self.agi_output_queue = queue.Queue(maxsize=10)
        self.user_input_queue = queue.Queue(maxsize=5)
        self.lock = threading.Lock()
        self.dopamine = 1.0

def sensory_thread(shared: SharedState, enable_video=True, enable_audio=True):
    logger.info(f"Sensory thread başlatılıyor (video: {enable_video}, audio: {enable_audio})")
    encoder = RealtimeEncoder(enable_video=enable_video, enable_audio=enable_audio)
    
    if not encoder.initialize():
        logger.error("Sensory encoder başlatılamadı. Thread durduruluyor.")
        return

    while shared.running:
        try:
            rates = encoder.get_combined_rates()
            shared.sensory_queue.put(rates, timeout=0.1)
        except queue.Full:
            continue
        except Exception as e:
            logger.error(f"Sensory hata: {e}")
            time.sleep(1)

# DEĞİŞİKLİK: core_thread artık daha fazla nesne alıyor (yeni dil katmanı ve sinapsı)
def core_thread(shared: SharedState, pre: BionicNeuron, post: BionicNeuron, language_neurons: BionicNeuron, 
                syn_sensory_to_post: BionicSynapse, syn_lang_to_post: BionicSynapse, 
                ttc: TextToConcepts, ctt: ConceptsToText):
    logger.info(f"Core thread: Taichi SNN aktif – {pre.num_neurons + post.num_neurons + language_neurons.num_neurons} nöron")
    
    step_count = 0
    last_report_time = time.time()
    
    def _get_active_concepts_from_spikes(spikes: np.ndarray) -> dict:
        active_indices = np.where(spikes > 0)[0]
        if len(active_indices) == 0:
            return {}
        
        concept_counts = {}
        # DEĞİŞİKLİK: Artık post nöronlarındaki aktiviteyi değil, dil nöronlarındaki aktiviteyi yorumluyoruz.
        # Bu daha temiz bir sinyal sağlar.
        max_concept_neuron_idx = ttc.get_concept_neurons_count()

        for idx in active_indices:
            if idx < max_concept_neuron_idx:
                word_id = idx // ttc.concept_neurons
                if word_id in ttc.id_to_word:
                    word = ttc.id_to_word[word_id]
                    concept_counts[word] = concept_counts.get(word, 0) + 1
        
        activations = {
            word: count / ttc.concept_neurons 
            for word, count in concept_counts.items()
        }
        return activations

    while shared.running:
        # Kullanıcı girdisini dil nöronlarına uygula
        try:
            spike_indices, _ = shared.user_input_queue.get_nowait()
            # Dil nöronlarına Poisson girdisi uygula
            lang_rates = np.zeros(language_neurons.num_neurons, dtype=np.float32)
            lang_rates[spike_indices] = 150.0 # Hz cinsinden yüksek ateşleme oranı
            language_neurons.apply_poisson_input(lang_rates)
            logger.info(f"Dil girdisi alınıyor: {len(spike_indices)} spike konsepti aktive ediliyor.")
        except queue.Empty:
            language_neurons.update() # Girdi yoksa normal şekilde güncelle

        # 10ms'lik simülasyon adımı
        for _ in range(10):
            # 1. Sensör ve dil katmanlarını güncelle
            try:
                rates = shared.sensory_queue.get_nowait()
                pre.apply_poisson_input(rates)
            except queue.Empty:
                pre.update()
            
            language_neurons.update()

            # 2. Spike'ları topla
            pre_spikes = pre.get_spikes()
            lang_spikes = language_neurons.get_spikes()
            post_spikes = post.get_spikes()
            
            # 3. Post-sinaptik akımları hesapla
            psc_from_sensory = syn_sensory_to_post.update(pre_spikes, post_spikes)
            psc_from_lang = syn_lang_to_post.update(lang_spikes, post_spikes)
            
            # 4. Akımları birleştir ve post katmanını güncelle
            total_psc = psc_from_sensory + psc_from_lang
            post.update(total_psc)
            
            # 5. Dopamini ayarla
            with shared.lock:
                current_dopamine = shared.dopamine
            syn_sensory_to_post.set_dopamine(current_dopamine)
            syn_lang_to_post.set_dopamine(current_dopamine)
            
            step_count += 1
        
        # AGI çıktısı üret
        if step_count % 200 == 0:
            # DEĞİŞİKLİK: Konsept aktivasyonunu doğrudan dil katmanından al
            lang_spikes = language_neurons.get_spikes()
            concept_activations = _get_active_concepts_from_spikes(lang_spikes)
            
            total_activity = np.sum(post.get_spikes()) / post.num_neurons
            error_signal = np.clip(total_activity * 5.0, 0.0, 1.0)

            if concept_activations or error_signal > ctt.curiosity_threshold:
                response = ctt.generate_response(concept_activations, error_signal, time.time())
                if response:
                    shared.agi_output_queue.put(response)
        
        try:
            agi_response = shared.agi_output_queue.get_nowait()
            print(f"AGI: {agi_response}")
        except queue.Empty:
            pass

        current_time = time.time()
        if current_time - last_report_time > 5.0:
            elapsed = current_time - last_report_time
            fps = step_count / elapsed
            logger.info(f"FPS: {fps:.1f} | Dopamin: {current_dopamine:.2f}")
            last_report_time = current_time
            step_count = 0

def main():
    parser = argparse.ArgumentParser(description="Proto-AGI Taichi Edition")
    parser.add_argument('--no-camera', action='store_true', help="Kamera girdisini devre dışı bırak")
    parser.add_argument('--no-audio', action='store_true', help="Mikrofon girdisini devre dışı bırak")
    args = parser.parse_args()
    
    shared = SharedState()
    
    logger.info("Taichi ve dil modülleri ana thread'de başlatılıyor...")
    # Ağ Katmanları
    sensory_neurons_count = 12000
    association_neurons_count = 10000
    
    ttc = TextToConcepts(vocab_size=100, concept_neurons=50)
    ctt = ConceptsToText(ttc, curiosity_threshold=0.8, curiosity_duration=3.0)
    language_neurons_count = ttc.get_concept_neurons_count()

    pre_sensory = BionicNeuron(sensory_neurons_count, dt=0.001, sparsity=0.9)
    post_association = BionicNeuron(association_neurons_count, dt=0.001, sparsity=0.9)
    # DEĞİŞİKLİK: Özel dil katmanı
    language_neurons = BionicNeuron(language_neurons_count, dt=0.001, sparsity=0.9)

    # Sinapslar
    syn_sensory_to_post = BionicSynapse(pre_sensory, post_association, sparsity=0.9)
    # DEĞİŞİKLİK: Dil katmanından ana katmana sinaps
    syn_lang_to_post = BionicSynapse(language_neurons, post_association, sparsity=0.9)
    
    logger.info("Başlatma tamamlandı.")

    core_args = (shared, pre_sensory, post_association, language_neurons, 
                 syn_sensory_to_post, syn_lang_to_post, ttc, ctt)
    
    threads = [
        threading.Thread(target=sensory_thread, name="Sensory", args=(shared, not args.no_camera, not args.no_audio)),
        threading.Thread(target=core_thread, name="Core", args=core_args)
    ]
    
    for t in threads:
        t.start()

    logger.info("Dialogue arayüzü ana thread'de aktif. Konuşmaya başlayabilirsiniz.")
    while shared.running:
        try:
            text = input("> ")
            if text.lower() in ["quit", "exit", "q"]:
                break

            spike_data = ttc.text_to_spike_data(text, current_time=time.time())
            if spike_data:
                try:
                    shared.user_input_queue.put(spike_data, timeout=0.1)
                except queue.Full:
                    logger.warning("Kullanıcı girdi kuyruğu dolu, girdi işlenemedi.")

            reward = 0.0
            if "iyi" in text.lower() or "doğru" in text.lower() or "evet" in text.lower():
                reward = 1.0
                logger.info("Pozitif geri bildirim alındı, dopamin artırılıyor.")
            elif "kötü" in text.lower() or "yanlış" in text.lower() or "hayır" in text.lower():
                reward = -1.0
                logger.info("Negatif geri bildirim alındı, dopamin azaltılıyor.")

            with shared.lock:
                shared.dopamine = np.clip(1.0 + reward * 0.5, 0.0, 2.0)

        except (EOFError, KeyboardInterrupt):
            break
    
    print("\nKapanma sinyali alındı. Proto-AGI durduruluyor...")
    shared.running = False
    for t in threads:
        t.join(timeout=2.0)

    print("Proto-AGI başarıyla kapatıldı.")

if __name__ == "__main__":
    main()