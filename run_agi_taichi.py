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
        # DEĞİŞİKLİK: Kullanıcı girdisi için yeni kuyruk
        self.user_input_queue = queue.Queue(maxsize=5)
        self.lock = threading.Lock()
        self.dopamine = 1.0

def sensory_thread(shared: SharedState, enable_video=True, enable_audio=True):
    """Sensör verilerini (kamera/mikrofon veya simülasyon) işler ve ana kuyruğa yazar."""
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

# DEĞİŞİKLİK: dialogue_thread artık ttc nesnesini alıyor ve spike verisi üretiyor
def dialogue_thread(shared: SharedState, ttc: TextToConcepts):
    """Kullanıcı girdisini alır, spike verisine çevirir ve dopamini günceller."""
    logger.info("Dialogue thread hazır. Konuşmaya başlayabilirsiniz.")
    
    while shared.running:
        try:
            text = input("> ")
            if text.lower() in ["quit", "exit", "q"]:
                shared.running = False
                break

            # Adım 1: Metni spike verisine çevir ve kuyruğa koy
            spike_data = ttc.text_to_spike_data(text, current_time=time.time())
            if spike_data:
                try:
                    shared.user_input_queue.put(spike_data, timeout=0.1)
                except queue.Full:
                    logger.warning("Kullanıcı girdi kuyruğu dolu, girdi işlenemedi.")

            # Adım 2: Dopamini ayarla
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
            shared.running = False
            break
    logger.info("Dialogue thread durduruldu.")

def core_thread(shared: SharedState, pre: BionicNeuron, post: BionicNeuron, syn: BionicSynapse, ttc: TextToConcepts, ctt: ConceptsToText):
    """Ana sinir ağı simülasyonunu Taichi üzerinde çalıştırır."""
    logger.info(f"Core thread: Taichi SNN aktif – {pre.num_neurons + post.num_neurons} nöron")
    
    step_count = 0
    last_report_time = time.time()
    
    def _get_active_concepts_from_spikes(spikes: np.ndarray) -> dict:
        active_indices = np.where(spikes > 0)[0]
        if len(active_indices) == 0:
            return {}
        
        concept_counts = {}
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
        # DEĞİŞİKLİK: Kullanıcı girdisinden gelen spike'ları işlemek için akım vektörü
        language_current = np.zeros(post.num_neurons, dtype=np.float32)
        try:
            spike_indices, _ = shared.user_input_queue.get_nowait()
            # Dil konseptleriyle ilişkili nöronlara güçlü bir akım darbesi uygula
            language_current[spike_indices] = 5.0 
            logger.debug(f"Dil girdisi enjekte ediliyor: {len(spike_indices)} spike.")
        except queue.Empty:
            pass

        for _ in range(10):
            # Sensory input
            try:
                rates = shared.sensory_queue.get_nowait()
                pre.apply_poisson_input(rates)
            except queue.Empty:
                pre.update()

            pre_spikes = pre.get_spikes()
            post_spikes = post.get_spikes()
            
            psc = syn.update(pre_spikes, post_spikes)
            
            # Post-sinaptik akıma dil girdisinden gelen akımı ekle
            psc += language_current
            
            post.update(psc)
            
            # Dil akımını bir sonraki adım için sıfırla (sadece bir anlık darbe)
            language_current.fill(0)
            
            with shared.lock:
                current_dopamine = shared.dopamine
            syn.set_dopamine(current_dopamine)
            
            step_count += 1
        
        if step_count % 200 == 0:
            post_spikes = post.get_spikes()
            concept_activations = _get_active_concepts_from_spikes(post_spikes)
            
            total_activity = np.sum(post_spikes) / post.num_neurons
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
    input_neurons = 12000
    l1_neurons = 10000
    pre = BionicNeuron(input_neurons, dt=0.001, sparsity=0.9)
    post = BionicNeuron(l1_neurons, dt=0.001, sparsity=0.9)
    syn = BionicSynapse(pre, post, sparsity=0.9)
    ttc = TextToConcepts(vocab_size=100, concept_neurons=50)
    ctt = ConceptsToText(ttc, curiosity_threshold=0.8, curiosity_duration=3.0)
    logger.info("Başlatma tamamlandı.")

    core_args = (shared, pre, post, syn, ttc, ctt)
    # DEĞİŞİKLİK: dialogue_thread'e ttc nesnesi veriliyor
    dialogue_args = (shared, ttc)
    
    threads = [
        threading.Thread(target=sensory_thread, name="Sensory", args=(shared, not args.no_camera, not args.no_audio)),
        threading.Thread(target=dialogue_thread, name="Dialogue", args=dialogue_args),
        threading.Thread(target=core_thread, name="Core", args=core_args)
    ]
    
    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nKapanma sinyali alındı. Proto-AGI durduruluyor...")
        shared.running = False
        for t in threads:
            t.join(timeout=2.0)

    print("Proto-AGI başarıyla kapatıldı.")

if __name__ == "__main__":
    main()