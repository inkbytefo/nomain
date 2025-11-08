from brian2 import ms, Synapses
from ..core.neuron import BionicNeuron
from ..core.synapse import BionicSynapse

class BionicColumn:
    """
    PSINet'in temel hesaplama birimi.
    
    Uyarıcı (excitatory) ve engelleyici (inhibitory) nöron popülasyonlarını
    ve aralarındaki bağlantıları içerir. "Winner-Take-All" benzeri
    bir rekabetçi dinamik oluşturur.
    """
    def __init__(self, num_excitatory=80, num_inhibitory=20):
        """
        Bir Biyonik Sütun oluşturur.

        Args:
            num_excitatory (int): Uyarıcı nöron sayısı.
            num_inhibitory (int): Engelleyici nöron sayısı.
        """
        print(f"BionicColumn oluşturuluyor: {num_excitatory} Uyarıcı, {num_inhibitory} Engelleyici...")
        
        # 1. Nöron Popülasyonlarını Oluştur
        # Engelleyici nöronlar genellikle daha hızlı tepki verir (daha küçük tau)
        self.excitatory_neurons = BionicNeuron(num_neurons=num_excitatory, tau=10*ms)
        self.inhibitory_neurons = BionicNeuron(num_neurons=num_inhibitory, tau=5*ms)
        
        # 2. İç Bağlantıları Kur (Sinapslar)
        # Kendi kendine öğrenme için uyarıcı nöronlar arasında STDP sinapsları
        self.E_to_E_synapse = BionicSynapse(self.excitatory_neurons, self.excitatory_neurons)
        # Başlangıçta zayıf bağlantılar
        self.E_to_E_synapse.synapses.w = 'rand() * 0.1' 
        
        # Uyarıcı -> Engelleyici bağlantısı (Rekabeti başlatır)
        # Bu bağlantı öğrenmez, sabittir.
        self.E_to_I_synapse = BionicSynapse(self.excitatory_neurons, self.inhibitory_neurons)
        self.E_to_I_synapse.synapses.connect() # Her uyarıcı her engelleyiciyi besler
        self.E_to_I_synapse.synapses.w = 2.0 # Daha güçlü bağlantı

        # Engelleyici -> Uyarıcı bağlantısı (Rekabeti uygular)
        # Bu, "Winner-Take-All" mekanizmasının anahtarıdır.
        # Bu da öğrenmez.
        self.I_to_E_synapse = BionicSynapse(self.inhibitory_neurons, self.excitatory_neurons)
        self.I_to_E_synapse.synapses.connect() # Her engelleyici her uyarıcıyı baskılar
        self.I_to_E_synapse.synapses.w = -3.0 # Çok güçlü, negatif (baskılayıcı) bağlantı

        # L1 içi zayıf lateral engelleme: Uyarıcılar birbirini hafifçe baskılar (soft WTA)
        self.E_lateral_inhib = Synapses(self.excitatory_neurons.group,
                                        self.excitatory_neurons.group,
                                        on_pre='v_post -= 0.2')
        self.E_lateral_inhib.connect(condition='i != j')

        print("BionicColumn oluşturuldu.")

    @property
    def all_objects(self):
        """
        Brian2 simülatörünün bilmesi gereken tüm bileşenleri döndürür.
        """
        return [
            self.excitatory_neurons.group,
            self.inhibitory_neurons.group,
            self.E_to_E_synapse.synapses,
            self.E_to_I_synapse.synapses,
            self.I_to_E_synapse.synapses,
            self.E_lateral_inhib
        ]