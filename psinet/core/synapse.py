## Developer: inkbytefo
## Modified: 2025-11-08
from brian2 import Synapses, ms
from .learning_rules import STDP_EQUATION, STDP_ON_PRE, STDP_ON_POST

class BionicSynapse:
    """
    Nöron grupları arasında öğrenen bağlantıları yönetir.
    """
    def __init__(self, pre_neurons, post_neurons, tau_pre=20*ms, tau_post=20*ms,
                 w_max=0.01, A_pre=0.01, A_post=-0.0105, initial_weight_max=0.01):
        """
        STDP öğrenme kuralına sahip bir sinaps grubu oluşturur.

        Args:
            pre_neurons (BionicNeuron): Kaynak nöron grubu.
            post_neurons (BionicNeuron): Hedef nöron grubu.
            tau_pre/tau_post (Quantity): STDP izlerinin zaman sabiteleri.
            w_max (float): Maksimum sinaptik ağırlık.
            A_pre/A_post (float): Her ateşlemede izlerdeki artış/azalış miktarı.
                                  A_post'un A_pre'den biraz daha negatif olması
            initial_weight_max (float): Başlangıç sinaptik ağırlıklarının maksimum değeri.
                                  genellikle daha kararlı bir öğrenme sağlar.
        """
        pre_group = pre_neurons.group if hasattr(pre_neurons, 'group') else pre_neurons
        post_group = post_neurons.group if hasattr(post_neurons, 'group') else post_neurons
        
        self.synapses = Synapses(
            pre_group, 
            post_group,
            model=STDP_EQUATION,
            on_pre=STDP_ON_PRE,
            on_post=STDP_ON_POST
        )
        
        self.synapses.connect()
        
        # Öğrenme parametrelerini ayarla (defaults; caller may override after object creation)
        self.synapses.taupre = tau_pre
        self.synapses.taupost = tau_post
        self.synapses.wmax = w_max
        self.synapses.Apre = A_pre
        self.synapses.Apost = A_post
        
        self.synapses.w = f'rand() * {initial_weight_max}'
    
    def set_learning_params(self, tau_pre=None, tau_post=None, w_max=None, A_pre=None, A_post=None):
        if tau_pre is not None:
            self.synapses.taupre = tau_pre
        if tau_post is not None:
            self.synapses.taupost = tau_post
        if w_max is not None:
            self.synapses.wmax = w_max
        if A_pre is not None:
            self.synapses.Apre = A_pre
        if A_post is not None:
            self.synapses.Apost = A_post
    
    def __repr__(self):
        return f"BionicSynapse connecting {self.synapses.source.N} to {self.synapses.target.N} neurons."