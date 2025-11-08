## Developer: inkbytefo
## Modified: 2025-11-08
# Gerekli Brian2 bileşenlerini içe aktaralım
from brian2 import NeuronGroup, ms, pA

# Nöronumuzun davranışını tanımlayan diferansiyel denklem.
# Bu, Leaky Integrate-and-Fire (LIF) modelinin temelidir.
LIF_EQUATION = '''
dv/dt = (I - v) / tau : 1 (unless refractory)
I : 1
tau : second
'''

# Adaptif eşikli LIF denklemi - spike-rate adaptation için
LIF_ADAPTIVE_EQUATION = '''
dv/dt = (I - v) / tau : 1 (unless refractory)
dtheta/dt = -theta / tau_theta : 1  # Adaptif eşiğin zamanla sıfıra dönmesi
I : 1
tau : second
tau_theta : second # Adaptasyon zaman sabitesi
'''

class BionicNeuron:
    """
    PSINet mimarisinin temel işlem birimi.
    
    Brian2 kütüphanesini kullanarak bir Leaky Integrate-and-Fire (LIF)
    nöron grubunu sarmalar (encapsulates) with spike-rate adaptation.
    """
    def __init__(self, num_neurons, tau=10*ms, threshold_initial=1.0, reset_v=0.0, refractory=5*ms,
                 tau_theta=100*ms, delta_theta=0.1):
        """
        Bir grup Biyonik Nöron oluşturur.

        Args:
            num_neurons (int): Bu grupta oluşturulacak nöron sayısı.
            tau (Quantity): Membran potansiyeli sızıntı zaman sabitesi.
            threshold_initial (float): Nöronun temel ateşleme eşiği.
            reset_v (float): Ateşlemeden sonra potansiyelin sıfırlanacağı değer.
            refractory (Quantity): Ateşlemeden sonraki tepkisizlik süresi.
            tau_theta (Quantity): Adaptif eşiğin sıfıra dönme zaman sabitesi.
            delta_theta (float): Her ateşlemede eşiğe eklenecek miktar.
        """
        # Brian2'nin NeuronGroup'unu kullanarak nöronları yaratıyoruz.
        # Brian2 bu denklemleri alıp arka planda verimli C++ koduna çevirir.
        self.group = NeuronGroup(
            num_neurons,
            model=LIF_ADAPTIVE_EQUATION,
            threshold=f'v > (theta + {threshold_initial})', # Eşik artık dinamik
            reset=f'v = {reset_v}; theta += {delta_theta}', # Ateşlemede eşiği artır
            refractory=refractory,
            method='exact' # Bu basit denklem için en doğru çözücü
        )
        
        # Nöronların parametrelerini ayarlıyoruz
        self.group.tau = tau
        self.group.tau_theta = tau_theta
        self.group.theta = 0  # Başlangıçta adaptasyon sıfır
        self.num_neurons = num_neurons

    def __repr__(self):
        return f"BionicNeuron(N={self.num_neurons})"