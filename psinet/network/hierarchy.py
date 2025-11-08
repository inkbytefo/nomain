from .column import BionicColumn
from ..core.synapse import BionicSynapse
from brian2 import Network, ms

class Hierarchy:
    """
    Generic multi-layer hierarchy of BionicColumns with learnable inter-layer connections.
    The first layer receives input from an external input layer (PoissonGroup),
    and each subsequent layer receives from the previous layer's excitatory neurons.
    """
    def __init__(self, input_layer, layers_config, connections_params=None):
        """
        Args:
            input_layer: PoissonGroup providing input spikes.
            layers_config: List of dicts defining each layer. Example item:
                {
                    'name': 'L1',
                    'num_excitatory': 100,
                    'num_inhibitory': 25,
                    'enable_lateral_inhibition': True,
                    'lateral_strength': 0.2,
                }
            connections_params: Dict with STDP params for connections. Keys:
                - 'inp_<first_layer_name_lower>' e.g., 'inp_l1'
                - '<prev>_<curr>' e.g., 'l1_l2'
                Each value: {'w_max': float, 'a_plus': float, 'a_minus': float}
        """
        self.input_layer = input_layer
        self.layers_in_order = []
        self.layers_by_name = {}
        self.connections = {}
        self.input_to_first_syn = None

        # Build layers
        for layer_def in layers_config:
            name = layer_def.get('name', f"L{len(self.layers_in_order)+1}")
            ne = int(layer_def.get('num_excitatory', 100))
            ni = int(layer_def.get('num_inhibitory', 25))
            eli = bool(layer_def.get('enable_lateral_inhibition', True))
            lat = float(layer_def.get('lateral_strength', 0.2))

            print(f"\nKatman oluşturuluyor ({name})...")
            col = BionicColumn(ne, ni, enable_lateral_inhibition=eli, lateral_strength=lat)
            self.layers_by_name[name] = col
            self.layers_in_order.append(name)

        # Build connections (learning-enabled)
        cp = connections_params or {}

        # Input -> first layer
        first = self.layers_in_order[0]
        key_inp = f"inp_{first.lower()}"
        if key_inp not in cp:
            raise ValueError(f"Missing required connection parameters for '{key_inp}' in connections_params")
        p_inp = cp[key_inp]
        print(f"Girdi -> {first} sinapsı (öğrenen) kuruluyor...")
        self.input_to_first_syn = BionicSynapse(
            pre_neurons=self.input_layer,
            post_neurons=self.layers_by_name[first].excitatory_neurons,
            w_max=p_inp['w_max'],
            A_pre=p_inp['a_plus'],
            A_post=p_inp['a_minus'],
            tau_pre=p_inp['tau_plus_ms']*ms,
            tau_post=p_inp['tau_minus_ms']*ms,
        )
        self.connections[key_inp] = self.input_to_first_syn

        # Inter-layer connections
        for prev_name, curr_name in zip(self.layers_in_order[:-1], self.layers_in_order[1:]):
            key = f"{prev_name.lower()}_{curr_name.lower()}"
            if key not in cp:
                raise ValueError(f"Missing required connection parameters for '{key}' in connections_params")
            p = cp[key]
            print(f"{prev_name} -> {curr_name} sinapsı (öğrenen) kuruluyor...")
            syn = BionicSynapse(
                pre_neurons=self.layers_by_name[prev_name].excitatory_neurons,
                post_neurons=self.layers_by_name[curr_name].excitatory_neurons,
                w_max=p['w_max'],
                A_pre=p['a_plus'],
                A_post=p['a_minus'],
                tau_pre=p['tau_plus_ms']*ms,
                tau_post=p['tau_minus_ms']*ms,
            )
            self.connections[key] = syn

    # Backwards compatibility helpers
    @property
    def layer1(self):
        return self.layers_by_name[self.layers_in_order[0]]

    @property
    def input_to_l1_synapse(self):
        return self.input_to_first_syn

    def build_network(self, *monitors):
        """
        Assemble a Brian2 network with all layers, inter-layer synapses, and monitors.
        """
        all_components = [self.input_layer]
        # Include columns
        for name in self.layers_in_order:
            col = self.layers_by_name[name]
            all_components.extend(col.all_objects)
        # Include synapses (BionicSynapse exposes .synapses)
        for syn in self.connections.values():
            all_components.append(syn.synapses)
        # Include monitors
        all_components.extend(monitors)
        return Network(all_components)

# Keep legacy alias for compatibility
SimpleHierarchy = Hierarchy