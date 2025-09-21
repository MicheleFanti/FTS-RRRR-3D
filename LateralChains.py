class SideChain:
    _residue_props = {
        'G': {'length': 0, 'terminal': 'none'},
        'Q': {'length': 3, 'terminal': 'both'},
        'K': {'length': 4, 'terminal': 'donor'},
        'L': {'length': 3, 'terminal': 'none'},
        'V': {'length': 2, 'terminal': 'none'},
        'F': {'length': 3, 'terminal': 'none'}, #Propriamente aromatico...
        'A': {'length': 1, 'terminal': 'none'},
        'E': {'length': 3, 'terminal': 'acceptor'},
        'S': {'length': 2, 'terminal': 'both'},
        'Y': {'length': 3, 'terminal': 'both'},
        'N': {'length': 2, 'terminal': 'both'}
    }

    def __init__(self, name):
        self.name = name
        props = self._residue_props.get(name)
        if props is None:
            raise ValueError(f"Residuo {name} non definito")
        self.length = props['length']
        self.terminal = props['terminal']