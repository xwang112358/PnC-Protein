class AttrMapping:
    def __init__(self, dataset_name, mapping_type='integer', node_attr_dims=None, edge_attr_dims=None):
        self.node_attr_dims = node_attr_dims
        self.edge_attr_dims = edge_attr_dims
        if dataset_name == 'MUTAG':
            self.node_attr_values = [['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']]
            self.edge_attr_values = [['A', 'S', 'D', 'T']]
        elif dataset_name == 'PTC_MR':
            self.node_attr_values = [['In','P','O','N','Na','C','Cl','S','Br','F','K','Cu','Zn','I','Ba','Sn','Pb','Ca']]
            self.edge_attr_values = [['T', 'D', 'S', 'A']]
            # triple, double, single, aromatic
        elif dataset_name == 'ZINC':
            self.node_attr_values =[['C', 'O', 'N', 'F', 'C H1', 'S', 'Cl', 'O -', 'N H1 +', 'Br', 'N H3 +',
             'N H2 +', 'N +', 'N -', 'S -', 'I', 'P', 'O H1 +', 'N H1 -', 'O +',
             'S +', 'P H1', 'P H2', 'C H2 -', 'P +', 'S H1 +', 'C H1 -', 'P H1 +']]
            self.edge_attr_values = [['N', 'S', 'D', 'T']]
        elif dataset_name == 'proteinshake':
            # 20 standard amino acids in ARNDCEQGHILKMFPSTWYV order
            self.node_attr_values = [['A', 'R', 'N', 'D', 'C', 
                             'E', 'Q', 'G', 'H', 'I', 
                             'L', 'K', 'M', 'F', 'P', 
                             'S', 'T', 'W', 'Y', 'V']]
            # For protein graphs, edges typically represent spatial proximity
            self.edge_attr_values = []  # or [] if no edge attributes
        # else:
        #     raise NotImplementedError
        self.mapping_type = mapping_type
    def map(self, node_features, edge_features=None):
        # print(f'DEBUG: attr_mapping.map called with mapping_type: {self.mapping_type}')
        # print(f'DEBUG: node_features shape: {node_features.shape}')
        
        if self.mapping_type == 'integer':
            node_attrs = node_features.unsqueeze(1) if node_features.dim()==1 else node_features
            if edge_features is not None:
                if edge_features.numel() != 0:
                    edge_attrs = edge_features.unsqueeze(1) if edge_features.dim() == 1 else edge_features
                else:
                    edge_attrs = edge_features
            else:
                edge_attrs = None
            return node_attrs, edge_attrs
        elif self.mapping_type == 'one_hot':
            node_attrs = node_features.argmax(1, keepdim=True)
            # print(f'DEBUG: argmax result: {node_attrs.flatten()}')
            if edge_features is not None:
                if edge_features.numel()!=0:
                    edge_attrs = edge_features.argmax(1, keepdim=True)
                else:
                    edge_attrs = edge_features[:,0:1]
            else:
                edge_attrs = None
            return node_attrs, edge_attrs
        else:
            raise NotImplementedError