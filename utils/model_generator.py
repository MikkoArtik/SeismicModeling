import json
import os
from typing import List, NamedTuple


class Edge(NamedTuple):
    depth: float
    vp: float


class Layer(NamedTuple):
    top: float
    bottom: float
    vp: float

    @property
    def vs(self) -> float:
        return self.vp / 2

    @property
    def density(self) -> float:
        return 1741 * (self.vp / 1000) ** 0.25

    def dict_conversion(self) -> dict:
        return {
            'top': self.top, 'bottom': self.bottom, 'v_p': self.vp,
            'v_s': self.vs, 'density': self.density
        }


def parse_file(path: str, h_column: int, v_column: int,
               skip_rows=0, delimiter='\t') -> List[Edge]:
    result = []
    with open(path) as file_ctx:
        for _ in range(skip_rows):
            next(file_ctx)
        for line in file_ctx:
            line = line.rstrip()
            if not line:
                continue
            split_line = line.split(delimiter)
            h_val = float(split_line[h_column])
            v_val = float(split_line[v_column])
            result.append(Edge(h_val, v_val))
    result.sort(key=lambda x: x.depth)
    return result


def create_layers_from_edges(edges: List[Edge]) -> List[Layer]:
    layers = []
    for i in range(len(edges) - 1):
        top, bottom = edges[i].depth, edges[i + 1].depth
        vp = edges[i].vp
        layers.append(Layer(top, bottom, vp))
    return layers


def convert_layers_to_dict(layers: List[Layer]) -> dict:
    dict_layer = []
    for layer in layers:
        layer_dict = layer.dict_conversion()
        layer_dict['density'] = round(layer_dict['density'] / 1000, 2)
        dict_layer.append(layer_dict)
    return {'model': dict_layer}


if __name__ == '__main__':
    root = '/media/michael/Data/Projects/Ulyanovskoye_deposit/Modeling/VSP_Well_1617'
    model_file = 'VSP_report.txt'
    export_file = 'VSP_report-tmp.json'

    path = os.path.join(root, model_file)
    edges = parse_file(path, 1, 5, 1, '\t')
    layers = create_layers_from_edges(edges)
    dict_val = convert_layers_to_dict(layers)
    with open(os.path.join(root, export_file), 'w') as file_ctx:
        file_ctx.write(json.dumps(dict_val, indent=4))

