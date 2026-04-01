from heapq import nsmallest
from itertools import product

from analytical_model.eyeriss import (
    EyerissAnalyzer,
    AnalysisResult,
    EyerissHardwareParam,
    EyerissMappingParam,
    PSUM_DATA_SIZE,
)
from layer_info import Conv2DShapeParam, MaxPool2DShapeParam


class EyerissMapper:
    cnt = 0

    def __init__(
        self,
        name: str | None,
    ) -> None:
        self.name = name if name is not None else f"mapping_{EyerissMapper.cnt}"
        self.analyzer = EyerissAnalyzer(name=self.name)
        EyerissMapper.cnt += 1

    def run(
        self,
        conv2d: Conv2DShapeParam,
        maxpool: MaxPool2DShapeParam | None = None,
        num_solutions: int = 1,
        mode: str | None = None,
    ) -> list[AnalysisResult]:
        self.analyzer.conv_shape = conv2d
        self.analyzer.maxpool_shape = maxpool
        self.analyzer.mode = self.mode = mode
        results = []

        for hardware in self.generate_hardware():
            self.hardware = hardware

            for mapping in self.generate_mappings():
                self.analyzer.mapping = mapping
                res = self.analyzer.summary
                res["m"] = mapping.m
                res["n"] = mapping.n
                res["e"] = mapping.e
                res["p"] = mapping.p
                res["q"] = mapping.q
                res["r"] = mapping.r
                res["t"] = mapping.t
                results.append(res)

        if num_solutions > 0:
            results = nsmallest(num_solutions, results, key=self.evaluate)
        return results

    def evaluate(self, metrics: AnalysisResult) -> float:
        score = 0
        #! <<<========= Implement here =========>>>
        edp = metrics["energy_total"] * metrics["latency"]
        score = edp
        #score = 1.0 / edp
        # score = metrics["latency"] * metrics["power_total"]
        return score

    @property
    def hardware(self) -> EyerissHardwareParam:
        return self.analyzer.hardware

    @hardware.setter
    def hardware(self, hardware_param: EyerissHardwareParam) -> None:
        assert isinstance(hardware_param, EyerissHardwareParam)
        self.analyzer.hardware = hardware_param

    def p_avaliable(self) -> list[int]:
        p_max = self.hardware.psum_spad_size // PSUM_DATA_SIZE
        return list(range(1, p_max + 1))

    def q_avaliable(self) -> list[int]:
        q_max = self.hardware.ifmap_spad_size // self.analyzer.conv_shape.S
        return list(range(1, q_max + 1))

    def e_available(self) -> list[int]:
        hw_strips = self.hardware.pe_array_h // self.analyzer.conv_shape.R
        e_max = self.hardware.pe_array_w * hw_strips
        return list(range(1, min(e_max, self.analyzer.conv_shape.E) + 1))

    def r_available(self) -> list[int]:
        r_max = self.hardware.pe_array_h // self.analyzer.conv_shape.R
        return list(range(1, r_max + 1))

    def t_available(self) -> list[int]:
        num_pes = self.hardware.pe_array_h * self.hardware.pe_array_w
        t_max = num_pes // self.analyzer.conv_shape.R
        return list(range(1, t_max + 1))

    def m_available(self) -> list[int]:
        m_max = self.analyzer.conv_shape.M
        return list(
            m for m in range(1, m_max + 1) if self.analyzer.conv_shape.M % m == 0
        )

    def validate(self, mapping) -> bool:
        m, n, e, p, q, r, t = mapping
        self.analyzer.mapping = EyerissMappingParam(*mapping)

        # pq constraints
        if p * q > self.hardware.filter_spad_size // self.analyzer.conv_shape.S:
            return False

        # e constraints
        if (
            e % self.hardware.pe_array_w != 0
            and e != self.hardware.pe_array_w // 2
            and self.analyzer.conv_shape.E != e
        ):
            return False

        # rt constraints
        if (
            r * t
            != self.hardware.pe_array_h
            * self.hardware.pe_array_w
            // self.analyzer.conv_shape.R
            // e
        ):
            return False

        # m constraints
        if m % p != 0:
            return False

        return self.analyzer.glb_size_legal

    def generate_mappings(self, verbose: bool = False) -> list[EyerissMappingParam]:
        candidate_solutions = []
        #! <<<========= Implement here =========>>>
        n_available_list = [1]
        p_available_list = self.p_avaliable()
        q_available_list = self.q_avaliable()
        e_available_list = self.e_available()
        r_available_list = self.r_available()
        t_available_list = self.t_available()
        m_available_list = self.m_available()

        all_combinations = product(
            m_available_list,
            n_available_list,
            e_available_list,
            p_available_list,
            q_available_list,
            r_available_list,
            t_available_list,
        )
        
        for mapping_tuple in all_combinations:
            if self.validate(mapping_tuple):

                valid_mapping = EyerissMappingParam(*mapping_tuple)
                candidate_solutions.append(valid_mapping)
                    
                if verbose:
                    print(f"Valid mapping found: {valid_mapping}")

                #print(f"{valid_mapping.m}, {valid_mapping.n}, {valid_mapping.e}, {valid_mapping.p}, {valid_mapping.q}, {valid_mapping.r}, {valid_mapping.t}")
        return candidate_solutions

    def generate_hardware(self) -> list[EyerissHardwareParam]:
        candidate_solutions = []
        pe_array_h_list = [6]  # add more values to explore more solutions
        pe_array_w_list = [8]  # add more values to explore more solutions
        ifmap_spad_size_list = [12]
        filter_spad_size_list = [48]
        psum_spad_size_list = [16]
        glb_size_list = [64 * 2**10]
        bus_bw_list = [4]
        noc_bw_list = [4]
        candidate_solutions = product(
            pe_array_h_list,
            pe_array_w_list,
            ifmap_spad_size_list,
            filter_spad_size_list,
            psum_spad_size_list,
            glb_size_list,
            bus_bw_list,
            noc_bw_list,
        )
        candidate_solutions = [EyerissHardwareParam(*m) for m in candidate_solutions]
        return candidate_solutions
